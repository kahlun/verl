# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
XPU-compatible attention implementations that replace flash_attn_varlen_func
(and HF's _flash_attention_forward) with torch SDPA for Intel XPU devices.

This enables VLM sequence packing on XPU at the cost of O(n^2) memory per
sub-sequence (vs flash attention's O(n) tiling).  Acceptable because verl
packs at most ~8-16K total tokens per micro-batch.

Key insight: after unpad_input removes padding tokens, every sub-sequence
contains NO all-masked rows, so SDPA is NaN-safe (no softmax(all -inf)).
"""

import torch
import torch.nn.functional as F


def xpu_varlen_sdpa(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=True,
    **kwargs,
):
    """Drop-in replacement for ``flash_attn_varlen_func`` using per-sequence
    ``torch.nn.functional.scaled_dot_product_attention``.

    Args:
        q:  (total_q, nheads, head_dim)
        k:  (total_k, nheads_k, head_dim)
        v:  (total_k, nheads_k, head_dim)
        cu_seqlens_q:  (batch_size + 1,)  cumulative sequence lengths for Q
        cu_seqlens_k:  (batch_size + 1,)  cumulative sequence lengths for K
        max_seqlen_q:  unused, kept for API compatibility
        max_seqlen_k:  unused, kept for API compatibility

    Returns:
        (total_q, nheads, head_dim)
    """
    batch_size = cu_seqlens_q.shape[0] - 1
    output = torch.empty_like(q)
    is_training = q.requires_grad

    for i in range(batch_size):
        q_s, q_e = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        k_s, k_e = cu_seqlens_k[i].item(), cu_seqlens_k[i + 1].item()

        # (seq, nheads, head_dim) -> (1, nheads, seq, head_dim)
        qi = q[q_s:q_e].unsqueeze(0).transpose(1, 2)
        ki = k[k_s:k_e].unsqueeze(0).transpose(1, 2)
        vi = v[k_s:k_e].unsqueeze(0).transpose(1, 2)

        oi = F.scaled_dot_product_attention(
            qi,
            ki,
            vi,
            dropout_p=dropout_p if is_training else 0.0,
            is_causal=causal,
            scale=softmax_scale,
        )  # (1, nheads, seq_q, head_dim)

        # (1, nheads, seq_q, head_dim) -> (seq_q, nheads, head_dim)
        output[q_s:q_e] = oi.squeeze(0).transpose(0, 1)

    return output


def xpu_flash_attention_forward(
    query_states,
    key_states,
    value_states,
    attention_mask,
    query_length,
    is_causal=True,
    sliding_window=None,
    use_top_left_mask=False,
    deterministic=None,
    **kwargs,
):
    """Drop-in replacement for HF's ``_flash_attention_forward`` using SDPA.

    Handles two cases:
    1. Packed sequences (non-monotonic ``position_ids``): routes through
       ``xpu_varlen_sdpa`` for correct per-sub-sequence attention with no
       cross-sequence leakage.
    2. Normal / generation: plain causal SDPA.

    Input layout  : (batch_size, seq_len, num_heads, head_dim)  — FA2 format
    Output layout : (batch_size, seq_len, num_heads, head_dim)
    """
    dropout_p = kwargs.pop("dropout", 0.0)
    softmax_scale = kwargs.pop("softmax_scale", None)
    position_ids = kwargs.pop("position_ids", None)
    kwargs.pop("softcap", None)  # not supported by SDPA

    is_training = query_states.requires_grad

    # --- Packed-sequence path ---------------------------------------------------
    # Detected by non-monotonically-increasing position_ids (position resets to 0
    # at each sub-sequence boundary), same heuristic as HF / qwen2_vl.py.
    if (
        position_ids is not None
        and query_length != 1
        and position_ids.ndim == 2
        and not (torch.diff(position_ids, dim=-1) >= 0).all()
    ):
        batch_size = query_states.size(0)
        # FA2 layout (B, S, H, D) → flat (total, H, D)
        q_flat = query_states.contiguous().view(-1, query_states.size(-2), query_states.size(-1))
        k_flat = key_states.contiguous().view(-1, key_states.size(-2), key_states.size(-1))
        v_flat = value_states.contiguous().view(-1, value_states.size(-2), value_states.size(-1))

        pos_flat = position_ids.view(-1)
        cu_seqlens = torch.cat([
            (pos_flat == 0).nonzero().view(-1).to(torch.int32),
            torch.tensor([pos_flat.size(0)], device=pos_flat.device, dtype=torch.int32),
        ])
        max_seqlen = cu_seqlens.diff().max()

        attn_out = xpu_varlen_sdpa(
            q_flat, k_flat, v_flat,
            cu_seqlens, cu_seqlens,
            max_seqlen, max_seqlen,
            dropout_p=dropout_p if is_training else 0.0,
            softmax_scale=softmax_scale,
            causal=is_causal,
        )
        # (total, H, D) → (B, S, H, D)
        return attn_out.view(batch_size, -1, attn_out.size(-2), attn_out.size(-1))

    # --- Normal path -----------------------------------------------------------
    # (B, S, H, D) -> (B, H, S, D)
    q = query_states.transpose(1, 2)
    k = key_states.transpose(1, 2)
    v = value_states.transpose(1, 2)

    # SDPA raises if both is_causal and attn_mask are set
    attn_mask = None if is_causal else attention_mask

    attn_output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p if is_training else 0.0,
        is_causal=is_causal,
        scale=softmax_scale,
    )

    # (B, H, S, D) -> (B, S, H, D)
    return attn_output.transpose(1, 2)
