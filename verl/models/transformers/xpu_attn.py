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
XPU-compatible attention implementations that replace ``flash_attn_varlen_func``
(and HF's ``_flash_attention_forward``) with ``torch.nn.functional.scaled_dot_product_attention``
for Intel XPU devices.

This enables VLM sequence packing on XPU at the cost of O(n²) memory per
sub-sequence (vs flash attention's O(n) tiling).  Acceptable because verl
packs at most ~8-16K total tokens per micro-batch.

Key insight: after ``unpad_input`` removes padding tokens, every sub-sequence
contains NO all-masked rows, so SDPA is NaN-safe (no ``softmax(all -inf)``).

Reference implementations:
  - ``flash_attn.flash_attn_interface.flash_attn_varlen_func``
    https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py
  - ``transformers.modeling_flash_attention_utils._flash_attention_forward``
    https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_flash_attention_utils.py
  - ``transformers.modeling_flash_attention_utils.prepare_fa_kwargs_from_position_ids``
    (cu_seqlens construction from position_ids — same ``(pos == 0).nonzero()`` approach)
"""

import warnings

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sliding-window mask helper
# ---------------------------------------------------------------------------

def _build_window_mask(seq_len_q, seq_len_k, window_size, causal, device, dtype):
    """Build a 2-D additive attention mask combining causality and sliding window.

    Follows the same ``window_size`` semantics as ``flash_attn``:
    query at position *i* attends to keys in
    ``[max(0, i - left), min(seqlen_k - 1, i + right)]``.

    Args:
        seq_len_q: Query sequence length.
        seq_len_k: Key sequence length.
        window_size: ``(left, right)`` — maximum look-back / look-ahead
            distances (inclusive).  ``-1`` means unlimited on that side.
        causal: Whether to apply causal (lower-triangular) masking.
        device: Torch device for the output tensor.
        dtype: Torch dtype for the output tensor.

    Returns:
        ``(1, 1, seq_len_q, seq_len_k)`` additive mask (``0.0`` = attend,
        ``-inf`` = block).  Broadcastable over batch and head dimensions.
    """
    q_idx = torch.arange(seq_len_q, device=device).unsqueeze(1)
    k_idx = torch.arange(seq_len_k, device=device).unsqueeze(0)
    diff = q_idx - k_idx  # positive when query position > key position

    attend = torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=device)

    if causal:
        attend = attend & (diff >= 0)

    left, right = window_size
    if left >= 0:
        attend = attend & (diff <= left)
    if right >= 0:
        attend = attend & (-diff <= right)

    mask = torch.where(attend, 0.0, float("-inf"))
    return mask.to(dtype).unsqueeze(0).unsqueeze(0)


def _has_window(window_size):
    """Return ``True`` if *window_size* actually constrains attention."""
    if window_size is None:
        return False
    left, right = window_size
    return left >= 0 or right >= 0


# ---------------------------------------------------------------------------
# flash_attn_varlen_func replacement
# ---------------------------------------------------------------------------

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
    window_size=(-1, -1),
    return_attn_probs=False,
    **kwargs,
):
    """Drop-in replacement for ``flash_attn_varlen_func`` using per-sequence
    ``torch.nn.functional.scaled_dot_product_attention``.

    Args:
        q:  ``(total_q, nheads, head_dim)``
        k:  ``(total_k, nheads_k, head_dim)``
        v:  ``(total_k, nheads_k, head_dim)``
        cu_seqlens_q:  ``(batch_size + 1,)``  cumulative sequence lengths for Q.
        cu_seqlens_k:  ``(batch_size + 1,)``  cumulative sequence lengths for K.
        max_seqlen_q:  Unused — kept for API compatibility with ``flash_attn``.
        max_seqlen_k:  Unused — kept for API compatibility with ``flash_attn``.
        window_size:  ``(left, right)`` sliding-window limits. ``-1`` means
            unlimited on that side.  Same semantics as ``flash_attn``.
        return_attn_probs:  Accepted for API compatibility; always ignored
            (SDPA does not expose attention probabilities).

    Returns:
        ``(total_q, nheads, head_dim)``
    """
    batch_size = cu_seqlens_q.shape[0] - 1
    output = torch.empty_like(q)
    is_training = q.requires_grad
    use_window = _has_window(window_size)
    drop = dropout_p if is_training else 0.0

    for i in range(batch_size):
        q_s, q_e = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        k_s, k_e = cu_seqlens_k[i].item(), cu_seqlens_k[i + 1].item()

        # (seq, nheads, head_dim) -> (1, nheads, seq, head_dim)
        qi = q[q_s:q_e].unsqueeze(0).transpose(1, 2)
        ki = k[k_s:k_e].unsqueeze(0).transpose(1, 2)
        vi = v[k_s:k_e].unsqueeze(0).transpose(1, 2)

        # GQA expansion: SYCL-TLA Flash does not broadcast K/V over Q heads;
        # repeat_interleave to make nheads_k == nheads_q.
        nheads_q = qi.shape[1]
        nheads_k_actual = ki.shape[1]
        if nheads_k_actual != nheads_q and nheads_k_actual != 1:
            assert nheads_q % nheads_k_actual == 0, (
                f"nheads_q={nheads_q} must be divisible by nheads_k={nheads_k_actual}"
            )
            repeats = nheads_q // nheads_k_actual
            ki = ki.repeat_interleave(repeats, dim=1)
            vi = vi.repeat_interleave(repeats, dim=1)

        if use_window:
            mask = _build_window_mask(
                q_e - q_s, k_e - k_s, window_size, causal,
                q.device, q.dtype,
            )
            oi = F.scaled_dot_product_attention(
                qi, ki, vi,
                attn_mask=mask,
                dropout_p=drop,
                is_causal=False,  # causality is baked into the mask
                scale=softmax_scale,
            )
        else:
            oi = F.scaled_dot_product_attention(
                qi, ki, vi,
                dropout_p=drop,
                is_causal=causal,
                scale=softmax_scale,
            )

        # (1, nheads, seq_q, head_dim) -> (seq_q, nheads, head_dim)
        output[q_s:q_e] = oi.squeeze(0).transpose(0, 1)

    return output


# ---------------------------------------------------------------------------
# HF _flash_attention_forward replacement
# ---------------------------------------------------------------------------

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

    1. **Packed sequences** (non-monotonic ``position_ids``, ``B = 1``):
       routes through :func:`xpu_varlen_sdpa` for correct per-sub-sequence
       attention with no cross-sequence leakage.
    2. **Normal / generation**: plain causal SDPA.

    The packed-sequence path requires ``batch_size == 1``, matching the
    ``_is_packed_sequence`` guard in HF ``transformers`` (sequences are
    concatenated into a single row with ``position_ids`` resetting to 0 at
    each sub-sequence boundary).

    Input layout  : ``(batch_size, seq_len, num_heads, head_dim)`` — FA2 format.
    Output layout : ``(batch_size, seq_len, num_heads, head_dim)``.
    """
    dropout_p = kwargs.pop("dropout", 0.0)
    softmax_scale = kwargs.pop("softmax_scale", None)
    position_ids = kwargs.pop("position_ids", None)
    softcap = kwargs.pop("softcap", None)

    # Forward-compat: accept (and discard) newer HF kwargs that SDPA cannot use.
    for _k in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k",
               "target_dtype", "attn_implementation"):
        kwargs.pop(_k, None)

    if softcap is not None:
        warnings.warn(
            f"xpu_flash_attention_forward: softcap={softcap} was requested but "
            "torch SDPA does not support attention-logit soft-capping. "
            "Outputs will differ from the original model.",
            stacklevel=2,
        )

    is_training = query_states.requires_grad
    batch_size = query_states.size(0)
    seq_len_k = key_states.size(1)
    drop = dropout_p if is_training else 0.0

    # --- Packed-sequence path ---------------------------------------------------
    # Detected by non-monotonically-increasing position_ids (position resets
    # to 0 at each sub-sequence boundary), same heuristic as HF / qwen2_vl.
    # Requires B=1 — the standard packed layout.  With B>1 the flattened
    # position_ids would silently merge batch boundaries and cause
    # cross-example attention leakage, so we fall through to the normal path.
    if (
        position_ids is not None
        and query_length != 1
        and batch_size == 1
        and position_ids.ndim == 2
        and not (torch.diff(position_ids, dim=-1) >= 0).all()
    ):
        # (1, S, H, D) → (S, H, D)
        q_flat = query_states.reshape(-1, query_states.size(-2), query_states.size(-1))
        k_flat = key_states.reshape(-1, key_states.size(-2), key_states.size(-1))
        v_flat = value_states.reshape(-1, value_states.size(-2), value_states.size(-1))

        pos = position_ids.reshape(-1)
        starts = (pos == 0).nonzero(as_tuple=False).view(-1).to(torch.int32)
        if starts.numel() == 0 or starts[0] != 0:
            raise ValueError(
                "Packed position_ids must begin with 0. "
                f"Got first zero-position at index "
                f"{starts[0].item() if starts.numel() else 'N/A'}."
            )
        cu_seqlens = torch.cat([
            starts,
            torch.tensor([pos.size(0)], device=pos.device, dtype=torch.int32),
        ])
        max_seqlen = cu_seqlens.diff().max().item()

        # Convert HF sliding_window (total window size) → flash_attn window_size.
        window_size = (-1, -1)
        if sliding_window is not None and sliding_window > 0 and max_seqlen > sliding_window:
            window_size = (sliding_window - 1, sliding_window - 1)

        attn_out = xpu_varlen_sdpa(
            q_flat, k_flat, v_flat,
            cu_seqlens, cu_seqlens,
            max_seqlen, max_seqlen,
            dropout_p=drop,
            softmax_scale=softmax_scale,
            causal=is_causal,
            window_size=window_size,
        )
        # (S, H, D) → (1, S, H, D)
        return attn_out.unsqueeze(0)

    # --- Normal path -----------------------------------------------------------
    # (B, S, H, D) -> (B, H, S, D)
    q = query_states.transpose(1, 2)
    k = key_states.transpose(1, 2)
    v = value_states.transpose(1, 2)

    use_causal = is_causal
    attn_mask = None

    if sliding_window is not None and sliding_window > 0 and seq_len_k > sliding_window:
        # Sliding-window requires an explicit mask; causality is baked in.
        window_size = (sliding_window - 1, -1 if is_causal else sliding_window - 1)
        attn_mask = _build_window_mask(
            q.size(2), seq_len_k, window_size, is_causal,
            q.device, q.dtype,
        )
        use_causal = False
        # Merge with any existing padding mask.
        if attention_mask is not None:
            attn_mask = attn_mask + attention_mask
    elif not is_causal:
        attn_mask = attention_mask

    attn_output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=drop,
        is_causal=use_causal,
        scale=softmax_scale,
    )

    # (B, H, S, D) -> (B, S, H, D)
    return attn_output.transpose(1, 2)
