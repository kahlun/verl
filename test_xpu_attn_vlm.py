#!/usr/bin/env python3
"""End-to-end tests for xpu_attn.py on Intel XPU.

Tests:
  1. xpu_varlen_sdpa — packed sequences, correctness vs per-sequence SDPA reference
  2. xpu_varlen_sdpa — sliding window support
  3. xpu_flash_attention_forward — packed-sequence path (non-monotonic position_ids)
  4. xpu_flash_attention_forward — normal causal path
  5. xpu_flash_attention_forward — sliding window in normal path
  6. xpu_flash_attention_forward — softcap warning
  7. xpu_flash_attention_forward — batch_size>1 with non-monotonic pos_ids falls through safely
  8. Cross-sequence leakage test (packed sequences must NOT attend across boundaries)
  9. Qwen2-VL integration — forward + backward with monkey-patched attention

Usage:
  ZE_AFFINITY_MASK=3 python test_xpu_attn_vlm.py
"""

import sys
import os
import warnings
import traceback

import torch
import torch.nn.functional as F

# Ensure verl is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verl.models.transformers.xpu_attn import (
    xpu_varlen_sdpa,
    xpu_flash_attention_forward,
    _build_window_mask,
)

DEVICE = "xpu"
DTYPE = torch.bfloat16

PASS = 0
FAIL = 0


def report(name, passed, detail=""):
    global PASS, FAIL
    tag = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if passed:
        PASS += 1
    else:
        FAIL += 1


def ref_varlen_sdpa(q, k, v, cu_q, cu_k, causal, scale):
    """Reference: per-sequence SDPA, no window."""
    n = cu_q.shape[0] - 1
    out = torch.empty_like(q)
    for i in range(n):
        qs, qe = cu_q[i].item(), cu_q[i + 1].item()
        ks, ke = cu_k[i].item(), cu_k[i + 1].item()
        qi = q[qs:qe].unsqueeze(0).transpose(1, 2)
        ki = k[ks:ke].unsqueeze(0).transpose(1, 2)
        vi = v[ks:ke].unsqueeze(0).transpose(1, 2)
        oi = F.scaled_dot_product_attention(qi, ki, vi, is_causal=causal, scale=scale)
        out[qs:qe] = oi.squeeze(0).transpose(0, 1)
    return out


# ───────────────────────────────────────────────────────────────────────────
# Test 1: xpu_varlen_sdpa basic correctness
# ───────────────────────────────────────────────────────────────────────────
def test_varlen_sdpa_basic():
    torch.manual_seed(42)
    nheads, hdim = 8, 64
    seqlens = [4, 6, 3]  # 3 packed sub-sequences
    total = sum(seqlens)
    cu = torch.tensor([0] + [sum(seqlens[:i+1]) for i in range(len(seqlens))],
                       dtype=torch.int32, device=DEVICE)

    q = torch.randn(total, nheads, hdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(total, nheads, hdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(total, nheads, hdim, device=DEVICE, dtype=DTYPE)
    scale = 1.0 / (hdim ** 0.5)

    out = xpu_varlen_sdpa(q, k, v, cu, cu, max(seqlens), max(seqlens),
                          softmax_scale=scale, causal=True)
    ref = ref_varlen_sdpa(q, k, v, cu, cu, causal=True, scale=scale)

    diff = (out - ref).abs().max().item()
    report("xpu_varlen_sdpa basic (3 packed seqs)", diff < 1e-3, f"max_diff={diff:.6f}")


# ───────────────────────────────────────────────────────────────────────────
# Test 2: xpu_varlen_sdpa with sliding window
# ───────────────────────────────────────────────────────────────────────────
def test_varlen_sdpa_window():
    torch.manual_seed(123)
    nheads, hdim = 4, 32
    seq_len = 16
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device=DEVICE)

    q = torch.randn(seq_len, nheads, hdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(seq_len, nheads, hdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(seq_len, nheads, hdim, device=DEVICE, dtype=DTYPE)

    # window_size=(3, 0) → causal, look back at most 3 positions
    out_win = xpu_varlen_sdpa(q, k, v, cu, cu, seq_len, seq_len,
                              causal=True, window_size=(3, 0))
    out_full = xpu_varlen_sdpa(q, k, v, cu, cu, seq_len, seq_len,
                               causal=True, window_size=(-1, -1))

    # With a window, outputs must differ from full attention for long sequences
    diff = (out_win - out_full).abs().max().item()
    report("xpu_varlen_sdpa sliding window", diff > 1e-4,
           f"window vs full diff={diff:.6f} (should be >0)")


# ───────────────────────────────────────────────────────────────────────────
# Test 3: xpu_flash_attention_forward — packed-sequence path
# ───────────────────────────────────────────────────────────────────────────
def test_flash_forward_packed():
    torch.manual_seed(7)
    nheads, hdim = 8, 64
    # 2 sub-sequences packed into B=1: [0,1,2, 0,1,2,3,4]
    pos_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]], device=DEVICE)
    seq_len = pos_ids.size(1)
    B = 1

    q = torch.randn(B, seq_len, nheads, hdim, device=DEVICE, dtype=DTYPE, requires_grad=True)
    k = torch.randn(B, seq_len, nheads, hdim, device=DEVICE, dtype=DTYPE, requires_grad=True)
    v = torch.randn(B, seq_len, nheads, hdim, device=DEVICE, dtype=DTYPE, requires_grad=True)

    out = xpu_flash_attention_forward(
        q, k, v,
        attention_mask=None,
        query_length=seq_len,
        is_causal=True,
        position_ids=pos_ids,
    )
    ok_shape = out.shape == (B, seq_len, nheads, hdim)
    ok_nan = not torch.isnan(out).any().item()

    # Backward
    loss = out.sum()
    loss.backward()
    ok_grad = q.grad is not None and not torch.isnan(q.grad).any().item()

    report("flash_forward packed path (shape)", ok_shape, f"shape={tuple(out.shape)}")
    report("flash_forward packed path (no NaN)", ok_nan)
    report("flash_forward packed path (backward)", ok_grad)


# ───────────────────────────────────────────────────────────────────────────
# Test 4: xpu_flash_attention_forward — normal causal path
# ───────────────────────────────────────────────────────────────────────────
def test_flash_forward_normal():
    torch.manual_seed(99)
    B, S, nheads, hdim = 2, 16, 4, 32
    q = torch.randn(B, S, nheads, hdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, S, nheads, hdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, S, nheads, hdim, device=DEVICE, dtype=DTYPE)

    out = xpu_flash_attention_forward(
        q, k, v,
        attention_mask=None,
        query_length=S,
        is_causal=True,
    )

    # Reference: plain SDPA
    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        is_causal=True,
    ).transpose(1, 2)

    diff = (out - ref).abs().max().item()
    report("flash_forward normal causal", diff < 1e-3, f"max_diff={diff:.6f}")


# ───────────────────────────────────────────────────────────────────────────
# Test 5: xpu_flash_attention_forward — sliding window (normal path)
# ───────────────────────────────────────────────────────────────────────────
def test_flash_forward_sliding_window():
    torch.manual_seed(55)
    B, S, nheads, hdim = 1, 32, 4, 32
    q = torch.randn(B, S, nheads, hdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, S, nheads, hdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, S, nheads, hdim, device=DEVICE, dtype=DTYPE)
    # Monotonic position_ids → normal path
    pos = torch.arange(S, device=DEVICE).unsqueeze(0)

    out_full = xpu_flash_attention_forward(
        q, k, v, attention_mask=None, query_length=S,
        is_causal=True, sliding_window=None, position_ids=pos,
    )
    out_win = xpu_flash_attention_forward(
        q, k, v, attention_mask=None, query_length=S,
        is_causal=True, sliding_window=8, position_ids=pos,
    )

    diff = (out_full - out_win).abs().max().item()
    report("flash_forward sliding window (normal path)", diff > 1e-4,
           f"full vs windowed diff={diff:.6f} (should be >0)")


# ───────────────────────────────────────────────────────────────────────────
# Test 6: softcap warning
# ───────────────────────────────────────────────────────────────────────────
def test_softcap_warning():
    B, S, nheads, hdim = 1, 8, 2, 16
    q = torch.randn(B, S, nheads, hdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, S, nheads, hdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, S, nheads, hdim, device=DEVICE, dtype=DTYPE)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = xpu_flash_attention_forward(
            q, k, v, attention_mask=None, query_length=S,
            is_causal=True, softcap=50.0,
        )
        got_warn = any("softcap" in str(x.message) for x in w)

    report("softcap emits warning", got_warn)
    report("softcap output valid (no NaN)", not torch.isnan(out).any().item())


# ───────────────────────────────────────────────────────────────────────────
# Test 7: B>1 with non-monotonic pos_ids → falls through to normal path
# ───────────────────────────────────────────────────────────────────────────
def test_batch_gt1_fallthrough():
    torch.manual_seed(11)
    B, S, nheads, hdim = 2, 8, 4, 32
    q = torch.randn(B, S, nheads, hdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, S, nheads, hdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, S, nheads, hdim, device=DEVICE, dtype=DTYPE)
    # Non-monotonic but B=2 → should NOT enter packed path
    pos = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4],
                        [0, 1, 0, 1, 2, 3, 0, 1]], device=DEVICE)

    out = xpu_flash_attention_forward(
        q, k, v, attention_mask=None, query_length=S,
        is_causal=True, position_ids=pos,
    )
    ok = out.shape == (B, S, nheads, hdim) and not torch.isnan(out).any().item()
    report("B>1 non-monotonic pos_ids → normal path", ok, f"shape={tuple(out.shape)}")


# ───────────────────────────────────────────────────────────────────────────
# Test 8: Cross-sequence leakage test
# ───────────────────────────────────────────────────────────────────────────
def test_no_cross_sequence_leakage():
    """Perturb tokens in sub-sequence 1; sub-sequence 0 output must not change."""
    torch.manual_seed(77)
    nheads, hdim = 4, 32
    # Two sub-sequences: [0,1,2] and [0,1,2,3]
    pos_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 3]], device=DEVICE)
    B, S = 1, 7

    q = torch.randn(B, S, nheads, hdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, S, nheads, hdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, S, nheads, hdim, device=DEVICE, dtype=DTYPE)

    out1 = xpu_flash_attention_forward(
        q.clone(), k.clone(), v.clone(),
        attention_mask=None, query_length=S, is_causal=True,
        position_ids=pos_ids,
    )

    # Perturb sub-sequence 1 (positions 3..6)
    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    q2[:, 3:, :, :] += 100.0
    k2[:, 3:, :, :] += 100.0
    v2[:, 3:, :, :] += 100.0

    out2 = xpu_flash_attention_forward(
        q2, k2, v2,
        attention_mask=None, query_length=S, is_causal=True,
        position_ids=pos_ids,
    )

    # Sub-sequence 0 (positions 0..2) must be identical
    diff_seq0 = (out1[:, :3] - out2[:, :3]).abs().max().item()
    # Sub-sequence 1 (positions 3..6) must be different
    diff_seq1 = (out1[:, 3:] - out2[:, 3:]).abs().max().item()

    report("no cross-sequence leakage (seq0 unchanged)", diff_seq0 < 1e-6,
           f"seq0 diff={diff_seq0:.8f}")
    report("no cross-sequence leakage (seq1 changed)", diff_seq1 > 1.0,
           f"seq1 diff={diff_seq1:.4f}")


# ───────────────────────────────────────────────────────────────────────────
# Test 9: Qwen2-VL integration (if model is available)
# ───────────────────────────────────────────────────────────────────────────
def test_qwen2vl_integration():
    """Full forward+backward through Qwen2-VL with the patched attention."""
    model_name = "Qwen/Qwen2-VL-2B-Instruct"

    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    except ImportError:
        report("Qwen2-VL integration (import)", False, "transformers not available")
        return

    # Check if model is cached locally
    from huggingface_hub import try_to_load_from_cache
    cached = try_to_load_from_cache(model_name, "config.json")
    if cached is None:
        report("Qwen2-VL integration", True, "SKIP — model not cached locally")
        return

    print("    Loading Qwen2-VL-2B-Instruct...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        attn_implementation="eager",
    ).to(DEVICE)
    model.train()

    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer

    # Simple text-only input (no image, tests the attention path)
    text = "What is 2 + 2?"
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]
    labels = input_ids.clone()

    # Forward
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    ok_loss = loss is not None and not torch.isnan(loss).item()
    report("Qwen2-VL forward (loss valid)", ok_loss,
           f"loss={loss.item():.4f}" if ok_loss else "NaN!")

    # Backward
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    ok_grad = grad_norm > 0 and not (grad_norm != grad_norm)
    report("Qwen2-VL backward (grad valid)", ok_grad, f"grad_norm={grad_norm:.2f}")

    # Now test with monkey-patched attention (VLM packed-sequence path)
    try:
        from verl.models.transformers.qwen2_vl import (
            qwen2_vl_attn_forward,
            qwen2_vl_forward,
        )

        # Apply monkey patches like verl does
        from transformers.models.qwen2_vl import modeling_qwen2_vl
        lang_model = model.model.language_model if hasattr(model.model, 'language_model') else model.model
        decoder_layers = lang_model.layers if hasattr(lang_model, 'layers') else lang_model.model.layers
        for layer in decoder_layers:
            layer.self_attn.forward = qwen2_vl_attn_forward.__get__(
                layer.self_attn, type(layer.self_attn)
            )
        modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = qwen2_vl_forward

        model.zero_grad()
        # verl's qwen2_vl_forward requires 4D position_ids: (4, batch, seq_len)
        # channels: text_pos, temporal, height, width — for text-only, all are arange
        seq_len = input_ids.size(1)
        pos_1d = torch.arange(seq_len, device=DEVICE).unsqueeze(0)  # (1, seq_len)
        position_ids = pos_1d.unsqueeze(0).expand(4, 1, seq_len)    # (4, 1, seq_len)
        # verl's monkey-patched forward returns raw model output (no loss);
        # loss is computed separately in verl's training loop
        outputs2 = model(input_ids=input_ids, position_ids=position_ids)
        # Get logits from the output and compute loss manually
        if hasattr(outputs2, 'logits'):
            logits = outputs2.logits
        elif hasattr(outputs2, 'last_hidden_state'):
            logits = outputs2.last_hidden_state
        else:
            logits = outputs2[0] if isinstance(outputs2, (tuple, list)) else outputs2
        ok_logits = logits is not None and not torch.isnan(logits).any().item()
        report("Qwen2-VL monkey-patched forward", ok_logits,
               f"output shape={tuple(logits.shape)}")

        loss2 = logits.sum()
        loss2.backward()
        grad_norm2 = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        ok_grad2 = grad_norm2 > 0
        report("Qwen2-VL monkey-patched backward", ok_grad2,
               f"grad_norm={grad_norm2:.2f}")
    except Exception as e:
        report("Qwen2-VL monkey-patch integration", False, f"{e}")

    del model
    torch.xpu.empty_cache()


# ───────────────────────────────────────────────────────────────────────────
# Test 10: _build_window_mask correctness
# ───────────────────────────────────────────────────────────────────────────
def test_window_mask():
    # Causal + window(2, 0): query i attends to keys [max(0,i-2), i]
    mask = _build_window_mask(4, 4, (2, 0), causal=True, device=DEVICE, dtype=DTYPE)
    mask_2d = mask.squeeze()  # (4, 4)

    # Expected attend pattern (0=attend, -inf=block):
    # Row 0: [attend, block, block, block]  → keys [0]
    # Row 1: [attend, attend, block, block]  → keys [0,1]
    # Row 2: [attend, attend, attend, block]  → keys [0,1,2]
    # Row 3: [block, attend, attend, attend]  → keys [1,2,3] (left=2 cuts off key 0)
    expected_attend = torch.tensor([
        [True,  False, False, False],
        [True,  True,  False, False],
        [True,  True,  True,  False],
        [False, True,  True,  True],
    ], device=DEVICE)

    actual_attend = mask_2d > float("-inf")
    ok = (actual_attend == expected_attend).all().item()
    report("_build_window_mask causal+window(2,0)", ok)


# ───────────────────────────────────────────────────────────────────────────
# Test 11: sliding window in packed-sequence path
# ───────────────────────────────────────────────────────────────────────────
def test_flash_forward_packed_window():
    torch.manual_seed(33)
    nheads, hdim = 4, 32
    # Single long sub-sequence where window matters
    S = 32
    pos_ids = torch.tensor([list(range(8)) + list(range(S - 8))], device=DEVICE)  # 2 packed seqs
    B = 1

    q = torch.randn(B, S, nheads, hdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, S, nheads, hdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, S, nheads, hdim, device=DEVICE, dtype=DTYPE)

    out_full = xpu_flash_attention_forward(
        q, k, v, attention_mask=None, query_length=S,
        is_causal=True, sliding_window=None, position_ids=pos_ids,
    )
    out_win = xpu_flash_attention_forward(
        q, k, v, attention_mask=None, query_length=S,
        is_causal=True, sliding_window=8, position_ids=pos_ids,
    )

    diff = (out_full - out_win).abs().max().item()
    report("flash_forward packed + sliding window", diff > 1e-4,
           f"diff={diff:.6f} (should be >0)")


# ───────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*70}")
    print(f"  xpu_attn.py VLM Test Suite")
    print(f"  Device: {DEVICE}, ZE_AFFINITY_MASK={os.environ.get('ZE_AFFINITY_MASK', 'not set')}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  dtype: {DTYPE}")
    print(f"{'='*70}\n")

    # Verify XPU is available
    if not torch.xpu.is_available():
        print("ERROR: XPU not available!")
        sys.exit(1)

    print(f"  XPU device: {torch.xpu.get_device_name(0)}")
    print()

    tests = [
        ("1. varlen_sdpa basic", test_varlen_sdpa_basic),
        ("2. varlen_sdpa sliding window", test_varlen_sdpa_window),
        ("3. flash_forward packed path", test_flash_forward_packed),
        ("4. flash_forward normal causal", test_flash_forward_normal),
        ("5. flash_forward sliding window", test_flash_forward_sliding_window),
        ("6. softcap warning", test_softcap_warning),
        ("7. B>1 fallthrough", test_batch_gt1_fallthrough),
        ("8. cross-sequence leakage", test_no_cross_sequence_leakage),
        ("9. _build_window_mask", test_window_mask),
        ("10. packed + sliding window", test_flash_forward_packed_window),
        ("11. Qwen2-VL integration", test_qwen2vl_integration),
    ]

    for name, fn in tests:
        print(f"  --- {name} ---")
        try:
            fn()
        except Exception as e:
            report(name, False, f"EXCEPTION: {e}")
            traceback.print_exc()
        print()

    print(f"{'='*70}")
    print(f"  Results: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
    print(f"{'='*70}")
    sys.exit(1 if FAIL > 0 else 0)


if __name__ == "__main__":
    main()
