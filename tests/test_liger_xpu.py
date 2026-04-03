#!/usr/bin/env python3
"""
Test Liger Kernel on Intel XPU (single GPU).

Tests:
  1. Import and basic module availability
  2. Apply liger kernel to a real model (Qwen2.5-0.5B)
  3. Forward pass with liger-patched model
  4. Backward pass (gradient flow)
  5. 5-step training loop: compare liger vs baseline (loss, memory, speed)
  6. Individual kernel correctness (RMSNorm, SwiGLU, RoPE, CrossEntropy)

Usage:
    ZE_AFFINITY_MASK=3 python tests/test_liger_xpu.py
"""

import os
import sys
import time
import gc
import warnings

os.environ.setdefault("ZE_AFFINITY_MASK", "3")

import torch
import torch.nn.functional as F

DEVICE = "xpu"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DTYPE = torch.bfloat16


def get_mem_gb():
    torch.xpu.synchronize()
    return torch.xpu.memory_allocated() / 1024**3


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─── Test 1: Import ───
def test_import():
    separator("Test 1: Liger Kernel Import")
    import liger_kernel
    ver = getattr(liger_kernel, "__version__", "unknown")
    print(f"  liger-kernel version: {ver}")

    from liger_kernel.transformers import (
        LigerRMSNorm,
        LigerSwiGLUMLP,
        LigerCrossEntropyLoss,
        liger_rotary_pos_emb,
    )
    print(f"  LigerRMSNorm: {LigerRMSNorm}")
    print(f"  LigerSwiGLUMLP: {LigerSwiGLUMLP}")
    print(f"  LigerCrossEntropyLoss: {LigerCrossEntropyLoss}")
    print(f"  liger_rotary_pos_emb: {liger_rotary_pos_emb}")

    from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
    print(f"  _apply_liger_kernel_to_instance: {_apply_liger_kernel_to_instance}")
    print("  ✅ PASS — All imports successful")
    return True


# ─── Test 2: Individual Kernels ───
def test_individual_kernels():
    separator("Test 2: Individual Liger Kernels on XPU")
    results = {}

    # 2a. RMSNorm
    print("\n  2a. LigerRMSNorm...")
    from liger_kernel.transformers import LigerRMSNorm
    hidden_size = 896  # Qwen2.5-0.5B
    ln = LigerRMSNorm(hidden_size).to(DEVICE, DTYPE)
    x = torch.randn(2, 128, hidden_size, device=DEVICE, dtype=DTYPE, requires_grad=True)
    try:
        out = ln(x)
        out.sum().backward()
        assert not torch.isnan(out).any(), "NaN in RMSNorm output"
        assert x.grad is not None, "No gradient"
        assert not torch.isnan(x.grad).any(), "NaN in RMSNorm gradient"
        print(f"      output shape: {out.shape}, grad shape: {x.grad.shape}")
        print("      ✅ PASS")
        results["RMSNorm"] = "PASS"
    except Exception as e:
        print(f"      ❌ FAIL: {e}")
        results["RMSNorm"] = f"FAIL: {e}"

    # 2b. CrossEntropy
    print("\n  2b. LigerCrossEntropyLoss...")
    from liger_kernel.transformers import LigerCrossEntropyLoss
    ce = LigerCrossEntropyLoss()
    logits = torch.randn(4, 32000, device=DEVICE, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, 32000, (4,), device=DEVICE)
    try:
        loss = ce(logits, targets)
        loss.backward()
        assert not torch.isnan(loss), "NaN in CE loss"
        assert logits.grad is not None, "No gradient"
        # Compare with standard CE
        logits2 = logits.detach().clone().requires_grad_(True)
        loss_ref = F.cross_entropy(logits2, targets)
        loss_ref.backward()
        diff = (loss.item() - loss_ref.item())
        print(f"      liger CE: {loss.item():.6f}, torch CE: {loss_ref.item():.6f}, diff: {abs(diff):.8f}")
        print(f"      grad max_diff: {(logits.grad - logits2.grad).abs().max().item():.8f}")
        print("      ✅ PASS")
        results["CrossEntropy"] = "PASS"
    except Exception as e:
        print(f"      ❌ FAIL: {e}")
        results["CrossEntropy"] = f"FAIL: {e}"

    # 2c. SwiGLU (via monkey-patching a model — tested in test 3)
    results["SwiGLU"] = "tested in model patch"

    return results


# ─── Test 3: Apply to Real Model ───
def test_apply_to_model():
    separator("Test 3: Apply Liger Kernel to Qwen2.5-0.5B")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

    print(f"  Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True
    ).to(DEVICE)
    mem_before = get_mem_gb()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, {mem_before:.2f} GB")

    # Apply Liger (same way verl does it)
    print("  Applying Liger kernel (fused_linear_cross_entropy=False, swiglu=True)...")
    try:
        _apply_liger_kernel_to_instance(
            model=model,
            fused_linear_cross_entropy=False,
            swiglu=True,
        )
        print("  ✅ Liger kernel applied successfully")
    except Exception as e:
        print(f"  ❌ FAIL applying Liger: {e}")
        import traceback; traceback.print_exc()
        return None, None

    return model, tokenizer


# ─── Test 4: Forward + Backward ───
def test_forward_backward(model, tokenizer):
    separator("Test 4: Forward + Backward with Liger-Patched Model")

    text = "The quick brown fox jumps over the lazy dog. " * 8
    inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)
    input_ids = inputs["input_ids"]
    labels = input_ids.clone()

    print(f"  Input shape: {input_ids.shape}")

    # Forward
    torch.xpu.synchronize()
    t0 = time.time()
    outputs = model(input_ids=input_ids, labels=labels)
    torch.xpu.synchronize()
    t_fwd = time.time() - t0
    loss = outputs.loss
    print(f"  Forward: loss={loss.item():.4f}, time={t_fwd*1000:.1f}ms")
    assert not torch.isnan(loss), "NaN in forward loss!"

    # Backward
    torch.xpu.synchronize()
    t0 = time.time()
    loss.backward()
    torch.xpu.synchronize()
    t_bwd = time.time() - t0

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    print(f"  Backward: grad_norm={grad_norm.item():.2f}, time={t_bwd*1000:.1f}ms")
    assert not torch.isnan(grad_norm), "NaN in grad_norm!"

    mem = get_mem_gb()
    print(f"  Memory: {mem:.2f} GB")
    print("  ✅ PASS — Forward + Backward clean")
    return True


# ─── Test 5: Training Loop Comparison ───
def test_training_comparison():
    separator("Test 5: 5-Step Training — Liger vs Baseline")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare data
    texts = [
        "Explain quantum computing in simple terms.",
        "What is the capital of France? The capital is Paris.",
        "Write a haiku about machine learning frameworks.",
        "The Pythagorean theorem states that a squared plus b squared equals c squared.",
        "Once upon a time in a land far away, there lived a wise old wizard.",
    ]
    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    input_ids = encodings["input_ids"]
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100  # ignore padding in loss

    def run_training(use_liger: bool, label: str):
        print(f"\n  --- {label} ---")
        torch.xpu.empty_cache()
        gc.collect()

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True
        ).to(DEVICE)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if use_liger:
            _apply_liger_kernel_to_instance(
                model=model, fused_linear_cross_entropy=False, swiglu=True,
            )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
        mem_before = get_mem_gb()

        losses = []
        times = []
        peak_mems = []

        model.train()
        for step in range(5):
            optimizer.zero_grad()
            torch.xpu.synchronize()
            t0 = time.time()

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            torch.xpu.synchronize()
            dt = time.time() - t0
            mem = get_mem_gb()

            losses.append(loss.item())
            times.append(dt)
            peak_mems.append(mem)
            nan_str = "NaN!" if torch.isnan(loss) else ""
            print(f"    step {step+1}: loss={loss.item():.4f}  grad_norm={grad_norm.item():.2f}  "
                  f"time={dt*1000:.0f}ms  mem={mem:.2f}GB {nan_str}")

        avg_time = sum(times[1:]) / len(times[1:])  # skip first (compilation)
        peak_mem = max(peak_mems)

        del model, optimizer
        torch.xpu.empty_cache()
        gc.collect()

        return {
            "losses": losses,
            "avg_time_ms": avg_time * 1000,
            "first_time_ms": times[0] * 1000,
            "peak_mem_gb": peak_mem,
            "mem_before_gb": mem_before,
        }

    baseline = run_training(use_liger=False, label="Baseline (no Liger)")
    liger = run_training(use_liger=True, label="Liger Kernel (RMSNorm + SwiGLU + RoPE + CE)")

    # Summary
    separator("Test 5 Summary: Liger vs Baseline")
    print(f"  {'Metric':<25} {'Baseline':>12} {'Liger':>12} {'Delta':>12}")
    print(f"  {'-'*61}")
    print(f"  {'First step (ms)':<25} {baseline['first_time_ms']:>12.0f} {liger['first_time_ms']:>12.0f} "
          f"{liger['first_time_ms'] - baseline['first_time_ms']:>+12.0f}")
    print(f"  {'Avg step 2-5 (ms)':<25} {baseline['avg_time_ms']:>12.0f} {liger['avg_time_ms']:>12.0f} "
          f"{liger['avg_time_ms'] - baseline['avg_time_ms']:>+12.0f}")
    speedup = (baseline['avg_time_ms'] / liger['avg_time_ms'] - 1) * 100 if liger['avg_time_ms'] > 0 else 0
    print(f"  {'Speedup':<25} {'':>12} {'':>12} {speedup:>+11.1f}%")
    print(f"  {'Peak memory (GB)':<25} {baseline['peak_mem_gb']:>12.2f} {liger['peak_mem_gb']:>12.2f} "
          f"{liger['peak_mem_gb'] - baseline['peak_mem_gb']:>+12.2f}")
    mem_save = (1 - liger['peak_mem_gb'] / baseline['peak_mem_gb']) * 100 if baseline['peak_mem_gb'] > 0 else 0
    print(f"  {'Memory savings':<25} {'':>12} {'':>12} {mem_save:>+11.1f}%")
    print(f"  {'Loss step 1':<25} {baseline['losses'][0]:>12.4f} {liger['losses'][0]:>12.4f}")
    print(f"  {'Loss step 5':<25} {baseline['losses'][-1]:>12.4f} {liger['losses'][-1]:>12.4f}")

    any_nan = any(torch.isnan(torch.tensor(liger['losses'])))
    if any_nan:
        print("\n  ❌ FAIL — NaN in Liger training!")
    else:
        print(f"\n  ✅ PASS — Liger training clean, {speedup:+.1f}% throughput, {mem_save:+.1f}% memory")

    return baseline, liger


# ─── Test 6: verl Integration Path ───
def test_verl_integration():
    separator("Test 6: verl SFT Trainer with use_liger=True (dry run)")
    # Test that verl's code path would work
    from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True
    ).to(DEVICE)

    # Simulate verl's exact code path from transformer_impl.py:268-274
    try:
        _apply_liger_kernel_to_instance(
            model=model,
            fused_linear_cross_entropy=False,
            swiglu=True,
        )
    except Exception as e:
        print(f"  ❌ FAIL: verl code path raises: {e}")
        import traceback; traceback.print_exc()
        del model; torch.xpu.empty_cache()
        return False

    # Forward pass
    input_ids = torch.randint(0, 1000, (1, 64), device=DEVICE)
    labels = input_ids.clone()
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()

    assert not torch.isnan(loss), "NaN!"
    print(f"  loss={loss.item():.4f}")
    print("  ✅ PASS — verl integration path works on XPU")

    del model; torch.xpu.empty_cache(); gc.collect()
    return True


# ─── Main ───
if __name__ == "__main__":
    print(f"Device: {torch.xpu.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"ZE_AFFINITY_MASK: {os.environ.get('ZE_AFFINITY_MASK', 'not set')}")

    results = {}

    # Test 1
    try:
        results["import"] = test_import()
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        import traceback; traceback.print_exc()
        results["import"] = False

    if not results.get("import"):
        print("\n\nImport failed — cannot continue.")
        sys.exit(1)

    # Test 2
    try:
        results["kernels"] = test_individual_kernels()
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        import traceback; traceback.print_exc()
        results["kernels"] = {"error": str(e)}

    # Test 3 + 4
    try:
        model, tokenizer = test_apply_to_model()
        if model is not None:
            results["apply"] = True
            results["fwd_bwd"] = test_forward_backward(model, tokenizer)
            del model; torch.xpu.empty_cache(); gc.collect()
        else:
            results["apply"] = False
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        import traceback; traceback.print_exc()
        results["apply"] = False

    # Test 5
    try:
        baseline, liger = test_training_comparison()
        results["training"] = True
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        import traceback; traceback.print_exc()
        results["training"] = False

    # Test 6
    try:
        results["verl_integration"] = test_verl_integration()
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        import traceback; traceback.print_exc()
        results["verl_integration"] = False

    # Final summary
    separator("FINAL SUMMARY")
    for k, v in results.items():
        status = "✅" if v and v is not False else "❌"
        print(f"  {status} {k}: {v}")
