"""
VERL XPU vs CUDA (A100) Benchmark
===================================
Run this script on EACH machine separately, then compare the JSON outputs.

Usage:
  # On XPU machine:
  python3 benchmark/xpu_vs_cuda_benchmark.py --device xpu --output results_xpu.json

  # On A100 machine:
  python3 benchmark/xpu_vs_cuda_benchmark.py --device cuda --output results_a100.json

  # Compare:
  python3 benchmark/xpu_vs_cuda_benchmark.py --compare results_xpu.json results_a100.json

Results are saved as JSON for easy transfer between machines.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F


# ─── Utilities ────────────────────────────────────────────────────────────────

def get_device_info(device: str) -> dict:
    if device == "xpu":
        return {
            "device": device,
            "name": torch.xpu.get_device_name(0),
            "total_memory_gb": torch.xpu.get_device_properties(0).total_memory / 1024**3,
            "count": torch.xpu.device_count(),
            "pytorch_version": torch.__version__,
        }
    elif device == "cuda":
        return {
            "device": device,
            "name": torch.cuda.get_device_name(0),
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "count": torch.cuda.device_count(),
            "pytorch_version": torch.__version__,
        }
    else:
        return {"device": device, "pytorch_version": torch.__version__}


def sync(device: str):
    if device == "xpu":
        torch.xpu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def peak_memory_gb(device: str) -> float:
    if device == "xpu":
        return torch.xpu.max_memory_allocated() / 1024**3
    elif device == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0.0


def reset_memory(device: str):
    if device == "xpu":
        torch.xpu.reset_peak_memory_stats()
        torch.xpu.empty_cache()
    elif device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def time_it(fn, warmup=3, repeat=10, device="xpu"):
    """Time a function, return (mean_ms, std_ms)."""
    for _ in range(warmup):
        fn()
    sync(device)

    times = []
    for _ in range(repeat):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - t0) * 1000)

    import statistics
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0


# ─── Benchmarks ───────────────────────────────────────────────────────────────

def bench_attention_forward(device, dtype=torch.bfloat16):
    """Compare eager vs SDPA attention forward pass."""
    results = {}
    batch, seq_len, nheads, head_dim = 4, 512, 16, 64

    q = torch.randn(batch, seq_len, nheads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seq_len, nheads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seq_len, nheads, head_dim, device=device, dtype=dtype)

    # Eager
    def eager_attn():
        qt = q.transpose(1, 2)
        kt = k.transpose(1, 2)
        vt = v.transpose(1, 2)
        return F.scaled_dot_product_attention(qt, kt, vt, is_causal=True)

    mean_ms, std_ms = time_it(eager_attn, device=device)
    results["eager_sdpa_ms"] = round(mean_ms, 3)
    results["eager_sdpa_std_ms"] = round(std_ms, 3)

    # SDPA with flash attention (CUDA only)
    if device == "cuda":
        try:
            from flash_attn import flash_attn_func
            def flash_fn():
                return flash_attn_func(q, k, v, causal=True)
            mean_ms, std_ms = time_it(flash_fn, device=device)
            results["flash_attn_ms"] = round(mean_ms, 3)
            results["flash_attn_std_ms"] = round(std_ms, 3)
        except ImportError:
            results["flash_attn_ms"] = None

    results["config"] = {"batch": batch, "seq_len": seq_len, "nheads": nheads, "head_dim": head_dim}
    return results


def bench_transformer_forward_backward(device, model_name="Qwen/Qwen2.5-0.5B-Instruct",
                                        dtype=torch.bfloat16, steps=5):
    """Full model forward + backward timing and loss curve."""
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

    attn_impl = "eager" if device == "xpu" else "flash_attention_2"
    try:
        config = AutoConfig.from_pretrained(model_name, attn_implementation=attn_impl)
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config,
                                                      torch_dtype=dtype).to(device)
    except Exception:
        attn_impl = "eager"
        config = AutoConfig.from_pretrained(model_name, attn_implementation=attn_impl)
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config,
                                                      torch_dtype=dtype).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer("Solve: what is 2+2?", return_tensors="pt").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    losses, step_times, grad_norms = [], [], []
    peak_mems = []

    model.train()
    reset_memory(device)

    for step in range(steps):
        sync(device)
        t0 = time.perf_counter()

        out = model(**inputs, labels=inputs.input_ids)
        out.loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()
        optimizer.zero_grad()

        sync(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        losses.append(round(out.loss.item(), 4))
        step_times.append(round(elapsed_ms, 1))
        grad_norms.append(round(gn, 4))
        peak_mems.append(round(peak_memory_gb(device), 2))

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    return {
        "model": model_name,
        "params_m": round(params_m, 1),
        "attn_impl": attn_impl,
        "losses": losses,
        "loss_final": losses[-1],
        "step_times_ms": step_times,
        "mean_step_ms": round(sum(step_times) / len(step_times), 1),
        "grad_norms": grad_norms,
        "peak_memory_gb": max(peak_mems),
        "nan_count": sum(1 for l in losses if l != l),  # check for NaN
    }


def bench_fused_kernel(device, dtype=torch.bfloat16):
    """Benchmark fused entropy kernel vs eager reference."""
    results = {}
    batch, hidden_dim, vocab = 16, 512, 32000

    try:
        from verl.utils.kernel.kernels import efficient_entropy_forward

        hidden = torch.randn(batch, hidden_dim, device=device, dtype=dtype).contiguous()
        weight = torch.randn(vocab, hidden_dim, device=device, dtype=dtype).contiguous()
        labels = torch.randint(0, vocab, (batch,), device=device)

        def fused_fn():
            return efficient_entropy_forward(hidden, weight, labels)

        mean_ms, std_ms = time_it(fused_fn, device=device)
        results["fused_kernel_ms"] = round(mean_ms, 3)
        results["fused_kernel_std_ms"] = round(std_ms, 3)
        results["fused_kernel_available"] = True
    except Exception as e:
        results["fused_kernel_available"] = False
        results["fused_kernel_error"] = str(e)

    # Eager reference
    hidden_f = torch.randn(batch, hidden_dim, device=device, dtype=dtype)
    weight_f = torch.randn(vocab, hidden_dim, device=device, dtype=dtype)
    labels_f = torch.randint(0, vocab, (batch,), device=device)

    def eager_entropy():
        logits = hidden_f @ weight_f.T
        log_p = torch.log_softmax(logits.float(), dim=-1)
        return -(log_p.exp() * log_p).sum(dim=-1).mean()

    mean_ms, std_ms = time_it(eager_entropy, device=device)
    results["eager_entropy_ms"] = round(mean_ms, 3)
    results["config"] = {"batch": batch, "hidden_dim": hidden_dim, "vocab": vocab}
    return results


def bench_seqlen_balancing_distributed():
    """Distributed correctness check (single-process smoke)."""
    try:
        from verl.utils.seqlen_balancing import rearrange_micro_batches
        return {"available": True, "note": "Full test requires 2-GPU torchrun"}
    except Exception as e:
        return {"available": False, "error": str(e)}


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_benchmarks(device: str, model: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  VERL XPU/CUDA Benchmark — {device.upper()}")
    print(f"{'='*60}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "device_info": get_device_info(device),
    }

    print("\n[1/4] Attention forward (eager SDPA)...")
    results["attention"] = bench_attention_forward(device)
    print(f"      eager SDPA: {results['attention']['eager_sdpa_ms']:.2f} ms")
    if "flash_attn_ms" in results["attention"] and results["attention"]["flash_attn_ms"]:
        print(f"      flash_attn: {results['attention']['flash_attn_ms']:.2f} ms")

    print(f"\n[2/4] Transformer forward+backward ({model})...")
    try:
        results["transformer"] = bench_transformer_forward_backward(device, model)
        r = results["transformer"]
        print(f"      mean step: {r['mean_step_ms']} ms | peak mem: {r['peak_memory_gb']} GB")
        print(f"      loss: {r['losses']}")
        print(f"      NaN count: {r['nan_count']}")
    except Exception as e:
        results["transformer"] = {"error": str(e)}
        print(f"      FAILED: {e}")

    print("\n[3/4] Fused entropy kernel...")
    results["fused_kernel"] = bench_fused_kernel(device)
    fk = results["fused_kernel"]
    if fk.get("fused_kernel_available"):
        speedup = round(fk["eager_entropy_ms"] / fk["fused_kernel_ms"], 2)
        print(f"      fused: {fk['fused_kernel_ms']:.3f} ms | eager: {fk['eager_entropy_ms']:.3f} ms | speedup: {speedup}x")
    else:
        print(f"      fused kernel unavailable: {fk.get('fused_kernel_error', '')}")
        print(f"      eager entropy: {fk['eager_entropy_ms']:.3f} ms")

    print("\n[4/4] Memory summary...")
    reset_memory(device)
    results["summary"] = {
        "device": device,
        "attention_eager_ms": results["attention"]["eager_sdpa_ms"],
        "transformer_mean_step_ms": results.get("transformer", {}).get("mean_step_ms"),
        "transformer_peak_mem_gb": results.get("transformer", {}).get("peak_memory_gb"),
        "fused_kernel_ms": fk.get("fused_kernel_ms"),
        "eager_entropy_ms": fk.get("eager_entropy_ms"),
    }

    return results


def compare_results(file_a: str, file_b: str):
    """Compare two benchmark result JSON files."""
    with open(file_a) as f: a = json.load(f)
    with open(file_b) as f: b = json.load(f)

    dev_a = a["device_info"]["device"].upper()
    dev_b = b["device_info"]["device"].upper()

    print(f"\n{'='*70}")
    print(f"  VERL Benchmark Comparison: {dev_a} vs {dev_b}")
    print(f"{'='*70}")

    def name(r):
        di = r["device_info"]
        return f"{di['device'].upper()} ({di.get('name','?')})"

    print(f"\n{'Metric':<40} {name(a):>15} {name(b):>15} {'Ratio':>8}")
    print("-" * 80)

    def row(label, va, vb, unit="ms", lower_is_better=True):
        if va is None or vb is None:
            print(f"  {label:<38} {'N/A':>15} {'N/A':>15} {'N/A':>8}")
            return
        ratio = va / vb if vb != 0 else float("inf")
        faster = dev_a if (ratio < 1 and lower_is_better) else dev_b
        marker = f"← {faster} faster" if abs(ratio - 1) > 0.05 else "≈ equal"
        print(f"  {label:<38} {va:>12.2f}{unit} {vb:>12.2f}{unit} {ratio:>7.2f}x  {marker}")

    # Attention
    row("Attn eager SDPA",
        a["attention"]["eager_sdpa_ms"], b["attention"]["eager_sdpa_ms"])

    # Transformer
    ta = a.get("transformer", {})
    tb = b.get("transformer", {})
    row("Transformer step",
        ta.get("mean_step_ms"), tb.get("mean_step_ms"))
    row("Peak memory",
        ta.get("peak_memory_gb"), tb.get("peak_memory_gb"), unit=" GB", lower_is_better=True)

    # Fused kernel
    fka = a.get("fused_kernel", {})
    fkb = b.get("fused_kernel", {})
    row("Eager entropy",
        fka.get("eager_entropy_ms"), fkb.get("eager_entropy_ms"))
    row("Fused kernel",
        fka.get("fused_kernel_ms"), fkb.get("fused_kernel_ms"))

    # Loss comparison
    print(f"\n{'Loss curves':}")
    la = ta.get("losses", [])
    lb = tb.get("losses", [])
    if la and lb:
        print(f"  {dev_a}: {la}")
        print(f"  {dev_b}: {lb}")
        if len(la) == len(lb):
            diffs = [abs(a - b) for a, b in zip(la, lb)]
            print(f"  Diff:  {[round(d, 4) for d in diffs]}")
            print(f"  Max loss diff: {max(diffs):.4f} ({'OK ✓' if max(diffs) < 0.5 else 'LARGE ⚠'})")

    print(f"\n  NaN check:")
    print(f"  {dev_a}: {ta.get('nan_count', '?')} NaN steps")
    print(f"  {dev_b}: {tb.get('nan_count', '?')} NaN steps")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["xpu", "cuda", "cpu"], default="xpu")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", default=None)
    parser.add_argument("--compare", nargs=2, metavar=("FILE_A", "FILE_B"),
                        help="Compare two result JSON files")
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
    else:
        results = run_benchmarks(args.device, args.model)

        output = args.output or f"results_{args.device}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output}")
