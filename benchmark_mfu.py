"""
Standalone MFU benchmark for Intel XPU vs NVIDIA GPU comparison.
Runs pure forward+backward training loop (no Ray, no vLLM, no RL overhead).
Equivalent to TorchTitan SFT MFU measurement.

Usage:
  docker exec pt bash -c 'cd /host/home/sdp/kl/verl_test_xpu && ZE_AFFINITY_MASK=3 python3 benchmark_mfu.py'

Results (2026-04-06, Arc Pro B60, Qwen2.5-0.5B):
  bs=4 eager:    20.1% MFU, 19.3 TFLOPS, 6518 tok/s
  bs=8 eager:    22.6% MFU, 21.7 TFLOPS, 7307 tok/s
  bs=4 compiled: 27.5% MFU, 26.4 TFLOPS, 8902 tok/s
"""
import torch
import time
import json


def benchmark(model, optimizer, input_ids, labels, device, warmup=3, steps=10, label=""):
    """Run forward+backward benchmark and return MFU stats."""
    N = sum(p.numel() for p in model.parameters())
    bs, seq = input_ids.shape
    tokens = bs * seq

    # Device peak BF16 TFLOPS
    PEAKS = {"Arc(TM) Pro B60": 96e12, "A100": 312e12, "H100": 989e12}
    dev_name = torch.xpu.get_device_name(0) if device == "xpu" else torch.cuda.get_device_name(0)
    peak = next((v for k, v in PEAKS.items() if k in dev_name), 96e12)

    sync = torch.xpu.synchronize if device == "xpu" else torch.cuda.synchronize

    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad()
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        optimizer.step()
    sync()

    # Measure
    times = []
    for _ in range(steps):
        sync()
        t0 = time.perf_counter()
        optimizer.zero_grad()
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        optimizer.step()
        sync()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    flops_per_step = 6 * N * tokens  # fwd + bwd = 3x fwd, each fwd = 2x params
    tflops = flops_per_step / (avg * 1e12)
    mfu = tflops / (peak / 1e12) * 100
    tok_s = tokens / avg
    print(f"  {label}: {avg:.3f}s/step  {tflops:.1f} TFLOPS  MFU={mfu:.1f}%  {tok_s:.0f} tok/s")
    return {"time": avg, "tflops": round(tflops, 1), "mfu": round(mfu, 1), "tok_s": round(tok_s)}


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM

    device = "xpu" if torch.xpu.is_available() else "cuda"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"Loading {model_name} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device)
    model.train()
    N = sum(p.numel() for p in model.parameters())
    vocab = model.config.vocab_size
    print(f"Parameters: {N/1e6:.0f}M\n")

    results = {}

    # Eager mode benchmarks
    for bs in [4, 8]:
        seq = 512
        ids = torch.randint(0, vocab, (bs, seq), device=device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
        r = benchmark(model, opt, ids, ids.clone(), device, label=f"bs={bs} eager")
        results[f"bs{bs}_eager"] = r

    # torch.compile benchmark
    print("\nCompiling model...")
    compiled = torch.compile(model)
    ids = torch.randint(0, vocab, (4, 512), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    r = benchmark(compiled, opt, ids, ids.clone(), device, warmup=3, steps=10, label="bs=4 compiled")
    results["bs4_compiled"] = r

    # Save
    if device == "xpu":
        mem = torch.xpu.max_memory_allocated() / 1e9
    else:
        mem = torch.cuda.max_memory_allocated() / 1e9

    results["meta"] = {
        "model": model_name, "params_M": round(N / 1e6),
        "device": torch.xpu.get_device_name(0) if device == "xpu" else torch.cuda.get_device_name(0),
        "peak_mem_gb": round(mem, 1), "pytorch": torch.__version__,
    }

    out_path = "evidence/mfu_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
