---
name: xpu-benchmark
description: Run the XPU performance benchmark and optionally compare with an A100 result. Produces JSON output that can be compared with results from another machine. Use when you want to measure XPU vs CUDA (A100) speed, loss parity, memory usage.
argument-hint: "[run|compare <file_a> <file_b>] default: run"
---

Run XPU benchmark at `/home/sdp/kl/verl_test_xpu`.

Action: $ARGUMENTS (default: run and save to benchmark/)

## Run benchmark (XPU)

```bash
cd /home/sdp/kl/verl_test_xpu
python3 benchmark/xpu_vs_cuda_benchmark.py \
  --device xpu \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output benchmark/results_xpu_$(date +%Y%m%d_%H%M).json
```

## Compare two results

If $ARGUMENTS contains two file paths, compare them:

```bash
cd /home/sdp/kl/verl_test_xpu
python3 benchmark/xpu_vs_cuda_benchmark.py \
  --compare <file_a> <file_b>
```

## If A100 results are needed (instructions for remote machine)

The benchmark script is self-contained. On the A100 machine:
```bash
# Copy only this one file:
scp benchmark/xpu_vs_cuda_benchmark.py user@a100:/tmp/

# Run on A100:
python3 /tmp/xpu_vs_cuda_benchmark.py --device cuda --output results_a100.json

# Copy result back:
scp user@a100:/tmp/results_a100.json benchmark/results_a100.json
```

Then compare:
```bash
python3 benchmark/xpu_vs_cuda_benchmark.py \
  --compare benchmark/results_xpu_b60.json benchmark/results_a100.json
```

## XPU B60 baseline (reference)

From `benchmark/results_xpu_b60.json`:
- Attn eager SDPA: 0.58 ms
- Transformer step (0.5B): 874.6 ms
- Peak memory: 4.64 GB
- Loss trajectory (5 steps): 3.80 → 1.98 → 1.01 → 0.49 → 0.24
- NaN count: 0

## Key things to check in comparison

1. **NaN count** — must be 0 on both machines (correctness)
2. **Loss diff** — should be < 0.05 per step (same random seed, same model)
3. **Speed ratio** — A100 expected ~3-4x faster for 0.5B
4. **Memory** — should be similar (model size dominated)
