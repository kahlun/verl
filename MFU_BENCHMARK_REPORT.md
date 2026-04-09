# MFU Benchmark Report — Llama-3.1-8B SFT on 4× A100 80GB PCIe

## Test Environment

| Item | Detail |
|------|--------|
| **Server** | skyocean, shared multi-user |
| **GPUs** | 4× NVIDIA A100 80GB **PCIe** (host GPUs 1,4,5,6) |
| **Interconnect** | PCIe Gen4 x16 (~32 GB/s bidirectional, ~16 GB/s effective per direction) |
| **No NVLink** | These are PCIe cards, not SXM. NVLink (600 GB/s) is only on A100 SXM systems (DGX/HGX). |
| **Model** | Meta-Llama-3.1-8B-Instruct (~8B parameters, bfloat16) |
| **Task** | Supervised Fine-Tuning (SFT) on GSM8K math dataset |
| **Dataset** | GSM8K train.parquet / test.parquet |
| **Steps** | 30 training steps + 1 validation |
| **Framework** | verl (volcengine RL framework) |
| **Container** | Docker container `kahlun` with CUDA 12.8, PyTorch 2.8+ |

---

## What is MFU?

**MFU (Model FLOP Utilization)** measures what percentage of a GPU's peak compute is doing useful math.

$$\text{MFU} = \frac{\text{Measured TFLOPS}}{\text{Peak TFLOPS per GPU} \times \text{Number of GPUs}} \times 100\%$$

- **A100 peak**: 312 TFLOPS (BF16 Tensor Core)
- **4× A100 peak**: 1,248 TFLOPS total
- If training achieves 10% MFU → only ~125 TFLOPS of 1,248 is useful math; rest is communication, memory, overhead

**Higher = better.** In practice:
- 40-50% on a single GPU is good
- 30-40% on multi-GPU NVLink is good
- On PCIe, expect much lower because communication is slow

### How MFU Is Computed in verl

All engines use the same code: `verl/utils/flops_counter.py`

```
estimated_flops = (6 × model_params × tokens_processed + attention_flops) / step_time / 1e12
MFU = estimated_flops / (312 TFLOPS × num_GPUs)
```

The `6×` factor accounts for forward (2×) + backward (4×) passes. The MFU formula is **identical** across all three engines (TorchTitan, Megatron, FSDP), so the numbers are directly comparable.

---

## Glossary of Parallelism Strategies

### TP (Tensor Parallelism)
**Splits each weight matrix across GPUs.** Each GPU holds a slice of every layer.
- Communication: All-reduce on **activations** after each layer (small, proportional to hidden_size × batch × seq_len)
- Memory: Each GPU holds 1/TP of model weights
- Best for: Within a node (needs fast interconnect)

### PP (Pipeline Parallelism)
**Splits layers across GPUs.** GPU 0 does layers 0-15, GPU 1 does layers 16-31.
- Communication: Point-to-point **send/recv** of hidden states between stages (tiny — just one tensor per micro-batch)
- Memory: Each GPU holds 1/PP of model layers
- Downside: Pipeline bubbles (GPUs idle waiting for their turn)

### DP (Data Parallelism) — Megatron style
**Each GPU holds a full copy of the model, splits the data.** After backward pass, gradients are all-reduced.
- Communication: All-reduce **gradients** once per step (~16GB for 8B model in bf16)
- Memory: Each GPU needs to fit the full model + optimizer states

### FSDP2 (Fully Sharded Data Parallelism v2) — TorchTitan style
**Shards model parameters AND data across GPUs.** Before each layer's forward, it must all-gather the full weights, then discard after use.
- Communication: **All-gather weights** before every forward layer + **reduce-scatter gradients** after backward
- Memory: Each GPU holds 1/FSDP_size of model weights (much less memory than DP)
- Trade-off: Saves memory vs DP, but communicates much more data per step

### Key Difference: DP vs FSDP2

| Aspect | Megatron DP | TorchTitan FSDP2 |
|--------|------------|------------------|
| What each GPU stores | **Full model copy** | **1/N of model weights** |
| When communication happens | Once after backward (gradient sync) | Before **every layer** forward (weight gather) + after backward |
| Total bytes moved per step | ~16GB (gradients only) | ~32GB+ (weights forward + gradients backward) |
| Memory per GPU | High (full model + optim) | Low (sharded) |
| Best for | Enough memory, fast bus | Memory-constrained, fast bus |

---

## Config Differences Between Engines

The experiments are **NOT identical configs** because each engine has different constraints:

| Setting | TorchTitan experiments | Megatron experiments | FSDP experiment |
|---------|----------------------|---------------------|-----------------|
| Engine | `torchtitan` | `megatron` (mcore via mbridge) | `fsdp` |
| `train_batch_size` | 128 | 64 | 128 |
| `max_token_len_per_gpu` | 2048 | 1024 | 2048 |
| `micro_batch_size_per_gpu` | (auto) | 1 | (auto) |
| Tokens per step (approx) | ~25,000 | ~12,500 | ~25,000 |
| Optimizer | AdamW (native) | Distributed Optimizer (mcore) | AdamW (native) |
| Attention | FlexAttention (Triton) | FlashAttention2 + TransformerEngine | FlashAttention2 |
| Sequence Parallel | TP Shard(1) on DTensor | Megatron SP (allgather+scatter) | None |
| lr schedule | cosine (warmup ratio 0.2) | constant after warmup (6 steps) | cosine (warmup ratio 0.2) |
| `use_remove_padding` | Yes | Yes | Yes |
| dtype | bfloat16 | bfloat16 | bfloat16 |

**Why different batch sizes?** Megatron's distributed optimizer + mcore has higher memory overhead, so `train_batch_size=64` and `max_token_len_per_gpu=1024` were needed to avoid OOM. TorchTitan/FSDP could handle 128/2048 because FSDP2 shards parameters.

**Impact on MFU:** MFU measures FLOPS/time, so different batch sizes don't invalidate the comparison — larger batches mean more tokens per step but also proportionally more compute. The MFU formula normalizes by actual tokens processed.

---

## Results — Full Matrix

### Successfully Completed (30 steps + validation)

| Rank | Engine | Parallelism | Steady MFU | Val Loss | ~Tokens/step | Notes |
|------|--------|------------|-----------|----------|-------------|-------|
| 1 | **Megatron** | TP=2, PP=2, DP=1 | **21.84%** | 0.5627 | ~12,300 | Best overall. Checkpoint save has mcore bug. |
| 2 | **Megatron** | PP=2, DP=2 | **19.93%** | 0.5617 | ~12,300 | Checkpoint save bug only. |
| 3 | **Megatron** | TP=2, DP=2 | **16.23%** | 0.5609 | ~12,300 | Fully working, no bugs. |
| 4 | **TorchTitan** | TP=2, FSDP2=2 | **10.62%** | 0.4921 | ~25,000 | Best TorchTitan. Required seq padding fix. |
| 5 | **TorchTitan** | TP=4, FSDP2=1 | **9.47%** | 0.4922 | ~25,000 | Pure TP, still limited by PCIe. |
| 6 | **TorchTitan** | FSDP2=4, TP=1 | **5.85%** | 0.4921 | ~25,000 | Pure FSDP2, worst PCIe bottleneck. |
| 7 | **FSDP (verl)** | DP=4 | **5.79%** | 0.4683 | ~25,000 | Baseline. Similar to TorchTitan FSDP2=4. |

**MFU values above are averaged from the last 10 steps (steps 21-30) to exclude warmup.**

### Failed Experiments

| Engine | Parallelism | Failure | Reason |
|--------|------------|---------|--------|
| **TorchTitan** | PP=2, FSDP2=2 | Crash | verl's PP integration has API mismatch with TorchTitan's pipeline schedule |
| **TorchTitan** | PP=2, TP=2 | Crash | Same PP integration bug |
| **TorchTitan** | PP=2, FSDP2=1 | Crash | Same PP integration bug |
| **Megatron** | TP=1, DP=4 | OOM | 8B model × 4 full copies + mcore optimizer doesn't fit in 80GB/GPU |

### Observations on Validation Loss

- **Megatron val/loss (~0.56)** vs **TorchTitan val/loss (~0.49)**: These are NOT comparable! Megatron used `train_batch_size=64` with `max_token_len_per_gpu=1024`, while TorchTitan used `128/2048`. Different effective batch sizes and learning rate schedules lead to different convergence behavior in 30 steps. Both would converge to similar values given enough steps.
- **FSDP val/loss (0.468)**: Uses cosine LR decay which was more aggressive in 30 steps, leading to apparently lower loss.

---

## Detailed Data — Steady State (Steps 21-30)

### Megatron TP=2, PP=2, DP=1 (Best)

| Step | MFU% | Loss | Grad Norm | Tokens |
|------|------|------|-----------|--------|
| 21 | 21.65 | 0.585 | 3.77 | 12,315 |
| 22 | 21.68 | 0.590 | 3.67 | 12,681 |
| 23 | 21.71 | 0.520 | 3.42 | 11,659 |
| 24 | 21.89 | 0.570 | 3.60 | 12,650 |
| 25 | 21.91 | 0.575 | 3.23 | 12,623 |
| 26 | 22.19 | 0.558 | 3.28 | 12,229 |
| 27 | 21.74 | 0.540 | 3.27 | 12,437 |
| 28 | 22.12 | 0.570 | 3.19 | 12,246 |
| 29 | 21.54 | 0.514 | 3.72 | 11,355 |
| 30 | 22.03 | 0.528 | 3.06 | 12,165 |
| **Avg** | **21.84** | **0.555** | | |
| **Val** | | **0.5627** | | |

### TorchTitan TP=2, FSDP2=2 (Best TorchTitan)

| Step | MFU% | Loss | Grad Norm | Tokens |
|------|------|------|-----------|--------|
| 21 | 10.15 | 0.513 | 5.22 | 24,508 |
| 22 | 10.22 | 0.495 | 4.55 | 25,226 |
| 23 | 10.26 | 0.485 | 3.45 | 25,156 |
| 24 | 10.45 | 0.497 | 3.40 | 25,895 |
| 25 | 11.82 | 0.467 | 3.01 | 23,596 |
| 26 | 10.42 | 0.454 | 2.84 | 25,361 |
| 27 | 10.13 | 0.452 | 2.79 | 24,841 |
| 28 | 10.41 | 0.455 | 2.54 | 25,908 |
| 29 | 10.33 | 0.490 | 2.51 | 25,378 |
| 30 | 12.02 | 0.449 | 2.41 | 24,311 |
| **Avg** | **10.62** | **0.476** | | |
| **Val** | | **0.4921** | | |

### TorchTitan FSDP2=4, TP=1 (Pure FSDP2)

| Step | MFU% | Loss | Grad Norm | Tokens |
|------|------|------|-----------|--------|
| 21 | 5.59 | 0.513 | 5.16 | 24,508 |
| 22 | 5.73 | 0.495 | 4.50 | 25,226 |
| 23 | 5.68 | 0.485 | 3.48 | 25,156 |
| 24 | 5.88 | 0.497 | 3.45 | 25,895 |
| 25 | 7.12 | 0.467 | 3.03 | 23,596 |
| 26 | 5.74 | 0.454 | 2.82 | 25,361 |
| 27 | 5.61 | 0.452 | 2.80 | 24,841 |
| 28 | 5.85 | 0.456 | 2.54 | 25,908 |
| 29 | 5.78 | 0.490 | 2.47 | 25,378 |
| 30 | 5.50 | 0.450 | 2.44 | 24,311 |
| **Avg** | **5.85** | **0.476** | | |
| **Val** | | **0.4921** | | |

---

## Why Megatron Beats TorchTitan on This Hardware

### Reason 1: FSDP2 vs TP+PP Communication Volume

On **A100 PCIe (~16 GB/s effective)**, communication dominates:

| Configuration | What gets sent over PCIe per step | Est. bytes | Est. time at 16 GB/s |
|--------------|-----------------------------------|-----------|---------------------|
| **Megatron TP=2, PP=2, DP=1** | TP activation all-reduce (small) + PP P2P hidden states (tiny) | ~1-2 GB | ~0.1s |
| **TorchTitan TP=2, FSDP2=2** | TP activation all-reduce + FSDP2 all-gather weights (~8GB) + reduce-scatter grads (~8GB) | ~17 GB | ~1s |
| **TorchTitan FSDP2=4, TP=1** | FSDP2 all-gather full weights (~16GB) + reduce-scatter grads (~16GB) | ~32 GB | ~2s |

Megatron TP=2+PP=2 has **DP=1** (no data parallelism at all), so there is **no weight or gradient synchronization across GPUs**. Each GPU permanently holds its own weights. TorchTitan FSDP2 must shuffle weights every step.

### Reason 2: TransformerEngine Fused Kernels

Megatron uses NVIDIA's TransformerEngine which fuses matmul + communication into a single GPU kernel (overlapping compute with TP all-reduce). TorchTitan uses PyTorch DTensor which runs them sequentially.

### Reason 3: Not a Fair PP Comparison

The fairest TorchTitan comparison to Megatron TP=2+PP=2 would be `TorchTitan TP=2+PP=2+FSDP2=1`, which would also avoid FSDP2 overhead entirely. But this config crashed due to a verl integration bug, so we couldn't test it.

---

## Would This Change on NVLink / Bigger Clusters?

**Yes, dramatically.**

| System | PCIe BW | FSDP2 all-gather 16GB | Impact |
|--------|---------|----------------------|--------|
| Your 4× A100 PCIe | 16 GB/s | ~1 second | 90% of step = communication |
| 8× A100 SXM (NVLink) | 600 GB/s | ~0.027s | 3% of step = communication |
| 256× H100 cluster + IB | 450+ GB/s | negligible | Compute-bound |

On NVLink/SXM systems, FSDP2 communication is **37× faster**. The gap between TorchTitan and Megatron would shrink dramatically. Meta uses FSDP2 (via TorchTitan) for training Llama-3 405B on thousands of GPUs — it scales well on proper hardware.

**FSDP2 does NOT get worse with more GPUs** — it gets worse with **slow interconnect**. Your PCIe system is the worst case.

---

## What TorchTitan Is and Why It Uses FSDP2

TorchTitan is PyTorch's **official reference implementation** for distributed training. It is designed to showcase PyTorch-native primitives:

| Feature | TorchTitan | Megatron |
|---------|-----------|---------|
| Data parallelism | **FSDP2 (always)** — shards params via DTensor | DDP (full model copy per GPU) |
| Tensor parallelism | DTensor Shard/Replicate plans | Custom column/row parallel linear |
| Pipeline parallelism | `torch.distributed.pipelining` | Custom interleaved 1F1B schedule |
| Kernels | PyTorch native + FlexAttention (Triton) | TransformerEngine (NVIDIA custom) |
| torch.compile | Supported | Not supported |

**There is no "plain DP" mode in TorchTitan.** When you set `data_parallel_shard_size=4`, it always means FSDP2. This is by design — TorchTitan's purpose is to prove FSDP2+DTensor as the modern PyTorch way.

---

## Known Bugs and Issues Found

### 1. TorchTitan PP Integration Bug (verl)
TorchTitan's Pipeline Parallel schedule API expects different input format than verl passes. All PP configs crashed. This is a verl integration issue, not a TorchTitan bug.

### 2. TorchTitan TP Sequence Length Alignment (fixed)
When `use_remove_padding=True`, packed sequences have arbitrary total length. TorchTitan's TP Sequence Parallel requires `total_tokens % (tp × cp × 2) == 0`. Root cause: PyTorch DTensor complex number bug ([pytorch/pytorch#130646](https://github.com/pytorch/pytorch/issues/130646)) forced TorchTitan to use `use_local_output=True`.

**Fix applied in verl**: Pad input_ids/position_ids to nearest multiple of `seq_len_divisor` in `prepare_model_inputs()`, strip padding in `prepare_model_outputs()`. See `transformer_impl.py` lines 632-651.

### 3. Megatron Checkpoint Save Bug
Training completes successfully, but checkpoint saving crashes with mcore `pipeline_model_parallel_size > 1`. Affects TP=2+PP=2 and PP=2+DP=2 configs. Training metrics and val/loss are valid.

### 4. Megatron TP=1, DP=4 OOM
Each GPU needs full model copy + mcore distributed optimizer states. 8B model in bf16 = ~16GB weights + ~48GB optimizer states + activations > 80GB.

---

## Raw Data Files

All results are stored in `/home/kahlun/verl/mfu_comparison/`:

| File | Engine | Config | Steps | Status |
|------|--------|--------|-------|--------|
| `mcore-llama8b-tp2-pp2.jsonl` | Megatron | TP=2, PP=2, DP=1 | 30 | OK |
| `mcore-llama8b-pp2-dp2.jsonl` | Megatron | PP=2, DP=2 | 30 | OK |
| `mcore-llama8b-tp2-dp2.jsonl` | Megatron | TP=2, DP=2 | 30 | OK |
| `mcore-llama8b-tp1-dp4.jsonl` | Megatron | TP=1, DP=4 | 8 | OOM after 8 steps |
| `tt-llama8b-tp2-fsdp2-fixed.jsonl` | TorchTitan | TP=2, FSDP2=2 | 30 | OK (with seq padding fix) |
| `tt-llama8b-tp4-fsdp1.jsonl` | TorchTitan | TP=4, FSDP2=1 | 30 | OK |
| `tt-llama8b-tp1-fsdp4.jsonl` | TorchTitan | FSDP2=4, TP=1 | 30 | OK |
| `fsdp-llama8b-dp4.jsonl` | FSDP (verl) | DP=4 | 30 | OK |
| `tt-llama8b-pp2-fsdp2.jsonl` | TorchTitan | PP=2, FSDP2=2 | 0 | CRASHED |
| `tt-llama8b-pp2-fsdp2-v2.jsonl` | TorchTitan | PP=2, FSDP2=2 | 0 | CRASHED (retry) |
| `tt-llama8b-pp2-dp1.jsonl` | TorchTitan | PP=2, FSDP2=1 | 0 | CRASHED |
| `tt-llama8b-pp2-tp2.jsonl` | TorchTitan | PP=2, TP=2 | 0 | CRASHED |
| `fsdp-llama8b-1gpu.jsonl` | FSDP (verl) | 1 GPU | 0 | CRASHED (OOM) |

Experiment scripts: `/home/kahlun/verl/scripts/mfu_exps/`

---

## Corrections to Earlier Notes

### Incorrect: "TorchTitan beats Megatron on every config"
This was only true for **single-GPU small model** (Qwen 0.5B). For multi-GPU Llama-8B on our PCIe system, Megatron wins every completed config by 1.5-3.8×.

### Incorrect: "FSDP2 only syncs gradients, not activations"
FSDP2 syncs **weights** (all-gather before forward) AND **gradients** (reduce-scatter after backward). It's more communication than plain DP, not less. The advantage is **memory saving**, not communication reduction.

### Misleading: Comparing val/loss across engines
Val/loss differences (0.49 vs 0.56) reflect different batch sizes, learning rate schedules, and tokens per step — not model quality differences. With enough training, both converge similarly.

### Context needed: "40-50% single-GPU is considered good"
True for compute-only workloads. Our 21.84% best-case includes all parallelism overhead and is reasonable for 4× PCIe GPUs on an 8B model.
