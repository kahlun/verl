# NVIDIA A100 Benchmark Results

**Date**: 2026-04-03  
**Hardware**: NVIDIA A100 80GB PCIe  
**Software**: torch 2.10.0+cu129 · vllm 0.17.0 · verl 0.8.0.dev · flash_attn ✓  
**Purpose**: NVIDIA baseline for comparison against XPU (Intel Arc Pro B60 24GB)

---

## 1. Microbenchmark (`xpu_vs_cuda_benchmark.py`)

| Benchmark | XPU B60 | CUDA A100 | A100 Speedup |
|-----------|---------|-----------|-------------|
| Attention SDPA (eager) | 0.584 ms | 0.060 ms | **10.3×** |
| Attention (flash_attn) | N/A | 0.090 ms | — |
| Transformer fwd+bwd | 874.6 ms | 164.5 ms | **5.3×** |
| Fused entropy kernel | 5.029 ms | 0.684 ms | **7.4×** |
| Peak memory | 4.64 GB | 4.66 GB | ≈ equal |
| Loss curve numerical diff | — | max diff 0.0485 | ✓ match |

Raw data: `results_a100.json` (CUDA), `results_xpu_b60.json` (XPU baseline)

---

## 2. SFT 4-GPU (FSDP)

| Setting | Value |
|---------|-------|
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Dataset | GSM8K (converted to SFT format) |
| GPUs | 4× A100 80GB |
| FSDP | ✓ |
| Loss start → end | 7.0 → 5.3 |
| Status | ✅ Completed |

---

## 3. GRPO Training

### Single GPU (1× A100)

| Setting | Value |
|---------|-------|
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Dataset | GSM8K |
| Epochs | 1 (29 steps) |
| Final val accuracy | **50.7%** (from ~0.1% baseline) |

### 4-GPU (4× A100)

| Setting | Value |
|---------|-------|
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Dataset | GSM8K |
| Epochs | 1 (116 steps, batch=64) |
| Avg step time | 12.2 s/step |
| Avg throughput | 2,585 tokens/s |
| Final val accuracy | **53.7%** |

Accuracy curve (every 5 steps):
`0.5% → 7.9% → 20.7% → 33.5% → 42.5% → 46.1% → 47.1% → ... → 53.7%`

---

## 4. VLM SFT (1× A100)

| Setting | Value |
|---------|-------|
| Model | Qwen/Qwen2-VL-2B-Instruct |
| Dataset | llamafactory/pokemon-gpt4o-captions (1499 train) |
| Epochs | 1 (187 steps) |
| Avg step time | ~1.12 s/step |
| Total time | ~4 min |
| Avg MFU | 11.4% |
| Loss start → end | 2.676 → 1.780 |
| FSDP strategy | fsdp2, ulysses SP=1 |
| Status | ✅ Completed |

---

## Files

| File | Description |
|------|-------------|
| `results_a100.json` | Raw microbenchmark results (CUDA A100) |
| `results_xpu_b60.json` | Raw microbenchmark results (XPU B60, baseline) |
| `cuda_a100_findings.json` | Full structured findings (all tasks) |
| `xpu_vs_cuda_benchmark.py` | Benchmark script (run with `--device cuda` or `--device xpu`) |

## Reproduce

```bash
# Microbenchmark
python3 benchmark/nvidia/xpu_vs_cuda_benchmark.py --device cuda --output results_a100.json

# Compare with XPU
python3 benchmark/nvidia/xpu_vs_cuda_benchmark.py --compare results_xpu_b60.json results_a100.json

# GRPO single GPU
CUDA_VISIBLE_DEVICES=0 python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=/root/data/gsm8k/train.parquet \
  data.val_files=/root/data/gsm8k/test.parquet \
  data.train_batch_size=256 data.max_prompt_length=512 data.max_response_length=512 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=5 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.n_gpus_per_node=1 trainer.total_epochs=1

# VLM SFT single GPU
torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  -m verl.trainer.sft_trainer \
  data.train_files=/root/data/pokemon/train.parquet \
  data.train_batch_size=8 data.max_length=2048 \
  data.pad_mode=no_padding data.use_dynamic_bsz=True \
  data.max_token_len_per_gpu=16384 \
  model.path=Qwen/Qwen2-VL-2B-Instruct \
  engine=fsdp optim=fsdp optim.lr=2e-5 \
  engine.ulysses_sequence_parallel_size=1 engine.strategy=fsdp2 \
  trainer.total_epochs=1
```
