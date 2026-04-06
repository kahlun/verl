# VERL XPU Reproducibility Guide

**Hardware:** 4× Intel Arc Pro B60 (Battlemage BMG-G21, 24 GB VRAM each, PCIe)
**Container:** `intel/vllm:0.14.1-xpu` (PyTorch 2.10.0+xpu)
**Host:** PyTorch 2.11.0+xpu, bitsandbytes 0.49.1
**Date of last run:** 2026-04-06

---

## Quick Start — Run All Unit Tests (< 30 seconds)

```bash
docker exec pt bash -c '
  cd /host/home/sdp/kl/verl_test_xpu &&
  ZE_AFFINITY_MASK=3 python3 test_t10_xpu_units.py
'
```

**Expected:** `Total: 12 PASS, 0 FAIL out of 12`

---

## Environment Setup (Required Before Any Multi-GPU Run)

```bash
# 1. Job timeout (sudo, one-time per reboot)
find /sys/devices -name "job_timeout_ms" -not -path "*/.defaults/*" | \
  while read f; do echo 10000 | sudo tee "$f" > /dev/null; done

# 2. Always set before torchrun / Ray
export CCL_BUFFER_CACHE=0
export RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1

# 3. Single-GPU tests: use GPU 3
export ZE_AFFINITY_MASK=3
```

---

## Per-Test Reproduction Commands

### P0 Tests

#### T1.1 — 1-GPU GRPO LoRA (Qwen2.5-0.5B, GSM8K)

```bash
docker exec pt bash -c '
  cd /host/home/sdp/kl/verl_test_xpu &&
  ZE_AFFINITY_MASK=3 \
  CCL_BUFFER_CACHE=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled \
  python3 run_xpu_ppo.py \
    data.train_files=/host/home/sdp/data/gsm8k/train.parquet \
    data.val_files=/host/home/sdp/data/gsm8k/test.parquet \
    data.train_batch_size=8 data.max_prompt_length=256 data.max_response_length=128 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.15 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    trainer.total_epochs=1 trainer.save_freq=-1 trainer.test_freq=-1 \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node=1 trainer.device=xpu \
    trainer.use_legacy_worker_impl=enable \
    "trainer.logger=[console]" \
    algorithm.adv_estimator=grpo \
    2>&1 | tee evidence/t1_1.log
'
```

**Expected:** 16+ steps complete, `perf/throughput` ≈ 80-126 tokens/sec, `actor/entropy` decreasing.

**Recorded metrics (step 20):**
| Metric | Value |
|--------|-------|
| actor/entropy | 0.497 |
| actor/loss | 8.6e-7 |
| perf/throughput | 125.6 tokens/sec |
| perf/time_per_step | 23.1s |
| response_length/mean | 252.2 |

#### T1.2 — 2-GPU GRPO LoRA

Same as T1.1 but with:
```bash
trainer.n_gpus_per_node=2
# Remove ZE_AFFINITY_MASK to use 2 GPUs
```

#### T1.3 — 1-GPU GRPO Full (no LoRA)

Same as T1.1 but add:
```bash
actor_rollout_ref.actor.use_lora=false
```

#### T1.4 — 4-GPU GRPO LoRA

Same as T1.1 but with:
```bash
trainer.n_gpus_per_node=4
# Remove ZE_AFFINITY_MASK to use all 4 GPUs
```

**Recorded:** 2 steps, 41s/step (legacy workers). Requires fresh reboot if GPU state is stale.

### T10.1 — OPO E2E (1-GPU)

Same as T1.1 but with:
```bash
algorithm.adv_estimator=opo
```

**Recorded:** Completed 2+ steps. Config validated, dataset loaded (7473 train, 1319 val).

---

### P1 Algorithm Variants (1-GPU)

All use the T1.1 base command with `algorithm.adv_estimator=<name>`:

| Test | adv_estimator | Extra Config | Status |
|------|--------------|--------------|--------|
| T2.1 | `grpo` | — | PASS (16 steps) |
| T2.2 | `rloo` | — | PASS (17 steps) |
| T2.3 | `reinforce_plus_plus` | — | PASS (19 steps) |
| T2.4 | `dppo_tv` / `dppo_kl` | `actor.loss_mode=dppo_tv` | PASS (19 steps) |
| T9.1 | `remax` | — | PASS (17 steps) |
| T9.2 | `grpo` | `actor.loss_mode=gspo` | PASS (17 steps) |
| T9.3 | `grpo` | `actor.loss_mode=sapo` | PASS (17 steps) |
| T9.4 | `grpo` | `actor.loss_mode=clip_cov` | PASS (19 steps) |
| T9.5 | `gpg` | `actor.loss_mode=gpg` | PASS (19 steps) |
| T9.7 | `grpo` | `actor.loss_mode=geo_mean` | PASS (17 steps) |
| T9.9 | `optimal_token` | `trainer.use_legacy_worker_impl=enable` | PASS (39 steps) |

---

### P2 Unit Tests — T10 Gap Coverage

These run as unit tests (no Ray/vLLM overhead):

```bash
docker exec pt bash -c '
  cd /host/home/sdp/kl/verl_test_xpu &&
  ZE_AFFINITY_MASK=3 python3 test_t10_xpu_units.py
'
```

| Test | Feature | Key Assertion |
|------|---------|---------------|
| T10.1 | OPO advantage | Output mean≈0, range [-13.7, 10.4] |
| T10.2 | kl_cov policy loss | Loss≈0.101, `torch.topk()` works on XPU |
| T10.3 | GRPO_PASSK | Only best-per-group gets nonzero advantage |
| T10.4 | RLOO_VECTORIZED | `torch.bincount()` leave-one-out works |
| T10.5 | GRPO_VECTORIZED | Vectorized group mean/std normalization |
| T10.6 | File logger | FileLogger creates JSONL output |
| T10.7 | Tensorboard logger | `_TensorboardAdapter` writes events |
| T10.8 | DAPO reward manager | Class registered, importable, has `run_single()` |
| T10.reg | All 14 estimators | All 14 advantage estimators resolve by name |
| T10.9.6 | GDPO advantage | 2-dim reward (accuracy+format), attention_mask must be `torch.long` |
| T10.9.10 | FAPO asymmetric clip | `clip_ratio_low=0.2`, `clip_ratio_high=0.28`, loss=0.108 |
| T10.9.11 | Distillation loss | 7 KL modes (k1,k3,kl,abs,mse,k2,low_var_kl) all finite on XPU |

---

### Multi-GPU Tests

#### T5.1 — Ulysses Sequence Parallelism (2-GPU)

```bash
docker exec pt bash -c '
  cd /host/home/sdp/kl/verl_test_xpu &&
  CCL_BUFFER_CACHE=0 RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  HYDRA_FULL_ERROR=1 WANDB_MODE=disabled \
  python3 run_xpu_ppo.py \
    data.train_files=/host/home/sdp/data/gsm8k/train.parquet \
    data.val_files=/host/home/sdp/data/gsm8k/test.parquet \
    data.train_batch_size=8 data.max_prompt_length=256 data.max_response_length=128 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.15 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    trainer.total_epochs=1 trainer.save_freq=-1 trainer.test_freq=-1 \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node=2 trainer.device=xpu \
    trainer.use_legacy_worker_impl=enable \
    "trainer.logger=[console]" \
    algorithm.adv_estimator=grpo \
    2>&1 | tee evidence/t5_1.log
'
```

**Recorded:** 16 steps, sp_size=2 on 2-GPU.

---

### SFT Tests

#### T6.1 — SFT 1-GPU

```bash
docker exec pt bash -c '
  cd /host/home/sdp/kl/verl_test_xpu &&
  ZE_AFFINITY_MASK=3 \
  CCL_BUFFER_CACHE=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python3 -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/host/home/sdp/data/gsm8k/train.parquet \
    data.val_files=/host/home/sdp/data/gsm8k/test.parquet \
    data.prompt_key=prompt data.response_key=answer \
    model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct \
    trainer.default_local_dir=./sft_output \
    trainer.total_epochs=1 \
    trainer.project_name=sft_xpu \
    trainer.experiment_name=sft_1gpu \
    trainer.logger=console \
    2>&1 | tee evidence/t6_1.log
'
```

---

### QLoRA (B1) — Verified on Host

```bash
python3 -c "
import torch
import bitsandbytes as bnb
print(f'bnb version: {bnb.__version__}')
linear = bnb.nn.Linear4bit(32, 32, bias=False, compute_dtype=torch.bfloat16)
linear = linear.to('xpu')
x = torch.randn(1, 32, dtype=torch.bfloat16, device='xpu')
y = linear(x)
print(f'Output shape: {y.shape}, device: {y.device}')
print('QLoRA forward pass: PASS')
"
```

**Requires:** Host environment (PyTorch 2.11+xpu, bitsandbytes 0.49.1).
Container needs `pip install bitsandbytes` first.

---

## Evidence Files

| File | Contents | In Git? |
|------|----------|---------|
| `test_t10_xpu_units.py` | 12 XPU unit tests (T10.1-T10.8, T10.reg, T9.6, T9.10, T9.11) | ✅ Yes |
| `run_xpu_ppo.py` | Ray XPU resource fix wrapper | ✅ Yes |
| `run_t10_batch.sh` | Gap-coverage test runner | ✅ Yes |
| `REPRODUCE.md` | This file — exact commands for every test | ✅ Yes |
| `evidence/` | Captured log excerpts from actual runs | ✅ Yes |
| `outputs/` | Full training logs (auto-generated, large) | ❌ Gitignored |

---

## MFU (Model FLOPs Utilization)

MFU is implemented in `verl/utils/flops_counter.py`. Arc Pro B60 added to
`_DEVICE_FLOPS` at ~96 TFLOPS BF16 (160 Xe2-cores × 256 BF16 ops/cycle × 2.35 GHz).

MFU will now appear in training logs as `perf/mfu/actor`, `perf/mfu/actor_infer`.
Previously reported 0.0 because Intel XPU wasn't in the device FLOPS dictionary.

---

## Capture Evidence Script

To re-run and capture all evidence:

```bash
# 1. Unit tests (fast, < 30s)
docker exec pt bash -c '
  cd /host/home/sdp/kl/verl_test_xpu &&
  ZE_AFFINITY_MASK=3 python3 test_t10_xpu_units.py 2>&1
' | tee evidence/t10_unit_tests.log

# 2. E2E 1-GPU GRPO (≈7 min for 16 steps)
# Use the T1.1 command above, tee to evidence/t1_1.log

# 3. Multi-GPU runs require Ray cleanup between runs:
docker exec pt bash -c 'ray stop --force; sleep 5'
```
