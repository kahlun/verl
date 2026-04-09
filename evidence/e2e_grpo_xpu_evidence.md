# E2E GRPO Test Evidence — Intel XPU

**Date:** 2026-04-03
**Hardware:** Intel Arc Pro B60 (Battlemage), 24 GB VRAM, PCIe
**Container:** `intel/vllm:0.14.1-xpu`

---

## Environment

| Component | Version |
|-----------|---------|
| PyTorch | 2.10.0+xpu (Intel fork) |
| IPEX (Intel Extension for PyTorch) | 2.10.10.post1+xpu |
| vLLM | Intel fork (`intel/vllm`) |
| verl | kahlun/verl fork (PRs: A0+A1+A2+B+D applied) |
| Ray | 2.x |
| Python | 3.12 |

**Required verl fork PRs (all applied together):**

| PR | What it fixes |
|----|--------------|
| [A0](https://github.com/kahlun/verl/pull/12) | XPU device detection — `get_device_name()`, `get_nccl_backend()`, `auto_set_device()` |
| [A1](https://github.com/kahlun/verl/pull/13) | XCCL workarounds — `ReduceOp.AVG/MAX` not supported in oneCCL |
| [A2](https://github.com/kahlun/verl/pull/14) | Ray resource mapping — Ray sees `xpu` resource, not `GPU` |
| [B](https://github.com/kahlun/verl/pull/5) | FSDP workers — removes force-override to `eager` attention, uses SDPA |
| [D](https://github.com/kahlun/verl/pull/7) | vLLM rollout — `ONEAPI_DEVICE_SELECTOR`, `level_zero:` prefix, sleep mode fix |

**Environment variables required:**
```bash
ZE_AFFINITY_MASK=0                          # Pin to GPU 0
CCL_ATL_SHM=1                               # Use shared memory transport (bypasses L0 IPC bug on B60 PCIe)
CCL_BUFFER_CACHE=0                          # Required with CCL_ATL_SHM
RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1  # Prevent Ray overriding device selector
```

---

## Test: GRPO 1-GPU, Qwen2.5-0.5B-Instruct, GSM8K, 20 steps

**Full e2e loop:** vLLM rollout → reward scoring → FSDP train update → repeat

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    trainer.n_gpus_per_node=1 \
    trainer.total_training_steps=20
```

**Results (20 steps):**

```
step= 1 | loss=1.34e-06 | grad_norm=1.41e-04 | rollout=16.3s | train=6.8s
step= 5 | loss=1.30e-06 | grad_norm=1.35e-04 | rollout=9.1s  | train=4.8s
step=10 | loss=1.00e-06 | grad_norm=1.22e-04 | rollout=8.8s  | train=4.8s
step=15 | loss=8.53e-07 | grad_norm=9.87e-05 | rollout=15.9s | train=4.8s
step=20 | loss=8.63e-07 | grad_norm=7.62e-05 | rollout=8.9s  | train=4.9s
```

Loss decreasing (1.34e-06 → 8.63e-07), training stable. ✓

Full log: `evidence/t1_1_grpo_1gpu.log`

---

## Test: SFT 2-GPU, Qwen2.5-0.5B-Instruct, GSM8K, 3 steps

**Date:** 2026-04-09
**Backend:** FSDP, 2× Intel Arc Pro B60, `ZE_AFFINITY_MASK=0,1`, `CCL_ATL_SHM=1`

```
step=1 | train/loss=0.618 | grad_norm=22.7  | xpu:0 + xpu:1 (FSDP sharded)
step=2 | train/loss=3.552 | grad_norm=140.2
step=3 | train/loss=3.588 | grad_norm=62.9  | val/loss=2.620
```

Checkpoint saved: `global_step_3/model_world_size_2_rank_0.pt` + `rank_1.pt` ✓

Full log: `evidence/t2_2_sft_2gpu.log`

---

## Test: Attention Verification

**Date:** 2026-04-09

```python
from verl.utils.device import get_default_attention_implementation
get_default_attention_implementation()  # → "sdpa"
```

- `is_xpu_available = True` ✓
- `get_default_attention_implementation() = "sdpa"` (not "eager") ✓
- Model loaded with `attn_implementation="sdpa"` ✓
- Forward pass: no NaN, logits shape `[1, 64, 151936]` ✓

**What this means:** verl on XPU uses Intel's SYCL-TLA Flash kernel (built into PyTorch, dispatched via `F.scaled_dot_product_attention`).
Not the 2017 naive O(S²) matmul ("eager"). Not the CUDA-only `flash_attn` C++ package.
PyTorch's built-in SDPA — works on any backend (CUDA, XPU, MPS) with no extra package.
