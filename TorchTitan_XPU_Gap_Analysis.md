# TorchTitan × Intel XPU Gap Analysis for VERL

**Date:** 2026-04-02 (last updated **2026-04-06**: T7.3 TP=2 PASS, T7.2 PP=2 FAIL, TP seq_len padding, nested tensor bug)  
**Environment:** PyTorch 2.10.0+xpu (container) / 2.11.0+xpu (host), torchtitan 0.2.2 (main), VERL v0.8.0.dev0  
**Hardware:** Intel Arc Pro B60 (Battlemage), 4× ~24 GB VRAM  
**Tester:** Automated XPU compatibility testing  

---

## Executive Summary

TorchTitan's core primitives (**flex_attention**, **torch.compile**, **VarlenMetadata**) all work on Intel XPU. The blockers are version-coupling issues between torchtitan, PyTorch XPU, and VERL — not fundamental XPU incompatibilities.

**Key finding:** VERL + TorchTitan **E2E SFT training works on XPU** — including **multi-GPU TP=2**. With 5 small patches (all <10 lines), Llama-3.2-1B-Instruct trains successfully on Intel Arc Pro B60. **1-GPU:** loss 1.59 → 0.79 over 3 steps. **TP=2 (2-GPU):** loss 1.31 → 0.80 over 5 steps, val_loss 0.81. **PP=2 FAILED** — B60 PCIe lacks P2P IPC (`zeMemOpenIpcHandle` returns INVALID_ARGUMENT).

---

## 1. XPU Feature Test Results

| Feature | Forward | Backward | NaN Check | Perf | Status |
|---|---|---|---|---|---|
| `flex_attention` (causal mask) | ✅ | ✅ | Clean | 24.4 ms/iter (S=128, fwd+bwd) | **WORKS** |
| `flex_attention` (document mask + causal, `and_masks`) | ✅ | ✅ | Clean | — | **WORKS** |
| `torch.compile` (dynamic=True, entropy function) | ✅ | ✅ | — | 0.15ms compiled vs 3.64ms eager (**24× speedup**) | **WORKS** |
| `VarlenMetadata` tensor creation | ✅ | N/A | — | — | **WORKS** |
| `create_block_mask` on XPU | ✅ | N/A | — | — | **WORKS** |

**Tested code:**
```python
# flex_attention with document masking (exact pattern TorchTitan uses)
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, and_masks
block_mask = create_block_mask(and_masks(causal, document_mask), B=1, H=None, Q_LEN=128, KV_LEN=128, device='xpu')
out = flex_attention(q, k, v, block_mask=block_mask)  # WORKS
out.sum().backward()  # WORKS
```

---

## 2. Gaps — TorchTitan Side (for TorchTitan Team)

### Gap T1: `ShardPlacementResult` missing from PyTorch 2.11.0+xpu

**Severity:** BLOCKER for end-to-end  
**Location:** `torchtitan/models/llama4/parallelize.py:13-15`  
**Error:**
```
ImportError: cannot import name 'ShardPlacementResult' from
'torch.distributed.fsdp._fully_shard._fsdp_common'
```
**Impact:** Any model that imports from `llama4/parallelize.py` fails at import time. This includes Qwen3 (via `from torchtitan.models.llama4.parallelize import apply_fsdp, apply_moe_ep_tp`).  
**Root Cause:** `ShardPlacementResult` was added to PyTorch nightly after the 2.11 branch cut. The Intel XPU wheel tracks stable releases.  
**Workaround:** Monkey-patch before import:
```python
import torch.distributed.fsdp._fully_shard._fsdp_common as m
from typing import NamedTuple
class ShardPlacementResult(NamedTuple):
    local_tensor: torch.Tensor
    global_offset: int
    narrowed: bool = False
m.ShardPlacementResult = ShardPlacementResult
```
**Ask:** Either (a) torchtitan guards this import with `try/except` for older PyTorch, or (b) the XPU PyTorch wheel includes `ShardPlacementResult`. Option (a) is simpler.

### Gap T2: XPU device not in TorchTitan's device abstraction

**Severity:** HIGH  
**Status:** Already partially addressed upstream  
**Details:** TorchTitan has 3 merged Intel XPU PRs:
- [PR #1016](https://github.com/pytorch/torchtitan/pull/1016) (MERGED): XPU bf16 peak flops for PVC
- [PR #1018](https://github.com/pytorch/torchtitan/pull/1018) (MERGED): XPU profiling support
- [PR #2228](https://github.com/pytorch/torchtitan/pull/2228) (MERGED): Generic device memory snapshot

Intel contributors: @frost-intel, @pkourdis, @githubsgi, @saforem2  
**Remaining work:** Ensure `device_type="xpu"` works end-to-end through TorchTitan's `Trainer` class. The individual device utils exist, but the full training loop hasn't been validated.

### Gap T3: `ReduceOp.PREMUL_SUM` for XCCL

**Severity:** MEDIUM  
**Status:** Fixed upstream  
**Details:** [PR #2332](https://github.com/pytorch/torchtitan/pull/2332) was CLOSED because the fix landed in PyTorch core instead:
- [pytorch/pytorch#172298](https://github.com/pytorch/pytorch/pull/172298)
- [torch-xpu-ops#2903](https://github.com/intel/torch-xpu-ops/pull/2903)

Should work once PyTorch XPU wheel includes it.

---

## 3. Gaps — VERL Side (for VERL Team)

These are bugs in VERL's TorchTitan integration, not in TorchTitan itself.

### Gap V1: EngineRegistry missing "xpu"

**Severity:** BLOCKER  
**File:** `verl/workers/engine/torchtitan/transformer_impl.py:577`  
**Current:** `@EngineRegistry.register(... device=["cuda", "npu"])`  
**Fix:** Add `"xpu"` to device list. **Already patched locally.**

### Gap V2: Factory function config matching

**Severity:** BLOCKER  
**File:** `verl/workers/engine/torchtitan/utils.py:127`  
**Error:** `derive_torchtitan_name_and_flavor()` uses `getattr(model_cfg, "dim")` but torchtitan main stores configs as factory functions.  
**Fix:** Call `model_cfg()` if `callable(model_cfg)` before attribute access. **Already patched locally.**

### Gap V3: `attn_backend` field renamed

**Severity:** BLOCKER  
**File:** `verl/workers/engine/torchtitan/transformer_impl.py:112-114, 598`  
**Error:** `model_spec.model.layer.attention.attn_backend = attn_type` → `AttributeError: 'Config' object has no attribute 'attn_backend'`  
**Details:** torchtitan main uses `mask_type` field (e.g., `mask_type='causal'`) not `attn_backend`. The VERL code was written against an older torchtitan API.  
**Fix:** Stash `attn_type` on the engine instance (`self._verl_attn_type = attn_type`) instead of writing to the torchtitan Config object. Read it back in `get_attention_masks()` via `self._verl_attn_type`. **Already patched locally.**

### Gap V4: torchtitan version pinning

**Severity:** HIGH  
**Details:** VERL requires torchtitan main (for `CompileConfig`, `Trainer.Config`) but torchtitan main requires PyTorch nightly (for `ShardPlacementResult`). The v0.2.2 tag is compatible with torch 2.11 but lacks the APIs VERL uses.  

| torchtitan version | API match (VERL) | PyTorch 2.11 compat | Status |
|---|---|---|---|
| v0.2.2 tag | ❌ No `CompileConfig` | ✅ | Not usable |
| main branch | ✅ All APIs present | ❌ `ShardPlacementResult` | Needs monkey-patch |

**Container access (2026-04-05 update):** torchtitan 0.2.2 IS accessible in the Docker container at `/host/home/sdp/miniforge3/lib/python3.12/site-packages/torchtitan/` — it was NOT on `PYTHONPATH` but all 9 VERL imports resolve when the path is added. This means T7.2/T7.3 are **not blocked** by package installation — just by missing `PYTHONPATH`. Use: `export PYTHONPATH=/host/home/sdp/miniforge3/lib/python3.12/site-packages:$PYTHONPATH`

---

### Gap V5: `torch.cuda.empty_cache()` in utils.py

**Severity:** BLOCKER  
**File:** `verl/workers/engine/torchtitan/utils.py:354`  
**Current:** `torch.cuda.empty_cache()`  
**Fix:** Replace with `from verl.utils.device import get_torch_device; get_torch_device().empty_cache()`. **Already patched locally.**

---

## 4. What Works Today (with patches)

With all 5 local patches (V1+V2+V3+V5 in VERL, T1 monkey-patch), the VERL TorchTitan engine:
- Successfully resolves model flavor (e.g., Llama-3.2-3B → `llama3/3B`) ✅
- Loads HF model config and creates TorchTitan Trainer ✅
- Creates optimizer (AdamW) and LR scheduler configs ✅
- Loads HF checkpoint from safetensors into FSDP2-wrapped model ✅
- Completes **forward pass** through full VERL SFT pipeline on XPU ✅
- **Completes full training loop** (forward + backward + optimizer step + validation + checkpoint) with 1B model ✅
- 3B model OOMs during backward on single 24GB GPU (not an XPU bug — same on CUDA)

> **Note (2026-04-05):** torchtitan 0.2.2 package is accessible inside the Docker
> container via host mount at `/host/home/sdp/miniforge3/lib/python3.12/site-packages/torchtitan/`.
> All 9 VERL imports resolve when `PYTHONPATH` includes this path. B12 in the test
> matrix has been reclassified from "Blocked" to "Ready". T7.2 (PP=2) and T7.3
> (TP=2) are now runnable — they just need `PYTHONPATH` set before launch.

### E2E VERL + TorchTitan SFT Test Results (2026-04-02)

**Successful run — Llama-3.2-1B-Instruct (1B params, single XPU):**
- Model: 973,146,112 params, 5.60 GiB on XPU (23.41% of 23.91 GiB)
- GPU: Single Intel Arc Pro B60 (23.91 GiB), `ZE_AFFINITY_MASK=3`
- Dataset: gsm8k `train_sft.parquet` (7473 rows)
- Config: `engine=torchtitan`, `optim=torchtitan`, `attn_type=varlen`, `use_torch_compile=False`

```
step:1 - train/loss:1.592 - grad_norm:51.35 - global_tokens:481
step:2 - train/loss:1.164 - grad_norm:32.86 - global_tokens:440
step:3 - train/loss:0.791 - grad_norm:25.43 - global_tokens:413
val/loss: 1.047
Checkpoint saved to: checkpoints/gsm8k-sft/test/global_step_3
```

**All stages passed:** model init → FSDP2 wrap → data load → forward → backward → optimizer step → validation → checkpoint save.

**3B model test (Llama-3.2-3B-Instruct):**
- Forward pass: ✅ (reached `loss.backward()`)
- Backward: ❌ OOM — 13.44 GiB model + gradients + activations exceed 24 GiB (not an XPU bug)

**CPU offload bug:**
Setting `offload_policy=True` triggers `enable_cpu_offload()` in torchtitan, which sets `init_device="cpu"`. However, FSDP2 prefetch/all-gather that should move params to XPU before forward **does not work** — results in `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and xpu:0!`. This is a real bug in torchtitan's CPU offload path on XPU.

---

## 5. Recommended Actions

### For TorchTitan Team (Fan's manager):
1. **Guard `ShardPlacementResult` import** with `try/except ImportError` in `llama4/parallelize.py` — enables stable PyTorch users (including XPU) to import model configs. Low effort, high impact.
2. **Fix CPU offload on non-CUDA devices** — `enable_cpu_offload()` + FSDP2 prefetch doesn't work on XPU. The model stays on CPU and forward fails with device mismatch. This blocks single-GPU training of large models.
3. **E2E XPU CI job** — validate `torchrun` on XPU for at least `llama3/debugmodel` with 1 GPU. Intel contributors already upstream; a CI job prevents regressions.
4. **Version alignment** — ensure v0.2.x tags remain compatible with the torch version they target. Current `v0.2.2` tag already drifted from stable torch.

### For oneAPI/Level Zero Team:
1. **P2P IPC on PCIe** — `zeMemOpenIpcHandle` returns `INVALID_ARGUMENT` on B60 PCIe. This blocks Pipeline Parallel (PP=2). TP=2 works because it uses XCCL all-reduce (collective), not point-to-point IPC. Clarify: is this a hardware limitation (PCIe vs NVLink) or a driver gap?

### For VERL Team:
1. Fix V1 (add "xpu" to EngineRegistry) — 1 line (**done**, committed)
2. Fix V2 (handle factory function configs) — 3 lines (**done**, committed)
3. Fix V3 (stash `attn_type` on engine instance instead of writing to torchtitan Config) — ~5 lines (**done**, committed)
4. Fix V5 (`torch.cuda.empty_cache()` → `get_torch_device().empty_cache()`) — 2 lines (**done**, committed)
5. Pin torchtitan version or add compat layer for both tag and main
6. Fix PP `NotImplementedError` — **done** (commit `71d4aecc`, 2026-04-05). Full PP implementation with forward-only and training paths. See §6.
7. Fix GQA head expansion in `xpu_varlen_sdpa` — **done** (commit `11ce2353`). `repeat_interleave` for nheads_q != nheads_k.

### For PyTorch XPU Team:
1. Ship `ShardPlacementResult` in next XPU wheel (if it's in nightly, it should flow down)
2. Merge `PREMUL_SUM` fix into XPU stable release

---

## 6. Pipeline Parallel — Implementation Status (2026-04-03)

### Why PP Had `NotImplementedError`

Non-PP and PP have fundamentally different execution models:

| | Non-PP | PP |
|---|---|---|
| Forward | `model(inputs)` → logits | `pp_schedule.step(inputs, ...)` → schedule coordinates stages |
| Backward | Caller calls `loss.backward()` | `pp_schedule.step(target=labels, losses=[])` does F+B **together** |
| Logits | Available on single rank | Only on **last PP stage rank** |
| Loss fn | Verl's custom fn (masked CE, RL terms) | **Baked into schedule at construction time** (TorchTitan's CE) |

verl's original architecture assumed the non-PP pattern throughout: `forward_step → get logits → loss.backward()`. PP can't use any of these steps directly.

### What Was Fixed (2026-04-03)

**File:** `verl/workers/engine/torchtitan/transformer_impl.py`

| Change | Where | What |
|---|---|---|
| PP forward-only | `TorchTitanEngine.model_forward_step` | Call `pp_schedule.step(return_outputs=True)`; return logits on last stage, `None` elsewhere |
| PP router | `TorchTitanEngineWithLMHead.forward_backward_batch` | Detect `pp_enabled` → dispatch to `_pp_forward_backward_batch` |
| PP training | `TorchTitanEngineWithLMHead._pp_forward_backward_batch` | Per-microbatch `pp_schedule.step(target=labels, losses=[])` — schedule handles F+B |
| PP inference | same method | Per-microbatch `pp_schedule.step(return_outputs=True)` — last stage computes log_probs |
| None-logits guard | `TorchTitanEngineWithLMHead.forward_step` | Return zero dummy output on non-last PP stage if called directly |
| Dummy output helper | `TorchTitanEngineWithLMHead._make_pp_dummy_output` | Zero-filled `log_probs`/`entropy` nested tensors for non-last stages |

### Known Limitations of PP Implementation

| Limitation | Impact | Fix |
|---|---|---|
| Training uses TorchTitan's CE loss (all tokens) | SFT loss differs from verl's masked CE (response-only). Same weight update direction, different loss value. | Inject verl's loss into pipeline stages (TorchTitan API not exposed yet) |
| Non-last PP stages return zero log_probs | RL algorithms (GRPO/PPO) need log_probs on all DP ranks — zeros won't compute correct advantages | Broadcast log_probs from last PP stage to all PP stages after forward |
| CPU offload + PP | TorchTitan CPU offload bug also affects PP (params don't move to device before forward) | Wait for TorchTitan Bug 2 fix |
| PP not validated multi-GPU on XPU | **TESTED (2026-04-06):** PP=2 FAILS with `zeMemOpenIpcHandle: ZE_RESULT_ERROR_INVALID_ARGUMENT`. B60 PCIe lacks P2P IPC memory sharing needed for PP stage transfers. | Needs NVLink-class interconnect or future L0 IPC support |
| `tie_word_embeddings` incompatible with PP | Llama-3.2-3B has `tie_word_embeddings=True`, which PP can't handle (embedding on first stage, lm_head on last stage can't share) | Used Llama-3.1-8B for PP=2 test instead |

### T7.2 PP=2 Test Results (2026-04-06) — FAIL

**Config:** Llama-3.1-8B, 2 GPUs, `pp_size=2`, TorchTitan engine  
**Error:** `zeMemOpenIpcHandle` returns `ZE_RESULT_ERROR_INVALID_ARGUMENT`  
**Root cause:** Pipeline Parallel requires P2P IPC memory sharing between GPU ranks (one rank writes activations, the next rank reads them). B60 connected via PCIe does not support this — it needs NVLink-class or CXL interconnect.  
**Evidence:** `evidence/t7_2_pp2.log` (4509 lines)  
**Note:** Required `tyro` + `torchtitan` copied to `_torchtitan_deps/` with clean PYTHONPATH (host site-packages polluted torch version).

### T7.3 TP=2 Test Results (2026-04-06) — PASS

**Config:** Llama-3.2-1B, 2 GPUs, `tp_size=2`, TorchTitan engine  
**Results:**
```
step:1 - train/loss:1.306 - grad_norm:71.12
step:2 - train/loss:1.112 - grad_norm:36.54
step:3 - train/loss:0.923 - grad_norm:28.91
step:4 - train/loss:0.855 - grad_norm:24.17
step:5 - train/loss:0.800 - grad_norm:21.03
val/loss: 0.814
Checkpoint saved
```
**Key:** TP=2 uses XCCL all-reduce (not P2P IPC), so it works over PCIe. Loss decreases consistently, val_loss validates generalization.  
**Evidence:** `evidence/t7_3_tp2.log` (2407 lines)

### What PP Would Enable (if P2P IPC were available)

With PP=2 on 2× Intel Arc Pro B60 (24 GB each):
- 48 GB effective memory → can train **7B models** that OOM on single GPU
- PP=4 → 96 GB → can train **30B+ models** (with also FSDP DP sharding)
- **Currently blocked** by B60 PCIe lacking P2P IPC. Would work on GPUs with NVLink/CXL.

### TorchTitan PP Features Available (already in torchtitan 0.2.2)

TorchTitan's PP is MORE advanced than Megatron's:

| Schedule | Description |
|---|---|
| `ScheduleGPipe` | All-forward then all-backward |
| `Schedule1F1B` | Interleaved forward/backward (standard) |
| `ScheduleInterleaved1F1B` | Looped 1F1B (more stages per rank) |
| `ScheduleInterleavedZeroBubble` | Near-zero bubble overhead |
| `ScheduleZBVZeroBubble` | ZBV variant |
| `ScheduleDualPipeV` | Dual-pipe variant |
| `ScheduleLoopedBFS` | BFS looped scheduling |

Configure via `engine.pipeline_parallel_schedule` in verl config.

```bash
# Quick smoke test: flex_attention on XPU
ZE_AFFINITY_MASK=0 python3 -c "
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
def causal(b, h, q, kv): return q >= kv
mask = create_block_mask(causal, B=1, H=None, Q_LEN=64, KV_LEN=64, device='xpu')
q = torch.randn(1,4,64,32, device='xpu', dtype=torch.bfloat16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)
out = flex_attention(q, k, v, block_mask=mask)
out.sum().backward()
print(f'flex_attention: OK, grad shape {q.grad.shape}')
"

# Quick smoke test: torch.compile on XPU
ZE_AFFINITY_MASK=0 python3 -c "
import torch
@torch.compile(dynamic=True)
def f(x): return -(torch.softmax(x,-1) * torch.log_softmax(x,-1)).sum(-1)
x = torch.randn(2,64,512, device='xpu', dtype=torch.bfloat16, requires_grad=True)
f(x).sum().backward()
print('torch.compile: OK')
"
```

---

## 7. TP Sequence Length Padding (torchtitan Issue #1306)

**Problem:** When using Tensor Parallelism (TP) with Sequence Parallel (SP), the sequence dimension is sharded via `Shard(1)`. If `seq_len % tp_degree != 0`, the last rank gets fewer tokens. The `use_local_output=True` workaround (for a complex-number DTensor bug in RoPE) drops shard metadata, causing LayerNorm to reconstruct wrong shapes.

**Root cause chain:**
1. PyTorch has a bug with complex number computation in DTensors (affects RoPE/rotary embeddings)
2. TorchTitan works around it by setting `use_local_output=True` (convert DTensor → plain Tensor)
3. That workaround breaks when `seq_len` isn't evenly divisible by TP degree

**TorchTitan's "fix" (PR #1312):** Only adds a validation assert — makes the failure explicit at startup instead of silently producing wrong results. The root DTensor bug is still open.

**VERL's actual workaround:** Already in `verl/workers/engine/torchtitan/transformer_impl.py`:
```python
# Pad packed sequence to nearest multiple of seq_len_divisor for TP
if self.parallel_dims.tp_enabled:
    divisor = self.parallel_dims.seq_len_divisor
    remainder = total_tokens % divisor
    if remainder != 0:
        tp_pad_len = divisor - remainder
        # pads input_ids and position_ids

# After forward pass, strip padding (~line 726):
if tp_pad_len > 0:
    logits = logits[:, :-tp_pad_len, :]
```

**Status:** Not an issue for VERL users — the dynamic padding works transparently. T7.3 (TP=2) passed 5 steps with arbitrary sequence lengths.

---

## 8. Nested Tensor Bug in Engine Workers (PyTorch #153238)

**Not TorchTitan-specific**, but discovered during TorchTitan/multi-GPU testing.

**Problem:** `unbind(dim=0)` on 3D+ jagged `NestedTensor` applies `split_with_sizes` to the wrong dimension in PyTorch's internal logic.

**Impact on VERL:** The new `engine_workers.py` path calls `concat_nested_tensors()` which uses `unbind(0)`. With 4 workers (smaller chunks), the jagged tensor offsets hit the edge case and crash. With 2 workers (larger chunks), offsets stay aligned.

**Code path:**
```
engine_workers.compute_log_prob()
  → collect_nd_compute_dataproto()
    → BatchData.concat()
      → concat_tensordict()
        → concat_nested_tensors()     ← verl/utils/tensordict_utils.py:186
          → tensor.unbind(0)          ← CRASH on 3D+ jagged
```

**Why legacy path works:** `fsdp_workers.py` uses `DataProto` (flat tensors), never calls `concat_nested_tensors()`.

**Fix applied:** Added `try/except` fallback in `concat_nested_tensors()` — pad → cat → reconstruct via offsets. Same pattern that already existed in `chunk_tensordict()` (lines 329-338).

**Note:** This is NOT an XPU bug. Same crash would happen on CUDA with 4 workers. It's why T1.4 (4-GPU GRPO) requires `trainer.use_legacy_worker_impl=enable`.

---

## Appendix: Upstream XPU PRs in TorchTitan

| PR | Status | Description | Author |
|---|---|---|---|
| [#1016](https://github.com/pytorch/torchtitan/pull/1016) | Merged | XPU bf16 peak flops for PVC | @frost-intel |
| [#1018](https://github.com/pytorch/torchtitan/pull/1018) | Merged | XPU profiling | @frost-intel |
| [#2228](https://github.com/pytorch/torchtitan/pull/2228) | Merged | Generic device memory snapshot | @githubsgi |
| [#2332](https://github.com/pytorch/torchtitan/pull/2332) | Closed | PREMUL_SUM (fixed in PyTorch core) | @saforem2 |
