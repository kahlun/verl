# Email Draft — TorchTitan × Intel XPU Status Update

**To:** [Manager's Name]  
**From:** [Your Name]  
**Date:** April 7, 2026  
**Subject:** XPU/TorchTitan Integration Status — TP=2 PASS, PP=2 BLOCKED, 5 VERL Patches Committed

---

Hi [Manager],

I wanted to share a status update on the TorchTitan × Intel XPU integration work for VERL. The short version: **end-to-end training works on XPU with 5 small patches**, multi-GPU Tensor Parallel (TP=2) passes, but Pipeline Parallel (PP=2) is blocked by a hardware limitation on the B60.

---

## What Works Today

With 5 patches (all committed to the VERL repo), the full VERL + TorchTitan SFT training pipeline works on Intel Arc Pro B60:

| Config | Result |
|---|---|
| 1-GPU, Llama-3.2-1B | ✅ Loss 1.59 → 0.79, 3 steps, checkpoint saved |
| 2-GPU TP=2, Llama-3.2-1B | ✅ Loss 1.31 → 0.80, 5 steps, val_loss 0.81 |
| 2-GPU PP=2, Llama-3.1-8B | ❌ Blocked — hardware P2P IPC issue |
| 1-GPU, Llama-3.2-3B | ❌ OOM (13.4 GiB model + gradients > 24 GiB) |

---

## The 5 Patches (All Committed)

These were bugs in VERL's TorchTitan integration, **not** in TorchTitan or XPU itself:

1. **EngineRegistry** — Added `"xpu"` to the device list (was CUDA/NPU only)
2. **Factory config matching** — TorchTitan main stores model configs as factory functions; VERL was reading attributes before calling them
3. **`attn_backend` rename** — TorchTitan main renamed this field to `mask_type`; VERL still used the old name
4. **`torch.cuda.empty_cache()`** — Replaced with the device-agnostic `get_torch_device().empty_cache()`
5. **PP forward/backward** — Implemented the Pipeline Parallel code path (was raising `NotImplementedError`)

---

## Why PP=2 Fails (Hardware Blocker)

Pipeline Parallel requires point-to-point IPC memory sharing between GPU ranks — one rank writes activations, the next reads them. On the B60 connected via PCIe, `zeMemOpenIpcHandle` returns `ZE_RESULT_ERROR_INVALID_ARGUMENT`.

**TP=2 works** because it uses XCCL all-reduce (collective broadcast), not P2P IPC.  
**PP=2 is blocked** until either:
- The Level Zero / oneAPI driver adds PCIe IPC support, or
- We test on hardware with NVLink-class or CXL interconnect

This is worth a conversation with the oneAPI/Level Zero team to clarify whether it's a driver gap or a fundamental PCIe limitation.

---

## One Open Ask for the TorchTitan Team

There is one low-effort TorchTitan-side fix that would help XPU users (and anyone on stable PyTorch):

> **Guard the `ShardPlacementResult` import** in `torchtitan/models/llama4/parallelize.py` with `try/except ImportError`.

`ShardPlacementResult` was added to PyTorch nightly after the 2.11 branch cut (which is what the XPU wheel tracks). Until it ships in a stable release, any code importing from `llama4/parallelize.py` fails — including Llama4 and Qwen3 models. We're currently working around it with a monkey-patch, but a 3-line guard in TorchTitan would fix it cleanly. I can open the PR if helpful.

---

## Next Steps

| Item | Owner | Status |
|---|---|---|
| TP=2 E2E SFT validated | Me | ✅ Done |
| PP implementation committed | Me | ✅ Done |
| `ShardPlacementResult` guard PR to TorchTitan | Me / TT team | Pending |
| P2P IPC on B60 PCIe | oneAPI/L0 team | Needs investigation |
| PP test on NVLink/CXL hardware | TBD | Blocked on hardware access |

---

Let me know if you'd like the full gap analysis doc — I have it written up in detail with evidence logs for each test case.

Best,  
[Your Name]
