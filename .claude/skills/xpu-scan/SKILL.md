---
name: xpu-scan
description: Scan the entire verl codebase for remaining CUDA-specific code that could affect XPU. Finds is_cuda asserts, torch.cuda.* calls, hardcoded "cuda"/"nccl" strings in the FSDP+vLLM training path. Use when upstream verl adds new files or you want to check if a new PR introduced regressions.
argument-hint: "[path] e.g. verl/workers/ to limit scope"
---

Scan the verl codebase at `/home/sdp/kl/verl_test_xpu` for remaining CUDA-specific code.

Scope: $ARGUMENTS (default: `verl/` excluding mcore, megatron, trtllm, mooncake, nixl, nccl_checkpoint_engine)

## Run these scans

```bash
cd /home/sdp/kl/verl_test_xpu

EXCLUDE="mcore|megatron|trtllm|mooncake|nixl|nccl_checkpoint"

echo "=== 1. is_cuda asserts (blockers if hit on XPU) ==="
grep -rn "assert.*\.is_cuda" verl/ --include="*.py" | grep -vE "$EXCLUDE|__pycache__"

echo ""
echo "=== 2. torch.cuda.* direct calls (excluding is_available/device_count) ==="
grep -rn "torch\.cuda\." verl/ --include="*.py" \
  | grep -vE "$EXCLUDE|__pycache__|is_cuda_available|is_available\(\)|device_count|#" \
  | grep -vE "torch\.cuda\.is_available|torch\.cuda\.device_count"

echo ""
echo "=== 3. Hardcoded 'cuda' string (not in comments) ==="
grep -rn '"cuda"' verl/ --include="*.py" \
  | grep -vE "$EXCLUDE|__pycache__|#|docstring|== .cuda|is_cuda" \
  | head -20

echo ""
echo "=== 4. ReduceOp.AVG/MAX/MIN without XPU workaround ==="
grep -rn "ReduceOp\.AVG\|ReduceOp\.MAX\|ReduceOp\.MIN" verl/ --include="*.py" \
  | grep -vE "$EXCLUDE|__pycache__|is_xpu|all_reduce_avg|cpu\(\)" | head -15

echo ""
echo "=== 5. flash_attn direct imports without XPU guard ==="
grep -rn "^from flash_attn\|^import flash_attn" verl/ --include="*.py" \
  | grep -vE "$EXCLUDE|__pycache__|is_xpu|is_npu|try:|except" | head -10
```

## For each finding, report

- File path and line number
- Whether it is in the FSDP+vLLM training path (check by tracing call chain)
- Whether it is guarded by `is_cuda_available`, `use_fused_kernels`, or similar
- Severity: BLOCKER / CORRECTNESS / COSMETIC / ALREADY GUARDED

## Known fixed items (ignore these)

- `kernels.py` — `is_cuda or is_xpu` (fixed)
- `linear_cross_entropy.py` — nvtx guarded with nullcontext (fixed)
- `engine/utils.py` — torch.cuda.manual_seed in `else:` branch (guarded)
- `device.py` — all torch.cuda.* guarded by `is_cuda_available`
