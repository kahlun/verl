---
name: xpu-status
description: Show current XPU support status — what's verified, what's pending, PR state, known issues. Use at the start of a session to get context, or to update the stakeholder report.
---

Show the current XPU support status for the verl fork at `/home/sdp/kl/verl_test_xpu`.

## Check PR state

```bash
cd /home/sdp/kl/verl_test_xpu
echo "=== Fork PRs ==="
gh pr list --repo kahlun/verl --state open 2>&1

echo ""
echo "=== Branch commit counts ahead of upstream/main ==="
for b in xpu/1-core-device-abstraction xpu/2-training-integration xpu/3-test-compatibility; do
  count=$(git log origin/main..$b --oneline 2>/dev/null | wc -l)
  echo "  $b: $count commit(s) ahead"
done
```

## Check what's verified vs pending

Read the analysis doc:
```bash
grep -A 100 "## 7. What Is Still Pending" /home/sdp/kl/verl_test_xpu/VERL_XPU_Code_Analysis.md | head -30
```

## Hardware check

```bash
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'XPU devices: {torch.xpu.device_count()}')
for i in range(torch.xpu.device_count()):
    props = torch.xpu.get_device_properties(i)
    print(f'  XPU {i}: {torch.xpu.get_device_name(i)} ({props.total_memory/1024**3:.1f} GB)')
" 2>&1 | grep -v WARNING
```

## Key facts to summarize

- Hardware: 4x Arc Pro B60 (~24 GB each), PCIe, no XeLink
- PyTorch 2.11.0+xpu, no IPEX, vLLM 0.14.1 (Docker)
- All 10 RL algorithms validated (single-GPU)
- All 6 VLM models supported (xpu_attn.py SDPA fallback)
- Fused Triton kernel enabled for XPU
- SDPA NaN bug: fixed in PyTorch 2.11

## Known blockers (external)

- Multi-GPU E2E with vLLM: needs Docker multi-GPU or host vLLM install
- SGLang: version mismatch (verl 0.4.x vs XPU needs 0.5.4+)
- Ulysses SP: needs flash_attn XPU port
