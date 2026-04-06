# Full MFU Comparison — Llama-3.1-8B SFT on 4× A100 80GB PCIe

## Results (steady-state MFU, steps 5-30)

| # | Engine | Parallelism | MFU | ~sec/step | Val Loss | Batch | max_token |
|---|--------|-------------|-----|-----------|----------|-------|-----------|
| 1 | TorchTitan | PP=2, FSDP2=2 | **25.85%** | 3.7s | 0.514 | 128 | 2048 |
| 2 | Megatron | TP=2, PP=2, DP=1 | 21.94% | 2.2s | 0.563 | 64 | 1024 |
| 3 | Megatron | PP=2, DP=2 | 20.05% | 2.4s | 0.562 | 64 | 1024 |
| 4 | Megatron | TP=2, DP=2 | 16.27% | 2.9s | 0.561 | 64 | 1024 |
| 5 | TorchTitan | TP=2, FSDP2=2 | 10.53% | 9.1s | 0.492 | 128 | 2048 |
| 6 | TorchTitan | TP=4, FSDP2=1 | 9.45% | 10.1s | 0.492 | 128 | 2048 |
| 7 | TorchTitan | FSDP2=4, TP=1 | 5.82% | 16.4s | 0.492 | 128 | 2048 |
| 8 | FSDP (verl) | DP=4 | 5.76% | 16.6s | 0.468 | 128 | 512 |

## Notes

- **Hardware**: 4× NVIDIA A100 80GB PCIe (NO NVLink, PCIe Gen4 x16)
- **Peak BF16 FLOPS**: 312 TFLOPS per GPU
- **MFU** = actual_FLOPS / peak_FLOPS, normalized — valid for comparing hardware utilization even with different batch settings
- **Batch settings differ**: TorchTitan uses batch=128/max_token=2048 (~25K tokens/step); Megatron uses batch=64/max_token=1024 (~12K tokens/step); FSDP baseline uses batch=128/max_token=512
- **Val loss discrepancy**: TorchTitan PP uses all-token loss vs verl's response-masked loss, explaining the higher val_loss for PP config
- **TorchTitan PP fix**: commit `cc5b0859` — uses `_step_microbatches()` directly with fixed-length padding to avoid PipeliningShapeError
- **JSONL results**: stored in container at `/workspace/verl/mfu_comparison/`

## Key Findings

1. **Pipeline Parallel is the clear winner** on PCIe-connected GPUs — PP=2+FSDP2=2 achieves 25.85% MFU, ~4.5× better than pure FSDP
2. **Megatron PP configs** (20-22% MFU) also outperform TP-heavy configs on PCIe interconnect
3. **TP is expensive on PCIe** — all-reduce over PCIe is the bottleneck; TP=4 (9.45%) barely beats pure FSDP (5.82%)
4. PCIe bandwidth (~25 GB/s bidirectional) is the limiting factor; NVLink would significantly change the TP results
