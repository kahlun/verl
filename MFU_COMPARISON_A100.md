# Full MFU Comparison — NVIDIA A100 80GB PCIe

## 1. Parallelism MFU Results (Llama-3.1-8B SFT, steady-state steps 5-30)

| # | Engine | Parallelism | GPUs | MFU | ~sec/step | Val Loss | Batch | max_token |
|---|--------|-------------|------|-----|-----------|----------|-------|-----------|
| 1 | TorchTitan | PP=2, FSDP2=2 | 4 | **25.85%** | 3.7s | 0.514 | 128 | 2048 |
| 2 | Megatron | TP=2, PP=2, DP=1 | 4 | 21.94% | 2.2s | 0.563 | 64 | 1024 |
| 3 | Megatron | PP=2, DP=2 | 4 | 20.05% | 2.4s | 0.562 | 64 | 1024 |
| 4 | Megatron | TP=2, DP=2 | 4 | 16.27% | 2.9s | 0.561 | 64 | 1024 |
| 5 | TorchTitan | TP=2, FSDP2=2 | 4 | 10.53% | 9.1s | 0.492 | 128 | 2048 |
| 6 | TorchTitan | TP=4, FSDP2=1 | 4 | 9.45% | 10.1s | 0.492 | 128 | 2048 |
| 7 | TorchTitan | FSDP2=4, TP=1 | 4 | 5.82% | 16.4s | 0.492 | 128 | 2048 |
| 8 | FSDP (verl) | DP=4 | 4 | 5.76% | 16.6s | 0.468 | 128 | 512 |

## 2. Context Parallel & Expert Parallel (2-GPU, Megatron)

| # | Engine | Parallelism | Model | GPUs | MFU | ~sec/step | Val Loss | Notes |
|---|--------|-------------|-------|------|-----|-----------|----------|-------|
| 9 | Megatron | CP=2 | Llama-3.2-1B | 2 | 15.8% | 1.3s | 1.175 | Sequence split across GPUs |
| 10 | Megatron | EP=2 | Qwen1.5-MoE-A2.7B (14.3B total) | 2 | 0.37% | ~20s | N/A | CPU optim offload needed; MFU low due to CPU<->GPU transfer |

## 3. Standalone PyTorch Benchmark (`benchmark_mfu.py`, Qwen2.5-0.5B, 1 GPU)

Direct hardware comparison with XPU (Arc Pro B60):

| Config | Arc Pro B60 (96 TFLOPS) | A100 PCIe (312 TFLOPS) | A100/B60 tok/s ratio |
|--------|------------------------|------------------------|---------------------|
| bs=4 eager | 20.1% MFU, 6518 tok/s | 22.5% MFU, 23684 tok/s | 3.6× |
| bs=8 eager | 22.6% MFU, 7307 tok/s | 26.8% MFU, 28221 tok/s | 3.9× |
| bs=4 compiled | 27.5% MFU, 8902 tok/s | 28.8% MFU, 30299 tok/s | 3.4× |

**Key insight**: MFU% is nearly identical (only ~2-4% gap) — the raw throughput difference is driven by hardware TFLOPS (312 vs 96), not software efficiency.

## Notes

- **Hardware**: NVIDIA A100 80GB PCIe (NO NVLink, PCIe Gen4 x16)
- **Peak BF16 FLOPS**: 312 TFLOPS per GPU
- **MFU** = actual_FLOPS / peak_FLOPS, normalized
- **Batch settings differ**: TorchTitan uses batch=128/max_token=2048; Megatron uses batch=64/max_token=1024; FSDP baseline uses batch=128/max_token=512
- **TorchTitan PP fix**: commit `cc5b0859` — uses `_step_microbatches()` directly with fixed-length padding
- **EP=2 OOM**: Qwen1.5-MoE (14.3B params) needs CPU optimizer offloading on 2× A100; MFU is artificially low due to CPU<->GPU transfer overhead, not a parallelism limitation
- **JSONL results**: `evidence/mfu_results/` and container `/workspace/verl/mfu_comparison/`

## Key Findings

1. **Pipeline Parallel is the clear winner** on PCIe — PP=2+FSDP2=2 achieves 25.85% MFU, ~4.5× better than pure FSDP
2. **Megatron PP configs** (20-22%) also outperform TP-heavy configs on PCIe interconnect
3. **TP is expensive on PCIe** — all-reduce over PCIe is the bottleneck; TP=4 (9.45%) barely beats pure FSDP (5.82%)
4. **CP=2 works** with ~16% MFU on Llama-1B (smaller model due to 2-GPU memory constraint)
5. **EP=2 works** — experts correctly sharded, training converges, but needs CPU offload for 14B MoE on 2 GPUs

## What's Left for XPU Side

### Must do (to match NVIDIA parallelism coverage):
- [ ] **T7.2**: TorchTitan SFT PP=2 on 2× B60
- [ ] **T7.3**: TorchTitan SFT TP=2 on 2× B60
- [ ] Run `benchmark_mfu.py` compiled on B60 with bs=8 (missing from XPU data)

### Optional (new parallelism axes, available in both Megatron and TorchTitan):
- [ ] CP=2 on XPU (Megatron or TorchTitan, Llama-3.2-1B, 2 GPUs)
- [ ] EP=2 on XPU (Megatron, Qwen1.5-MoE, 2 GPUs — needs CPU optim offload)
- [ ] TorchTitan CP=2 on NVIDIA or XPU (config: `engine.context_parallel_size=2`)
- [ ] TorchTitan EP=2 on NVIDIA or XPU (config: `engine.expert_parallel_size=2`, needs MoE model)

### Not tested on either side (lower priority):
- [ ] Distributed Optimizer sharding (only useful with DP>1)
- [ ] CPU param/grad offload (`engine.param_offload=True`)
- [ ] Sequence Parallel (SP) — usually paired with TP (`engine.sequence_parallel=True`)
