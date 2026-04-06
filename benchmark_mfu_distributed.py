"""
Multi-GPU MFU benchmark for Intel XPU: DDP and FSDP.
Runs pure forward+backward training loop across 1/2/4 GPUs.
Comparable to TorchTitan/Megatron MFU measurement methodology.

Usage (inside docker):
  # 1-GPU baseline
  torchrun --nproc_per_node=1 benchmark_mfu_distributed.py --mode ddp --bs 4
  # 2-GPU DDP
  torchrun --nproc_per_node=2 benchmark_mfu_distributed.py --mode ddp --bs 4
  # 4-GPU DDP
  torchrun --nproc_per_node=4 benchmark_mfu_distributed.py --mode ddp --bs 4
  # 4-GPU FSDP
  torchrun --nproc_per_node=4 benchmark_mfu_distributed.py --mode fsdp --bs 4
"""
import argparse
import json
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from transformers import AutoModelForCausalLM, AutoConfig


# Peak BF16 TFLOPS per device
DEVICE_PEAKS = {
    "Arc(TM) Pro B60": 96e12,
    "A100": 312e12,
    "H100": 989e12,
    "RTX 4090": 330e12,
}


def get_peak_flops(device_name):
    for k, v in DEVICE_PEAKS.items():
        if k in device_name:
            return v
    return 96e12  # default to B60


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # Pin to local rank BEFORE init_process_group to prevent cross-GPU access
    os.environ["ZE_AFFINITY_MASK"] = str(local_rank)
    torch.xpu.set_device(local_rank)
    dist.init_process_group(backend="xccl")
    return local_rank, dist.get_world_size(), dist.get_rank()


def cleanup():
    dist.destroy_process_group()


def run_benchmark(args):
    local_rank, world_size, rank = setup_distributed()
    device = f"xpu:{local_rank}"
    is_main = rank == 0

    model_name = args.model
    bs = args.bs
    seq_len = args.seq_len
    warmup = args.warmup
    steps = args.steps
    mode = args.mode

    if is_main:
        print(f"{'='*60}")
        print(f"Multi-GPU MFU Benchmark")
        print(f"  Mode: {mode.upper()}, GPUs: {world_size}, BS/GPU: {bs}")
        print(f"  Model: {model_name}, Seq: {seq_len}")
        print(f"  Global BS: {bs * world_size}")
        print(f"{'='*60}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device)
    model.train()

    N = sum(p.numel() for p in model.parameters())
    vocab = model.config.vocab_size

    if is_main:
        print(f"Parameters: {N/1e6:.0f}M")

    # Wrap model
    if mode == "ddp":
        model = DDP(model, device_ids=[local_rank])
    elif mode == "fsdp":
        bf16_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        model = FSDP(model, mixed_precision=bf16_policy, device_id=local_rank)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Create data
    input_ids = torch.randint(0, vocab, (bs, seq_len), device=device)
    labels = input_ids.clone()

    # Device info
    dev_name = torch.xpu.get_device_name(local_rank)
    peak = get_peak_flops(dev_name)

    if is_main:
        print(f"Device: {dev_name}")
        print(f"Peak BF16: {peak/1e12:.0f} TFLOPS per GPU")
        print()

    # Warmup
    for i in range(warmup):
        optimizer.zero_grad()
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        optimizer.step()
        if is_main and i == 0:
            print(f"  Warmup step 0 loss: {out.loss.item():.4f}")

    torch.xpu.synchronize()
    dist.barrier()

    # Measure
    times = []
    losses = []
    for i in range(steps):
        torch.xpu.synchronize()
        dist.barrier()
        t0 = time.perf_counter()

        optimizer.zero_grad()
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        optimizer.step()

        torch.xpu.synchronize()
        times.append(time.perf_counter() - t0)
        losses.append(out.loss.item())

    # Compute MFU
    avg_time = sum(times) / len(times)
    tokens_per_step = bs * seq_len * world_size  # global tokens
    flops_per_step = 6 * N * tokens_per_step
    tflops_achieved = flops_per_step / (avg_time * 1e12)
    # MFU = achieved / (peak * world_size)
    mfu = tflops_achieved / (peak / 1e12 * world_size) * 100
    tok_s = tokens_per_step / avg_time

    # Per-GPU metrics
    per_gpu_tflops = tflops_achieved / world_size
    per_gpu_tok_s = tok_s / world_size

    if is_main:
        print(f"Results ({mode.upper()}, {world_size}×GPU, bs={bs}/gpu, seq={seq_len}):")
        print(f"  Avg step time: {avg_time:.3f}s")
        print(f"  Global throughput: {tok_s:.0f} tok/s ({per_gpu_tok_s:.0f} tok/s/gpu)")
        print(f"  Aggregate TFLOPS: {tflops_achieved:.1f} ({per_gpu_tflops:.1f}/gpu)")
        print(f"  MFU: {mfu:.1f}%")
        print(f"  Loss (last): {losses[-1]:.4f}")
        print(f"  Step times: {[round(t, 3) for t in times]}")

        # Memory
        mem = torch.xpu.max_memory_allocated(local_rank) / 1e9
        print(f"  Peak memory: {mem:.1f} GB")

        result = {
            "mode": mode,
            "world_size": world_size,
            "bs_per_gpu": bs,
            "global_bs": bs * world_size,
            "seq_len": seq_len,
            "avg_step_time": round(avg_time, 4),
            "global_tok_s": round(tok_s),
            "per_gpu_tok_s": round(per_gpu_tok_s),
            "aggregate_tflops": round(tflops_achieved, 1),
            "per_gpu_tflops": round(per_gpu_tflops, 1),
            "mfu_pct": round(mfu, 1),
            "peak_mem_gb": round(mem, 1),
            "loss_last": round(losses[-1], 4),
            "model": model_name,
            "params_M": round(N / 1e6),
            "device": dev_name,
            "peak_bf16_tflops": peak / 1e12,
            "pytorch": torch.__version__,
        }

        # Save results
        out_dir = "evidence"
        os.makedirs(out_dir, exist_ok=True)
        fname = f"mfu_{mode}_{world_size}gpu_bs{bs}.json"
        out_path = os.path.join(out_dir, fname)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Saved: {out_path}")

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ddp", "fsdp"], default="ddp")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--bs", type=int, default=4, help="batch size per GPU")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    run_benchmark(args)
