"""
VeOmni MFU Benchmark — NVIDIA A100 80GB PCIe
GPU indices 1 & 2 (CUDA_VISIBLE_DEVICES=1,2)
Date: 2026-04-09

Modes (one torchrun per mode):
  baseline  — 1-GPU, plain HF model, no veomni (reference)
  dp2       — VeOmni FSDP2 dp_shard=2
  sp2       — VeOmni Ulysses SP=2 (dp_shard=1, ulysses=2)
  ep2       — VeOmni Expert Parallel ep_size=2 (Qwen3-MoE)

MFU formula (same as VERL_TEST_MATRIX.md §10):
  flops_per_step = 6 * N * (bs_per_gpu * seq_len * world_size)
  For DP: world_size batches processed → global_tokens = bs*seq*world
  For SP: 1 batch split over world_size GPUs → global_tokens = bs*seq*1
          (SP trades throughput for memory/longer-seq capability)
  MFU = flops_per_step / (step_time * peak_per_gpu * world_size)

Usage:
  # baseline (1 GPU)
  CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29520 \\
      /workspace/verl/benchmark_mfu_veomni.py --mode baseline

  # DP=2
  CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --master_port=29521 \\
      /workspace/verl/benchmark_mfu_veomni.py --mode dp2

  # SP=2
  CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --master_port=29522 \\
      /workspace/verl/benchmark_mfu_veomni.py --mode sp2

  # EP=2 (MoE)
  CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --master_port=29523 \\
      /workspace/verl/benchmark_mfu_veomni.py --mode ep2
"""
import argparse
import json
import os
import time

import torch
import torch.distributed as dist

# ── Model paths ──────────────────────────────────────────────────────────────
DENSE_MODEL = (
    "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct"
    "/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
)
MOE_MODEL = (
    "/root/.cache/huggingface/hub/models--trl-internal-testing--tiny-Qwen3MoeForCausalLM"
    "/snapshots/3d0483f76c0218a17d90d170d4dd2084882e429f"
)

A100_PEAK_BF16 = 312e12  # 312 TFLOPS per A100 80GB


def rank0_print(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs, flush=True)


def sync_and_time():
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    return time.perf_counter()


# ── Baseline: 1-GPU, plain HF model, no veomni ───────────────────────────────
def run_baseline(bs, seq_len, warmup, steps):
    rank0_print("=" * 60)
    rank0_print(f"BASELINE  1-GPU plain HF  bs={bs}  seq={seq_len}")
    rank0_print("=" * 60)
    from transformers import AutoModelForCausalLM

    device = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(
        DENSE_MODEL, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device).train()
    N = sum(p.numel() for p in model.parameters())
    rank0_print(f"  model: {N/1e6:.1f}M params  device={device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ids = torch.randint(0, model.config.vocab_size, (bs, seq_len), device=device)

    for _ in range(warmup):
        optimizer.zero_grad()
        model(input_ids=ids, labels=ids).loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    times, losses = [], []
    for i in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.zero_grad()
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        losses.append(out.loss.item())
        rank0_print(f"  step {i+1:2d}  loss={losses[-1]:.4f}  time={times[-1]:.3f}s")

    return _report("baseline", 1, bs, seq_len, N, times, losses,
                   global_tokens_fn=lambda: bs * seq_len * 1)


# ── DP=2: VeOmni FSDP2 sharding ──────────────────────────────────────────────
def run_dp2(bs, seq_len, warmup, steps):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    world = dist.get_world_size()

    rank0_print("=" * 60)
    rank0_print(f"DP=2 (VeOmni FSDP2)  world={world}  bs={bs}/gpu  seq={seq_len}")
    rank0_print("=" * 60)

    from veomni.distributed import parallel_state as ps
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.models.auto import build_foundation_model
    from veomni.optim import build_optimizer

    ps.init_parallel_state(
        dp_size=world, dp_shard_size=world, dp_replicate_size=1,
        ulysses_size=1, ep_size=1, dp_mode="fsdp2",
    )
    state = ps.get_parallel_state()
    rank0_print(f"  mesh: {state.device_mesh}")

    model = build_foundation_model(
        config_path=DENSE_MODEL, weights_path=None, init_device="meta",
        attn_implementation="flash_attention_2", force_use_huggingface=True,
    )
    N = sum(p.numel() for p in model.parameters())
    vocab_size = model.config.vocab_size
    rank0_print(f"  model: {N/1e6:.1f}M params")

    model = build_parallelize_model(
        model=model, weights_path=DENSE_MODEL,
        enable_gradient_checkpointing=False, init_device="meta",
    )
    optimizer = build_optimizer(model, lr=1e-4)
    ids = torch.randint(0, vocab_size, (bs, seq_len), device=device)

    for _ in range(warmup):
        optimizer.zero_grad()
        model(input_ids=ids, labels=ids, use_cache=False).loss.backward()
        optimizer.step()

    dist.barrier()
    torch.cuda.synchronize()
    times, losses = [], []
    for i in range(steps):
        dist.barrier(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.zero_grad()
        out = model(input_ids=ids, labels=ids, use_cache=False)
        out.loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        losses.append(out.loss.item())
        rank0_print(f"  step {i+1:2d}  loss={losses[-1]:.4f}  time={times[-1]:.3f}s")

    # DP processes different batches per rank → global_tokens scales with world
    return _report("veomni_dp2", world, bs, seq_len, N, times, losses,
                   global_tokens_fn=lambda: bs * seq_len * world)


# ── SP=2: VeOmni Ulysses sequence parallel ────────────────────────────────────
def run_sp2(bs, seq_len, warmup, steps):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    world = dist.get_world_size()

    rank0_print("=" * 60)
    rank0_print(f"SP=2 (VeOmni Ulysses)  world={world}  bs={bs}  seq={seq_len}")
    rank0_print("  Note: SP splits ONE batch across GPUs — throughput same as 1-GPU,")
    rank0_print("        measures efficiency per GPU when sharing a single long sequence.")
    rank0_print("=" * 60)

    from veomni.distributed import parallel_state as ps
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.models.auto import build_foundation_model
    from veomni.optim import build_optimizer

    ps.init_parallel_state(
        dp_size=1, dp_shard_size=1, dp_replicate_size=1,
        ulysses_size=world, ep_size=1, dp_mode="fsdp2",
    )
    state = ps.get_parallel_state()
    rank0_print(f"  mesh: {state.device_mesh}  ulysses={state.ulysses_size}")

    model = build_foundation_model(
        config_path=DENSE_MODEL, weights_path=None, init_device="meta",
        attn_implementation="flash_attention_2", force_use_huggingface=True,
    )
    N = sum(p.numel() for p in model.parameters())
    vocab_size = model.config.vocab_size
    rank0_print(f"  model: {N/1e6:.1f}M params")

    model = build_parallelize_model(
        model=model, weights_path=DENSE_MODEL,
        enable_gradient_checkpointing=False, init_device="meta",
    )
    optimizer = build_optimizer(model, lr=1e-4)
    # seq_len must be divisible by ulysses_size
    ids = torch.randint(0, vocab_size, (bs, seq_len), device=device)

    for _ in range(warmup):
        optimizer.zero_grad()
        model(input_ids=ids, labels=ids, use_cache=False).loss.backward()
        optimizer.step()

    dist.barrier()
    torch.cuda.synchronize()
    times, losses = [], []
    for i in range(steps):
        dist.barrier(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.zero_grad()
        out = model(input_ids=ids, labels=ids, use_cache=False)
        out.loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        losses.append(out.loss.item())
        rank0_print(f"  step {i+1:2d}  loss={losses[-1]:.4f}  time={times[-1]:.3f}s")

    # SP: same 1 batch across all GPUs → global_tokens = bs * seq (NOT * world)
    return _report("veomni_sp2", world, bs, seq_len, N, times, losses,
                   global_tokens_fn=lambda: bs * seq_len * 1,
                   note="SP splits 1 batch: throughput ≠ DP×world; GPU-hours = 2× baseline for same tokens")


# ── EP=2: VeOmni Expert Parallel ─────────────────────────────────────────────
def run_ep2(bs, seq_len, warmup, steps):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    world = dist.get_world_size()

    rank0_print("=" * 60)
    rank0_print(f"EP=2 (VeOmni Expert Parallel, Qwen3-MoE)  world={world}  bs={bs}/gpu  seq={seq_len}")
    rank0_print("=" * 60)

    from veomni.distributed import parallel_state as ps
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.models.auto import build_foundation_model
    from veomni.optim import build_optimizer

    ps.init_parallel_state(
        dp_size=world, dp_shard_size=1, dp_replicate_size=world,
        ulysses_size=1, ep_size=world, dp_mode="fsdp2",
    )
    state = ps.get_parallel_state()
    rank0_print(f"  mesh: {state.device_mesh}  ep_size={state.ep_size}")

    model = build_foundation_model(
        config_path=MOE_MODEL, weights_path=None, init_device="meta",
        attn_implementation="flash_attention_2",
        moe_implementation="fused", force_use_huggingface=False,
    )
    N = sum(p.numel() for p in model.parameters())
    vocab_size = model.config.vocab_size
    rank0_print(f"  model: {N/1e6:.1f}M params  type={model.config.model_type}")

    model = build_parallelize_model(
        model=model, weights_path=MOE_MODEL,
        enable_gradient_checkpointing=False, init_device="meta",
    )
    optimizer = build_optimizer(model, lr=1e-4)
    ids = torch.randint(0, vocab_size, (bs, seq_len), device=device)

    for _ in range(warmup):
        optimizer.zero_grad()
        model(input_ids=ids, labels=ids, use_cache=False).loss.backward()
        optimizer.step()

    dist.barrier()
    torch.cuda.synchronize()
    times, losses = [], []
    for i in range(steps):
        dist.barrier(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.zero_grad()
        out = model(input_ids=ids, labels=ids, use_cache=False)
        out.loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        losses.append(out.loss.item())
        rank0_print(f"  step {i+1:2d}  loss={losses[-1]:.4f}  time={times[-1]:.3f}s")

    # EP: each rank processes its own batch → global_tokens scales with world
    return _report("veomni_ep2", world, bs, seq_len, N, times, losses,
                   global_tokens_fn=lambda: bs * seq_len * world)


# ── Report helper ─────────────────────────────────────────────────────────────
def _report(mode, world, bs, seq_len, N, times, losses,
            global_tokens_fn, note=None):
    avg_time = sum(times) / len(times)
    global_tokens = global_tokens_fn()
    flops = 6 * N * global_tokens
    tflops = flops / (avg_time * 1e12)
    mfu = tflops / (A100_PEAK_BF16 / 1e12 * world) * 100
    tok_s = global_tokens / avg_time

    rank0_print()
    rank0_print(f"── {mode} results ──")
    rank0_print(f"  avg step time : {avg_time:.3f}s")
    rank0_print(f"  global tok/s  : {tok_s:,.0f}  ({tok_s/world:,.0f} tok/s/GPU)")
    rank0_print(f"  agg TFLOPS    : {tflops:.1f}  ({tflops/world:.1f}/GPU)")
    rank0_print(f"  MFU           : {mfu:.2f}%  (A100 peak = {A100_PEAK_BF16/1e12:.0f} TFLOPS)")
    rank0_print(f"  step times    : {[round(t,3) for t in times]}")
    if note:
        rank0_print(f"  note          : {note}")

    mem = torch.cuda.max_memory_allocated() / 1e9
    rank0_print(f"  peak GPU mem  : {mem:.2f} GB")

    result = {
        "mode": mode, "world_size": world,
        "bs_per_gpu": bs, "global_bs": bs * world, "seq_len": seq_len,
        "model_params_M": round(N / 1e6, 1),
        "avg_step_time_s": round(avg_time, 4),
        "global_tok_s": round(tok_s),
        "per_gpu_tok_s": round(tok_s / world),
        "global_tflops": round(tflops, 2),
        "per_gpu_tflops": round(tflops / world, 2),
        "mfu_pct": round(mfu, 2),
        "peak_mem_gb": round(mem, 2),
        "loss_last": round(losses[-1], 4),
        "step_times": [round(t, 4) for t in times],
        "a100_peak_bf16_tflops": A100_PEAK_BF16 / 1e12,
    }
    if note:
        result["note"] = note

    out_dir = "/workspace/verl/evidence"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"mfu_veomni_{mode}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    rank0_print(f"  saved → {out_path}")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "dp2", "sp2", "ep2"], required=True)
    parser.add_argument("--bs", type=int, default=4,
                        help="Batch size per GPU")
    parser.add_argument("--seq", type=int, default=512,
                        help="Sequence length (must be divisible by 2 for SP)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    if args.mode != "baseline":
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        rank0_print(f"[dist] world={dist.get_world_size()}  rank={dist.get_rank()}  "
                    f"device=cuda:{local_rank}")

    if args.mode == "baseline":
        result = run_baseline(args.bs, args.seq, args.warmup, args.steps)
    elif args.mode == "dp2":
        result = run_dp2(args.bs, args.seq, args.warmup, args.steps)
    elif args.mode == "sp2":
        result = run_sp2(args.bs, args.seq, args.warmup, args.steps)
    elif args.mode == "ep2":
        result = run_ep2(args.bs, args.seq, args.warmup, args.steps)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
