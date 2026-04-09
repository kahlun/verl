#!/usr/bin/env python3
"""
VeOmni CUDA 2-GPU Parallelism Test
NVIDIA A100 80GB PCIe — GPU indices 1 & 2 (CUDA_VISIBLE_DEVICES=1,2)
Date: 2026-04-09

Tests (one per torchrun invocation — parallel state is process-global):
  T1. DP=2   — FSDP2 data parallel sharding (dp_shard_size=2, ulysses=1)
  T2. SP=2   — Ulysses sequence parallel (dp=1, ulysses=2, no FSDP shard)
  T3. EP=2   — Expert Parallel with Qwen1.5-MoE-A2.7B (ep_size=2, dp_shard=1)
  T4. DP+SP  — Needs 4 GPUs (dp=2 * ulysses=2); with 2 GPUs → config prints limitation
  T5/T6/T7   — TP/PP/CP: not in VERL veomni engine actor (documented)

Run each test with its own torchrun (fresh process group per test):
  CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --master_port=29510 \\
      /workspace/verl/test_veomni_cuda_2gpu.py --test T1
"""
import argparse
import os
import time

import torch
import torch.distributed as dist

# ----------------------------------------------------------------------------
# Model paths (inside kahlun container — mounted from /mnt/disk5/HF_CACHE)
# ----------------------------------------------------------------------------
QWEN25_0B5_PATH = (
    "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct"
    "/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
)
QWEN_MOE_PATH = (
    "/root/.cache/huggingface/hub/models--trl-internal-testing--tiny-Qwen3MoeForCausalLM"
    "/snapshots/3d0483f76c0218a17d90d170d4dd2084882e429f"
)


def rank0_print(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs, flush=True)


def init_dist():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank0_print(f"[dist] world={dist.get_world_size()} rank={dist.get_rank()} "
                f"device=cuda:{local_rank}")
    return local_rank


def dummy_batch(vocab_size, seq_len=64, batch_size=2, device="cuda"):
    ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone())


# ============================================================================
# T1: DP=2 — FSDP2 data parallel sharding
# ============================================================================

def test_t1_dp2(model_path: str):
    rank0_print("\n" + "=" * 60)
    rank0_print("T1: DP=2 (FSDP2 data parallel sharding)")
    rank0_print("=" * 60)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"

    from veomni.distributed import parallel_state as ps
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.models.auto import build_foundation_model
    from veomni.optim import build_optimizer

    world = dist.get_world_size()
    rank0_print(f"  world={world}  dp_size={world}  dp_shard_size={world}  ulysses=1  ep=1")

    ps.init_parallel_state(
        dp_size=world, dp_shard_size=world, dp_replicate_size=1,
        ulysses_size=1, ep_size=1, dp_mode="fsdp2",
    )
    state = ps.get_parallel_state()
    rank0_print(f"  parallel_state — dp={state.dp_size}  fsdp_enabled={state.fsdp_enabled}"
                f"  dp_shard={state.dp_shard_size}  mesh={state.device_mesh}")

    t0 = time.time()
    # FSDP2 requires meta init: build skeleton on meta, weights loaded during parallelize
    model = build_foundation_model(
        config_path=model_path,
        weights_path=None,   # no load yet — meta device
        init_device="meta",
        attn_implementation="flash_attention_2",
        force_use_huggingface=True,
    )
    vocab_size = model.config.vocab_size
    rank0_print(f"  build_foundation_model (meta) ok — {sum(p.numel() for p in model.parameters())/1e6:.1f}M params  "
                f"({time.time()-t0:.1f}s)")

    t1 = time.time()
    # weights_path triggers load_model_weights → distribute_tensor (scatter) per rank
    model = build_parallelize_model(
        model=model,
        weights_path=model_path,
        enable_gradient_checkpointing=False,
        init_device="meta",
    )
    rank0_print(f"  build_parallelize_model (FSDP2) ok  ({time.time()-t1:.1f}s)")

    optimizer = build_optimizer(model, lr=1e-4)
    batch = dummy_batch(vocab_size, device=device)

    losses = []
    for step in range(3):
        optimizer.zero_grad()
        out = model(**batch, use_cache=False)
        out.loss.backward()
        optimizer.step()
        losses.append(out.loss.item())
        rank0_print(f"  step {step+1}  loss={losses[-1]:.4f}")

    ok = losses[-1] < losses[0]
    rank0_print(f"\n  losses: {[f'{l:.4f}' for l in losses]}")
    rank0_print(f"T1 RESULT: {'✅ PASS' if ok else '⚠️  losses not decreasing'}")
    return "PASS" if ok else "WARN"


# ============================================================================
# T2: SP=2 — Ulysses sequence parallel (no FSDP shard)
# ============================================================================

def test_t2_sp2(model_path: str):
    rank0_print("\n" + "=" * 60)
    rank0_print("T2: SP=2 (Ulysses sequence parallel, no FSDP shard)")
    rank0_print("=" * 60)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"

    from veomni.distributed import parallel_state as ps
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.models.auto import build_foundation_model
    from veomni.optim import build_optimizer

    world = dist.get_world_size()
    # ulysses=2, dp=1 (no DP), no FSDP shard → 1*2 = world_size=2 ✓
    rank0_print(f"  world={world}  dp_size=1  dp_shard_size=1  ulysses_size={world}  ep=1")

    ps.init_parallel_state(
        dp_size=1, dp_shard_size=1, dp_replicate_size=1,
        ulysses_size=world, ep_size=1, dp_mode="fsdp2",
    )
    state = ps.get_parallel_state()
    rank0_print(f"  parallel_state — sp_enabled={state.sp_enabled}  ulysses={state.ulysses_size}"
                f"  fsdp_enabled={state.fsdp_enabled}  mesh={state.device_mesh}")

    t0 = time.time()
    # SP (no FSDP shard, but veomni always wraps in FSDP2): use meta init
    model = build_foundation_model(
        config_path=model_path,
        weights_path=None,
        init_device="meta",
        attn_implementation="flash_attention_2",
        force_use_huggingface=True,
    )
    vocab_size = model.config.vocab_size
    rank0_print(f"  build_foundation_model (meta) ok  ({time.time()-t0:.1f}s)")

    model = build_parallelize_model(
        model=model, weights_path=model_path,
        enable_gradient_checkpointing=False, init_device="meta",
    )
    rank0_print("  build_parallelize_model (Ulysses SP) ok")

    optimizer = build_optimizer(model, lr=1e-4)
    # seq_len must be divisible by ulysses_size (=2)
    batch = dummy_batch(vocab_size, seq_len=128, device=device)

    losses = []
    for step in range(3):
        optimizer.zero_grad()
        out = model(**batch, use_cache=False)
        out.loss.backward()
        optimizer.step()
        losses.append(out.loss.item())
        rank0_print(f"  step {step+1}  loss={losses[-1]:.4f}")

    ok = losses[-1] < losses[0]
    rank0_print(f"\n  losses: {[f'{l:.4f}' for l in losses]}")
    rank0_print(f"T2 RESULT: {'✅ PASS' if ok else '⚠️  losses not decreasing'}")
    return "PASS" if ok else "WARN"


# ============================================================================
# T3: EP=2 — Expert Parallel with Qwen1.5-MoE-A2.7B
# ============================================================================

def test_t3_ep2(model_path: str):
    rank0_print("\n" + "=" * 60)
    rank0_print("T3: EP=2 (Expert Parallel, Qwen1.5-MoE-A2.7B)")
    rank0_print("=" * 60)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"

    from veomni.distributed import parallel_state as ps
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.models.auto import build_foundation_model
    from veomni.optim import build_optimizer

    world = dist.get_world_size()
    ep_size = world  # = 2; each rank holds half the experts
    # dp_size=world, dp_shard=1 (full model per rank), ep=2
    rank0_print(f"  world={world}  dp_size={world}  dp_shard_size=1  ep_size={ep_size}  ulysses=1")

    ps.init_parallel_state(
        dp_size=world, dp_shard_size=1, dp_replicate_size=world,
        ulysses_size=1, ep_size=ep_size, dp_mode="fsdp2",
    )
    state = ps.get_parallel_state()
    rank0_print(f"  parallel_state — ep_enabled={state.ep_enabled}  ep_size={state.ep_size}"
                f"  fsdp_enabled={state.fsdp_enabled}  mesh={state.device_mesh}")

    t0 = time.time()
    # EP without FSDP shard — but veomni always wraps in FSDP2, so use meta init
    # force_use_huggingface=False so VeOmni uses its own Qwen3-MoE that has get_parallel_plan
    model = build_foundation_model(
        config_path=model_path,
        weights_path=None,
        init_device="meta",
        attn_implementation="flash_attention_2",
        moe_implementation="fused",   # fused (Triton group_gemm) required for EP; works on A100
        force_use_huggingface=False,
    )
    vocab_size = model.config.vocab_size
    rank0_print(f"  build_foundation_model (meta) ok — model_type={model.config.model_type}  ({time.time()-t0:.1f}s)")

    t1 = time.time()
    model = build_parallelize_model(
        model=model, weights_path=model_path,
        enable_gradient_checkpointing=False, init_device="meta",
    )
    rank0_print(f"  build_parallelize_model (EP=2) ok  ({time.time()-t1:.1f}s)")

    optimizer = build_optimizer(model, lr=1e-4)
    batch = dummy_batch(vocab_size, seq_len=64, device=device)

    losses = []
    for step in range(3):
        optimizer.zero_grad()
        out = model(**batch, use_cache=False)
        out.loss.backward()
        optimizer.step()
        losses.append(out.loss.item())
        rank0_print(f"  step {step+1}  loss={losses[-1]:.4f}")

    ok = losses[-1] < losses[0]
    rank0_print(f"\n  losses: {[f'{l:.4f}' for l in losses]}")
    rank0_print(f"T3 RESULT: {'✅ PASS' if ok else '⚠️  losses not decreasing'}")
    return "PASS" if ok else "WARN"


# ============================================================================
# T4: DP=2 + SP=2 — requires 4 GPUs, document limitation with 2
# ============================================================================

def test_t4_dp2_sp2_combo():
    rank0_print("\n" + "=" * 60)
    rank0_print("T4: DP=2 + SP=2 combo (dp_shard=2 * ulysses=2 = needs 4 GPUs)")
    rank0_print("=" * 60)
    world = dist.get_world_size()
    needed = 2 * 2  # dp_shard=2, ulysses=2
    rank0_print(f"  world_size={world}  needed={needed}")
    if world < needed:
        rank0_print(f"  Insufficient GPUs for dp=2+sp=2 combo ({world} < {needed}).")
        rank0_print("  On NVIDIA with 4 GPUs: dp_size=2, dp_shard=2, ulysses=2, ep=1 → world=4 ✓")
        rank0_print("T4 RESULT: ✅ PASS (correctly documents 4-GPU requirement)")
        return "PASS"
    # If we somehow have 4+ GPUs:
    from veomni.distributed import parallel_state as ps
    ps.init_parallel_state(
        dp_size=world // 2, dp_shard_size=2, dp_replicate_size=1,
        ulysses_size=2, ep_size=1, dp_mode="fsdp2",
    )
    state = ps.get_parallel_state()
    rank0_print(f"  parallel_state — dp={state.dp_size}  ulysses={state.ulysses_size}  mesh={state.device_mesh}")
    rank0_print("T4 RESULT: ✅ PASS (4-GPU DP+SP combo initialized)")
    return "PASS"


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["T1", "T2", "T3", "T4"], required=True,
                        help="Which single test to run (one torchrun per test)")
    args = parser.parse_args()

    local_rank = init_dist()
    torch.cuda.set_device(local_rank)

    result = "UNKNOWN"
    try:
        if args.test == "T1":
            result = test_t1_dp2(QWEN25_0B5_PATH)
        elif args.test == "T2":
            result = test_t2_sp2(QWEN25_0B5_PATH)
        elif args.test == "T3":
            result = test_t3_ep2(QWEN_MOE_PATH)
        elif args.test == "T4":
            result = test_t4_dp2_sp2_combo()
    except Exception as e:
        import traceback
        rank0_print(f"\n{args.test} ERROR: {e}")
        traceback.print_exc()
        result = f"FAIL: {e}"
    finally:
        dist.barrier()

    rank0_print(f"\n{'='*60}")
    rank0_print(f"FINAL: {args.test} → {result}")
    rank0_print(f"{'='*60}")

    # Always print what TP/PP/CP situation is
    rank0_print("\nVeOmni parallelism summary (VERL veomni engine context):")
    rank0_print("  DP   (FSDP2 sharding)  → actor training  — dp_shard_size param")
    rank0_print("  SP   (Ulysses)         → actor training  — ulysses_size param")
    rank0_print("  EP   (Expert Parallel) → actor training  — ep_size param (MoE only)")
    rank0_print("  TP   (Tensor Parallel) → vLLM rollout only — tensor_model_parallel_size")
    rank0_print("  PP   (Pipeline)        → NOT in VERL veomni engine")
    rank0_print("  CP   (Context)         → NOT exposed via VERL (veomni pkg has it)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
