"""
Extended VeOmni test suite — NVIDIA A100 PCIe GPU1+GPU2 (2026-04-09)

Tests:
  --mode dp   DP=2 FSDP2 A/B scatter test (reproduces test_src_data_rank_ab.py baseline)
  --mode sp   SP=2 Ulysses A/B scatter test — confirms scatter fires in SP mode too
  --mode cp   CP=2 Context Parallel capability probe — ring-attn infrastructure check

--fix flag:  use src_data_rank=None (local split, zero scatter) instead of default=0

Run (GPU1+GPU2 only):
  # SP=2 DEFAULT (scatter active):
  CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --master_port=29600 \
      /workspace/verl/test_veomni_extended.py --mode sp

  # SP=2 FIX (scatter suppressed):
  CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --master_port=29601 \
      /workspace/verl/test_veomni_extended.py --mode sp --fix

  # CP=2 probe:
  CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --master_port=29602 \
      /workspace/verl/test_veomni_extended.py --mode cp
"""

import argparse
import os
import time

import torch
import torch.distributed as dist

# ── parse BEFORE any distributed/model init ───────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["dp", "sp", "cp"], default="dp",
                    help="Parallelism mode: dp=FSDP2, sp=Ulysses SP=2, cp=Context Parallel CP=2")
parser.add_argument("--fix", action="store_true",
                    help="Use src_data_rank=None (zero-scatter fix) instead of default src_data_rank=0")
parser.add_argument("--model", default=None,
                    help="Model path override (default: Qwen2.5-0.5B-Instruct)")
args = parser.parse_args()

# ── monkey-patch distribute_tensor BEFORE veomni import ──────────────────────
import torch.distributed.tensor as _tdt

_orig_dt = _tdt.distribute_tensor
_scatter_shard_calls = [0]
_scatter_shard_bytes = [0]
_replicate_calls     = [0]

def _instrumented_dt(tensor, device_mesh=None, placements=None, *, src_data_rank=0):
    from torch.distributed.tensor.placement_types import Shard, Replicate
    has_shard = any(isinstance(p, Shard) for p in (placements or []))
    has_repl  = any(isinstance(p, Replicate) for p in (placements or []))

    if args.fix:
        return _orig_dt(tensor, device_mesh, placements, src_data_rank=None)
    else:
        if has_shard:
            _scatter_shard_calls[0] += 1
            _scatter_shard_bytes[0] += tensor.numel() * tensor.element_size()
        if has_repl:
            _replicate_calls[0] += 1
        return _orig_dt(tensor, device_mesh, placements, src_data_rank=src_data_rank)

_tdt.distribute_tensor = _instrumented_dt

# ── now safe to import veomni ─────────────────────────────────────────────────
from veomni.distributed.parallel_state import init_parallel_state
from veomni.models.auto import build_foundation_model
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.optim import build_optimizer

# ── distributed init ──────────────────────────────────────────────────────────
dist.init_process_group(backend="nccl")
rank       = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
torch.manual_seed(42)

MODE_TAG = f"{args.mode.upper()}=2 {'FIX' if args.fix else 'DEFAULT'}"

if rank == 0:
    print(f"\n{'='*65}")
    print(f"  VeOmni Extended Test — {MODE_TAG}")
    print(f"  world_size={world_size}  GPUs=A100 PCIe (no NVLink)")
    print(f"{'='*65}\n", flush=True)

MODEL_PATH = args.model or (
    "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct"
    "/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
)

# ── parallelism config per mode ───────────────────────────────────────────────
if args.mode == "dp":
    # FSDP2 DP=2: fully sharded data parallel, 2 ranks share one model
    init_parallel_state(
        dp_size=2, dp_shard_size=2, dp_replicate_size=1,
        ulysses_size=1, cp_size=1, ep_size=1, dp_mode="fsdp2",
    )
    # sequence length has no constraint for DP
    SEQ_TEXT = (
        "The quick brown fox jumps over the lazy dog. "
        "It keeps running through the fields, leaping fences effortlessly."
    )

elif args.mode == "sp":
    # Ulysses SP=2: sequence is split across 2 ranks, full model on each (no FSDP shard)
    # Constraint: seq_len must be divisible by ulysses_size=2
    init_parallel_state(
        dp_size=1, dp_shard_size=1, dp_replicate_size=1,
        ulysses_size=2, cp_size=1, ep_size=1, dp_mode="fsdp2",
    )
    # Use longer text; tokenizer output length must be even (pad if needed)
    SEQ_TEXT = (
        "The quick brown fox jumps over the lazy dog and the cat sat on the mat watching patiently. "
        "Context parallel and sequence parallel both require the sequence length to be divisible."
    )

elif args.mode == "cp":
    # Context Parallel CP=2: VeOmni sets up a CP process group; ring-attention probe
    # Constraint: seq_len must be divisible by cp_size=2
    init_parallel_state(
        dp_size=1, dp_shard_size=1, dp_replicate_size=1,
        ulysses_size=1, cp_size=2, ep_size=1, dp_mode="fsdp2",
    )
    SEQ_TEXT = (
        "The quick brown fox jumps over the lazy dog and the cat sat on the mat watching patiently. "
        "Context parallel and sequence parallel both require the sequence length to be divisible."
    )

# ── build model on meta device ────────────────────────────────────────────────
model = build_foundation_model(
    config_path=MODEL_PATH,
    weights_path=None,
    init_device="meta",
    attn_implementation="flash_attention_2",
    force_use_huggingface=True,
)

# ── timed weight loading ──────────────────────────────────────────────────────
dist.barrier()
t0 = time.perf_counter()
model = build_parallelize_model(
    model,
    weights_path=MODEL_PATH,
    enable_gradient_checkpointing=False,
    init_device="meta",
)
dist.barrier()
load_time = time.perf_counter() - t0

if rank == 0:
    print(f"[{MODE_TAG}] Weight load time : {load_time:.3f} s", flush=True)
    if not args.fix:
        wire_bytes = _scatter_shard_bytes[0] * (world_size - 1) / world_size
        print(f"[{MODE_TAG}] Shard distribute_tensor calls : {_scatter_shard_calls[0]}")
        print(f"[{MODE_TAG}] Replicate calls               : {_replicate_calls[0]}")
        print(f"[{MODE_TAG}] Full-tensor bytes per rank     : {_scatter_shard_bytes[0]/1e9:.3f} GB")
        print(f"[{MODE_TAG}] Wire bytes wasted by scatter   : {wire_bytes/1e9:.3f} GB")
    else:
        print(f"[{MODE_TAG}] scatter suppressed (src_data_rank=None) — 0 bytes wasted")
    print(flush=True)

# ── training run ──────────────────────────────────────────────────────────────
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokens = tokenizer(SEQ_TEXT, return_tensors="pt")
input_ids = tokens["input_ids"].cuda()
seq_len = input_ids.shape[1]

# For SP=2 / CP=2: pad sequence to be divisible by parallelism degree
if args.mode in ("sp", "cp"):
    degree = 2  # ulysses_size or cp_size
    if seq_len % degree != 0:
        pad = degree - (seq_len % degree)
        pad_ids = torch.full((1, pad), tokenizer.pad_token_id or 0, dtype=input_ids.dtype).cuda()
        input_ids = torch.cat([input_ids, pad_ids], dim=1)
        seq_len = input_ids.shape[1]
    if rank == 0:
        print(f"[{MODE_TAG}] seq_len={seq_len} (divisible by {degree})", flush=True)

labels = input_ids.clone()
optimizer = build_optimizer(model, lr=1e-4)
model.train()

losses = []
step_times = []

for step in range(5):
    optimizer.zero_grad()
    t_step = time.perf_counter()
    try:
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss.backward()
        optimizer.step()
        dist.barrier()
        elapsed = time.perf_counter() - t_step
        losses.append(round(loss.item(), 6))
        step_times.append(round(elapsed, 4))
    except Exception as e:
        if rank == 0:
            print(f"[{MODE_TAG}] ERROR at step {step}: {type(e).__name__}: {e}", flush=True)
        dist.barrier()
        break

if rank == 0 and losses:
    avg_step = sum(step_times[1:]) / max(len(step_times[1:]), 1)
    print(f"[{MODE_TAG}] Losses (5 steps) : {losses}")
    print(f"[{MODE_TAG}] Step times (s)   : {step_times}")
    print(f"[{MODE_TAG}] Avg step (warmup excluded): {avg_step:.4f} s")
    print()
    if not args.fix:
        print("CORRECTNESS CHECK: re-run with --fix and compare Losses lines.")
    else:
        print("CORRECTNESS CHECK: compare Losses with DEFAULT run — must be identical.")
    print(flush=True)

dist.destroy_process_group()
