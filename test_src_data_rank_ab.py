"""
A/B test: VeOmni weight loading with vs without the src_data_rank=None fix.

DEFAULT mode (current VeOmni):
    distribute_tensor(t, mesh, placements)
    → src_data_rank=0 → mesh_scatter() → dist.scatter() over NVLink/PCIe
    Even though every rank already read the full weights from disk.

FIX mode (proposed fix):
    distribute_tensor(t, mesh, placements, src_data_rank=None)
    → each rank slices its local chunk → ZERO collective communication

Expected results:
    - Losses must be IDENTICAL (same model weights after load)
    - Load time in FIX mode should be faster (no scatter overhead)
    - scatter_calls in FIX mode = 0

Usage (GPU1+GPU2 only):
    # DEFAULT (current VeOmni behaviour):
    CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --master_port=29530 \
        /workspace/verl/test_src_data_rank_ab.py

    # FIX (src_data_rank=None):
    CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --master_port=29531 \
        /workspace/verl/test_src_data_rank_ab.py --fix
"""

import argparse
import os
import time

import torch
import torch.distributed as dist

# ── parse args BEFORE any distributed init ──────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--fix", action="store_true",
                    help="Use src_data_rank=None (proposed fix)")
parser.add_argument("--model", default=None,
                    help="Override model path (default: Qwen2.5-0.5B-Instruct)")
args = parser.parse_args()

# ── monkey-patch distribute_tensor BEFORE veomni is imported ────────────────
# torch_parallelize.py does a lazy `from torch.distributed.tensor import
# distribute_tensor` inside the function body, so patching the module
# attribute here is sufficient — the lazy import will see our wrapper.

import torch.distributed.tensor as _tdt

_orig_distribute_tensor = _tdt.distribute_tensor

_scatter_Shard_calls    = [0]   # number of Shard distribute_tensor calls
_scatter_Shard_bytes    = [0]   # total bytes in the FULL tensors being scattered
_scatter_Replicate_calls= [0]

def _instrumented_dt(tensor, device_mesh=None, placements=None, *, src_data_rank=0):
    from torch.distributed.tensor.placement_types import Shard, Replicate

    has_shard = any(isinstance(p, Shard) for p in (placements or []))
    has_repl  = any(isinstance(p, Replicate) for p in (placements or []))

    if args.fix:
        # FIX: local split/copy — zero scatter collective
        return _orig_distribute_tensor(tensor, device_mesh, placements,
                                       src_data_rank=None)
    else:
        # DEFAULT: count scatter traffic for Shard params
        if has_shard:
            _scatter_Shard_calls[0] += 1
            _scatter_Shard_bytes[0] += tensor.numel() * tensor.element_size()
        if has_repl:
            _scatter_Replicate_calls[0] += 1
        return _orig_distribute_tensor(tensor, device_mesh, placements,
                                       src_data_rank=src_data_rank)   # default=0

_tdt.distribute_tensor = _instrumented_dt

# ── now safe to import veomni ────────────────────────────────────────────────
from veomni.distributed.parallel_state import init_parallel_state
from veomni.models.auto import build_foundation_model
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.optim import build_optimizer

# ── distributed init ─────────────────────────────────────────────────────────
dist.init_process_group(backend="nccl")
rank       = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
torch.manual_seed(42)

MODE = "FIX(src_data_rank=None)" if args.fix else "DEFAULT(src_data_rank=0)"

if rank == 0:
    print(f"\n{'='*60}")
    print(f"  MODE: {MODE}")
    print(f"  world_size={world_size}, ranks using GPU1+GPU2 (A100 PCIe, no NVLink)")
    print(f"{'='*60}\n", flush=True)

# ── parallel state: DP=2 FSDP2 ──────────────────────────────────────────────
init_parallel_state(
    dp_size=2, dp_shard_size=2, dp_replicate_size=1,
    ulysses_size=1, ep_size=1, dp_mode="fsdp2",
)

MODEL_PATH = args.model or (
    "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct"
    "/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
)

# ── build model on meta device ───────────────────────────────────────────────
model = build_foundation_model(
    config_path=MODEL_PATH,
    weights_path=None,
    init_device="meta",
    attn_implementation="flash_attention_2",
    force_use_huggingface=True,
)

# ── time the weight loading ──────────────────────────────────────────────────
dist.barrier()
t_load_start = time.perf_counter()

model = build_parallelize_model(
    model,
    weights_path=MODEL_PATH,
    enable_gradient_checkpointing=False,
    init_device="meta",
    # broadcast_model_weights_from_rank0 NOT set  →  load_model_weights path
    # (every rank reads from disk, distribute_tensor is called per-param)
)

dist.barrier()
t_load_end = time.perf_counter()
load_time = t_load_end - t_load_start

if rank == 0:
    print(f"[{MODE}] Weight load time : {load_time:.3f} s", flush=True)
    if not args.fix:
        # Scatter stats — bytes sent on wire ≈ sum(param_size) * (world-1)/world
        # because rank0 sends (world-1) slices to other ranks
        wire_bytes = _scatter_Shard_bytes[0] * (world_size - 1) / world_size
        print(f"[{MODE}] Shard distribute_tensor calls : {_scatter_Shard_calls[0]}")
        print(f"[{MODE}] Replicate distribute_tensor calls: {_scatter_Replicate_calls[0]}")
        print(f"[{MODE}] Full-tensor bytes loaded (per rank) : "
              f"{_scatter_Shard_bytes[0]/1e9:.3f} GB")
        print(f"[{MODE}] Wire bytes wasted by scatter (≈)   : "
              f"{wire_bytes/1e9:.3f} GB  [sent over PCIe for data every rank already had]")
    else:
        print(f"[{MODE}] No scatter collective was fired — zero wire bytes wasted")
    print(flush=True)

# ── short training run — losses MUST match between modes ────────────────────
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
text = "The quick brown fox jumps over the lazy dog and then keeps running"
tokens = tokenizer(text, return_tensors="pt")
input_ids = tokens["input_ids"].cuda()
labels = input_ids.clone()

optimizer = build_optimizer(model, lr=1e-4)
model.train()

losses = []
step_times = []
for step in range(5):
    optimizer.zero_grad()
    t0 = time.perf_counter()
    out = model(input_ids=input_ids, labels=labels)
    loss = out.loss
    loss.backward()
    optimizer.step()
    dist.barrier()
    t1 = time.perf_counter()
    losses.append(round(loss.item(), 6))
    step_times.append(round(t1 - t0, 4))

if rank == 0:
    avg_step = sum(step_times[1:]) / len(step_times[1:])   # skip warmup
    print(f"[{MODE}] Losses (5 steps) : {losses}")
    print(f"[{MODE}] Step times (s)   : {step_times}")
    print(f"[{MODE}] Avg step time    : {avg_step:.4f} s  (warmup step excluded)")
    print()
    print("CORRECTNESS CHECK: run both modes and compare the 'Losses' lines.")
    print("They must be identical to confirm the fix does not change model weights.")
    print(flush=True)

dist.destroy_process_group()
