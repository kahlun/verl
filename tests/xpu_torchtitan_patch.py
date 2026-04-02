"""Monkey-patch for torchtitan on PyTorch 2.11.0+xpu.

torchtitan main branch uses ShardPlacementResult from PyTorch nightly,
which doesn't exist in torch 2.11.0+xpu. This shim stubs it so the
llama4/qwen3 import chain succeeds. Only needed until PyTorch XPU
ships the symbol.
"""
import torch
import torch.distributed.fsdp._fully_shard._fsdp_common as _fsdp_common
from typing import NamedTuple

if not hasattr(_fsdp_common, "ShardPlacementResult"):

    class ShardPlacementResult(NamedTuple):
        local_tensor: torch.Tensor
        global_offset: int
        narrowed: bool = False

    _fsdp_common.ShardPlacementResult = ShardPlacementResult
