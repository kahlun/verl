# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The NIXL checkpoint engine must load and stage weights on hosts without CUDA.

A rollout node on non-NVIDIA hardware has no cupy and no ``torch.cuda``. These tests stub the ``nixl`` package and
block ``cupy`` so any CUDA-only import or call in the engine fails loudly.
"""

import importlib
import importlib.util
import sys
import types
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def nixl_module(monkeypatch):
    for name in ("nixl", "nixl._api", "nixl._bindings"):
        stub = types.ModuleType(name)
        stub.__getattr__ = lambda attr: MagicMock()  # any nixl symbol the engine touches at import time
        monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.setitem(sys.modules, "cupy", None)  # `import cupy` now raises ImportError
    monkeypatch.delitem(sys.modules, "verl.checkpoint_engine.nixl_checkpoint_engine", raising=False)
    module = importlib.import_module("verl.checkpoint_engine.nixl_checkpoint_engine")
    monkeypatch.setattr(module, "NixlAgent", MagicMock())
    return module


def test_module_imports_without_cupy(nixl_module):
    assert nixl_module.NIXLCheckpointEngine is not None


def test_cpu_buckets_are_host_uint8_buffers(nixl_module):
    engine = nixl_module.NIXLCheckpointEngine(bucket_size=4096, device="cpu")
    engine.prepare()

    for buf in (engine.send_buf, engine.recv_buf):
        assert buf.device.type == "cpu"
        assert buf.dtype == torch.uint8
        assert buf.numel() == 4096
    assert engine.agent.register_memory.call_count == 2


def test_cuda_buckets_still_need_cupy(nixl_module):
    engine = nixl_module.NIXLCheckpointEngine(bucket_size=4096, device="cuda")
    with pytest.raises(ImportError):
        engine.prepare()


def test_no_hardcoded_cuda_synchronize():
    source = importlib.util.find_spec("verl.checkpoint_engine.nixl_checkpoint_engine").origin
    with open(source) as f:
        assert "torch.cuda.synchronize" not in f.read()


@pytest.mark.parametrize("rollout_world_size", [1, 2, 3])
@pytest.mark.parametrize("actor_world_size", [1, 2])
def test_topology_is_a_chain_trainer_to_each_rollout_worker(nixl_module, actor_world_size, rollout_world_size):
    """Agent 0 feeds rollout worker 1, which feeds worker 2, and so on; a single rollout worker has no successor."""
    metadata = [f"agent{i}" for i in range(actor_world_size + rollout_world_size)]
    rollout_agents = metadata[actor_world_size:]

    actor, rollout = nixl_module.NIXLCheckpointEngine.build_topology(actor_world_size, rollout_world_size, metadata)

    assert actor["next_agent_metadata"] == [rollout_agents[0]] + [None] * (actor_world_size - 1)
    assert rollout["rank"] == list(range(1, rollout_world_size + 1))
    assert rollout["prev_agent_metadata"] == [metadata[0]] + rollout_agents[:-1]
    assert rollout["next_agent_metadata"] == rollout_agents[1:] + [None]
    for kwargs in (actor, rollout):
        assert {len(v) for v in kwargs.values()} == {len(kwargs["rank"])}
