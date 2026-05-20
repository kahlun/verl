# Copyright 2026 Intel Corporation
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
"""Intel XPU platform implementation (Arc Pro / Data Center GPU Max)."""

import logging
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional

import torch

from .platform_base import PlatformBase
from .platform_manager import PlatformRegistry

logger = logging.getLogger(__name__)


@PlatformRegistry.register(platform="xpu")
class PlatformXPU(PlatformBase):
    """Platform backend for Intel XPU (Arc Pro B-Series / Data Center GPU Max).

    Tested on Intel Arc Pro B60 (Battlemage, 24 GB VRAM) with driver 26.09.

    Known limitations (tracked upstream):
    - IPC not supported: XPU uses SYCL IPC handles; the CUDA 8-element tuple
      format would corrupt data. CPU shared memory fallback is used instead.
      Upstream: https://github.com/intel/torch-xpu-ops/issues/1678
    - vLLM memory check needs patching: Level-Zero context overhead from Ray
      pre-started workers inflates reported "used" memory. See xpu_patches.py.
      Upstream: https://github.com/vllm-project/vllm/pull/37149
    """

    # ------------------------------------------------------------------
    # Core device management
    # ------------------------------------------------------------------

    @property
    def device_name(self) -> str:
        return "xpu"

    @property
    def device_module(self) -> ModuleType:
        return torch.xpu

    def is_available(self) -> bool:
        try:
            return hasattr(torch, "xpu") and torch.xpu.is_available()
        except Exception:
            return False

    def current_device(self) -> int:
        return torch.xpu.current_device()

    def device_count(self) -> int:
        return torch.xpu.device_count()

    def set_device(self, device_index: int) -> None:
        torch.xpu.set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        torch.xpu.synchronize(device_index)

    # ------------------------------------------------------------------
    # Random number generator
    # ------------------------------------------------------------------

    def manual_seed(self, seed: int) -> None:
        torch.xpu.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.xpu.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def set_allocator_settings(self, settings: str) -> None:
        # XPU does not expose _set_allocator_settings; no-op.
        logger.debug("XPU does not support allocator settings, ignoring: %s", settings)

    def empty_cache(self) -> None:
        torch.xpu.empty_cache()

    # ------------------------------------------------------------------
    # Device properties
    # ------------------------------------------------------------------

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        # XPU has no CUDA-style compute capability; return (None, None).
        return (None, None)

    # ------------------------------------------------------------------
    # Distributed communication
    # ------------------------------------------------------------------

    def communication_backend_name(self) -> str:
        return "xccl"

    def visible_devices_envvar(self) -> str:
        # ZE_AFFINITY_MASK accepts bare device indices (0,1,2,...).
        # ONEAPI_DEVICE_SELECTOR needs "level_zero:N" prefix — avoid for device-restriction.
        return "ZE_AFFINITY_MASK"

    # ------------------------------------------------------------------
    # Profiling helpers
    # ------------------------------------------------------------------

    @contextmanager
    def nvtx_range(self, msg: str):
        # XPU does not have an NVTX equivalent; no-op.
        yield

    def profiler_start(self) -> None:
        pass

    def profiler_stop(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Low-level runtime API
    # ------------------------------------------------------------------

    def cudart(self) -> Any:
        return None
