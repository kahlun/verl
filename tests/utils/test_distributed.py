# Copyright (c) 2026 BAAI. All rights reserved.
"""Unit tests for verl.utils.distributed.set_numa_affinity's platform dispatch.

Regression coverage for a bug caught in review: the libnuma.so probe used to
live in this dispatcher, before the platform hook, so any image lacking
libnuma.so (plausible on XPU images -- they have no reason to ship an
NVIDIA-adjacent library) returned early and never called the platform's
override at all. The probe now lives inside PlatformBase's own NVML-specific
default implementation (see tests/plugin/test_platform_abstraction.py),
not here. These tests mock get_platform() entirely, so they pass regardless
of whether libnuma.so happens to be installed on the machine running them --
that independence from libnuma is the thing being tested.
"""

import os
from unittest import mock

import pytest

import verl.utils.distributed as distributed


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)


class TestSetNumaAffinityDispatch:
    def test_delegates_to_platform_with_local_rank_from_env(self, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "2")
        mock_platform = mock.MagicMock()
        mock_platform.ray_resource_name.return_value = "GPU"

        with mock.patch("verl.utils.distributed.get_platform", return_value=mock_platform):
            with mock.patch("verl.utils.distributed.get_resource_name", return_value="GPU"):
                with mock.patch("verl.utils.distributed.ray") as mock_ray:
                    mock_ray.is_initialized.return_value = False
                    distributed.set_numa_affinity()

        mock_platform.set_numa_affinity.assert_called_once_with(2)

    def test_delegates_to_platform_with_local_rank_from_ray(self):
        mock_platform = mock.MagicMock()

        with mock.patch("verl.utils.distributed.get_platform", return_value=mock_platform):
            with mock.patch("verl.utils.distributed.get_resource_name", return_value="GPU"):
                with mock.patch("verl.utils.distributed.ray") as mock_ray:
                    mock_ray.is_initialized.return_value = True
                    mock_ray.get_runtime_context.return_value.get_accelerator_ids.return_value = {"GPU": ["5"]}
                    distributed.set_numa_affinity()

        mock_platform.set_numa_affinity.assert_called_once_with(5)

    def test_dispatch_is_independent_of_libnuma(self, monkeypatch):
        """The regression: dispatch must not depend on libnuma.so at all anymore.

        Simulates libnuma.so being completely absent (raises OSError, as it
        would via ctypes.CDLL on a box that doesn't have it) and asserts the
        platform hook still runs. Before the fix, this scenario returned
        before ever reaching get_platform().set_numa_affinity(...).
        """
        monkeypatch.setenv("LOCAL_RANK", "0")
        mock_platform = mock.MagicMock()

        with mock.patch("ctypes.CDLL", side_effect=OSError("libnuma.so not found")):
            with mock.patch("verl.utils.distributed.get_platform", return_value=mock_platform):
                with mock.patch("verl.utils.distributed.get_resource_name", return_value="GPU"):
                    with mock.patch("verl.utils.distributed.ray") as mock_ray:
                        mock_ray.is_initialized.return_value = False
                        distributed.set_numa_affinity()

        mock_platform.set_numa_affinity.assert_called_once_with(0)

    def test_npu_platform_skips_entirely(self):
        mock_platform = mock.MagicMock()
        with mock.patch("verl.utils.distributed.is_npu_available", True):
            with mock.patch("verl.utils.distributed.get_platform", return_value=mock_platform):
                distributed.set_numa_affinity()
        mock_platform.set_numa_affinity.assert_not_called()

    def test_missing_local_rank_env_does_not_raise(self):
        mock_platform = mock.MagicMock()
        with mock.patch("verl.utils.distributed.get_platform", return_value=mock_platform):
            with mock.patch("verl.utils.distributed.get_resource_name", return_value="GPU"):
                with mock.patch("verl.utils.distributed.ray") as mock_ray:
                    mock_ray.is_initialized.return_value = False
                    distributed.set_numa_affinity()  # LOCAL_RANK unset, must not raise
        mock_platform.set_numa_affinity.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
