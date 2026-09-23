# Copyright (c) 2026 BAAI. All rights reserved.
"""Unit tests for the platform abstraction layer."""

import os
from contextlib import contextmanager
from unittest import mock

import pytest

from verl.plugin.platform import get_platform, set_platform
from verl.plugin.platform.platform_base import PlatformBase
from verl.plugin.platform.platform_manager import (
    PlatformRegistry,
    _create_platform,
    _detect_platform_name,
)


def _make_mock_platform(name="mock_xpu"):
    """Return a minimal concrete PlatformBase subclass for testing."""

    class _Mock(PlatformBase):
        @property
        def device_name(self):
            return name

        @property
        def vendor_name(self):
            return f"mock_{name}"

        @property
        def device_module(self):
            import torch

            return torch

        def is_available(self):
            return True

        def is_platform_available(self, use_smi_check=False):
            return True

        def current_device(self):
            return 0

        def device_count(self):
            return 1

        def set_device(self, device_index):
            pass

        def synchronize(self, device_index=None):
            pass

        def manual_seed(self, seed):
            pass

        def manual_seed_all(self, seed):
            pass

        def set_allocator_settings(self, settings):
            pass

        def empty_cache(self):
            pass

        def get_device_capability(self, device_index=0):
            return (None, None)

        def communication_backend_name(self):
            return "mock_ccl"

        def visible_devices_envvar(self):
            return "MOCK_VISIBLE_DEVICES"

        def ray_resource_name(self):
            return "MOCK"

        def ray_noset_envvars(self):
            return ["RAY_EXPERIMENTAL_NOSET_MOCK_VISIBLE_DEVICES"]

        def is_ipc_supported(self):
            return False

        @contextmanager
        def nvtx_range(self, msg):
            yield

        def profiler_start(self):
            pass

        def profiler_stop(self):
            pass

        def cudart(self):
            return None

    _Mock.__name__ = f"Mock_{name}"
    return _Mock


class TestPlatformDetection:
    """Test platform auto-detection logic."""

    def setup_method(self):
        import verl.plugin.platform.platform_manager as pm

        pm._current_platform = None

    def test_env_override_nvidia(self):
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "nvidia"}):
            assert _detect_platform_name() == "nvidia"

    def test_env_override_huawei(self):
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "huawei"}):
            assert _detect_platform_name() == "huawei"

    def test_env_override_amd(self):
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "amd"}):
            assert _detect_platform_name() == "amd"

    def test_invalid_value_passes_through(self):
        # When an explicit platform name is set, _detect_platform_name returns
        # it as-is (validation happens later in _create_platform).
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "invalid"}):
            assert _detect_platform_name() == "invalid"

    def test_case_insensitive(self):
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "NVIDIA"}):
            assert _detect_platform_name() == "nvidia"

    def test_empty_triggers_auto_detection(self):
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": ""}):
            assert _detect_platform_name() in ("nvidia", "huawei", "amd")


class TestPlatformCreation:
    """Test platform creation."""

    def test_cuda_warns_if_unavailable(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            with mock.patch("verl.plugin.platform.platform_manager.logger") as mock_logger:
                platform = _create_platform("nvidia")
                mock_logger.warning.assert_called_once()
                assert platform is not None

    def test_invalid_platform_raises(self):
        with pytest.raises(ValueError):
            _create_platform("invalid_platform")


class TestPlatformSingleton:
    """Test singleton and external injection."""

    def setup_method(self):
        import verl.plugin.platform.platform_manager as pm

        pm._current_platform = None

    def test_get_platform_returns_singleton(self):
        p1 = get_platform()
        p2 = get_platform()
        assert p1 is p2

    def test_set_platform_external_injection(self):
        """Simulates an entry_points plugin calling set_platform()."""
        custom = _make_mock_platform("mock_xpu")()
        set_platform(custom)
        assert get_platform() is custom
        assert get_platform().device_name == "mock_xpu"


class TestPlatformRegistry:
    """Test PlatformRegistry dynamic registration."""

    def setup_method(self):
        import verl.plugin.platform.platform_manager as pm

        pm._current_platform = None

    def test_builtin_platforms_registered(self):
        names = PlatformRegistry.registered_names()
        assert "nvidia" in names
        assert "huawei" in names
        assert "amd" in names

    def test_register_custom_platform(self):
        """External plugin registers a new platform via @PlatformRegistry.register()."""
        MockCls = _make_mock_platform("mock_test")
        PlatformRegistry.register(platform="mock_test")(MockCls)

        assert "mock_test" in PlatformRegistry.registered_names()
        assert PlatformRegistry.get("mock_test") is MockCls
        assert _create_platform("mock_test").device_name == "mock_test"

        del PlatformRegistry._platforms["mock_test"]

    def test_register_override(self):
        """Re-registering the same name overrides the previous class (last writer wins)."""
        original = PlatformRegistry.get("nvidia")
        FakeCls = _make_mock_platform("nvidia")
        PlatformRegistry.register(platform="nvidia")(FakeCls)
        assert PlatformRegistry.get("nvidia") is FakeCls
        PlatformRegistry._platforms["nvidia"] = original

    def test_unregistered_platform_raises(self):
        with pytest.raises(ValueError):
            _create_platform("nonexistent_platform")


class TestNumaAffinityDefault:
    """PlatformBase.set_numa_affinity's default (NVML) implementation.

    Covers the forwarding/success/missing-pynvml/cleanup paths flagged as
    untested in review. libnuma is mocked here rather than skipped, so this
    exercises the same logic regardless of whether libnuma.so happens to be
    installed on the machine running the test.
    """

    def _platform(self):
        return _make_mock_platform("mock_numa")()

    def test_no_numa_topology_skips_pynvml_entirely(self):
        platform = self._platform()
        fake_libnuma = mock.MagicMock()
        fake_libnuma.numa_available.return_value = -1
        with mock.patch("ctypes.CDLL", return_value=fake_libnuma):
            with mock.patch("builtins.__import__") as mocked_import:
                platform.set_numa_affinity(0)
                assert not any(call.args and call.args[0] == "pynvml" for call in mocked_import.call_args_list)

    def test_libnuma_missing_returns_without_raising(self):
        platform = self._platform()
        with mock.patch("ctypes.CDLL", side_effect=OSError("libnuma.so not found")):
            platform.set_numa_affinity(0)  # must not raise

    def test_success_path_calls_nvml_and_cleans_up(self):
        platform = self._platform()
        fake_libnuma = mock.MagicMock()
        fake_libnuma.numa_available.return_value = 0
        fake_pynvml = mock.MagicMock()
        fake_pynvml.nvmlDeviceGetHandleByIndex.return_value = "handle-3"

        with mock.patch("ctypes.CDLL", return_value=fake_libnuma):
            with mock.patch.dict("sys.modules", {"pynvml": fake_pynvml}):
                platform.set_numa_affinity(3)

        fake_pynvml.nvmlInit.assert_called_once()
        fake_pynvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(3)
        fake_pynvml.nvmlDeviceSetCpuAffinity.assert_called_once_with("handle-3")
        fake_pynvml.nvmlShutdown.assert_called_once()

    def test_pynvml_not_installed_returns_without_raising(self):
        platform = self._platform()
        fake_libnuma = mock.MagicMock()
        fake_libnuma.numa_available.return_value = 0

        real_import = __import__

        def _raise_for_pynvml(name, *args, **kwargs):
            if name == "pynvml":
                raise ImportError("no module named pynvml")
            return real_import(name, *args, **kwargs)

        with mock.patch("ctypes.CDLL", return_value=fake_libnuma):
            with mock.patch("builtins.__import__", side_effect=_raise_for_pynvml):
                platform.set_numa_affinity(0)  # must not raise

    def test_nvml_shutdown_called_even_on_failure_after_init(self):
        platform = self._platform()
        fake_libnuma = mock.MagicMock()
        fake_libnuma.numa_available.return_value = 0
        fake_pynvml = mock.MagicMock()
        fake_pynvml.nvmlDeviceGetHandleByIndex.side_effect = RuntimeError("boom")

        with mock.patch("ctypes.CDLL", return_value=fake_libnuma):
            with mock.patch.dict("sys.modules", {"pynvml": fake_pynvml}):
                platform.set_numa_affinity(0)  # must not raise

        fake_pynvml.nvmlShutdown.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
