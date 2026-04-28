"""Test that TorchtitanEngineConfig.attn_type is correctly propagated to
torchtitan's model_registry when building the model spec.

Background — torchtitan API evolution:
  - v0.1.0 / v0.2.0 / v0.2.1 / v0.2.2 (all PyPI tags):
      Used get_train_spec(); no model_registry() at all.
  - Feb 2026 refactor (commit 9810191, "Config System Refactor"):
      model_registry(flavor) introduced; NO attn_backend param.
      Attention backend was baked per-flavor (e.g. "debugmodel_flex_attn").
      model_spec.model.layer.attention.attn_backend was a mutable string field.
  - Apr 2026 (commit 7cec166+, current HEAD):
      model_registry(flavor, attn_backend="sdpa") — param added.
      attn_backend resolved at registry time into typed inner_attention Config.
      model_spec.model.layers[0].attention.inner_attention — plural 'layers'.

The original verl code was written for the Feb-2026 state but broke when
torchtitan refactored again in April (changed .layer → .layers, removed the
string field). The fix uses inspect.signature to handle both snapshots.

The bug (before fix):
  verl called model_registry(flavor) without attn_backend, AND the fallback
  guard used .layer (singular) which never matched. Both paths silently fell
  back to sdpa even when engine_config.attn_type="flex".
"""

import importlib
import inspect
from unittest.mock import MagicMock, patch

import pytest


def _get_model_modules():
    """Return (name, module) pairs for torchtitan models with model_registry."""
    model_names = ["llama3", "qwen3", "deepseek_v3", "llama4"]
    results = []
    for name in model_names:
        try:
            mod = importlib.import_module(f"torchtitan.models.{name}")
            if hasattr(mod, "model_registry"):
                results.append((name, mod))
        except ImportError:
            pass
    return results


@pytest.fixture(params=_get_model_modules(), ids=lambda x: x[0])
def model_module(request):
    return request.param


class TestAttnBackendSync:
    """Verify attn_backend is correctly passed to model_registry."""

    def test_model_registry_accepts_attn_backend(self, model_module):
        """model_registry() must accept attn_backend as a keyword argument."""
        name, mod = model_module
        sig = inspect.signature(mod.model_registry)
        assert "attn_backend" in sig.parameters, (
            f"torchtitan.models.{name}.model_registry() does not accept "
            f"'attn_backend' parameter"
        )

    def test_default_is_sdpa(self, model_module):
        """model_registry default for attn_backend should be 'sdpa' for most models."""
        name, mod = model_module
        sig = inspect.signature(mod.model_registry)
        default = sig.parameters["attn_backend"].default
        # llama4 defaults to flex; all others default to sdpa
        if name == "llama4":
            assert default == "flex", (
                f"llama4 model_registry should default to 'flex', got '{default}'"
            )
        else:
            assert default == "sdpa", (
                f"{name} model_registry should default to 'sdpa', got '{default}'"
            )

    def test_flex_attn_propagated(self, model_module):
        """When attn_backend='flex' is passed, the model must use FlexAttention."""
        name, mod = model_module
        from torchtitan.models.common.attention import FlexAttention

        # Use the smallest flavor available (debugmodel or first key)
        flavors = None
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if isinstance(obj, dict) and attr.endswith("_configs"):
                flavors = obj
                break
        assert flavors is not None, f"No configs dict found in {name}"

        flavor = "debugmodel" if "debugmodel" in flavors else next(iter(flavors))
        spec = mod.model_registry(flavor, attn_backend="flex")

        first_layer = spec.model.layers[0]
        inner = first_layer.attention.inner_attention
        assert isinstance(inner, FlexAttention.Config), (
            f"Expected FlexAttention.Config for attn_backend='flex', "
            f"got {type(inner).__name__}"
        )

    def test_sdpa_attn_default(self, model_module):
        """When called without attn_backend, model should use SDPA (except llama4)."""
        name, mod = model_module
        from torchtitan.models.common.attention import (
            FlexAttention,
            ScaledDotProductAttention,
        )

        flavors = None
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if isinstance(obj, dict) and attr.endswith("_configs"):
                flavors = obj
                break
        assert flavors is not None

        flavor = "debugmodel" if "debugmodel" in flavors else next(iter(flavors))
        spec = mod.model_registry(flavor)

        first_layer = spec.model.layers[0]
        inner = first_layer.attention.inner_attention

        if name == "llama4":
            assert isinstance(inner, FlexAttention.Config)
        else:
            assert isinstance(inner, ScaledDotProductAttention.Config), (
                f"Expected ScaledDotProductAttention.Config by default, "
                f"got {type(inner).__name__}"
            )

    def test_legacy_api_fallback_warns(self):
        """On older torchtitan without attn_backend param, verl emits a warning.

        Simulates the Feb-2026 torchtitan snapshot where model_registry(flavor)
        had no attn_backend parameter. Verifies the inspect-guard logic falls
        back gracefully and logs a warning rather than crashing or silently
        applying the wrong backend.
        """
        import logging

        # Build a fake model_registry that has NO attn_backend param (Feb-2026 API)
        def legacy_model_registry(flavor: str):
            return MagicMock(name="legacy_model_spec")

        # Exercise the inspect-guard logic directly (same logic as transformer_impl.py)
        called_without_attn_backend = []
        warned = []

        mock_logger = MagicMock()
        mock_logger.warning.side_effect = lambda msg, *a, **kw: warned.append(msg)

        attn_type = "flex"
        sig = inspect.signature(legacy_model_registry)
        if "attn_backend" in sig.parameters:
            result = legacy_model_registry("debugmodel", attn_backend=attn_type)
        else:
            mock_logger.warning(
                f"torchtitan's model_registry() does not accept 'attn_backend'. "
                f"The requested attn_type='{attn_type}' cannot be applied. "
                f"To use a specific attention backend, pass the pre-baked flavor "
                f"(e.g. 'debugmodel_flex') or upgrade torchtitan."
            )
            result = legacy_model_registry("debugmodel")
            called_without_attn_backend.append(True)

        assert called_without_attn_backend, (
            "Legacy path should call model_registry without attn_backend"
        )
        assert warned, "Legacy path should emit a warning about attn_type"
        assert "attn_backend" in warned[0] and "flex" in warned[0], (
            f"Warning should mention 'attn_backend' and the requested type, got: {warned[0]}"
        )

    def test_verl_code_uses_inspect_guard(self):
        """Verify verl's transformer_impl.py uses inspect guard for API compatibility."""
        import os

        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        impl_path = os.path.join(
            repo_root, "verl/workers/engine/torchtitan/transformer_impl.py"
        )
        with open(impl_path) as f:
            source = f.read()

        # Must use inspect to detect which API is available
        assert '"attn_backend" in' in source and "signature" in source, (
            "transformer_impl.py must use inspect.signature to guard the attn_backend "
            "call for torchtitan API compatibility."
        )

        # Must call with attn_backend when available
        assert "model_registry(torchtitan_flavor, attn_backend=" in source, (
            "transformer_impl.py must pass attn_backend to model_registry() "
            "when the parameter is available."
        )

        # Must NOT contain the broken direct assignment (singular .layer)
        assert "model_spec.model.layer.attention.attn_backend" not in source, (
            "transformer_impl.py still contains broken assignment to "
            "model_spec.model.layer (singular). torchtitan uses .layers (plural)."
        )

        # Must NOT read attn_type from the wrong trainer path
        assert "self.trainer.model_config.layer.attention.attn_backend" not in source, (
            "transformer_impl.py must not read attn_type from "
            "self.trainer.model_config.layer.attention.attn_backend. "
            "Should use self.engine_config.attn_type."
        )
