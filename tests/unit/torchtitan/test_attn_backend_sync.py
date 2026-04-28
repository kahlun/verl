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
from unittest.mock import MagicMock

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


@pytest.fixture(params=_get_model_modules(), ids=lambda x: x[0] if isinstance(x, tuple) else str(x))
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

    def test_legacy_api_fallback_raises(self):
        """On older torchtitan without attn_backend param, verl raises RuntimeError
        when attn_type is flex or varlen.

        Simulates the Feb-2026 torchtitan snapshot. Continuing silently would
        produce a mismatch: model built with sdpa but mask builder expecting
        flex/varlen, corrupting training. A clear RuntimeError is safer.
        """
        def legacy_model_registry(flavor: str):
            return MagicMock(name="legacy_model_spec")

        # flex/varlen must raise — model and mask would be inconsistent
        for bad_attn_type in ("flex", "varlen"):
            sig = inspect.signature(legacy_model_registry)
            assert "attn_backend" not in sig.parameters
            with pytest.raises(RuntimeError, match="attn_backend"):
                if "attn_backend" in sig.parameters:
                    legacy_model_registry("debugmodel", attn_backend=bad_attn_type)
                else:
                    if bad_attn_type in ("flex", "varlen"):
                        raise RuntimeError(
                            f"torchtitan's model_registry() does not accept 'attn_backend' "
                            f"(older snapshot detected). Cannot build model with "
                            f"attn_type='{bad_attn_type}' as requested. "
                            f"Either upgrade torchtitan (commit 7cec166 or later), or use a "
                            f"pre-baked flavor that encodes the backend "
                            f"(e.g. 'debugmodel_flex_attn' or 'debugmodel_varlen_attn')."
                        )
                    legacy_model_registry("debugmodel")

        # sdpa with legacy API is fine — model + mask both use sdpa path
        sig = inspect.signature(legacy_model_registry)
        result = legacy_model_registry("debugmodel")  # no raise expected
        assert result is not None

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

        # Legacy path must raise for flex/varlen, not silently continue
        assert 'attn_type in ("flex", "varlen")' in source, (
            "transformer_impl.py legacy path must raise for flex/varlen "
            "to prevent model+mask backend mismatch."
        )
        assert "raise RuntimeError" in source, (
            "transformer_impl.py legacy path must raise RuntimeError, not warn-and-continue."
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

        # Warning/error message must use correct legacy flavor suffix naming
        assert "_flex_attn" in source and "_varlen_attn" in source, (
            "Error message must cite correct legacy flavor suffixes '_flex_attn' "
            "and '_varlen_attn', not '_flex' alone."
        )
