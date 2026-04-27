"""Test that TorchtitanEngineConfig.attn_type is correctly propagated to
torchtitan's model_registry when building the model spec.

This is a unit test for the fix in transformer_impl.py that ensures the
attn_backend parameter is passed to model_registry() instead of relying on
a broken post-hoc override that used the wrong attribute path.

The bug: verl called model_registry(flavor) without passing attn_backend,
so every model was silently built with sdpa regardless of the user's config.
The post-hoc override referenced 'model_spec.model.layer' (singular) but
torchtitan uses 'model_spec.model.layers' (plural), so it never executed.
"""

import importlib
import inspect

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

        # Pick smallest flavor
        flavor = "debugmodel" if "debugmodel" in flavors else next(iter(flavors))
        spec = mod.model_registry(flavor, attn_backend="flex")

        # Verify the inner attention is FlexAttention
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
            # llama4 defaults to flex
            assert isinstance(inner, FlexAttention.Config)
        else:
            assert isinstance(inner, ScaledDotProductAttention.Config), (
                f"Expected ScaledDotProductAttention.Config by default, "
                f"got {type(inner).__name__}"
            )

    def test_verl_code_passes_attn_backend(self):
        """Verify verl's transformer_impl.py passes attn_backend to model_registry.

        This is a source-code level check to ensure the fix is in place.
        """
        impl_path = (
            "verl/workers/engine/torchtitan/transformer_impl.py"
        )
        import os

        # Find the file relative to the repo root
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        full_path = os.path.join(repo_root, impl_path)

        with open(full_path) as f:
            source = f.read()

        # The fix: model_registry must be called with attn_backend=
        assert "model_registry(torchtitan_flavor, attn_backend=" in source, (
            "transformer_impl.py must pass attn_backend to model_registry(). "
            "Without this, the model silently uses sdpa regardless of config."
        )

        # The bug: must NOT contain the broken direct assignment
        assert "model_spec.model.layer.attention.attn_backend" not in source, (
            "transformer_impl.py still contains the broken direct assignment "
            "to model_spec.model.layer.attention.attn_backend (singular 'layer'). "
            "torchtitan uses 'layers' (plural list), so this never executes."
        )

        # Also check the second bug site is fixed
        assert "self.trainer.model_config.layer.attention.attn_backend" not in source, (
            "transformer_impl.py still reads attn_type from "
            "self.trainer.model_config.layer.attention.attn_backend which uses "
            "wrong attribute path. Should use self.engine_config.attn_type."
        )
