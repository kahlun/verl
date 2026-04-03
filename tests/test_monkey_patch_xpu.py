"""Test that the monkey_patch.py XPU fix works correctly.

Validates:
1. _flash_attention_forward resolves to xpu_flash_attention_forward on XPU
2. _ulysses_flash_attention_forward can be called (sp_size=1 path)
3. attn_implementation defaults to "sdpa" on XPU
4. Model forward works with sdpa attn_implementation (SYCL-TLA Flash kernel)
5. Model forward + verl monkey_patch works with sdpa on XPU
"""

import torch
import pytest


@pytest.fixture(autouse=True)
def xpu_guard():
    if not torch.xpu.is_available():
        pytest.skip("Requires XPU")


def test_flash_attention_forward_resolves_to_xpu():
    """The module-level _flash_attention_forward should be xpu_flash_attention_forward on XPU."""
    from verl.models.transformers import monkey_patch
    from verl.models.transformers.xpu_attn import xpu_flash_attention_forward

    # The module-level name should point to our XPU implementation
    assert monkey_patch._flash_attention_forward is xpu_flash_attention_forward, (
        f"Expected xpu_flash_attention_forward, got {monkey_patch._flash_attention_forward}"
    )
    print("PASS: _flash_attention_forward correctly resolves to xpu_flash_attention_forward")


def test_default_attention_implementation():
    """get_default_attention_implementation() should return 'sdpa' on XPU."""
    from verl.utils.device import get_default_attention_implementation

    impl = get_default_attention_implementation()
    assert impl == "sdpa", f"Expected 'sdpa', got '{impl}'"
    print(f"PASS: default attention implementation = '{impl}'")


def test_ulysses_forward_sp1():
    """_ulysses_flash_attention_forward works in sp_size=1 (no-op) mode on XPU."""
    from verl.models.transformers.monkey_patch import _ulysses_flash_attention_forward

    device = "xpu"
    batch, seq_len, n_heads, head_dim = 2, 64, 4, 32
    # In real HF models, GQA expansion happens in the model layer before calling
    # the attention forward.  Use matching Q/KV heads for direct call testing.
    n_kv_heads = n_heads

    q = torch.randn(batch, seq_len, n_heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch, seq_len, n_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch, seq_len, n_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)

    # sp_size=1 path: should just call _flash_attention_forward (→ xpu_flash_attention_forward)
    out = _ulysses_flash_attention_forward(
        q, k, v,
        attention_mask=None,
        query_length=seq_len,
        position_ids=position_ids,
        is_causal=True,
    )

    assert out.shape == (batch, seq_len, n_heads, head_dim), f"Unexpected shape: {out.shape}"
    assert out.dtype == torch.bfloat16
    assert not torch.isnan(out).any(), "Output contains NaN"
    print(f"PASS: _ulysses_flash_attention_forward sp_size=1 output shape={out.shape}")


def test_model_forward_sdpa():
    """A small model loaded with attn_implementation='sdpa' works on XPU (SYCL-TLA Flash)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2.5-0.5B"
    device = "xpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(device)

    inputs = tokenizer("Hello world", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)

    assert out.logits is not None
    assert not torch.isnan(out.logits).any(), "SDPA produced NaN — SYCL-TLA kernel issue?"
    print(f"PASS: Model forward with sdpa on XPU, logits shape={out.logits.shape}")

    del model
    torch.xpu.empty_cache()


def test_model_forward_with_monkey_patch():
    """Model with verl's apply_monkey_patch works with sdpa on XPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from verl.models.transformers.monkey_patch import apply_monkey_patch

    model_name = "Qwen/Qwen2.5-0.5B"
    device = "xpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(device)

    # Apply verl's monkey patch (use_remove_padding patches _flash_attention_forward globally)
    apply_monkey_patch(model, use_remove_padding=False, use_fused_kernels=False)

    inputs = tokenizer("The quick brown fox", return_tensors="pt").to(device)
    labels = inputs["input_ids"].clone()

    out = model(**inputs, labels=labels)

    assert out.loss is not None
    assert not torch.isnan(out.loss), "Loss is NaN after monkey_patch + sdpa"
    print(f"PASS: Model + monkey_patch + sdpa on XPU, loss={out.loss.item():.4f}")

    del model
    torch.xpu.empty_cache()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
