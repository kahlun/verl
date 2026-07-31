# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Device-agnostic access to the FlashAttention sequence-packing helpers.

``index_first_axis``, ``pad_input``, ``rearrange`` and ``unpad_input`` are the
``bert_padding`` utilities that pack/unpack padded batches into the
variable-length layout attention kernels expect.  Which implementation to use
is a **per-device** decision, so it lives in the platform plugin layer:
each ``PlatformBase`` subclass returns the right callables from
``get_attention_functions()`` (CUDA/ROCm prefer the fused ``flash_attn``
kernels; NPU, XPU and other accelerators use the hardware-agnostic
transformers/einops implementations).

This module is a thin, lazily-resolved facade over that hook so callers can
simply ``from verl.utils.attention_utils import unpad_input`` without caring
about the active device.
"""

from typing import Any, Callable

_index_first_axis: Callable | None = None
_pad_input: Callable | None = None
_rearrange: Callable | None = None
_unpad_input: Callable | None = None


def _get_attention_functions() -> tuple[Callable, Callable, Callable, Callable]:
    """Resolve the attention helpers for the active platform (cached).

    Delegates to ``get_platform().get_attention_functions()`` so that the
    device-specific choice is made once, by the platform plugin, rather than
    via hardware ``if`` branches scattered across the codebase.
    """
    global _index_first_axis, _pad_input, _rearrange, _unpad_input

    if _index_first_axis is None:
        from verl.plugin.platform import get_platform

        _index_first_axis, _pad_input, _rearrange, _unpad_input = get_platform().get_attention_functions()

    return _index_first_axis, _pad_input, _rearrange, _unpad_input


def index_first_axis(*args: Any, **kwargs: Any) -> Any:
    """Gather rows of ``input`` at ``indices`` (device-dispatched)."""
    func, *_ = _get_attention_functions()
    return func(*args, **kwargs)


def pad_input(*args: Any, **kwargs: Any) -> Any:
    """Scatter unpadded ``hidden_states`` back to ``(batch, seqlen, ...)`` (device-dispatched)."""
    _, func, *_ = _get_attention_functions()
    return func(*args, **kwargs)


def rearrange(*args: Any, **kwargs: Any) -> Any:
    """``einops.rearrange`` (routed through the platform for parity with the others)."""
    *_, func, _ = _get_attention_functions()
    return func(*args, **kwargs)


def unpad_input(*args: Any, **kwargs: Any) -> Any:
    """Remove padding and return FlashAttention-compatible sequence metadata (device-dispatched)."""
    *_, func = _get_attention_functions()
    return func(*args, **kwargs)


__all__ = ["index_first_axis", "pad_input", "rearrange", "unpad_input"]
