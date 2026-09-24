# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0(the "License");
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

from types import SimpleNamespace

import torch

from verl.workers.engine.fsdp.transformer_impl import _drop_redundant_tied_lm_head


def test_tied_model_drops_redundant_lm_head():
    params = {
        "model.embed_tokens.weight": torch.randn(8, 4),
        "lm_head.weight": torch.randn(8, 4),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4),
    }

    _drop_redundant_tied_lm_head(params, SimpleNamespace(tie_word_embeddings=True))

    assert "lm_head.weight" not in params
    assert "model.embed_tokens.weight" in params
    assert "model.layers.0.self_attn.q_proj.weight" in params


def test_untied_model_keeps_lm_head():
    params = {
        "model.embed_tokens.weight": torch.randn(8, 4),
        "lm_head.weight": torch.randn(8, 4),
    }

    _drop_redundant_tied_lm_head(params, SimpleNamespace(tie_word_embeddings=False))

    assert "lm_head.weight" in params


def test_tied_model_without_lm_head_key_is_a_no_op():
    params = {"model.embed_tokens.weight": torch.randn(8, 4)}

    _drop_redundant_tied_lm_head(params, SimpleNamespace(tie_word_embeddings=True))

    assert params.keys() == {"model.embed_tokens.weight"}


def test_missing_tie_word_embeddings_attribute_defaults_to_untied():
    params = {"lm_head.weight": torch.randn(8, 4)}

    _drop_redundant_tied_lm_head(params, SimpleNamespace())

    assert "lm_head.weight" in params
