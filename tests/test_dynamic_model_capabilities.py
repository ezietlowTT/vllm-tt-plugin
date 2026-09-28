# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import pytest

from vllm_tt_plugin.platform import TTPlatform


def test_dynamic_capability_resolver_takes_precedence_over_legacy_dict():
    class DynamicModel:
        model_capabilities = {"output_tokens_per_step": 1}

        @classmethod
        def get_model_capabilities(cls):
            return {"output_tokens_per_step": 32, "tt_adaptive_block_output": True}

    assert TTPlatform._resolve_model_capabilities(DynamicModel) == {
        "output_tokens_per_step": 32,
        "tt_adaptive_block_output": True,
    }
    assert TTPlatform._resolve_output_tokens_per_step(DynamicModel) == 32


def test_static_capability_dict_remains_the_default_contract():
    class StaticModel:
        model_capabilities = {"output_tokens_per_step": 8}

    assert (
        TTPlatform._resolve_model_capabilities(StaticModel)
        == StaticModel.model_capabilities
    )
    assert TTPlatform._resolve_output_tokens_per_step(StaticModel) == 8


def test_dynamic_capability_resolver_rejects_non_dict():
    class BrokenModel:
        @classmethod
        def get_model_capabilities(cls):
            return 32

    with pytest.raises(TypeError, match="capabilities must be a dict or None"):
        TTPlatform._resolve_model_capabilities(BrokenModel)
