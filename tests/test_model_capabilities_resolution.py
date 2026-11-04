# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""``get_model_capabilities``: config time and runtime read one resolved dict."""

from types import SimpleNamespace

import pytest

from tests.test_block_request_validation import (
    BlockModel,
    _config,
    _patch_model_resolution,
)
from vllm_tt_plugin.config import (
    bind_tt_model_capabilities,
    get_tt_output_tokens_per_step,
    is_tt_adaptive_block_output_model,
    resolve_tt_model_capabilities,
)
from vllm_tt_plugin.loader import TTModelLoader
from vllm_tt_plugin.platform import TTPlatform

_PROFILE_CAPABILITIES = {
    **BlockModel.model_capabilities,
    "tt_adaptive_block_output": True,
}


class ProfileModel:
    """A class whose import-time dict is the width-1 profile; the hook is not."""

    model_capabilities = {"output_tokens_per_step": 1}

    @classmethod
    def get_model_capabilities(cls):
        return dict(_PROFILE_CAPABILITIES)

    @staticmethod
    def release_request(_slot):
        pass

    @staticmethod
    def release_persistent_capture():
        pass

    @staticmethod
    def note_state_slots_moved(_moves):
        pass


def test_hook_result_replaces_the_class_dict():
    assert resolve_tt_model_capabilities(ProfileModel) == _PROFILE_CAPABILITIES
    assert TTPlatform._resolve_output_tokens_per_step(ProfileModel) == 256


def test_class_dict_is_the_default_contract():
    class StaticModel:
        model_capabilities = {"output_tokens_per_step": 8}

    assert resolve_tt_model_capabilities(StaticModel) is StaticModel.model_capabilities
    assert resolve_tt_model_capabilities(SimpleNamespace()) is None
    assert TTPlatform._resolve_output_tokens_per_step(StaticModel) == 8


def test_hook_returning_a_non_dict_fails_the_launch():
    class BrokenModel:
        @classmethod
        def get_model_capabilities(cls):
            return 32

    with pytest.raises(TypeError, match=r"must be a dict or None, got int: 32"):
        resolve_tt_model_capabilities(BrokenModel)


def test_check_and_update_config_reads_the_resolved_dict(monkeypatch):
    """The width-256 adaptive contract exists only in the hook result."""
    config = _config()
    config.scheduler_config.max_num_seqs = 2
    _patch_model_resolution(monkeypatch, ProfileModel)

    TTPlatform.check_and_update_config(config)

    assert get_tt_output_tokens_per_step(config) == 256
    assert is_tt_adaptive_block_output_model(config) is True


def test_loader_binds_the_resolved_dict_onto_the_model(monkeypatch):
    built = []

    class Model:
        pass

    class LoadedProfileModel(ProfileModel):
        @classmethod
        def initialize_vllm_model(cls, *_args, **_kwargs):
            built.append(Model())
            return built[-1]

    monkeypatch.setattr(
        "vllm_tt_plugin.loader.get_model_architecture",
        lambda _model_config: (LoadedProfileModel, None),
    )
    model_config = SimpleNamespace(hf_config=None, max_model_len=1024)
    vllm_config = SimpleNamespace(
        additional_config={},
        device_config=SimpleNamespace(device=None),
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        parallel_config=SimpleNamespace(data_parallel_size=1),
    )
    monkeypatch.setattr("vllm_tt_plugin.loader.get_tt_max_batch_size", lambda _c: 2)

    model = TTModelLoader(SimpleNamespace()).load_model(vllm_config, model_config)

    assert model is built[-1]
    assert model.model_capabilities == _PROFILE_CAPABILITIES


def test_bind_keeps_an_instance_dict_the_model_built_itself():
    narrowed = {**_PROFILE_CAPABILITIES, "supports_sample_on_device": False}

    class NarrowingModel:
        def __init__(self):
            self.model_capabilities = narrowed

    model = NarrowingModel()
    bind_tt_model_capabilities(model, ProfileModel)
    assert model.model_capabilities is narrowed


def test_bind_leaves_a_class_without_the_hook_alone():
    class StaticModel:
        model_capabilities = {"output_tokens_per_step": 1}

    model = StaticModel()
    bind_tt_model_capabilities(model, StaticModel)
    assert "model_capabilities" not in vars(model)
