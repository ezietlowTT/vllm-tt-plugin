# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""``tt_block_output_multimodal``: adaptive block models keep encoder features.

The runner half needs no block-output branch: ``TTModelRunner`` gathers the
features of every multimodal model's prompt rows (``test_multimodal_gather.py``).
"""

import pytest
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager

from tests.test_block_request_validation import (
    BlockModel,
    _config,
    _patch_model_resolution,
)
from tests.test_block_scheduler import _request, _scheduler
from vllm_tt_plugin.config import is_tt_block_output_multimodal
from vllm_tt_plugin.platform import TTPlatform

FEATURE_TOKENS = 16


def _image_request(offset: int):
    request = _request(max_tokens=3)
    request.mm_features = [
        MultiModalFeatureSpec(
            data=None,
            modality="image",
            identifier="img-0",
            mm_position=PlaceholderRange(offset=offset, length=FEATURE_TOKENS),
        )
    ]
    return request


def _multimodal_scheduler(*, multimodal: bool, encoder_budget: int):
    scheduler = _scheduler(adaptive=True, max_num_seqs=2, multimodal=multimodal)
    # The test checkpoint is text-only; give the scheduler the encoder budget
    # and cache a multimodal model's mm budget would size.
    scheduler.max_num_encoder_input_tokens = encoder_budget
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=encoder_budget)
    return scheduler


def test_capable_model_schedules_a_feature_at_offset_zero():
    """The head-of-line case the drop guards against: a feature at offset 0 is
    encoded and the whole prompt prefills in one step, not zero tokens."""
    scheduler = _multimodal_scheduler(multimodal=True, encoder_budget=64)
    request = _image_request(offset=0)

    scheduler.add_request(request)
    step = scheduler.schedule()

    assert len(request.mm_features) == 1
    assert step.num_scheduled_tokens == {request.request_id: 32}
    assert step.scheduled_encoder_inputs == {request.request_id: [0]}


def test_without_the_capability_features_are_dropped():
    scheduler = _multimodal_scheduler(multimodal=False, encoder_budget=64)
    request = _image_request(offset=0)

    scheduler.add_request(request)

    assert request.mm_features == []
    assert scheduler.schedule().num_scheduled_tokens == {request.request_id: 32}


def test_zero_encoder_budget_still_drops_features():
    """A capable declaration on a model with no encoder budget (a text-only
    checkpoint) cannot schedule the feature: it would stall at offset 0, so the
    drop still applies."""
    scheduler = _multimodal_scheduler(multimodal=True, encoder_budget=0)
    request = _image_request(offset=0)

    scheduler.add_request(request)

    assert request.mm_features == []
    assert scheduler.schedule().num_scheduled_tokens == {request.request_id: 32}


class _MultimodalWithoutAdaptive(BlockModel):
    model_capabilities = {
        **BlockModel.model_capabilities,
        "tt_block_output_multimodal": True,
    }


class _AdaptiveMultimodal(BlockModel):
    model_capabilities = {
        **BlockModel.model_capabilities,
        "tt_adaptive_block_output": True,
        "tt_block_output_multimodal": True,
    }


def test_startup_stores_the_capability(monkeypatch):
    config = _config()
    _patch_model_resolution(monkeypatch, _AdaptiveMultimodal)

    TTPlatform.check_and_update_config(config)

    assert is_tt_block_output_multimodal(config) is True


def test_startup_rejects_the_capability_without_adaptive(monkeypatch):
    config = _config()
    _patch_model_resolution(monkeypatch, _MultimodalWithoutAdaptive)

    with pytest.raises(
        ValueError, match="tt_block_output_multimodal requires tt_adaptive_block_output"
    ):
        TTPlatform.check_and_update_config(config)
