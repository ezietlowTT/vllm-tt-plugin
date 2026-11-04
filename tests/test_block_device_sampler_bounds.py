# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""``max_device_top_k`` / ``supports_device_penalties`` on a block-output model.

They route rows to host sampling, which only a ``tt_plain_fallback`` plain-lane
decode or an adaptive prefill anchor may use; a block step stays on the
model-owned device sampler whatever its rows' controls.
"""

from types import SimpleNamespace

import pytest

from tests.test_adaptive_block_batched import _BatchedBlockModel
from tests.test_block_request_validation import _config, _patch_model_resolution
from tests.test_lane_routing import _PlainFallbackBlockModel
from vllm_tt_plugin.config import get_tt_output_tokens_per_step
from vllm_tt_plugin.input_batch import SamplingInputBatch
from vllm_tt_plugin.lane_routing import (
    TT_EXECUTION_LANE_DFLASH,
    TT_EXECUTION_LANE_PLAIN,
    TT_EXECUTION_LANE_PREFILL,
)
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.platform import TTPlatform

BOUNDS = {"max_device_top_k": 32, "supports_device_penalties": False}


class _BoundedPlainFallbackModel(_PlainFallbackBlockModel):
    model_capabilities = {**_PlainFallbackBlockModel.model_capabilities, **BOUNDS}


class _BoundedAdaptiveModel(_BatchedBlockModel):
    model_capabilities = {**_BatchedBlockModel.model_capabilities, **BOUNDS}


def test_startup_accepts_the_bound_on_a_plain_fallback_block_model(monkeypatch):
    config = _config(max_num_seqs=4)
    _patch_model_resolution(monkeypatch, _BoundedPlainFallbackModel)

    TTPlatform.check_and_update_config(config)

    assert get_tt_output_tokens_per_step(config) == 256


def test_startup_rejects_the_bound_without_plain_fallback(monkeypatch):
    config = _config(max_num_seqs=4)
    _patch_model_resolution(monkeypatch, _BoundedAdaptiveModel)

    with pytest.raises(ValueError, match="max_device_top_k.*without tt_plain_fallback"):
        TTPlatform.check_and_update_config(config)


def _runner(*, top_k: int, temperature: float = 0.7, penalties: bool = False):
    sampling = SamplingInputBatch(2)
    sampling.top_k[:] = top_k
    sampling.temperature[:] = temperature
    return SimpleNamespace(
        sample_on_device_mode="all",
        _is_block_output_model=True,
        _is_adaptive_block_output=True,
        seeded_sampling_policy="auto",
        num_devices=4,
        tt_data_parallel_size=1,
        model=SimpleNamespace(model_capabilities=dict(BOUNDS)),
        model_config=SimpleNamespace(logits_processors=[]),
        supports_device_grammar=False,
        input_batch=SimpleNamespace(
            sampling=sampling,
            req_id_to_index={"a": 0, "b": 1},
            no_penalties=not penalties,
            no_allowed_token_ids=True,
            max_num_logprobs=None,
        ),
    )


def _device(runner, *, is_decode: bool, lane: str) -> bool:
    return TTModelRunner.check_perform_device_sampling(
        runner, is_decode, False, execution_lane=lane
    )


@pytest.mark.parametrize(
    ("top_k", "penalties", "expected"),
    [
        (1, False, True),
        (32, False, True),
        (33, False, False),
        (64, False, False),
        (0, False, False),  # unbounded
        (20, True, False),  # penalties stay on the host sampler
    ],
)
def test_plain_lane_steps_apply_the_device_bounds(top_k, penalties, expected):
    runner = _runner(top_k=top_k, penalties=penalties)
    assert _device(runner, is_decode=True, lane=TT_EXECUTION_LANE_PLAIN) is expected


def test_a_greedy_plain_step_ignores_the_top_k_bound():
    runner = _runner(top_k=64, temperature=0.0)
    assert _device(runner, is_decode=True, lane=TT_EXECUTION_LANE_PLAIN)


@pytest.mark.parametrize(("top_k", "penalties"), [(64, False), (0, False), (20, True)])
def test_a_dflash_block_step_is_never_host_routed(top_k, penalties):
    """A sampled dflash request (tt_dflash_sampled) may carry any top_k and a
    presence penalty; its block step is model-sampled regardless."""
    runner = _runner(top_k=top_k, penalties=penalties)
    assert _device(runner, is_decode=True, lane=TT_EXECUTION_LANE_DFLASH)


def test_an_adaptive_prefill_anchor_applies_the_bound():
    runner = _runner(top_k=64)
    assert not _device(runner, is_decode=False, lane=TT_EXECUTION_LANE_PREFILL)
    runner = _runner(top_k=16)
    assert _device(runner, is_decode=False, lane=TT_EXECUTION_LANE_PREFILL)


def test_an_ordinary_model_keeps_the_bound_on_every_step():
    runner = _runner(top_k=64)
    runner._is_block_output_model = False
    assert not _device(runner, is_decode=True, lane=None)
