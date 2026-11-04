# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""``tt_dflash_sampled``: sampled requests on the DFlash lane."""

import itertools
from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.core.sched.output import SchedulerOutput

from tests.test_block_request_validation import (
    _config,
    _patch_model_resolution,
    _processor_harness,
)
from tests.test_block_scheduler import _request, _scheduler
from tests.test_lane_routing import _PlainFallbackBlockModel
from vllm_tt_plugin.config import (
    has_tt_dflash_sampled,
    store_tt_adaptive_block_output,
    store_tt_dflash_sampled,
    store_tt_plain_fallback,
)
from vllm_tt_plugin.lane_routing import (
    TT_EXECUTION_LANE_DFLASH,
    TT_EXECUTION_LANE_PLAIN,
    classify_tt_execution_lane,
    keeps_dflash_presence_penalty,
)
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.platform import TTPlatform, _neutralize_model_owned_sampling
from vllm_tt_plugin.scheduler import set_tt_block_step_decisions

_GRID = [
    dict(zip(("temperature", "presence_penalty", "frequency_penalty", "extra"), v))
    for v in itertools.product(
        (0.0, 0.7),
        (0.0, 1.5, -0.5),
        (0.0, 0.3),
        (None, "logprobs", "n", "structured", "min_p"),
    )
    if not (v[0] == 0.0 and v[3] == "n")  # vLLM rejects n > 1 with greedy
]


def _params(temperature, presence_penalty, frequency_penalty, extra):
    kwargs = dict(
        max_tokens=16,
        temperature=temperature,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        seed=42,
        top_k=20,
        top_p=0.95,
    )
    if extra == "logprobs":
        kwargs["logprobs"] = 0
    elif extra == "n":
        kwargs["n"] = 2
    elif extra == "structured":
        kwargs["structured_outputs"] = StructuredOutputsParams(json_object=True)
    elif extra == "min_p":
        kwargs["min_p"] = 0.1
    return SamplingParams(**kwargs)


@pytest.mark.parametrize("case", _GRID)
def test_sampled_classification(case):
    params = _params(**case)
    lane, reason = classify_tt_execution_lane(params, sampled_dflash=True)
    sampled = case["temperature"] > 0
    # vLLM resets min_p to neutral on a greedy request, so it never routes it.
    extra = (
        None if (case["extra"] == "min_p" and params.min_p == 0.0) else case["extra"]
    )
    eligible = (
        extra is None
        and case["frequency_penalty"] == 0.0
        and (
            case["presence_penalty"] == 0.0
            or (sampled and case["presence_penalty"] > 0.0)
        )
    )
    assert lane == (TT_EXECUTION_LANE_DFLASH if eligible else TT_EXECUTION_LANE_PLAIN)
    if eligible:
        assert reason == ("eligible_sampled" if sampled else "eligible_greedy")
    else:
        assert "temperature" not in reason.split(",")
    if not sampled:
        # Greedy requests classify exactly as without the capability.
        assert (lane, reason) == classify_tt_execution_lane(params)


@pytest.mark.parametrize("case", [c for c in _GRID if c["temperature"] > 0])
def test_without_the_capability_every_sampled_request_is_plain(case):
    lane, reason = classify_tt_execution_lane(_params(**case))
    assert lane == TT_EXECUTION_LANE_PLAIN
    assert "temperature" in reason.split(",")


def _sampled_config(flag: bool):
    config = SimpleNamespace(additional_config={})
    store_tt_plain_fallback(config, True)
    store_tt_dflash_sampled(config, flag)
    return config


def test_presence_penalty_is_model_owned_only_on_sampled_dflash():
    sampled = SamplingParams(max_tokens=8, temperature=1.0, presence_penalty=1.5)
    greedy = SamplingParams(max_tokens=8, temperature=0.0, presence_penalty=1.5)
    negative = SamplingParams(max_tokens=8, temperature=1.0, presence_penalty=-1.0)
    with_logprobs = SamplingParams(
        max_tokens=8, temperature=1.0, presence_penalty=1.5, logprobs=0
    )
    on, off = _sampled_config(True), _sampled_config(False)

    assert keeps_dflash_presence_penalty(on, sampled)
    assert not keeps_dflash_presence_penalty(off, sampled)
    assert not keeps_dflash_presence_penalty(on, greedy)  # plain lane
    assert not keeps_dflash_presence_penalty(on, negative)  # plain lane
    assert not keeps_dflash_presence_penalty(on, with_logprobs)  # plain lane

    params = SamplingParams(
        max_tokens=8, temperature=1.0, presence_penalty=1.5, repetition_penalty=1.2
    )
    ignored = _neutralize_model_owned_sampling(
        params, keep_transport_controls=True, keep_presence_penalty=True
    )
    assert params.presence_penalty == 1.5
    assert params.repetition_penalty == 1.0
    assert not any(x.startswith("presence_penalty") for x in ignored)


@pytest.mark.parametrize("flag", [False, True])
def test_input_processor_keeps_a_sampled_request_s_controls(monkeypatch, flag):
    processor_cls, processor = _processor_harness(monkeypatch)
    store_tt_adaptive_block_output(processor.vllm_config, True)
    store_tt_plain_fallback(processor.vllm_config, True)
    store_tt_dflash_sampled(processor.vllm_config, flag)
    params = SamplingParams(
        max_tokens=16,
        temperature=0.8,
        top_k=20,
        top_p=0.95,
        seed=3,
        presence_penalty=1.5,
    )

    request = processor_cls.process_inputs(
        processor, "sampled", {"prompt_token_ids": [1] * 200}, params
    )

    # dflash lane (flag) or plain lane (no flag): every control survives.
    sampling = request.sampling_params
    assert (sampling.temperature, sampling.top_k, sampling.top_p, sampling.seed) == (
        0.8,
        20,
        0.95,
        3,
    )
    assert sampling.presence_penalty == 1.5


class _SampledFallbackBlockModel(_PlainFallbackBlockModel):
    model_capabilities = {
        **_PlainFallbackBlockModel.model_capabilities,
        "tt_dflash_sampled": True,
    }


class _SampledWithoutFallback(_PlainFallbackBlockModel):
    model_capabilities = {
        **_PlainFallbackBlockModel.model_capabilities,
        "tt_plain_fallback": False,
        "tt_dflash_sampled": True,
    }


@pytest.mark.parametrize(
    ("model_class", "expected"),
    [(_SampledFallbackBlockModel, True), (_PlainFallbackBlockModel, False)],
)
def test_startup_reads_the_capability_and_no_environment(
    monkeypatch, model_class, expected
):
    # The fork's TT_DFLASH_SAMPLED switch is not read by the plugin.
    monkeypatch.setenv("TT_DFLASH_SAMPLED", "1" if not expected else "0")
    config = _config(max_num_seqs=4)
    _patch_model_resolution(monkeypatch, model_class)

    TTPlatform.check_and_update_config(config)

    assert has_tt_dflash_sampled(config) is expected
    assert config.additional_config.get("_tt_dflash_sampled") is expected


def test_startup_rejects_the_capability_without_plain_fallback(monkeypatch):
    config = _config(max_num_seqs=4)
    _patch_model_resolution(monkeypatch, _SampledWithoutFallback)

    with pytest.raises(
        ValueError, match="tt_dflash_sampled requires tt_plain_fallback"
    ):
        TTPlatform.check_and_update_config(config)


@pytest.mark.parametrize("flag", [False, True])
def test_scheduler_routes_sampled_requests_by_capability(flag):
    scheduler = _scheduler(
        adaptive=True,
        batched=True,
        max_num_seqs=2,
        plain_fallback=True,
        dflash_sampled=flag,
    )
    sampled = _request(64, request_id="sampled")
    sampled.sampling_params.temperature = 0.7
    greedy = _request(64, request_id="greedy")
    greedy.sampling_params.temperature = 0.0

    scheduler.add_request(sampled)
    scheduler.add_request(greedy)

    assert scheduler._lane_router.lane("greedy") == TT_EXECUTION_LANE_DFLASH
    assert scheduler._lane_router.lane("sampled") == (
        TT_EXECUTION_LANE_DFLASH if flag else TT_EXECUTION_LANE_PLAIN
    )


@pytest.mark.parametrize(
    ("flag", "decisions", "expected"),
    [
        (True, {"a": True, "b": True}, True),
        (True, {"a": True, "b": False}, False),  # a width-1 row reads them
        (True, {}, False),  # not a block-output step
        (False, {"a": True, "b": True}, False),
    ],
)
def test_block_steps_skip_the_penalty_history_only_when_model_owned(
    flag, decisions, expected
):
    runner = SimpleNamespace(_dflash_sampled=flag)
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {"a": 1, "b": 1}
    if decisions:
        set_tt_block_step_decisions(scheduler_output, decisions)

    assert (
        TTModelRunner._block_step_owns_penalties(runner, scheduler_output) is expected
    )
