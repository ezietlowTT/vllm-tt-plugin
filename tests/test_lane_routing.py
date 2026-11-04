# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""``tt_plain_fallback``: DFlash and plain execution lanes on block-output models."""

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams, StructuredOutputsParams

from tests.test_adaptive_block_batched import _BatchedBlockModel, _outputs
from tests.test_block_request_validation import (
    _config,
    _patch_model_resolution,
    _processor_harness,
    _set_platform_contract,
    _validate,
)
from tests.test_block_scheduler import CANVAS, _request, _runner_output, _scheduler
from vllm_tt_plugin.config import (
    has_tt_plain_fallback,
    store_tt_adaptive_block_output,
    store_tt_plain_fallback,
)
from vllm_tt_plugin.lane_routing import (
    TT_EXECUTION_LANE_DFLASH,
    TT_EXECUTION_LANE_PLAIN,
    classify_tt_execution_lane,
    get_tt_execution_lane,
    get_tt_request_execution_lanes,
)
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.platform import TTPlatform
from vllm_tt_plugin.scheduler import get_tt_block_step_decisions

# Classification


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"temperature": 0.7}, "temperature"),
        ({"temperature": 0.7, "n": 2}, "n,temperature"),
        ({"presence_penalty": 0.5}, "presence_penalty"),
        ({"frequency_penalty": 0.5}, "frequency_penalty"),
        ({"repetition_penalty": 1.1}, "repetition_penalty"),
        ({"temperature": 0.7, "min_p": 0.1}, "temperature,min_p"),
        ({"min_tokens": 2}, "min_tokens"),
        ({"logprobs": 0}, "logprobs"),
        (
            {"structured_outputs": StructuredOutputsParams(json_object=True)},
            "structured_outputs",
        ),
        ({"logit_bias": {3: 1.0}}, "logit_bias"),
        ({"allowed_token_ids": [2, 3]}, "allowed_token_ids"),
        ({"bad_words": ["x"]}, "bad_words"),
        ({"flat_logprobs": True}, "flat_logprobs"),
    ],
)
def test_every_host_control_routes_to_the_plain_lane(kwargs, reason):
    params = SamplingParams(max_tokens=8, **{"temperature": 0.0, **kwargs})
    assert classify_tt_execution_lane(params) == (TT_EXECUTION_LANE_PLAIN, reason)


def test_a_greedy_request_with_neutral_controls_routes_to_dflash():
    params = SamplingParams(max_tokens=8, temperature=0.0, top_k=5, seed=3)
    assert classify_tt_execution_lane(params) == (
        TT_EXECUTION_LANE_DFLASH,
        "eligible_greedy",
    )


# Scheduler


def _lane_scheduler():
    return _scheduler(adaptive=True, batched=True, max_num_seqs=2, plain_fallback=True)


def _lane_pair(scheduler):
    """A greedy (dflash) and a sampled (plain) request through their prefill."""
    dflash = _request(CANVAS * 4, request_id="dflash")
    plain = _request(CANVAS * 4, request_id="plain")
    dflash.sampling_params.temperature = 0.0
    plain.sampling_params.temperature = 0.7
    scheduler.add_request(dflash)
    scheduler.add_request(plain)
    prefill = scheduler.schedule()
    assert get_tt_execution_lane(prefill) == "prefill"
    assert get_tt_request_execution_lanes(prefill) == {
        "dflash": TT_EXECUTION_LANE_DFLASH,
        "plain": TT_EXECUTION_LANE_PLAIN,
    }
    scheduler.update_from_output(prefill, _outputs({"dflash": [5], "plain": [6]}))
    return dflash, plain


def test_decode_steps_are_homogeneous_and_alternate_lanes():
    scheduler = _lane_scheduler()
    dflash, plain = _lane_pair(scheduler)

    first = scheduler.schedule()
    assert get_tt_execution_lane(first) == TT_EXECUTION_LANE_DFLASH
    assert list(first.num_scheduled_tokens) == ["dflash"]
    assert get_tt_block_step_decisions(first) == {"dflash": True}
    scheduler.update_from_output(first, _runner_output(first, list(range(CANVAS))))

    second = scheduler.schedule()
    assert get_tt_execution_lane(second) == TT_EXECUTION_LANE_PLAIN
    assert list(second.num_scheduled_tokens) == ["plain"]
    # A plain-lane decode is a one-token step even on the batched contract.
    assert get_tt_block_step_decisions(second) == {"plain": False}
    assert plain.num_output_placeholders == 1
    scheduler.update_from_output(second, _runner_output(second, [9]))

    third = scheduler.schedule()
    assert get_tt_execution_lane(third) == TT_EXECUTION_LANE_DFLASH
    assert list(third.num_scheduled_tokens) == ["dflash"]


def test_a_lane_step_returns_the_hidden_lane_to_running():
    scheduler = _lane_scheduler()
    dflash, plain = _lane_pair(scheduler)

    step = scheduler.schedule()

    assert list(step.num_scheduled_tokens) == ["dflash"]
    assert {request.request_id for request in scheduler.running} == {
        "dflash",
        "plain",
    }
    assert scheduler._lane_router.lane("plain") == TT_EXECUTION_LANE_PLAIN


def test_a_single_lane_decodes_without_alternating():
    scheduler = _lane_scheduler()
    dflash, plain = _lane_pair(scheduler)
    scheduler.finish_requests("dflash", dflash.status.FINISHED_ABORTED)

    for _ in range(2):
        step = scheduler.schedule()
        assert get_tt_execution_lane(step) == TT_EXECUTION_LANE_PLAIN
        assert list(step.num_scheduled_tokens) == ["plain"]
        scheduler.update_from_output(step, _runner_output(step, [9]))


def test_finished_requests_leave_the_lane_map():
    scheduler = _lane_scheduler()
    dflash, plain = _lane_pair(scheduler)
    scheduler.finish_requests("plain", plain.status.FINISHED_ABORTED)

    scheduler.schedule()

    assert "plain" not in scheduler._lane_router._lanes


def test_plain_lane_requests_keep_their_max_tokens():
    """The block-request repair clamps max_tokens to whole canvases on the
    dflash lane only; a plain request decodes one token per step and keeps its
    limit."""
    scheduler = _lane_scheduler()
    plain = _request(230, request_id="plain")
    plain.sampling_params.temperature = 0.7
    block = _request(230, request_id="block")
    block.sampling_params.temperature = 0.0

    scheduler.add_request(plain)
    scheduler.add_request(block)

    assert plain.max_tokens == 230
    assert block.max_tokens == 224  # (256 - 32) // 16 * 16


def test_without_the_capability_steps_carry_no_lane():
    scheduler = _scheduler(adaptive=True, batched=True, max_num_seqs=2)
    request = _request(CANVAS * 2)
    request.sampling_params.temperature = 0.7
    scheduler.add_request(request)

    step = scheduler.schedule()

    assert get_tt_execution_lane(step) is None
    assert get_tt_request_execution_lanes(step) == {}


# Platform


class _PlainFallbackBlockModel(_BatchedBlockModel):
    model_capabilities = {
        **_BatchedBlockModel.model_capabilities,
        "tt_plain_fallback": True,
    }


class _PlainFallbackWithoutBatched(_BatchedBlockModel):
    model_capabilities = {
        **_BatchedBlockModel.model_capabilities,
        "tt_adaptive_block_batched": False,
        "tt_plain_fallback": True,
    }


def test_startup_stores_the_capability(monkeypatch):
    config = _config(max_num_seqs=4)
    _patch_model_resolution(monkeypatch, _PlainFallbackBlockModel)

    TTPlatform.check_and_update_config(config)

    assert has_tt_plain_fallback(config) is True
    assert config.additional_config["_tt_plain_fallback"] is True


def test_startup_rejects_the_capability_without_batched(monkeypatch):
    config = _config(max_num_seqs=4)
    _patch_model_resolution(monkeypatch, _PlainFallbackWithoutBatched)

    with pytest.raises(
        ValueError, match="tt_plain_fallback requires tt_adaptive_block_batched"
    ):
        TTPlatform.check_and_update_config(config)


def test_an_operator_key_does_not_enable_routing():
    config = SimpleNamespace(additional_config={"tt_plain_fallback": True})
    assert has_tt_plain_fallback(config) is False


def test_validation_admits_plain_lane_controls(monkeypatch):
    config = _set_platform_contract(monkeypatch)
    store_tt_adaptive_block_output(config, True)
    store_tt_plain_fallback(config, True)

    _validate(SamplingParams(max_tokens=16, logprobs=0))
    _validate(SamplingParams(max_tokens=16, allowed_token_ids=[2, 3]))
    _validate(
        SamplingParams(
            max_tokens=16,
            structured_outputs=StructuredOutputsParams(json_object=True),
        )
    )
    # A plain-lane request is not held to whole canvases: 257 tokens after a
    # 513-token prompt would need two 256-token canvases past max_model_len.
    _validate(SamplingParams(max_tokens=257, temperature=0.7), prompt_len=513)
    with pytest.raises(ValueError, match="thinking_token_budget"):
        _validate(SamplingParams(max_tokens=16, thinking_token_budget=8))


def test_input_processor_keeps_plain_lane_sampling_and_default_limit(monkeypatch):
    processor_cls, processor = _processor_harness(
        monkeypatch, output_size=256, max_model_len=1024
    )
    store_tt_adaptive_block_output(processor.vllm_config, True)
    store_tt_plain_fallback(processor.vllm_config, True)
    params = SamplingParams(max_tokens=None, temperature=0.7, seed=42, min_p=0.1)

    request = processor_cls.process_inputs(
        processor, "plain", {"prompt_token_ids": [1] * 200}, params
    )

    sampling = request.sampling_params
    assert (sampling.temperature, sampling.seed, sampling.min_p) == (0.7, 42, 0.1)
    assert sampling.max_tokens == 824  # vLLM's default, not whole canvases
    assert params.max_tokens is None


def test_input_processor_keeps_a_dflash_request_greedy(monkeypatch):
    """The dflash lane's anchor is sampled with the request's own controls; the
    block-output neutralization would turn temperature 0 into 1."""
    processor_cls, processor = _processor_harness(monkeypatch)
    store_tt_adaptive_block_output(processor.vllm_config, True)
    store_tt_plain_fallback(processor.vllm_config, True)

    request = processor_cls.process_inputs(
        processor,
        "dflash",
        {"prompt_token_ids": [1] * 200},
        SamplingParams(max_tokens=None, temperature=0.0),
    )

    assert request.sampling_params.temperature == 0.0
    assert request.sampling_params.max_tokens == 768  # whole canvases


# Runner


def _sampling_runner(*, greedy=True, penalties=False, active_processor=False):
    runner = object.__new__(TTModelRunner)
    runner.sample_on_device_mode = "decode_only"
    runner.model = SimpleNamespace(model_capabilities={})
    runner.seeded_sampling_policy = "auto"
    runner.supports_device_grammar = False
    runner.num_devices = 4
    runner.tt_data_parallel_size = 1
    runner.supports_topk_logprobs = False
    runner.model_config = SimpleNamespace(logits_processors=())
    runner.input_batch = SimpleNamespace(
        all_greedy=greedy,
        no_penalties=not penalties,
        no_allowed_token_ids=True,
        max_num_logprobs=None,
        sampling=SimpleNamespace(
            bad_words_token_ids={},
            has_active_logitsprocs=lambda: active_processor,
        ),
    )
    return runner


def _device(runner, lane, structured=False):
    return runner.check_perform_device_sampling(
        is_decode=True, has_structured_outputs=structured, execution_lane=lane
    )


def test_a_greedy_plain_step_samples_on_device():
    """A completed min_tokens constraint must not keep host readback forever."""
    assert _device(_sampling_runner(), TT_EXECUTION_LANE_PLAIN)


@pytest.mark.parametrize(
    ("runner", "structured"),
    [
        pytest.param(_sampling_runner(greedy=False), False, id="random"),
        pytest.param(_sampling_runner(penalties=True), False, id="penalties"),
        pytest.param(_sampling_runner(active_processor=True), False, id="processor"),
        pytest.param(_sampling_runner(), True, id="structured"),
    ],
)
def test_plain_steps_keep_semantic_controls_on_host(runner, structured):
    assert not _device(runner, TT_EXECUTION_LANE_PLAIN, structured)


def test_a_random_step_outside_the_plain_lane_is_unchanged():
    assert _device(_sampling_runner(greedy=False), None)
    assert _device(_sampling_runner(greedy=False), TT_EXECUTION_LANE_DFLASH)


def test_a_plain_decode_step_may_host_sample(monkeypatch):
    """The block-output host-sampling guard exempts plain-lane decodes only."""
    from tests import test_block_model_runner

    tt_out = torch.tensor([[[0.0, 1.0, 0.0]]])
    with pytest.raises(ValueError, match="fell back to host sampling"):
        test_block_model_runner._extract(
            4, tt_out, [1], device_sampling=False, is_decode=True
        )

    class ReachedHostSampling(Exception):
        pass

    def bind(runner):
        runner._check_host_logits = lambda *_a, **_k: (_ for _ in ()).throw(
            ReachedHostSampling()
        )
        return bind_committed_width(runner)

    bind_committed_width = test_block_model_runner._bind_committed_width
    monkeypatch.setattr(test_block_model_runner, "_bind_committed_width", bind)
    with pytest.raises(ReachedHostSampling):
        test_block_model_runner._extract(
            4,
            tt_out,
            [1],
            device_sampling=False,
            is_decode=True,
            execution_lane=TT_EXECUTION_LANE_PLAIN,
        )
