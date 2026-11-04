# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""``tt_adaptive_block_batched``: every decode request of a step blocks.

Without the capability an adaptive model blocks only on a solo decode that owns
its single session; ``test_block_scheduler.py``'s
``test_adaptive_batched_decode_commits_single_tokens`` pins that default.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from tests.test_block_request_validation import (
    BlockModel,
    _config,
    _patch_model_resolution,
)
from tests.test_block_scheduler import (
    CANVAS,
    _adaptive_anchor,
    _request,
    _scheduler,
)
from vllm_tt_plugin.config import is_tt_adaptive_block_batched
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.platform import TTPlatform
from vllm_tt_plugin.scheduler import get_tt_block_step_decisions


def _outputs(tokens_by_req: dict[str, list[int]]) -> ModelRunnerOutput:
    req_ids = list(tokens_by_req)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: idx for idx, req_id in enumerate(req_ids)},
        sampled_token_ids=[list(tokens_by_req[req_id]) for req_id in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _batched_pair(
    *,
    prompt_a: int = 32,
    prompt_b: int = 32,
    max_prompt: int = 0,
    ragged: bool = False,
    max_tokens_a: int = CANVAS * 4,
    ignore_eos: bool = True,
    kv_extent: int = 0,
):
    """Two requests driven through their batched prefill anchors (5 and 6)."""
    scheduler = _scheduler(
        adaptive=True,
        max_num_seqs=2,
        batched=True,
        ragged=ragged,
        max_prompt=max_prompt,
        kv_extent=kv_extent,
    )
    req_a = _request(
        max_tokens_a, request_id="req-a", prompt_length=prompt_a, ignore_eos=ignore_eos
    )
    req_b = _request(CANVAS * 4, request_id="req-b", prompt_length=prompt_b)
    scheduler.add_request(req_a)
    scheduler.add_request(req_b)
    prefill = scheduler.schedule()
    decisions = get_tt_block_step_decisions(prefill)
    assert decisions == {"req-a": False, "req-b": False}  # anchors stay width-1
    scheduler.update_from_output(prefill, _outputs({"req-a": [5], "req-b": [6]}))
    return scheduler, req_a, req_b


def test_batched_decode_step_blocks_every_request():
    scheduler, req_a, req_b = _batched_pair()

    decode = scheduler.schedule()

    assert get_tt_block_step_decisions(decode) == {"req-a": True, "req-b": True}
    for req in (req_a, req_b):
        assert req.num_output_placeholders == CANVAS
    block_a = list(range(100, 100 + CANVAS))
    block_b = list(range(200, 200 + CANVAS))
    outputs = scheduler.update_from_output(
        decode, _outputs({"req-a": block_a, "req-b": block_b})
    )
    committed = {o.request_id: o.new_token_ids for o in outputs[0].outputs}
    assert committed == {"req-a": block_a, "req-b": block_b}
    assert req_a.num_output_placeholders == 0
    assert req_b.num_output_placeholders == 0


def test_batched_solo_decode_blocks_without_owning_a_shared_session():
    """A request that never prefilled solo still blocks when it decodes alone."""
    scheduler, req_a, req_b = _batched_pair()
    scheduler.finish_requests("req-a", RequestStatus.FINISHED_ABORTED)

    decode = scheduler.schedule()

    assert get_tt_block_step_decisions(decode) == {"req-b": True}
    assert req_b.num_output_placeholders == CANVAS


def test_batched_applies_the_prompt_frontier_per_request():
    """A prompt above tt_adaptive_block_max_prompt_tokens stays width-1 in the
    same step that blocks its within-frontier neighbour."""
    scheduler, req_a, req_b = _batched_pair(prompt_a=32, prompt_b=64, max_prompt=48)

    decode = scheduler.schedule()

    assert get_tt_block_step_decisions(decode) == {"req-a": True, "req-b": False}
    assert req_a.num_output_placeholders == CANVAS
    assert req_b.num_output_placeholders == 1
    block_a = list(range(100, 100 + CANVAS))
    outputs = scheduler.update_from_output(
        decode, _outputs({"req-a": block_a, "req-b": [7]})
    )
    committed = {o.request_id: o.new_token_ids for o in outputs[0].outputs}
    assert committed == {"req-a": block_a, "req-b": [7]}


def test_batched_refuses_a_step_mixing_prefill_and_decode():
    """TTScheduler's modes never build the mix; the base scheduler does, and the
    batched contract refuses it rather than reserve one width for both."""
    scheduler = _scheduler(adaptive=True, max_num_seqs=2, batched=True)
    req_a = _request(CANVAS * 4, request_id="req-a")
    scheduler.add_request(req_a)
    _adaptive_anchor(scheduler, req_a)
    scheduler.add_request(_request(CANVAS * 4, request_id="req-b"))

    with pytest.raises(RuntimeError, match=r"mixes prefill and decode.*1 decode of 2"):
        AsyncScheduler.schedule(scheduler)


class _BatchedBlockModel(BlockModel):
    model_capabilities = {
        **BlockModel.model_capabilities,
        "tt_adaptive_block_output": True,
        "tt_adaptive_block_batched": True,
    }

    @staticmethod
    def note_state_slots_moved(_moves):
        pass


class _BatchedWithoutAdaptive(BlockModel):
    model_capabilities = {
        **BlockModel.model_capabilities,
        "tt_adaptive_block_batched": True,
    }


def test_startup_stores_the_batched_capability(monkeypatch):
    config = _config(max_num_seqs=4)
    _patch_model_resolution(monkeypatch, _BatchedBlockModel)

    TTPlatform.check_and_update_config(config)

    assert is_tt_adaptive_block_batched(config) is True


def test_startup_rejects_batched_without_adaptive(monkeypatch):
    config = _config()
    _patch_model_resolution(monkeypatch, _BatchedWithoutAdaptive)

    with pytest.raises(
        ValueError, match="tt_adaptive_block_batched requires tt_adaptive_block_output"
    ):
        TTPlatform.check_and_update_config(config)


def test_runner_commits_a_full_block_for_every_row():
    """The runner half of the contract: a batched block step's ``[N, W]`` output
    appends W tokens to each of the N requests and publishes W ids per row."""
    width = 8
    outputs: list[list[int]] = [[], []]
    runner = SimpleNamespace(
        _output_tokens_per_step=width,
        _is_adaptive_block_output=True,
        _is_adaptive_block_ragged=False,
        _plain_fallback=False,
        _dflash_sampled=False,
        input_batch=SimpleNamespace(
            num_reqs=2,
            req_ids=["a", "b"],
            num_tokens=np.array([3, 5], dtype=np.int32),
            token_ids_cpu=np.zeros((2, 32), dtype=np.int32),
            req_output_token_ids=outputs,
        ),
        model_config=SimpleNamespace(max_model_len=32),
    )
    runner._tt_committed_width = lambda toks: TTModelRunner._tt_committed_width(
        runner, toks
    )
    block = torch.arange(2 * width, dtype=torch.int32).reshape(2, width)

    TTModelRunner._apply_sampled_tokens_to_state(runner, block)
    output = TTModelRunner._build_runner_output(runner, block)

    assert runner.input_batch.num_tokens.tolist() == [3 + width, 5 + width]
    assert outputs == [list(range(width)), list(range(width, 2 * width))]
    assert output.sampled_token_ids == outputs
