# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""``tt_adaptive_block_ragged``: each row of a block step commits its own width.

Contract: a block step returns a rectangular int32 ``[num_reqs, W]`` tensor;
row ``i`` holds ``1 <= n_i <= W`` real token ids followed by
``TT_RAGGED_BLOCK_PAD_TOKEN_ID``. The runner strips the padding and commits
``n_i`` ids; the scheduler accepts ``1..W`` ids per request and consumes the
whole ``W`` reservation.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.request import RequestStatus

from tests.test_adaptive_block_batched import (
    _batched_pair,
    _BatchedBlockModel,
    _outputs,
)
from tests.test_block_request_validation import (
    BlockModel,
    _config,
    _patch_model_resolution,
)
from tests.test_block_scheduler import (
    BLOCK_SIZE,
    CANVAS,
    _adaptive_anchor,
    _request,
    _runner_output,
    _scheduler,
)
from vllm_tt_plugin.config import (
    TT_RAGGED_BLOCK_PAD_TOKEN_ID,
    is_tt_adaptive_block_ragged,
)
from vllm_tt_plugin.model_runner import (
    TTModelRunner,
    _committed_row_widths,
    _ragged_row_widths,
)
from vllm_tt_plugin.platform import TTPlatform
from vllm_tt_plugin.scheduler import get_tt_block_step_decisions

PAD = TT_RAGGED_BLOCK_PAD_TOKEN_ID
W = 8
PROMPT = 32


def _row(ids, width: int = W) -> list[int]:
    ids = list(ids)
    assert len(ids) <= width
    return [*ids, *([PAD] * (width - len(ids)))]


def _bind(runner: SimpleNamespace, *, ragged: bool, width: int = W):
    runner._output_tokens_per_step = width
    runner._is_adaptive_block_output = True
    runner._is_adaptive_block_ragged = ragged
    runner._tt_committed_width = lambda toks: TTModelRunner._tt_committed_width(
        runner, toks
    )
    return runner


def _runner(
    *,
    ragged: bool,
    num_tokens=(0, 0),
    max_model_len: int = 32,
    req_ids=("a", "b"),
    width: int = W,
):
    outputs: list[list[int]] = [[] for _ in req_ids]
    runner = _bind(
        SimpleNamespace(
            input_batch=SimpleNamespace(
                num_reqs=len(req_ids),
                req_ids=list(req_ids),
                num_tokens=np.array(num_tokens, dtype=np.int32),
                token_ids_cpu=np.zeros((len(req_ids), max_model_len), dtype=np.int32),
                req_output_token_ids=outputs,
            ),
            model_config=SimpleNamespace(max_model_len=max_model_len),
        ),
        ragged=ragged,
        width=width,
    )
    return runner, outputs


# Row parse


def test_ragged_row_widths_count_the_real_ids():
    rows = np.array([_row([1, 2, 3]), _row([4]), _row(range(10, 18))], dtype=np.int32)
    assert _ragged_row_widths(rows).tolist() == [3, 1, W]
    assert _ragged_row_widths(np.zeros((0, W), dtype=np.int32)).tolist() == []


def test_ragged_row_widths_reject_a_row_with_no_real_id():
    rows = np.array([_row([1, 2]), _row([])], dtype=np.int32)
    with pytest.raises(ValueError, match=r"commits no token.*row widths \[2, 0\]"):
        _ragged_row_widths(rows)


def test_ragged_row_widths_reject_padding_before_a_real_id():
    rows = np.array([_row([1, 2, 3]), [1, PAD, 3, PAD, PAD, PAD, PAD, PAD]])
    with pytest.raises(ValueError, match=r"before a real token id in row\(s\) \[1\]"):
        _ragged_row_widths(rows)


def test_committed_widths_parse_only_ragged_block_rows():
    rows = np.array([_row([1]), _row([2, 3])], dtype=np.int32)
    assert _committed_row_widths(rows, W, ragged=False).tolist() == [W, W]
    assert _committed_row_widths(rows, W, ragged=True).tolist() == [1, 2]
    anchors = np.array([[5], [6]], dtype=np.int32)
    assert _committed_row_widths(anchors, 1, ragged=True).tolist() == [1, 1]


# Runner commit paths


def test_ragged_rows_commit_only_their_real_ids():
    runner, outputs = _runner(ragged=True)
    block = torch.tensor([_row([11, 12, 13, 14, 15]), _row([21])], dtype=torch.int32)

    TTModelRunner._apply_sampled_tokens_to_state(runner, block)
    output = TTModelRunner._build_runner_output(runner, block)

    assert runner.input_batch.num_tokens.tolist() == [5, 1]
    assert runner.input_batch.token_ids_cpu[0, :5].tolist() == [11, 12, 13, 14, 15]
    assert runner.input_batch.token_ids_cpu[1, :1].tolist() == [21]
    assert not (runner.input_batch.token_ids_cpu < 0).any()
    assert outputs == [[11, 12, 13, 14, 15], [21]]
    assert output.sampled_token_ids == [[11, 12, 13, 14, 15], [21]]


def test_ragged_width_one_rows_are_plain_anchors():
    runner, outputs = _runner(ragged=True)
    anchors = torch.tensor([[5], [6]], dtype=torch.int32)

    TTModelRunner._apply_sampled_tokens_to_state(runner, anchors)
    output = TTModelRunner._build_runner_output(runner, anchors)

    assert runner.input_batch.num_tokens.tolist() == [1, 1]
    assert output.sampled_token_ids == [[5], [6]]


def test_ragged_off_keeps_fixed_width_rows_verbatim():
    """Without the capability nothing strips: the batched contract fills every
    row to W, so a padded row commits W ids exactly as before."""
    runner, outputs = _runner(ragged=False)
    block = torch.tensor([_row([11, 12, 13, 14, 15]), _row([21])], dtype=torch.int32)

    TTModelRunner._apply_sampled_tokens_to_state(runner, block)
    output = TTModelRunner._build_runner_output(runner, block)

    assert runner.input_batch.num_tokens.tolist() == [W, W]
    assert output.sampled_token_ids == [_row([11, 12, 13, 14, 15]), _row([21])]


def test_ragged_rows_keep_the_shape_check():
    runner, _ = _runner(ragged=True)
    with pytest.raises(ValueError, match="violates output_tokens_per_step"):
        TTModelRunner._build_runner_output(
            runner, torch.zeros((2, W - 1), dtype=torch.int32)
        )


def test_ragged_row_is_clipped_at_max_model_len():
    runner, outputs = _runner(ragged=True, num_tokens=(30, 0), max_model_len=32)
    block = torch.tensor(
        [_row([11, 12, 13, 14, 15]), _row([21, 22, 23])], dtype=torch.int32
    )

    TTModelRunner._apply_sampled_tokens_to_state(runner, block)

    assert runner.input_batch.num_tokens.tolist() == [32, 3]
    assert outputs == [[11, 12], [21, 22, 23]]


@pytest.mark.parametrize("in_batch", [("a", "b"), ("a",)])
def test_ragged_captured_rows_commit_only_their_real_ids(in_batch):
    """The deferred-apply path (explicit req_ids) strips the same way, including
    for a request already gone from the persistent batch."""
    states = {req_id: SimpleNamespace(output_token_ids=[]) for req_id in "ab"}
    runner = _bind(
        SimpleNamespace(
            requests=states,
            input_batch=SimpleNamespace(
                req_id_to_index={r: i for i, r in enumerate("ab") if r in in_batch},
                num_tokens=np.zeros(2, dtype=np.int32),
                token_ids_cpu=np.zeros((2, 32), dtype=np.int32),
            ),
            model_config=SimpleNamespace(max_model_len=32),
        ),
        ragged=True,
    )
    block = torch.tensor([_row([11, 12, 13]), _row([21, 22])], dtype=torch.int32)

    TTModelRunner._apply_sampled_tokens_to_state(runner, block, req_ids=["a", "b"])

    assert states["a"].output_token_ids == [11, 12, 13]
    assert states["b"].output_token_ids == [21, 22]
    assert runner.input_batch.num_tokens.tolist() == [3, 2 if "b" in in_batch else 0]


# Scheduler


def test_ragged_rows_commit_their_own_widths_and_keep_the_w_reservation():
    scheduler, req_a, req_b = _batched_pair(ragged=True)

    decode = scheduler.schedule()
    for req in (req_a, req_b):
        assert get_tt_block_step_decisions(decode)[req.request_id] is True
        assert req.num_output_placeholders == CANVAS
    block_a, block_b = list(range(100, 105)), [200]
    outputs = scheduler.update_from_output(
        decode, _outputs({"req-a": block_a, "req-b": block_b})
    )
    committed = {o.request_id: o.new_token_ids for o in outputs[0].outputs}
    assert committed == {"req-a": block_a, "req-b": block_b}
    for req, block in ((req_a, block_a), (req_b, block_b)):
        assert req.num_output_placeholders == 0  # W consumed, not n
        assert req.num_tokens == PROMPT + 1 + len(block)

    decode = scheduler.schedule()
    assert decode.num_scheduled_tokens == {"req-a": 5, "req-b": 1}
    for req in (req_a, req_b):
        assert req.num_computed_tokens == req.num_tokens
        assert req.num_output_placeholders == CANVAS


def test_ragged_solo_decode_commits_a_single_token_row():
    scheduler = _scheduler(adaptive=True, max_num_seqs=2, batched=True, ragged=True)
    request = _request(CANVAS * 2)
    scheduler.add_request(request)
    _adaptive_anchor(scheduler, request)

    decode = scheduler.schedule()
    outputs = scheduler.update_from_output(decode, _runner_output(decode, [9]))
    assert outputs[0].outputs[0].new_token_ids == [9]

    decode = scheduler.schedule()
    assert decode.num_scheduled_tokens == {"req-0": 1}
    assert request.num_computed_tokens == request.num_tokens


@pytest.mark.parametrize(
    ("row", "match"),
    [
        pytest.param([], r"0 not in 1\.\.16", id="empty-row"),
        pytest.param(list(range(CANVAS + 1)), r"17 not in 1\.\.16", id="over-wide"),
        pytest.param([7, -1, -1], "with padding", id="pad-reached-scheduler"),
    ],
)
def test_ragged_width_guards(row, match):
    scheduler, req_a, _ = _batched_pair(ragged=True)
    decode = scheduler.schedule()
    # update_from_output binds the step's decisions before committing.
    scheduler._committing_block_step_decisions = get_tt_block_step_decisions(decode)
    with pytest.raises(ValueError, match=match):
        # Direct call: the base update_from_output never forwards an empty row.
        scheduler._update_request_with_output(req_a, list(row))


def test_batched_without_ragged_keeps_the_fixed_width_check():
    scheduler, _, _ = _batched_pair(ragged=False)
    decode = scheduler.schedule()
    with pytest.raises(ValueError, match=r"5 != 16"):
        scheduler.update_from_output(
            decode, _outputs({"req-a": list(range(5)), "req-b": list(range(CANVAS))})
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"adaptive": False}, id="no-adaptive"),
        pytest.param({"adaptive": True, "batched": False}, id="no-batched"),
    ],
)
def test_scheduler_requires_the_batched_contract(kwargs):
    with pytest.raises(ValueError, match="requires tt_adaptive_block_batched"):
        _scheduler(max_num_seqs=2, ragged=True, **kwargs)


@pytest.mark.parametrize(
    ("max_tokens", "ignore_eos", "row", "kept", "status"),
    [
        # anchor + 3 of the 5-token row reach max_tokens=4
        (4, True, [10, 11, 12, 13, 14], [10, 11, 12], "FINISHED_LENGTH_CAPPED"),
        (CANVAS, False, [10, 2, 12], [10, 2], "FINISHED_STOPPED"),
    ],
    ids=["max_tokens", "eos"],
)
def test_ragged_row_stops_trim_and_consume_the_reservation(
    max_tokens, ignore_eos, row, kept, status
):
    scheduler, req_a, req_b = _batched_pair(
        ragged=True, max_tokens_a=max_tokens, ignore_eos=ignore_eos
    )
    decode = scheduler.schedule()
    outputs = scheduler.update_from_output(
        decode, _outputs({"req-a": row, "req-b": [200, 201]})
    )
    committed = {o.request_id: o.new_token_ids for o in outputs[0].outputs}
    assert committed == {"req-a": kept, "req-b": [200, 201]}
    assert req_a.status == RequestStatus[status]
    assert req_a.num_output_placeholders == 0
    assert req_b.num_output_placeholders == 0

    decode = scheduler.schedule()
    assert decode.num_scheduled_tokens == {"req-b": 2}
    assert req_b.num_computed_tokens == req_b.num_tokens


def test_ragged_kv_extent_covers_the_next_block_after_a_short_row():
    """After a short row the next schedule allocates from the tokens the
    request actually has: num_tokens plus the declared block KV extent."""
    extent = CANVAS * 2 + 8  # wider than the default 2 W lookahead
    scheduler, req_a, req_b = _batched_pair(
        ragged=True, prompt_a=80, prompt_b=80, kv_extent=extent
    )
    assert scheduler.num_lookahead_tokens == extent
    decode = scheduler.schedule()
    scheduler.update_from_output(
        decode, _outputs({"req-a": [100, 101], "req-b": [200]})
    )

    decode = scheduler.schedule()
    assert decode.num_scheduled_tokens == {"req-a": 2, "req-b": 1}
    for req in (req_a, req_b):
        blocks = scheduler.kv_cache_manager.get_block_ids(req.request_id)[0]
        need = req.num_tokens + extent
        assert len(blocks) == -(-need // BLOCK_SIZE)


def test_runner_output_drives_the_scheduler_over_several_ragged_steps():
    """End to end: the runner strips each padded row, the scheduler commits it,
    and over several steps num_computed_tokens never passes num_tokens while the
    runner's own token count stays equal to the scheduler's."""
    scheduler, req_a, req_b = _batched_pair(ragged=True)
    requests = {"req-a": req_a, "req-b": req_b}
    runner, outputs = _runner(
        ragged=True,
        num_tokens=(PROMPT + 1, PROMPT + 1),
        max_model_len=256,
        req_ids=("req-a", "req-b"),
        width=CANVAS,
    )
    steps = [
        {"req-a": list(range(100, 105)), "req-b": [200]},
        {"req-a": list(range(300, 300 + CANVAS)), "req-b": [400, 401, 402]},
        {"req-a": [500], "req-b": list(range(600, 600 + CANVAS))},
    ]
    committed = {"req-a": [], "req-b": []}
    for rows in steps:
        decode = scheduler.schedule()
        for req_id, req in requests.items():
            assert get_tt_block_step_decisions(decode)[req_id] is True
            assert req.num_computed_tokens <= req.num_tokens
        block = torch.tensor(
            [_row(rows[r], CANVAS) for r in runner.input_batch.req_ids],
            dtype=torch.int32,
        )
        TTModelRunner._apply_sampled_tokens_to_state(runner, block)
        output = TTModelRunner._build_runner_output(runner, block)
        scheduler.update_from_output(decode, output)
        for idx, (req_id, req) in enumerate(requests.items()):
            committed[req_id] += rows[req_id]
            assert req.num_tokens == int(runner.input_batch.num_tokens[idx])
            assert req.num_computed_tokens <= req.num_tokens
            assert list(req.output_token_ids)[1:] == committed[req_id]
    assert outputs == [committed["req-a"], committed["req-b"]]

    decode = scheduler.schedule()
    assert decode.num_scheduled_tokens == {"req-a": 1, "req-b": CANVAS}
    for req in requests.values():
        assert req.num_computed_tokens == req.num_tokens


class _RaggedBlockModel(_BatchedBlockModel):
    model_capabilities = {
        **_BatchedBlockModel.model_capabilities,
        "tt_adaptive_block_ragged": True,
    }


class _RaggedWithoutBatched(BlockModel):
    model_capabilities = {
        **BlockModel.model_capabilities,
        "tt_adaptive_block_output": True,
        "tt_adaptive_block_ragged": True,
    }

    @staticmethod
    def note_state_slots_moved(_moves):
        pass


def test_startup_stores_the_ragged_capability(monkeypatch):
    config = _config(max_num_seqs=4)
    _patch_model_resolution(monkeypatch, _RaggedBlockModel)

    TTPlatform.check_and_update_config(config)

    assert is_tt_adaptive_block_ragged(config) is True


def test_startup_rejects_ragged_without_batched(monkeypatch):
    config = _config(max_num_seqs=4)
    _patch_model_resolution(monkeypatch, _RaggedWithoutBatched)

    with pytest.raises(
        ValueError, match="tt_adaptive_block_ragged requires tt_adaptive_block_batched"
    ):
        TTPlatform.check_and_update_config(config)
