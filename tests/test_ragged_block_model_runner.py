# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_tt_plugin.config import TT_RAGGED_BLOCK_PAD_TOKEN_ID
from vllm_tt_plugin.model_runner import (
    TTModelRunner,
    _committed_row_widths,
    _ragged_row_widths,
)

PAD = TT_RAGGED_BLOCK_PAD_TOKEN_ID
W = 8


def _row(ids, width: int = W) -> list[int]:
    ids = list(ids)
    return [*ids, *([PAD] * (width - len(ids)))]


def _bind(runner: SimpleNamespace, *, ragged: bool) -> SimpleNamespace:
    runner._is_adaptive_block_output = True
    runner._is_adaptive_block_ragged = ragged
    runner._tt_committed_width = lambda toks: TTModelRunner._tt_committed_width(
        runner, toks
    )
    return runner


def _runner(*, ragged: bool, num_tokens=(0, 0), max_model_len: int = 32):
    outputs: list[list[int]] = [[], []]
    return (
        _bind(
            SimpleNamespace(
                _output_tokens_per_step=W,
                input_batch=SimpleNamespace(
                    num_reqs=2,
                    req_ids=["a", "b"],
                    num_tokens=np.array(num_tokens, dtype=np.int32),
                    token_ids_cpu=np.zeros((2, max_model_len), dtype=np.int32),
                    req_output_token_ids=outputs,
                ),
                model_config=SimpleNamespace(max_model_len=max_model_len),
            ),
            ragged=ragged,
        ),
        outputs,
    )


def test_ragged_row_widths_and_validation():
    rows = np.array([_row([1, 2, 3]), _row([4]), _row(range(10, 18))])
    assert _ragged_row_widths(rows).tolist() == [3, 1, W]
    assert _ragged_row_widths(np.zeros((0, W), dtype=np.int32)).tolist() == []
    with pytest.raises(ValueError, match="commits no token"):
        _ragged_row_widths(np.array([_row([1]), _row([])]))
    with pytest.raises(ValueError, match="before a real token"):
        _ragged_row_widths(np.array([_row([1]), [1, PAD, 3, PAD, PAD, PAD, PAD, PAD]]))


def test_ragged_rows_commit_and_publish_only_real_ids():
    runner, outputs = _runner(ragged=True)
    block = torch.tensor([_row([11, 12, 13]), _row([21])], dtype=torch.int32)
    TTModelRunner._apply_sampled_tokens_to_state(runner, block)
    output = TTModelRunner._build_runner_output(runner, block)
    assert runner.input_batch.num_tokens.tolist() == [3, 1]
    assert outputs == [[11, 12, 13], [21]]
    assert output.sampled_token_ids == [[11, 12, 13], [21]]
    assert not (runner.input_batch.token_ids_cpu < 0).any()


def test_full_width_and_width_one_preserve_existing_contracts():
    runner, outputs = _runner(ragged=True)
    full = torch.arange(2 * W, dtype=torch.int32).reshape(2, W)
    TTModelRunner._apply_sampled_tokens_to_state(runner, full)
    assert outputs == [list(range(W)), list(range(W, 2 * W))]

    runner, outputs = _runner(ragged=True)
    anchors = torch.tensor([[5], [6]], dtype=torch.int32)
    TTModelRunner._apply_sampled_tokens_to_state(runner, anchors)
    assert TTModelRunner._build_runner_output(runner, anchors).sampled_token_ids == [
        [5],
        [6],
    ]
    assert outputs == [[5], [6]]


def test_fixed_width_mode_is_unchanged():
    runner, outputs = _runner(ragged=False)
    block = torch.tensor([_row([11, 12]), _row([21])], dtype=torch.int32)
    TTModelRunner._apply_sampled_tokens_to_state(runner, block)
    assert runner.input_batch.num_tokens.tolist() == [W, W]
    assert outputs == [_row([11, 12]), _row([21])]


def test_ragged_rows_clip_at_max_model_len():
    runner, outputs = _runner(ragged=True, num_tokens=(30, 0), max_model_len=32)
    block = torch.tensor([_row([11, 12, 13]), _row([21, 22])], dtype=torch.int32)
    TTModelRunner._apply_sampled_tokens_to_state(runner, block)
    assert runner.input_batch.num_tokens.tolist() == [32, 2]
    assert outputs == [[11, 12], [21, 22]]


def test_deferred_rows_use_the_same_logical_widths():
    states = {
        "a": SimpleNamespace(output_token_ids=[]),
        "b": SimpleNamespace(output_token_ids=[]),
    }
    runner = _bind(
        SimpleNamespace(
            _output_tokens_per_step=W,
            requests=states,
            input_batch=SimpleNamespace(
                req_id_to_index={"a": 0},
                num_tokens=np.zeros(2, dtype=np.int32),
                token_ids_cpu=np.zeros((2, 32), dtype=np.int32),
            ),
            model_config=SimpleNamespace(max_model_len=32),
        ),
        ragged=True,
    )
    block = torch.tensor([_row([11, 12, 13]), _row([21, 22])], dtype=torch.int32)
    TTModelRunner._apply_sampled_tokens_to_state(runner, block, req_ids=["a", "b"])
    assert runner.input_batch.num_tokens.tolist() == [3, 0]
    assert states["a"].output_token_ids == [11, 12, 13]
    assert states["b"].output_token_ids == [21, 22]


def test_shape_and_default_flag_guards():
    runner, outputs = _runner(ragged=True)
    with pytest.raises(ValueError, match="violates output_tokens_per_step"):
        TTModelRunner._build_runner_output(
            runner, torch.zeros((2, W - 1), dtype=torch.int32)
        )

    del runner._is_adaptive_block_ragged
    block = torch.tensor([_row([1]), _row([2, 3])], dtype=torch.int32)
    TTModelRunner._apply_sampled_tokens_to_state(runner, block)
    assert outputs == [_row([1]), _row([2, 3])]
    rows = block.numpy()
    assert _committed_row_widths(rows, W, ragged=False).tolist() == [W, W]
    assert _committed_row_widths(rows, W, ragged=True).tolist() == [1, 2]
