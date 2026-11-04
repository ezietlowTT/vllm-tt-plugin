# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The model-facing lane kwargs of ``tt_plain_fallback``.

``prefill_forward`` receives ``request_execution_lanes`` (one lane per row) and
``decode_forward`` receives ``execution_lane`` only for a model that declares
the capability; every other model keeps its call shape.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.sched.output import SchedulerOutput

from tests.test_decode_reload_contract import (
    _submission_input,
    _verifying_controller,
)
from tests.test_lane_input_batch import (
    _add_to_lane,
    _lane_batch,
    _make_req,
    _without_speculation,
)
from tests.test_model_runner import _batch_with_one_request, _fake_runner, _prepare
from vllm_tt_plugin.lane_routing import (
    TT_EXECUTION_LANE_ATTR,
    TT_REQUEST_EXECUTION_LANES_ATTR,
)
from vllm_tt_plugin.lane_scheduler import TTStepPlan
from vllm_tt_plugin.model_runner import TTModelRunner


def _tag_lanes(scheduler_output, lanes, step_lane="prefill"):
    setattr(scheduler_output, TT_EXECUTION_LANE_ATTR, step_lane)
    setattr(scheduler_output, TT_REQUEST_EXECUTION_LANES_ATTR, dict(lanes))
    return scheduler_output


@pytest.mark.parametrize("plain_fallback", [False, True])
def test_front_packed_prefill_rows_carry_lanes_only_with_the_capability(
    monkeypatch, plain_fallback
):
    batch, request = _batch_with_one_request(
        prompt_len=8, output_len=0, num_computed_tokens=0
    )
    runner = _fake_runner(batch, request)
    runner._plain_fallback = plain_fallback
    # Tag every step, as a lane-routing scheduler would: the runner, not the
    # field's presence, decides whether the model sees it.
    monkeypatch.setattr(
        SchedulerOutput,
        TT_EXECUTION_LANE_ATTR,
        "prefill",
        raising=False,
    )
    monkeypatch.setattr(
        SchedulerOutput,
        TT_REQUEST_EXECUTION_LANES_ATTR,
        {"r": "plain"},
        raising=False,
    )

    model_input = _prepare(runner, ("r", 8, 0, 0))

    assert model_input.prompt_lens is not None
    expected = ["plain"] if plain_fallback else None
    assert model_input.request_execution_lanes == expected


@pytest.mark.parametrize("plain_fallback", [False, True])
def test_lane_prefill_rows_carry_lanes_only_with_the_capability(plain_fallback):
    batch = _lane_batch(num_lanes=1, per_lane=1)
    request = _make_req("request", list(range(8)), [], dict(temperature=0.0))
    row = _add_to_lane(batch, request, lane=0)
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {"request": 8}
    scheduler_output.total_num_scheduled_tokens = 8
    _tag_lanes(scheduler_output, {"request": "dflash"})
    plan = TTStepPlan(
        is_decode=False,
        capacity=1,
        scheduled_req_ids=("request",),
        scheduled_rows=(row,),
        input_rows=(row,),
        req_id_to_row={"request": row},
        batch_size_per_dp=(1,),
        prefill_empty_slots=(row,),
    )
    runner = _without_speculation(
        SimpleNamespace(
            max_num_blocks_per_req=8,
            _block_tables_per_layer=lambda _: None,
            check_perform_device_sampling=lambda **_: True,
            _output_tokens_per_step=1,
            _plain_fallback=plain_fallback,
            model_config=SimpleNamespace(is_multimodal_model=False),
            requests={"request": request},
        )
    )

    model_input = batch.build_model_input(runner, scheduler_output, None, plan)

    assert model_input.request_execution_lanes == (
        ["dflash"] if plain_fallback else None
    )


def test_a_prefill_row_without_a_lane_raises():
    from vllm_tt_plugin.lane_routing import prefill_row_execution_lanes

    scheduler_output = _tag_lanes(SchedulerOutput.make_empty(), {"a": "plain"})
    with pytest.raises(ValueError, match=r"Prefill rows \['b'\] have no execution"):
        prefill_row_execution_lanes(scheduler_output, ["a", "b"])


@pytest.mark.parametrize("lanes", [None, ["plain", "dflash"]])
def test_submit_prefill_forwards_lanes_only_when_built(lanes):
    captured: dict = {}

    class FakeModel:
        def prefill_forward(self, **kwargs):
            captured.update(kwargs)
            return object()

    runner = SimpleNamespace(
        kv_caches=object(),
        trace_mode="none",
        request_specific_rope=False,
        supports_stage_outputs=False,
        model=FakeModel(),
        async_decode=SimpleNamespace(note_prefill_submitted=lambda: None),
    )
    model_input = SimpleNamespace(
        input_tokens=torch.zeros((2, 1), dtype=torch.int32),
        block_tables=torch.zeros((2, 1), dtype=torch.int32),
        prompt_lens=[1, 1],
        input_positions=torch.zeros((2,), dtype=torch.int32),
        block_tables_per_layer=None,
        stage_inputs=None,
        multi_modal_kwargs={},
        perform_device_sampling=False,
        prefill_empty_slots=[0, 1],
        request_execution_lanes=lanes,
    )

    TTModelRunner.submit_prefill(runner, model_input, [2])

    if lanes is None:
        assert "request_execution_lanes" not in captured
    else:
        assert captured["request_execution_lanes"] == lanes


@pytest.mark.parametrize("plain_fallback", [False, True])
def test_decode_forward_gets_the_lane_only_with_the_capability(plain_fallback):
    controller, model = _verifying_controller()
    controller.runner._plain_fallback = plain_fallback
    model_input = _submission_input(layout_changed=False)
    model_input.execution_lane = "plain" if plain_fallback else None

    controller.submit_decode(model_input, read_from_device=True)

    (call,) = model.calls
    if plain_fallback:
        assert call["execution_lane"] == "plain"
    else:
        assert "execution_lane" not in call
