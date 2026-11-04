# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Execution lanes for adaptive block-output models (``tt_plain_fallback``).

A block-output model owns its sampler on block steps, so it cannot honour
request controls that need host logits (logprobs, penalties, grammar, ``n``,
temperature, ...). A model declaring ``tt_plain_fallback`` serves those requests
with ordinary one-token decode instead. Each request is classified once, at
admission, onto one of two lanes and keeps it for life:

- ``dflash``: greedy with no control the model-owned block sampler cannot
  reproduce (with ``tt_dflash_sampled``, also temperature > 0 with top-k,
  top-p, seed and a non-negative presence penalty); decodes in blocks;
- ``plain``: anything else; decodes one host- or device-sampled token per step
  with every control intact.

``TTScheduler`` keeps each decode step homogeneous (one lane per step,
alternating when both have work) and tags every step with its lane.
"""

from typing import TYPE_CHECKING, Any

from vllm_tt_plugin.config import has_tt_dflash_sampled, has_tt_plain_fallback
from vllm_tt_plugin.logger import init_tt_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = init_tt_logger(__name__)

TT_EXECUTION_LANE_DFLASH = "dflash"
TT_EXECUTION_LANE_PLAIN = "plain"
# The step lane of a prefill step, which may hold requests of both lanes.
TT_EXECUTION_LANE_PREFILL = "prefill"

# TT fields on SchedulerOutput (registered in TT_SCHEDULER_OUTPUT_FIELDS).
TT_EXECUTION_LANE_ATTR = "_tt_execution_lane"
TT_REQUEST_EXECUTION_LANES_ATTR = "_tt_request_execution_lanes"


def classify_tt_execution_lane(
    params: Any, *, sampled_dflash: bool = False
) -> tuple[str, str]:
    """Return ``(lane, reason)`` for one request's sampling params.

    ``dflash`` needs a one-choice greedy request whose every response control
    is neutral; any control that needs host logits, per-token probabilities,
    grammar masking or history penalties routes to ``plain`` with that control
    intact. With ``sampled_dflash`` (``tt_dflash_sampled``) a temperature > 0
    request with a non-negative presence penalty is also ``dflash``: the model
    samples it, and applies the presence penalty, itself. The reason lists the
    offending fields (operator telemetry only).
    """
    incompatible: list[str] = []
    sampled = sampled_dflash and params.temperature > 0.0
    if params.n != 1:
        incompatible.append("n")
    if params.temperature != 0.0 and not sampled:
        incompatible.append("temperature")
    if sampled and params.presence_penalty < 0.0:
        incompatible.append("presence_penalty")
    for field, neutral in (
        ("presence_penalty", 0.0),
        ("frequency_penalty", 0.0),
        ("repetition_penalty", 1.0),
        ("min_p", 0.0),
        ("min_tokens", 0),
    ):
        if field == "presence_penalty" and sampled:
            continue  # model-owned on the sampled DFlash path (sign checked above)
        if getattr(params, field) != neutral:
            incompatible.append(field)
    for field in (
        "logprobs",
        "logprob_token_ids",
        "structured_outputs",
        "logit_bias",
        "allowed_token_ids",
    ):
        if getattr(params, field) is not None:
            incompatible.append(field)
    if params.flat_logprobs:
        incompatible.append("flat_logprobs")
    if params.bad_words:
        incompatible.append("bad_words")
    if incompatible:
        return TT_EXECUTION_LANE_PLAIN, ",".join(incompatible)
    if sampled:
        return TT_EXECUTION_LANE_DFLASH, "eligible_sampled"
    return TT_EXECUTION_LANE_DFLASH, "eligible_greedy"


def block_output_request_lane(vllm_config: "VllmConfig | None", params) -> str | None:
    """The lane a block-output request is admitted to, or ``None`` when the
    model declares no ``tt_plain_fallback`` (every request is a block request).
    """
    if vllm_config is None or not has_tt_plain_fallback(vllm_config):
        return None
    lane, _ = classify_tt_execution_lane(
        params, sampled_dflash=has_tt_dflash_sampled(vllm_config)
    )
    return lane


def keeps_dflash_presence_penalty(vllm_config: "VllmConfig", params) -> bool:
    """Whether a request's presence penalty is model-owned: a sampled request
    on a ``tt_dflash_sampled`` model's dflash lane, whose rejection sampler
    applies it over the model's own output history."""
    return (
        params.temperature > 0.0
        and params.presence_penalty > 0.0
        and block_output_request_lane(vllm_config, params) == TT_EXECUTION_LANE_DFLASH
    )


def get_tt_execution_lane(scheduler_output: "SchedulerOutput") -> str | None:
    """The step's lane: ``prefill``, a decode lane, or ``None`` when the
    scheduler routes no lanes (no ``tt_plain_fallback``)."""
    return getattr(scheduler_output, TT_EXECUTION_LANE_ATTR, None)


def get_tt_request_execution_lanes(
    scheduler_output: "SchedulerOutput",
) -> dict[str, str]:
    """The admission-pinned lane of every request scheduled in this step."""
    return dict(getattr(scheduler_output, TT_REQUEST_EXECUTION_LANES_ATTR, {}))


def prefill_row_execution_lanes(
    scheduler_output: "SchedulerOutput", row_req_ids
) -> list[str]:
    """The lane of each prefill row, ``row_req_ids`` in forward-row order.

    Call it only for a ``tt_plain_fallback`` model, whose scheduler tags every
    scheduled request; a row without a lane raises.
    """
    lanes = get_tt_request_execution_lanes(scheduler_output)
    missing = [req_id for req_id in row_req_ids if req_id not in lanes]
    if missing:
        raise ValueError(
            f"Prefill rows {missing} have no execution lane in the step's "
            f"lane map {lanes}"
        )
    return [lanes[req_id] for req_id in row_req_ids]


class TTLaneRouter:
    """Per-scheduler lane bookkeeping: admission, lookup, and decode alternation.

    Disabled (no ``tt_plain_fallback``), every request is a ``dflash`` request
    and ``next_decode_lane`` returns ``None``, so the scheduler neither filters
    decode steps nor tags them.
    """

    def __init__(self, enabled: bool, *, sampled_dflash: bool = False):
        self.enabled = enabled
        self.sampled_dflash = sampled_dflash
        self._lanes: dict[str, str] = {}
        self._last_decode_lane: str | None = None

    def admit(self, request: "Request") -> str:
        """Classify a new request onto its lane for life and return the lane."""
        if not self.enabled:
            return TT_EXECUTION_LANE_DFLASH
        lane, reason = classify_tt_execution_lane(
            request.sampling_params, sampled_dflash=self.sampled_dflash
        )
        self._lanes[request.request_id] = lane
        logger.info(
            "TT request %s routed to the %s lane (%s)",
            request.request_id,
            lane,
            reason,
        )
        return lane

    def lane(self, req_id: str) -> str:
        return self._lanes.get(req_id, TT_EXECUTION_LANE_DFLASH)

    def request_lanes(self, req_ids) -> dict[str, str]:
        return {req_id: self.lane(req_id) for req_id in req_ids}

    def forget(self, req_ids) -> None:
        for req_id in req_ids:
            self._lanes.pop(req_id, None)

    def next_decode_lane(self, running: list["Request"]) -> str | None:
        """The lane the next decode step serves, alternating when both lanes
        have running decode requests; ``None`` when routing is disabled."""
        if not self.enabled:
            return None
        lanes = {
            self.lane(request.request_id)
            for request in running
            if not request.is_prefill_chunk
        }
        if len(lanes) <= 1:
            return next(iter(lanes), TT_EXECUTION_LANE_DFLASH)
        if self._last_decode_lane == TT_EXECUTION_LANE_DFLASH:
            return TT_EXECUTION_LANE_PLAIN
        return TT_EXECUTION_LANE_DFLASH

    def record_decode(self, lane: str, scheduler_output: "SchedulerOutput") -> None:
        """Advance the alternation once a decode step of ``lane`` ran."""
        if scheduler_output.total_num_scheduled_tokens:
            self._last_decode_lane = lane
