"""Raw tracking to a possession timeline: frame models, draft decoding and spell corrections."""

import json
from pathlib import Path

import numpy as np

from .decoding import decode
from .early_sequence import refine_from_observed_exit
from .gru_inference import predict_gru
from .prepare_ball_out import prepare_ball_out
from .span_inference import predict_spans
from .tcn_inference import predict_pair_streams

TEAM_SWITCH_COST = 8.0
DEAD_SWITCH_COST = 6.0
TCN_WEIGHT = 0.5
# The spell corrections were fitted with four forecast contexts that share one excluded match.
CONTEXT_MATCH = "dfl_J03WMX"


def draft_timeline(members, tactics, period):
    """Decode the equal blend of the mean TCN forecast and the tactics GRU."""
    blended = TCN_WEIGHT * members.mean(axis=0) + (1 - TCN_WEIGHT) * tactics
    return decode(np.log(blended.clip(1e-7)), period, TEAM_SWITCH_COST, DEAD_SWITCH_COST)


def load_exit_delays(path):
    delays = json.loads(Path(path).read_text())
    values = delays["observed"], delays["crossing"]
    if not all(isinstance(v, int) and 0 <= v <= 10 for v in values):
        raise ValueError("Invalid ball-exit delays")
    return values


def predict_tracking(graded_match, weights, *, context_match=CONTEXT_MATCH, draft_pairs=None,
                     gru_members=None, device="cpu"):
    """Return one H/A/D code per graded frame.

    The four checkpoint pairs that exclude ``context_match`` supply the forecast contexts
    for the spell corrections. ``draft_pairs`` (default: all ten pairs) and ``gru_members``
    (default: all five) supply the forecasts that the draft timeline is decoded from.
    """
    weights = Path(weights)
    period = np.asarray(graded_match["period"])
    available = sorted(
        tuple(path.relative_to(weights / "typed").parts[:2])
        for path in (weights / "typed").glob("*/*/model.pt")
    )
    contexts = [pair for pair in available if context_match in pair]
    if len(contexts) != 4:
        raise ValueError("Expected four checkpoint pairs for the forecast contexts")
    draft_pairs = available if draft_pairs is None else sorted(tuple(sorted(p)) for p in draft_pairs)
    streams = predict_pair_streams(
        graded_match, weights, pairs=set(contexts) | set(draft_pairs), device=device
    )
    tactics = predict_gru(graded_match, weights, device, members=gru_members)
    # streams[pair] is (class forecasts of the three TCN families, typed transition streams);
    # average the class forecasts over the draft pairs, family by family.
    members = np.mean([streams[pair][0] for pair in draft_pairs], axis=0)
    draft = draft_timeline(members, tactics, period)
    forecasts = {pair: streams[pair] for pair in contexts}
    # The typed family is the first of the three (tcn_inference stacks it first).
    typed = [(forecasts[pair][0][0], forecasts[pair][1]) for pair in contexts]
    observed, crossing = load_exit_delays(weights / "exit-delays.json")
    # The draft is passed twice: as the ownership timeline and as the exit reference.
    reference = prepare_ball_out(graded_match, draft, draft, typed, tactics, observed, crossing)
    spans = predict_spans(graded_match, weights, pairs=contexts)
    spans = {tuple(sorted(pair)): value for pair, value in spans.items()}
    return refine_from_observed_exit(graded_match, reference, forecasts, spans, tactics, weights)
