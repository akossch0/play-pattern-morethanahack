"""Refine a possession timeline before the final spell corrections."""

from pathlib import Path

import numpy as np

from . import credit_timing, endpoint_runtime, stoppage_kernel
from .duel_consensus import merge_duel_consensus
from .late_sequence import refine
from .duel_inputs import load_models, predict_duel_inputs
from .stoppage_roles import load_roles, predict_roles
from .stoppage_validity import load_repeat, predict_repeat


def refine_branches(reference, tracking, typed_forecasts, tactics, weights_dir):
    """Return (timeline, flank, credit).

    timeline is the reference after the stoppage and duel-consensus stages. flank
    and credit are two alternative retimings of it (endpoint_runtime.apply_flank
    and credit_timing.apply) that late_sequence.refine later merges.

    Reference is the timeline after the observed ball-exit correction.
    Each typed forecast is one (frame probabilities, transition scores) pair.
    """
    weights = Path(weights_dir)
    period = tracking["period"]
    prepared = stoppage_kernel.feature_inputs(tracking)
    role = predict_roles(
        reference,
        tracking,
        typed_forecasts,
        tactics,
        load_roles(weights / "stoppage-roles.pt"),
        prepared=prepared,
    )
    repeat = predict_repeat(
        reference,
        tracking,
        typed_forecasts,
        tactics,
        load_repeat(weights / "stoppage-repeat.pt"),
        prepared=prepared,
    )
    agreed = stoppage_kernel.agree_stoppage_edges(reference, period, role, repeat)
    cleaned = stoppage_kernel.predict(
        agreed,
        tracking,
        typed_forecasts,
        tactics,
        stoppage_kernel.load(weights / "stoppage-kernel.npz"),
        prepared=prepared,
    )
    asymmetric, alternate = predict_duel_inputs(
        cleaned, tracking, typed_forecasts, load_models(weights / "duel-control.npz")
    )
    timeline = merge_duel_consensus(cleaned, period, asymmetric, alternate)
    flank_options = endpoint_runtime.candidate_bank(timeline, tracking, typed_forecasts)
    flank = endpoint_runtime.apply_flank(
        timeline, flank_options, endpoint_runtime.load_model(weights / "endpoint.npz")
    )
    probabilities, edges = (
        np.stack(views) for views in zip(*typed_forecasts, strict=True)
    )
    credit = credit_timing.apply(
        timeline,
        period,
        tracking["ball_observed"],
        probabilities,
        edges,
        credit_timing.load_weights(weights / "credit-timing.json"),
    )
    return timeline, flank, credit


def refine_from_observed_exit(match, reference, forecasts, spans, tactics, weights_dir):
    """Run the recovered sequence from an input timeline adjusted for ball exits."""
    # members[0] is the first of each pair's three class streams (tcn_inference stacks
    # the typed model first); pairs are taken in sorted-key order, as in refine().
    typed = [(members[0], edges) for _, (members, edges) in sorted(forecasts.items())]
    initial, flank, credit = refine_branches(reference, match, typed, tactics, weights_dir)
    return refine(match, initial, flank, credit, forecasts, spans, tactics, weights_dir)
