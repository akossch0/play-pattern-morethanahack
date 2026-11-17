"""Refine a decoded timeline from the two upstream boundary estimators."""

import json
from pathlib import Path

import numpy as np

from . import carrier, interruption, matched_context, motion_ranker, native_timing, rolling_ball, stoppage
from .features import extract_features, geometry
from .observed_exit import apply_observed_exit
from .short_return import apply_short_return, merge_timing_branches


def _cuts(labels, period):
    changes = (labels[1:] != labels[:-1]) | (period[1:] != period[:-1])
    return np.r_[0, np.flatnonzero(changes) + 1, len(labels)]


def refine(match, initial, flank_timing, credit_timing, forecasts, spans, tactics, weights_dir):
    """Run the final sequence of corrections on an aligned upstream timeline.

    Forecasts maps held-pair names to (three class streams, six transition streams).
    Spans maps the same names to (interval proposals, frame probabilities).
    """
    if len(forecasts) != 4 or forecasts.keys() != spans.keys():
        raise ValueError("The historical sequence needs four aligned forecast and span contexts")
    weights = Path(weights_dir)
    period = match["period"]
    tracking = extract_features(match)
    pitch = np.array([match["pitch_length_m"], match["pitch_width_m"]], dtype=np.float32)
    ball = np.where(match["ball_observed"][:, None], match["ball_xy"] * pitch, np.nan)
    teams = [np.where(match[f"{side}_observed"][:, :, None], match[f"{side}_xy"] * pitch, np.nan)
             for side in ("home", "away")]
    pairs = sorted(forecasts)
    banks = np.stack([forecasts[pair][0] for pair in pairs], axis=1)
    edges = np.stack([forecasts[pair][1] for pair in pairs]).mean(axis=0)
    members = banks.mean(axis=1)
    current = merge_timing_branches(initial, period, flank_timing, credit_timing)
    current = apply_short_return(current, match)
    current = rolling_ball.apply(current, period, banks, tactics, _cuts)
    prefix = motion_ranker.motion_prefix(*teams, period)
    rows, design, hidden = motion_ranker.candidates(current, period, tracking, members, edges, prefix, _cuts)
    current = motion_ranker.apply(current, rows, design, hidden, motion_ranker.load_models(weights / "motion-ranker.pt"))
    # banks is (3 class streams, 4 held pairs, frames, 3 classes); merge the first two axes into
    # the 12 streams apply_observed_exit expects, four held pairs per class stream.
    current = apply_observed_exit(current, match, banks.reshape(12, len(period), 3), tactics)
    parameters = json.loads((weights / "interruption.json").read_text())
    # Two interruption fits from interruption.json, 30 features each; predict_interruption takes
    # the element-wise minimum of their (utility, confidence) outputs.
    models = tuple(stoppage.predictor(parameters[name], 30) for name in ("natural", "mixed"))
    current = interruption.predict_interruption(current, period, tracking, members, tactics, models)
    current, _ = native_timing.apply(current, period, tracking, native_timing.load_table(weights / "native-timing.json"))
    groups = [matched_context.describe(current, period, tracking, spans[pair][0],
                                       forecasts[pair][0].mean(axis=0), spans[pair][1]) for pair in pairs]
    current, _ = matched_context.apply(current, groups, matched_context.load_index(weights / "matched-context-index.npz"))
    current, _ = carrier.apply(current, period, ball, teams, tactics, carrier.load_classifier(weights / "carrier.json"))
    movement = np.stack([geometry(*teams, period), geometry(*teams[::-1], period)], axis=1)
    correction = stoppage.predictor(json.loads((weights / "stoppage.json").read_text()))
    return stoppage.predict_suffix(current, period, tracking, members, edges, movement, tactics, correction)
