"""Retime hidden team turnovers with exact-overlap credit learned from typed forecasts."""

from bisect import bisect_left
import json
from pathlib import Path

import numpy as np


OFFSETS = np.arange(-21, 22, 3)
SAMPLE = np.array([-16, -8, -4, 0, 4, 8, 16])
FEATURES = 35
CREDIT_FLOOR = 0.05
RISK_CAP = 0.34


def _cuts(labels, period):
    return np.r_[0, np.flatnonzero((labels[1:] != labels[:-1]) | (period[1:] != period[:-1])) + 1, len(labels)]


def _forecast_features(positions, at, lo, hi, before, seen, forecasts):
    odds, disagreement, edge = forecasts
    time = positions[:, None] + SAMPLE
    # log-odds signed toward the team in possession before the cut
    old = odds[time] * (1 - 2 * before)
    probabilities = edge[time, before]
    shifts = (positions - at) / 21.0  # Normalize to [-1, 1] range matching OFFSETS bounds
    duration = np.log1p([at - lo, hi - at])
    base = np.column_stack((
        shifts, abs(shifts), shifts**2, np.tile(duration, (len(shifts), 1)),
        old, old[:, 0] - old[:, -1], old[:, 2] - old[:, 4], probabilities[:, 1:6],
        disagreement[time].mean(1), seen[time].mean(1),
    ))
    contrast = base[:, 5:19] - base[positions == at][0, 5:19]
    return np.column_stack((base, contrast))


def _candidates(reference, period, observed, forecasts):
    cuts = _cuts(reference, period)
    for lo, at, hi in zip(cuts[:-2], cuts[1:-1], cuts[2:], strict=True):
        before, after = map(int, reference[[at - 1, at]])
        if max(before, after) >= 2 or before == after or period[lo] != period[hi - 1]:
            continue
        if not 23 <= min(at - lo, hi - at) <= 125 or at < 16 or at + 16 >= len(observed):
            continue
        if observed[at - 12:at + 13].mean() > 0.22:
            continue
        positions = at + OFFSETS
        positions = positions[(positions >= max(lo + 16, 16)) & (positions <= min(hi - 16, len(period) - 17))]
        positions = positions[(period[positions - 16] == period[at]) & (period[positions + 16] == period[at])]
        if len(positions) < 5 or at not in positions:
            continue
        features = _forecast_features(positions, at, lo, hi, before, observed, forecasts)
        yield (lo, at, hi, before, after), positions, features


def _select_nonoverlapping(paths, weights):
    order = sorted((index for index, weight in enumerate(weights) if weight > 1e-7),
                   key=lambda index: paths[index][1])
    ends = [paths[index][1] for index in order]
    best = [0.0]
    backtrack = []
    for cursor, index in enumerate(order):
        previous = bisect_left(ends, paths[index][0], hi=cursor)
        utility = weights[index] + best[previous]
        wins = utility > best[-1]
        backtrack.append((wins, previous))
        best.append(utility if wins else best[-1])
    selected = []
    cursor = len(order)
    while cursor:
        wins, previous = backtrack[cursor - 1]
        if wins:
            selected.append(order[cursor - 1])
        cursor = previous if wins else cursor - 1
    return selected


def load_weights(path):
    """Read fitted means, scales, overlap coefficients and risk coefficients."""
    payload = json.loads(Path(path).read_text())
    shapes = {"center": (FEATURES,), "scale": (FEATURES,),
              "coefficients": (FEATURES + 1, 2), "risk": (FEATURES + 1,)}
    state = {}
    for name, shape in shapes.items():
        value = np.asarray(payload[name], dtype=float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid credit timing parameter {name}")
        state[name] = value
    if (state["scale"] <= 0).any():
        raise ValueError("Credit timing scales must be positive")
    return state


def apply(reference, period, ball_observed, probabilities, edges, weights):
    """Return a retimed copy of the reference timeline.

    Forecast probabilities have shape (members, frames, 3). The first two
    edge columns are the directional home-to-away and away-to-home scores.
    """
    reference, period, ball_observed = map(np.asarray, (reference, period, ball_observed))
    probabilities, edges = map(np.asarray, (probabilities, edges))
    if (reference.ndim != 1 or period.shape != reference.shape or ball_observed.shape != reference.shape
            or probabilities.ndim != 3 or probabilities.shape[1:] != (len(reference), 3)
            or not len(probabilities) or edges.ndim != 3 or edges.shape[:2] != probabilities.shape[:2]
            or edges.shape[2] < 2):
        raise ValueError("Tracking, reference and forecast arrays must be aligned")
    logs = np.log(probabilities[:, :, :2].clip(1e-6))
    odds = (logs[:, :, 0] - logs[:, :, 1]).clip(-6.0, 6.0)
    forecasts = (odds.mean(0), odds.std(0), edges[:, :, :2].mean(0))
    paths, scores = [], []
    for (lo, at, hi, before, after), positions, features in _candidates(reference, period, ball_observed, forecasts):
        design = np.column_stack((np.ones(len(features)), ((features - weights["center"]) / weights["scale"]).clip(-6, 6)))
        direct = (design @ weights["coefficients"])[:, 0]
        loss = 1 / (1 + np.exp(-(design @ weights["risk"]).clip(-18, 18)))
        utility = direct - .0007 * abs(positions - at)
        allowed = (positions != at) & (utility > CREDIT_FLOOR) & (loss < RISK_CAP)
        if not allowed.any():
            continue
        best = np.where(allowed, utility, -np.inf).argmax()
        stop = int(positions[best])
        a, b = sorted((at, stop))
        paths.append((lo, hi, a, b, after if stop < at else before))
        scores.append(float(utility[best] - CREDIT_FLOOR))
    result = reference.copy()
    for index in _select_nonoverlapping(paths, scores):
        _, _, a, b, side = paths[index]
        result[a:b] = side
    return result
