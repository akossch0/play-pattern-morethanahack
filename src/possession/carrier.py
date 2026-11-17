"""Merge short opposing spells when player and ball movement support continuous control."""

from bisect import bisect_left
import json
from pathlib import Path

import numpy as np

from datalake.core import D


FEATURE_COUNT = 57
MIN_CREDIT = 0.06
MIN_CONFIDENCE = 0.42
MAX_EDITS = 2


def co_motion(ball, players, period):
    """Estimate how closely each player follows the ball over an eight-frame span."""
    frames = np.arange(len(period))
    before = np.maximum(frames - 4, 0)
    after = np.minimum(frames + 4, len(period) - 1)
    same_period = (period[before] == period) & (period[after] == period)
    # Change from frame - 4 to frame + 4 (indices clamped to the array ends), scaled by 25 / 8:
    # per second at 25 fps wherever the span is the full 8 frames.
    ball_velocity = (ball[after] - ball[before]) * (25 / 8)
    player_velocity = (players[after] - players[before]) * (25 / 8)
    distance = np.linalg.norm(players - ball[:, None], axis=2)
    relative_speed = np.linalg.norm(player_velocity - ball_velocity[:, None], axis=2)
    available = np.isfinite(distance) & np.isfinite(relative_speed) & same_period[:, None]
    strength = np.exp(-np.nan_to_num(distance, nan=60.0) / 2.5
                      - np.nan_to_num(relative_speed, nan=60.0) / 6)
    return np.where(available, strength, 0.0), available


def _candidates(reference, period):
    changes = (reference[1:] != reference[:-1]) | (period[1:] != period[:-1])
    stops = np.r_[0, np.flatnonzero(changes) + 1, len(reference)]
    codes = reference[stops[:-1]]
    for i in range(1, len(codes) - 1):
        lo, start, stop, hi = map(int, stops[i - 1:i + 3])
        other = int(codes[i])
        if other >= D or not 9 <= stop - start <= 105 or period[lo] != period[hi - 1]:
            continue
        host = 1 - other
        lead, tail = int(codes[i - 1]), int(codes[i + 1])
        if lead not in (host, D) or tail not in (host, D) or lead == tail == D:
            continue
        if (lead == host and start - lo < 7) or (tail == host and hi - stop < 7):
            continue
        beginning = start if lead == D else lo
        ending = stop if tail == D else hi
        near_before = i >= 2 and codes[i - 2] == D and start - lo < 110
        near_after = i + 2 < len(codes) and codes[i + 2] == D and hi - stop < 110
        nearby = lead == D or tail == D or near_before or near_after
        yield beginning, ending, start, stop, host, lead == D, tail == D, nearby


def describe(reference, period, ball, players):
    """Return candidate intervals and the 57 features used by the carrier model."""
    if not len(reference):
        return [], np.empty((0, FEATURE_COUNT))
    movement = [co_motion(ball, side, period) for side in players]
    visibility = np.isfinite(ball).all(axis=1)
    rows, features = [], []
    for lo, hi, start, stop, host, lead_dead, tail_dead, nearby in _candidates(reference, period):
        left, right = max(lo, start - 24), min(hi, stop + 24)
        middle = (start + stop) // 2
        spans = ((left, start), (start, middle), (middle, stop), (stop, right))
        values = [np.log1p(stop - start), np.log1p(start - lo), np.log1p(hi - stop),
                  lead_dead, tail_dead, nearby]
        continuity = []
        for side in (host, 1 - host):
            control, seen = movement[side]
            for a, b in spans:
                if b <= a:
                    values.extend([0.0] * 5)
                    continue
                segment = control[a:b]
                strength = segment.max(axis=1)
                values.extend([strength.mean(), np.quantile(strength, 0.75),
                               float((strength >= 0.31).mean()), segment.mean(axis=0).max(),
                               seen[a:b].any(axis=1).mean()])
            player = int(control[left:right].mean(axis=0).argmax())
            inside = float(control[start:stop, player].mean())
            before = float(control[left:start, player].mean()) if left < start else 0.0
            after = float(control[stop:right, player].mean()) if stop < right else 0.0
            # Lowest of this player's mean control before, inside and after the spell. A window
            # that is empty because the spell borders a dead ball counts as 0.0.
            continuity.extend([min(before, after, inside),
                               float((control[start:stop].max(axis=1) >= 0.31).mean())])
            values.extend([inside, before, after])
        rows.append((lo, hi, start, stop, host, nearby))
        features.append([*values, *continuity, visibility[start:stop].mean()])
    return rows, np.asarray(features, dtype=float).reshape(-1, FEATURE_COUNT)


def load_classifier(path: Path):
    model = json.loads(path.read_text())
    center, scale, credit, odds = (np.asarray(model[name], dtype=float)
                                   for name in ("center", "scale", "credit", "odds"))
    if (center.shape != (FEATURE_COUNT,) or scale.shape != center.shape
            or credit.shape != (FEATURE_COUNT + 1,) or odds.shape != credit.shape
            or not all(np.isfinite(w).all() for w in (center, scale, credit, odds))
            or np.any(scale <= 0)):
        raise ValueError("Invalid carrier correction weights")

    def predict(features):
        inputs = np.column_stack((np.ones(len(features)), ((features - center) / scale).clip(-6, 6)))
        return np.column_stack((inputs @ credit, 1 / (1 + np.exp(-(inputs @ odds).clip(-18, 18)))))

    return predict


def _nonoverlapping(rows, weights, eligible):
    """Maximise credit over compatible edits, preserving the original tie-breaking."""
    order = sorted((i for i in eligible if weights[i] > 1e-7), key=lambda i: rows[i][1])
    stops = [rows[i][1] for i in order]
    maxima = [0.0]
    backtrack = []
    for cursor, i in enumerate(order):
        before = bisect_left(stops, rows[i][0], hi=cursor)
        utility = weights[i] + maxima[before]
        wins = utility > maxima[-1]
        backtrack.append((wins, before))
        maxima.append(utility if wins else maxima[-1])
    cursor = len(order)
    selected = []
    while cursor:
        wins, before = backtrack[cursor - 1]
        if wins:
            selected.append(order[cursor - 1])
        cursor = before if wins else cursor - 1
    return selected


def apply(reference, period, ball, players, tactics, model):
    """Apply the actor-only correction to an existing possession timeline."""
    rows, features = describe(reference, period, ball, players)
    revised = reference.copy()
    if not len(rows):
        return revised, 0
    scores = model(features)
    odds = np.log(tactics[:, 0].clip(1e-7)) - np.log(tactics[:, 1].clip(1e-7))
    accepted = []
    for i, (credit, confidence) in enumerate(scores):
        if credit < MIN_CREDIT or confidence < MIN_CONFIDENCE:
            continue
        _, _, start, stop, host, nearby = rows[i]
        if not nearby or stop - start > 72 or features[i, -1] < 0.36:
            continue
        if (1 - 2 * host) * odds[start:stop].mean() < 0.10:
            continue
        accepted.append(i)
    selected = _nonoverlapping(rows, scores[:, 0] - MIN_CREDIT + 1e-8, accepted)
    selected.sort(key=lambda i: scores[i, 0], reverse=True)
    for i in selected[:MAX_EDITS]:
        _, _, start, stop, host, _ = rows[i]
        revised[start:stop] = host
    np.testing.assert_array_equal(reference == D, revised == D)
    return revised, min(MAX_EDITS, len(selected))
