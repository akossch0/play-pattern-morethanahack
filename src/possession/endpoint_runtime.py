"""Retiming of hidden turnovers from typed forecasts and external boundary errors."""

from bisect import bisect_left
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as pm

# Candidate shifts, in frames, tried around each eligible home/away boundary.
OFFSETS = np.arange(-21, 22, 3)
# Frame offsets around each candidate position (not the original boundary) at which
# forecasts, ball visibility and player motion are read in describe().
SAMPLE = np.array([-16, -8, -4, 0, 4, 8, 16])


def player_motion(track):
    period = track["period"]
    frame = np.arange(len(period))
    left, right = (
        (frame - 4).clip(0, len(period) - 1),
        (frame + 4).clip(0, len(period) - 1),
    )
    size = np.array([track["pitch_length_m"], track["pitch_width_m"]], np.float32)
    xy = np.asarray([track[f"{side}_xy"] * size for side in ("home", "away")])
    seen = np.isfinite(xy).all(-1)
    velocity = (xy[:, right] - xy[:, left]) * 25.0 / 8
    velocity[:, period[left] != period[right]] = np.nan
    finite = np.isfinite(velocity).all(-1)
    velocity = np.where(finite[..., None], velocity, 0.0)
    speed = np.linalg.norm(velocity, axis=-1)
    observed = np.where(seen[..., None], xy, 0.0)
    centroid = observed.sum(2) / seen.sum(2).clip(1)[..., None]
    spread = np.linalg.norm(observed - centroid[:, :, None, :], axis=-1)
    shape = [
        seen.mean(2),
        finite.mean(2),
        (speed * finite).sum(2) / finite.sum(2).clip(1),
        (spread * seen).sum(2) / seen.sum(2).clip(1),
        speed.max(2),
    ]
    team = np.moveaxis(np.asarray(shape, dtype=np.float32), 0, -1)
    pressure = [[], []]
    for side in (0, 1):
        opponent = 1 - side
        chunks = []
        for first in range(0, len(frame), 2048):
            chunk = slice(first, first + 2048)
            rows = np.arange(min(2048, len(frame) - first))
            delta = (
                observed[opponent, chunk, None, :, :]
                - observed[side, chunk, :, None, :]
            )
            squares = (delta * delta).sum(-1)
            legs = np.where(
                seen[side, chunk, :, None] & seen[opponent, chunk, None, :],
                squares,
                np.inf,
            )
            neighbor = legs.argmin(2)
            vectors = delta[rows[:, None], np.arange(xy.shape[2])[None, :], neighbor]
            distance = np.sqrt(legs.min(2))
            valid = (
                np.isfinite(distance)
                & finite[side, chunk]
                & finite[opponent, first + rows[:, None], neighbor]
            )
            unit = vectors / np.where(valid, distance.clip(0.25), 1.0)[..., None]
            into = (velocity[side, chunk] * unit).sum(-1)
            away = (velocity[opponent, first + rows[:, None], neighbor] * unit).sum(-1)
            relative = np.linalg.norm(
                velocity[side, chunk]
                - velocity[opponent, first + rows[:, None], neighbor],
                axis=-1,
            )
            local = valid & (distance <= 14)
            close = valid & (distance <= 6)
            near_weight = np.where(local, np.exp(-distance / 7), 0.0)
            mass = near_weight.sum(1).clip(0.05)
            chunks.append(
                np.column_stack(
                    (
                        local.mean(1),
                        close.mean(1),
                        (near_weight * into).sum(1) / mass,
                        (near_weight * away).sum(1) / mass,
                        (near_weight * (into - away)).sum(1) / mass,
                        (near_weight * relative).sum(1) / mass,
                        (close & (into >= 1)).mean(1),
                        (close & (into - away >= 1)).mean(1),
                    )
                )
            )
        pressure[side] = np.concatenate(chunks)
    values = np.asarray(pressure, dtype=np.float32)
    assert np.isfinite(team).all() and np.isfinite(values).all()
    return team, values


def cuts(codes, period):
    return np.r_[
        0,
        np.flatnonzero((codes[1:] != codes[:-1]) | (period[1:] != period[:-1])) + 1,
        len(codes),
    ]


def stream(forecasts):
    p = np.array([np.log(item[0][:, :2].clip(1e-6)) for item in forecasts])
    log_odds = (p[:, :, 0] - p[:, :, 1]).clip(-6.0, 6.0)
    e = np.stack([item[1][:, :2] for item in forecasts])
    return log_odds.mean(0), log_odds.std(0), e.mean(0)


def describe(positions, at, lo, hi, before, seen, forecasts, motion):
    odds, disagreement, edge = forecasts
    time = positions[:, None] + SAMPLE
    old = odds[time] * (1 - 2 * before)
    probabilities = edge[time, before]
    shifts = (positions - at) / 21.0
    duration = np.log1p([at - lo, hi - at])
    visibility = seen[time].mean(1)
    # A separate output row accounts for a forecast with no matching boundary.
    basic = np.column_stack(
        (
            shifts,
            abs(shifts),
            shifts**2,
            np.tile(duration, (len(shifts), 1)),
            old,
            old[:, 0] - old[:, -1],
            old[:, 2] - old[:, 4],
            probabilities[:, 1:6],
            disagreement[time].mean(1),
            visibility,
        )
    )
    team, pressure = motion
    moving = []
    for source in (team, pressure):
        series = source[before, time] - source[1 - before, time]
        past = series[:, :3].mean(1)
        after = series[:, 4:].mean(1)
        moving.extend((past, series[:, 3], after, after - past))
    all_features = np.column_stack((basic, *moving))
    none = all_features[positions == at].copy()
    return np.column_stack(
        (np.concatenate((all_features, none)), np.r_[np.zeros(len(positions)), 1.0])
    )


def proposals(codes, period, seen, forecasts, motion):
    endpoints = cuts(codes, period)
    observed = np.flatnonzero(seen)
    for lo, at, hi in zip(endpoints[:-2], endpoints[1:-1], endpoints[2:], strict=True):
        before, after = map(int, codes[[at - 1, at]])
        if max(before, after) >= 2 or before == after or period[lo] != period[hi - 1]:
            continue
        if not 23 <= min(at - lo, hi - at) <= 125 or at < 16 or at + 16 >= len(seen):
            continue
        if seen[at - 12 : at + 13].mean() > 0.22:
            continue
        positions = at + OFFSETS
        take = (positions >= max(lo + 16, 16)) & (
            positions <= min(hi - 16, len(period) - 17)
        )
        positions = positions[take]
        positions = positions[
            (period[positions - 16] == period[at])
            & (period[positions + 16] == period[at])
        ]
        if len(positions) < 5 or at not in positions:
            continue
        before_seen = (
            observed[observed.searchsorted(at) - 1]
            if observed.searchsorted(at) > 0
            else -1000
        )
        after_seen = (
            observed[observed.searchsorted(at)]
            if observed.searchsorted(at) < len(observed)
            else len(seen) + 1000
        )
        distance = min(at - before_seen, after_seen - at)
        features = describe(positions, at, lo, hi, before, seen, forecasts, motion)
        yield (lo, at, hi, before, after), positions, features, distance


def delta_matrix(positions, outer, lo, t, hi):
    change = positions[:, None, None]
    truth = positions[None, :, None]
    left = np.minimum(lo + outer[None, None, :, 0], truth - 9)
    right = np.maximum(hi + outer[None, None, :, 1], truth + 9)
    l_intersection = (np.minimum(change, truth) - np.maximum(lo, left)).clip(0)
    l_union = np.maximum(change, truth) - np.minimum(lo, left)
    r_intersection = (np.minimum(hi, right) - np.maximum(change, truth)).clip(0)
    r_union = np.maximum(hi, right) - np.minimum(change, truth)
    thresholds = np.asarray(pm.SPELL_IOUS)
    credit = ((l_intersection / l_union)[..., None] >= thresholds).astype(float)
    credit += (r_intersection / r_union)[..., None] >= thresholds
    return credit - credit[positions == t]


def expectation(positions, probability, outer, ratio, lo, t, hi):
    delta = delta_matrix(positions, outer, lo, t, hi)
    weight = probability[:-1, None] * ratio[None, :]
    pooled = delta.mean(-1)
    gain = np.sum(weight[None] * pooled, axis=(1, 2))
    loss = np.sum(weight[None] * (pooled < 0.0), axis=(1, 2))
    strict = np.sum(weight[None] * delta[..., 2:].mean(-1), axis=(1, 2))
    return gain, loss, strict


def select(paths, weights, indices):
    order = sorted((i for i in indices if weights[i] > 1e-7), key=lambda i: paths[i][1])
    right = [paths[i][1] for i in order]
    maxima = [0.0]
    last = []
    for cursor, i in enumerate(order):
        before = bisect_left(right, paths[i][0], hi=cursor)
        utility = weights[i] + maxima[before]
        wins = utility > maxima[-1]
        last.append((wins, before))
        maxima.append(utility if wins else maxima[-1])
    edits = []
    cursor = len(order)
    while cursor:
        wins, before = last[cursor - 1]
        if wins:
            edits.append(order[cursor - 1])
        cursor = before if wins else cursor - 1
    return edits


def _design(features, center, scale):
    normalized = ((features - center) / scale).clip(-6, 6)
    # Column indices follow describe(): 3-4 log1p spell lengths before/after the
    # boundary, 8 and 11 forecast log-odds for the pre-boundary side at SAMPLE offsets 0
    # and +16, 19 forecast disagreement, 20 ball visibility. The last feature column is
    # 1 only on the appended "no matching boundary" row, so these six terms are zero on
    # every other row.
    interaction = normalized[:, [3, 4, 8, 11, 19, 20]] * (features[:, -1:] > 0.5)
    return np.column_stack((normalized, interaction))


def posterior_probability(features, model):
    center, scale, coefficients = model
    logits = _design(features, center, scale) @ coefficients
    probability = np.exp(logits - logits.max())
    return probability / probability.sum()


def load_model(path):
    """Read the frozen numeric parameters for the endpoint timing stage."""
    with np.load(Path(path), allow_pickle=False) as archive:
        model = {
            name: archive[name]
            for name in (
                "center",
                "scale",
                "coefficients",
                "long_centers",
                "long_weights",
                "short_centers",
                "short_weights",
            )
        }
    if model["center"].shape != (74,) or model["scale"].shape != (74,):
        raise ValueError("Invalid hidden-turnover feature normalization")
    if model["coefficients"].shape != (80,) or np.any(model["scale"] <= 0):
        raise ValueError("Invalid hidden-turnover posterior coefficients")
    if not all(np.isfinite(value).all() for value in model.values()):
        raise ValueError("Hidden-turnover parameters must be finite")
    for name in ("long", "short"):
        centers, weights = model[name + "_centers"], model[name + "_weights"]
        if (
            centers.shape != (len(weights), 2)
            or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1)
        ):
            raise ValueError("Invalid empirical boundary error distribution")
    return model


def candidate_bank(reference, track, forecasts, motion=None):
    """Build the shared boundary candidate rows from period-aligned typed forecasts."""
    if motion is None:
        motion = player_motion(track)
    return list(
        proposals(
            reference,
            track["period"],
            track["ball_observed"],
            stream(forecasts),
            motion,
        )
    )


def apply_flank(reference, bank, model):
    """Apply the selected empirical-flank policy to a frozen incoming timeline."""
    posterior = tuple(model[name] for name in ("center", "scale", "coefficients"))
    options, weights = [], []
    for (lo, at, hi, before, after), positions, features, _ in bank:
        probability = posterior_probability(features, posterior)
        if probability[-1] > 0.60:
            continue
        group = "short" if min(at - lo, hi - at) <= 65 else "long"
        centers, ratio = model[group + "_centers"], model[group + "_weights"]
        if at - lo > hi - at:
            centers = centers[:, ::-1]
        gain, loss, _ = expectation(positions, probability, centers, ratio, lo, at, hi)
        utility = gain - 0.00085 * np.abs(positions - at)
        allowed = (positions != at) & (loss < 0.35) & (utility > 0.015)
        if not allowed.any():
            continue
        index = np.where(allowed, utility, -np.inf).argmax()
        destination = int(positions[index])
        start, stop = sorted((at, destination))
        options.append((lo, hi, start, stop, after if destination < at else before))
        weights.append(float(utility[index] - 0.015))
    chosen = select(options, weights, range(len(options)))
    result = reference.copy()
    for i in chosen:
        _, _, start, stop, side = options[i]
        result[start:stop] = side
    return result
