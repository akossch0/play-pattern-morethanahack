"""Retiming of hidden home-away boundaries using team and opponent motion."""

from bisect import bisect_left
from pathlib import Path

import numpy as np

# Frame offsets around each candidate boundary position at which scan() samples
# ball visibility and the team/pressure motion features.
SAMPLES = np.array([-20, -12, -6, 0, 6, 12, 20])


def motion(track):
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
    pressure = np.empty((2, len(period), 8), dtype=np.float32)
    for lo in range(0, len(period), 2048):
        rows = slice(lo, lo + 2048)
        for side in (0, 1):
            pressure[side, rows] = _pressure_for_side(
                observed[:, rows],
                seen[:, rows],
                velocity[:, rows],
                finite[:, rows],
                side,
            )
    if not np.isfinite(team).all() or not np.isfinite(pressure).all():
        raise ValueError("Non-finite player motion features")
    return team, pressure


def _pressure_for_side(observed, seen, velocity, finite, side):
    frame = np.arange(observed.shape[1])
    opponent = 1 - side
    delta = observed[opponent, :, None, :, :] - observed[side, :, :, None, :]
    squares = (delta * delta).sum(-1)
    legs = np.where(
        seen[side, :, :, None] & seen[opponent, :, None, :], squares, np.inf
    )
    neighbor = legs.argmin(2)
    vectors = delta[frame[:, None], np.arange(observed.shape[2])[None, :], neighbor]
    distance = np.sqrt(legs.min(2))
    valid = (
        np.isfinite(distance)
        & finite[side]
        & finite[opponent, frame[:, None], neighbor]
    )
    unit = vectors / np.where(valid, distance.clip(0.25), 1.0)[..., None]
    into = (velocity[side] * unit).sum(-1)
    away = (velocity[opponent, frame[:, None], neighbor] * unit).sum(-1)
    relative = np.linalg.norm(
        velocity[side] - velocity[opponent, frame[:, None], neighbor], axis=-1
    )
    local = valid & (distance <= 14)
    close = valid & (distance <= 6)
    # Weight by nearest-opponent distance: 7 m e-folding length, zero beyond 14 m.
    near_weight = np.where(local, np.exp(-distance / 7), 0.0)
    mass = near_weight.sum(1).clip(0.05)
    return np.column_stack(
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


def temporal_difference(tape, times, left, right):
    first, second = tape[left, times], tape[right, times]
    mean = 0.5 * (first + second)
    difference = first - second
    slope = np.diff(difference, axis=1)
    return np.column_stack(
        (
            mean.reshape(len(times), -1),
            difference.reshape(len(times), -1),
            slope.reshape(len(times), -1),
        )
    )


def scan(codes, period, track, forecasts, helpers, material):
    team, pressure = material
    emission = np.mean([p for p, _ in forecasts], axis=0)
    logits = (
        np.log(emission[:, 0].clip(1e-6)) - np.log(emission[:, 1].clip(1e-6))
    ).clip(-7, 7)
    prefix = np.r_[0, np.cumsum(logits, dtype=np.float64)]
    directional = np.mean([edge for _, edge in forecasts], axis=0)[:, :2]
    cuts = helpers["cuts"](codes, period)
    observed = track["ball_observed"]
    output = []
    for lo, t, hi in zip(cuts[:-2], cuts[1:-1], cuts[2:], strict=True):
        before, after = int(codes[t - 1]), int(codes[t])
        if max(before, after) >= 2 or before == after or period[lo] != period[hi - 1]:
            continue
        if min(t - lo, hi - t) < 26 or observed[t - 12 : t + 13].mean() > 0.22:
            continue
        lo_candidate, hi_candidate = max(lo + 18, t - 18), min(hi - 18, t + 18)
        position = t + np.arange(-18, 19, 3)
        eligible = (
            (position >= lo_candidate) & (position <= hi_candidate) & (position != t)
        )
        samples = position[eligible, None] + SAMPLES
        permitted = (samples[:, 0] >= 0) & (samples[:, -1] < len(period))
        positions, samples = position[eligible][permitted], samples[permitted]
        permitted = (period[samples] == period[t]).all(1)
        positions, samples = positions[permitted], samples[permitted]
        if not len(positions):
            continue
        moved = positions - t
        contrast = (prefix[positions] - prefix[t]) * (2 * after - 1) / abs(moved)
        edge = directional[positions, before] - directional[t, before]
        basic = np.column_stack(
            (
                moved / 18,
                abs(moved) / 18,
                contrast,
                edge,
                directional[positions, before],
                np.full(len(positions), np.log1p(t - lo)),
                np.full(len(positions), np.log1p(hi - t)),
                observed[samples].mean(1),
            )
        )
        human = temporal_difference(team, samples, before, after)
        inter = temporal_difference(pressure, samples, before, after)
        block = np.column_stack((basic, human, inter))
        clean = np.array([not observed[min(t, p) : max(t, p)].any() for p in positions])
        output.append(
            ((int(lo), int(t), int(hi), before, after), positions, block, clean)
        )
    return output


# Fitting/development helper: nothing on the predict.py path calls this.
def logistic(x, weight, target, penalty):
    coef = np.zeros(x.shape[1])
    total = np.dot(weight, target).clip(0.001, 0.999)
    coef[0] = np.log(total / (1 - total))
    for _ in range(10):
        estimate = 1.0 / (1.0 + np.exp(-(x @ coef).clip(-20, 20)))
        response = weight * estimate * (1.0 - estimate)
        gradient = x.T @ (weight * (estimate - target)) + penalty @ coef
        curvature = (x.T * response) @ x + penalty
        update = np.linalg.solve(curvature, gradient)
        coef -= update
        if np.linalg.norm(update) < 1e-7:
            break
    return coef


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


def _cuts(labels, periods):
    changes = (labels[1:] != labels[:-1]) | (periods[1:] != periods[:-1])
    return np.r_[0, np.flatnonzero(changes) + 1, len(labels)]


def _compact(features):
    teams = features[:, 8:108].reshape(-1, 20, 5)
    pressure = features[:, 108:268].reshape(-1, 20, 8)
    summaries = [features[:, :8]]
    for signal in (teams, pressure):
        difference = signal[:, 7:14]
        summaries.extend(
            (
                difference[:, :3].mean(axis=1),
                difference[:, 3],
                difference[:, 4:].mean(axis=1),
            )
        )
    return np.column_stack(summaries)


def _design(features, center, scale):
    normalized = ((_compact(features) - center) / scale).clip(-6, 6)
    return np.column_stack((np.ones(len(normalized)), normalized))


# Fitting/development helper: nothing on the predict.py path calls this.
def fit_control(parts):
    """One equal-weight external training group per match."""
    features, target, _ = (np.concatenate(group) for group in zip(*parts, strict=True))
    weights = np.concatenate(
        [weight / weight.sum() / len(parts) for _, _, weight in parts]
    )
    compact = _compact(features)
    center = weights @ compact
    scale = np.sqrt(weights @ ((compact - center) ** 2)).clip(0.06)
    x = _design(features, center, scale)
    penalty = 0.10 * np.eye(x.shape[1])
    penalty[0, 0] = 1e-6
    gain = np.linalg.solve((x.T * weights) @ x + penalty, x.T @ (weights * target))
    loss = logistic(x, weights, (target < -0.02).astype(float), penalty * 0.55)
    return center, scale, gain, loss


def load_models(path):
    with np.load(Path(path), allow_pickle=False) as weights:
        models = tuple(
            tuple(
                weights[f"{variant}_{field}"]
                for field in ("center", "scale", "gain", "loss")
            )
            for variant in ("reference", "alternate")
        )
    for center, scale, gain, loss in models:
        if (
            center.shape != (47,)
            or scale.shape != center.shape
            or gain.shape != (48,)
            or loss.shape != gain.shape
            or np.any(scale <= 0)
            or not all(np.isfinite(part).all() for part in (center, scale, gain, loss))
        ):
            raise ValueError("Invalid hidden-boundary timing weights")
    return models


def candidates(base, tracking, forecasts, movement):
    """Return aligned boundary proposals for fitting or inference."""
    return scan(
        base, tracking["period"], tracking, forecasts, {"cuts": _cuts}, movement
    )


def predict_branch(base, bank, model):
    center, scale, gain, loss = model
    edits, weights = [], []
    for (lo, at, hi, before, after), positions, features, _ in bank:
        if min(at - lo, hi - at) > 125:
            continue
        rows = _design(features, center, scale)
        expected = rows @ gain
        risk = 1 / (1 + np.exp(-(rows @ loss).clip(-20, 20)))
        utility = expected - 0.001 * np.abs(positions - at)
        safe = (utility > 0.065) & (risk < 0.30)
        if not safe.any():
            continue
        best = np.where(safe, utility, -np.inf).argmax()
        at_new = int(positions[best])
        start, stop = sorted((at_new, at))
        edits.append((lo, hi, start, stop, after if at_new < at else before))
        weights.append(float(utility[best] - 0.065))
    prediction = base.copy()
    for i in select(edits, weights, range(len(edits))):
        _, _, start, stop, side = edits[i]
        prediction[start:stop] = side
    return prediction


def predict_duel_inputs(base, tracking, forecasts, models):
    """Return the reference-draft and alternate-draft boundary timelines."""
    bank = candidates(base, tracking, forecasts, motion(tracking))
    return tuple(predict_branch(base, bank, model) for model in models)
