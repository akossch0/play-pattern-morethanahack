"""Correct a short rival spell adjacent to a stoppage."""

from pathlib import Path

import numpy as np

from datalake.remap_touch import _evidence
from .interruption import nonoverlapping


def _mean(segment):
    return segment.mean() if segment.size else 0.0


def receiver_motion(ev, lo, hi, side):
    # A candidate receiver must be observed on both endpoints. Compare ball motion
    # with the direction toward that player's later location.
    frames = np.arange(lo, hi - 5)
    future = frames + 5
    slot = ev["slot"][side][future]
    seen = (
        ev["seen"][side][future]
        & ev["ball_seen"][frames]
        & ev["observed"][side][frames, slot]
    )
    seen &= ev["dist"][side][future] <= 2.0
    if not seen.any():
        return 0.0, 0.0
    a, b, player = frames[seen], future[seen], slot[seen]
    ball = ev["ball_m"][a]
    step = ev["ball_m"][b] - ball
    ray = ev["player_m"][side][b, player] - ball
    alignment = (step * ray).sum(1) / (
        np.linalg.norm(ray, axis=1) * np.linalg.norm(step, axis=1)
    ).clip(0.1)
    approach = np.linalg.norm(ray, axis=1) - ev["dist"][side][b]
    # The ball's 5-frame step points within about 37 degrees of the receiver's later
    # position (alignment > 0.80) and brings the ball more than 1.25 m closer to it.
    receive = (alignment > 0.80) & (approach > 1.25)
    return float(receive.sum() / max(1, len(frames))), float(
        np.max(np.where(receive, approach, 0.0))
    )


def restart_edge_features(track, motion, p, edges, row):
    lo, a, b, hi, side = row
    team = 1 - side
    seen = track["ball_seen"]
    times = np.unique(np.linspace(a, b - 1, 9).round().astype(int))
    own = p[a:b, side]
    host = p[a:b, team]
    nearest = track["dist"][side][a:b]
    another = track["dist"][team][a:b]
    anchor = p[max(0, a - 20) : a, team] if a > lo else p[b : min(hi, b + 20), team]
    end = min(hi, b + 20) if b < hi else b + 20
    context = p[max(0, a - 20) : end, 2]
    contrast = motion[side, times] - motion[team, times]
    return np.r_[
        a == lo,
        np.log1p([a - lo, b - a, hi - b]),
        seen[a:b].mean(),
        _mean(seen[max(0, a - 16) : a]),
        _mean(seen[b : b + 16]),
        own.mean(),
        own.max(),
        host.mean(),
        host.max(),
        _mean(anchor),
        np.log(own.clip(1e-6)).mean() - np.log(host.clip(1e-6)).mean(),
        (own > 0.65).mean(),
        (host > 0.65).mean(),
        context.mean(),
        edges[max(0, a - 4) : a + 5, team].max(),
        edges[max(0, b - 4) : b + 5, side].max(),
        np.mean((nearest < 1.5) & track["seen"][side][a:b]),
        np.mean((another < 1.5) & track["seen"][team][a:b]),
        np.nan_to_num(nearest, nan=10.0, posinf=10.0).min(),
        np.nan_to_num(another, nan=10.0, posinf=10.0).min(),
        contrast.mean(0),
        contrast[0],
        contrast[-1],
        receiver_motion(track, a, b, side),
        receiver_motion(track, a, b, team),
    ].astype(np.float32)


def _candidates(codes, period, track, motion, pairs):
    p = np.mean([m for m, _ in pairs], axis=0)
    edge = np.mean([e for _, e in pairs], axis=0)
    bounds = np.r_[
        0,
        np.flatnonzero((codes[1:] != codes[:-1]) | (period[1:] != period[:-1])) + 1,
        len(codes),
    ]
    states = codes[bounds[:-1]]
    output = []
    for i in range(1, len(states) - 1):
        side = int(states[i])
        if side >= 2:
            continue
        lo, a, b, hi = map(int, bounds[i - 1 : i + 3])
        before, after = int(states[i - 1]), int(states[i + 1])
        if not 7 <= b - a <= 75 or period[lo] != period[hi - 1]:
            continue
        opening = False
        if before == 2 and after == 1 - side and hi - b >= 10:
            row, old, count = (a, a, b, hi, side), [(a, b, side), (b, hi, after)], 1
            opening = True
        elif after == 2 and before == 1 - side and a - lo >= 10:
            row, old, count = (lo, a, b, b, side), [(lo, a, before), (a, b, side)], 1
        elif before == after == 1 - side and min(a - lo, hi - b) >= 10:
            next_stop = (
                i + 2 < len(states)
                and states[i + 2] == 2
                and period[bounds[i + 2]] == period[a]
            )
            prev_stop = (
                i >= 2 and states[i - 2] == 2 and period[bounds[i - 2]] == period[a]
            )
            if not next_stop and not prev_stop:
                continue
            if next_stop and hi - b > 95 or prev_stop and not next_stop and a - lo > 95:
                continue
            opening = bool(prev_stop and not next_stop)
            row = lo, a, b, hi, side
            old = [(lo, a, before), (a, b, side), (b, hi, after)]
            count = 2
        else:
            continue
        x = restart_edge_features(track, motion, p, edge, row)
        shifted = [(old[0][0], old[-1][1], 1 - side)]
        odds = np.log(p[a:b, 1 - side].clip(1e-6)) - np.log(p[a:b, side].clip(1e-6))
        control = np.r_[
            count == 2,
            opening,
            odds.min(),
            np.quantile(odds, 0.25),
            (odds > 0).mean(),
            track["ball_seen"][a:b].mean(),
        ]
        output.append(
            (row, old, shifted, count, opening, np.r_[x, control].astype(np.float32))
        )
    return output


def _pressure_chunk(period, xy, start, stop):
    frame = np.arange(start, stop)
    left, right = (
        (frame - 4).clip(0, len(period) - 1),
        (frame + 4).clip(0, len(period) - 1),
    )
    seen = np.isfinite(xy[:, start:stop]).all(-1)
    # Central difference over frames t-4..t+4 (8 frames at 25 fps), in m/s. The clip
    # shortens that span in the first and last four frames of the match; the divisor stays 8.
    velocity = (xy[:, right] - xy[:, left]) * 25.0 / 8
    velocity[:, period[left] != period[right]] = np.nan
    finite = np.isfinite(velocity).all(-1)
    velocity = np.where(finite[..., None], velocity, 0.0)
    observed = np.where(seen[..., None], xy[:, start:stop], 0.0)
    pressure = [[], []]
    for side in (0, 1):
        opponent = 1 - side
        delta = observed[opponent, :, None, :, :] - observed[side, :, :, None, :]
        squares = (delta * delta).sum(-1)
        legs = np.where(
            seen[side, :, :, None] & seen[opponent, :, None, :], squares, np.inf
        )
        neighbor = legs.argmin(2)
        vectors = delta[
            np.arange(len(frame))[:, None], np.arange(xy.shape[2])[None, :], neighbor
        ]
        distance = np.sqrt(legs.min(2))
        valid = (
            np.isfinite(distance)
            & finite[side]
            & finite[opponent, np.arange(len(frame))[:, None], neighbor]
        )
        unit = vectors / np.where(valid, distance.clip(0.25), 1.0)[..., None]
        into = (velocity[side] * unit).sum(-1)
        away = (
            velocity[opponent, np.arange(len(frame))[:, None], neighbor] * unit
        ).sum(-1)
        relative = np.linalg.norm(
            velocity[side]
            - velocity[opponent, np.arange(len(frame))[:, None], neighbor],
            axis=-1,
        )
        local = valid & (distance <= 14)
        close = valid & (distance <= 6)
        near_weight = np.where(local, np.exp(-distance / 7), 0.0)
        mass = near_weight.sum(1).clip(0.05)
        pressure[side] = np.column_stack(
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
    return np.asarray(pressure, dtype=np.float32)


def _pressure(track):
    period = track["period"]
    # Nearest-opponent comparisons use bounded batches to fit long matches.
    size = np.array([track["pitch_length_m"], track["pitch_width_m"]], np.float32)
    xy = np.asarray([track[f"{side}_xy"] * size for side in ("home", "away")])
    chunks = [
        _pressure_chunk(period, xy, start, min(start + 4096, len(period)))
        for start in range(0, len(period), 4096)
    ]
    return np.concatenate(chunks, axis=1)


def contact_evidence(match):
    evidence = _evidence(match)
    evidence["observed"] = [match[f"{side}_observed"] for side in ("home", "away")]
    evidence["player_m"] = [
        match[f"{side}_xy"] * evidence["size"] for side in ("home", "away")
    ]
    return evidence


def feature_inputs(match):
    """Compute contact and player-motion inputs once for a match."""
    return contact_evidence(match), _pressure(match)


# Fitting/development helper: nothing on the predict.py path calls this.
def fit(parts):
    """Fit the selected linear ridge model from external candidate examples."""
    features, valid, gain = (np.concatenate(rows) for rows in zip(*parts, strict=True))
    center = features.mean(0)
    scale = features.std(0).clip(0.04)
    train = np.asarray(((features - center) / scale).clip(-6, 6), dtype=np.float64)
    targets = np.column_stack((gain, valid)).astype(np.float64)
    intercept = targets.mean(0)
    weights = np.linalg.solve(
        train.T @ train + 16 * np.eye(train.shape[1]), train.T @ (targets - intercept)
    )
    return dict(center=center, scale=scale, intercept=intercept, weights=weights)


def load(path: Path):
    with np.load(path, allow_pickle=False) as model:
        return {
            name: model[name] for name in ("center", "scale", "intercept", "weights")
        }


def predict(reference, match, typed_pairs, tactics, model, *, prepared=None):
    """Apply the selected linear correction to the preceding stoppage timeline."""
    if not typed_pairs:
        raise ValueError("At least one typed forecast pair is required")
    track, pressure = feature_inputs(match) if prepared is None else prepared
    bank = _candidates(reference, match["period"], track, pressure, typed_pairs)
    result = reference.copy()
    if not bank:
        return result
    features = np.asarray([row[-1] for row in bank], dtype=np.float32)
    normalized = np.asarray(
        ((features - model["center"]) / model["scale"]).clip(-6, 6), dtype=np.float64
    )
    gain, chance = (normalized @ model["weights"] + model["intercept"]).T
    proposals = []
    weights = []
    for i, ((_, start, stop, _, side), previous, _, count, _, _) in enumerate(bank):
        if count != 2 or gain[i] <= 0.095 or chance[i].clip(0, 1) > 0.30:
            continue
        if (tactics[start:stop, 1 - side] - tactics[start:stop, side]).mean() <= 0.06:
            continue
        proposals.append((previous[0][0], previous[-1][1], start, stop, side))
        weights.append(float(gain[i] - 0.095))
    for i in nonoverlapping(proposals, weights, range(len(proposals))):
        _, _, start, stop, side = proposals[i]
        result[start:stop] = 1 - side
    return result


def _removed_edges(base, period, proposal):
    cuts = np.r_[
        0,
        np.flatnonzero((base[1:] != base[:-1]) | (period[1:] != period[:-1])) + 1,
        len(base),
    ]
    removed = {}
    covered = np.zeros(len(base), dtype=bool)
    for lo, hi in zip(cuts[:-1], cuts[1:], strict=True):
        if base[lo] >= 2 or np.array_equal(base[lo:hi], proposal[lo:hi]):
            continue
        if lo == 0 or hi == len(base) or not np.all(proposal[lo:hi] == 1 - base[lo]):
            raise ValueError("A proposed removal must replace one complete live spell")
        if base[lo - 1] == 2:
            anchor = hi
        elif base[hi] == 2:
            anchor = lo - 1
        else:
            raise ValueError("A proposed removal must lie at a stoppage edge")
        removed[int(lo), int(hi)] = int(proposal[lo]), int(anchor)
        covered[lo:hi] = True
    if not np.array_equal(covered, base != proposal):
        raise ValueError(
            "The candidate timeline changed a frame outside a live stoppage edge"
        )
    return removed


def agree_stoppage_edges(reference, period, role_adjusted, independently_fitted):
    """Keep complete spell removals that both stop-edge fits chose."""
    left = _removed_edges(reference, period, role_adjusted)
    right = _removed_edges(reference, period, independently_fitted)
    prediction = reference.copy()
    selected = []
    for (lo, hi), (side, anchor) in sorted(left.items()):
        if right.get((lo, hi)) != (side, anchor):
            continue
        if any(a <= anchor < b or lo <= other < hi for a, b, other in selected):
            continue
        if reference[anchor] != side:
            raise ValueError("Proposed possession disagrees with the adjacent team")
        prediction[lo:hi] = side
        selected.append((lo, hi, anchor))
    return prediction
