"""Recover missed live takeovers at the end of a spell."""

import numpy as np

from datalake.evaluate import pm


# Candidate takeover lengths in frames: each proposes giving the last `length` frames of a
# team spell that ends in a stoppage to the rival. Ascending: the loop breaks at the first
# length that would leave the original team fewer than 13 frames.
WIDTHS = (10, 16, 23, 32, 44, 60, 82, 105)


# Charge that targets() subtracts from the mean change in exact["counts"] produced by a
# candidate's patch; the second score column is 1 only when that mean exceeds PRICE (else 0).
# With counts() below as exact["counts"], the mean is over pm.SPELL_IOUS of the change in
# matched ground-truth spells.
PRICE = 0.40


def spell_features(reference, period, track, members, edge, cuts):
    endpoints = cuts(reference, period)
    states = reference[endpoints[:-1]]
    odds = np.log(members[:, :, 0].clip(1e-7)) - np.log(members[:, :, 1].clip(1e-7))
    # track columns follow features.FEATURES: 18 ball_speed_mps, 19 ball_missing, 21/22
    # nearest_home_m/nearest_away_m (so 21 + rival is the rival side). margin is away minus home
    # and zero while the ball is missing: positive when the home side is nearer a tracked ball.
    visibility = track[:, 19] == 0
    margin = (track[:, 22] - track[:, 21]) * visibility
    rows, features = [], []
    for i in np.flatnonzero((states[:-1] < pm.D) & (states[1:] == pm.D)):
        lo, hi, stopped = map(int, endpoints[i : i + 3])
        if hi - lo < 24 or stopped - hi < 9 or period[lo] != period[stopped - 1]:
            continue
        side = int(states[i])
        rival = 1 - side
        sign = 1 - 2 * rival
        for length in WIDTHS:
            a = hi - length
            if a - lo < 13:
                break
            mid = (a + hi) // 2
            sections = ((a - 10, a), (a, mid), (mid, hi), (hi, hi + 9))
            forecasts = [sign * odds[:, u:v].mean(1) for u, v in sections]
            live = (1 - members[:, a:hi, 2]).mean(axis=1)
            cue = [edge[u - 4 : u + 5, col].max() for u, col in ((a, side), (hi, 4 + rival))]
            x = np.r_[
                np.log1p(length),
                np.log1p(a - lo),
                length / (hi - lo),
                np.log1p(stopped - hi),
                *forecasts,
                live,
                cue,
            ]
            # Tracking enters only the joint models.
            motion = []
            for u, v in sections:
                visible = visibility[u:v]
                motion.extend(
                    [
                        visible.mean(),
                        sign * margin[u:v].mean(),
                        np.mean((sign * margin[u:v] > 1.0) & visible),
                        np.mean((track[u:v, 21 + rival] < 1.6) & visible),
                        np.mean(track[u:v, 18] * visible),
                    ]
                )
            rows.append((lo, hi, a, rival))
            features.append(np.r_[x, motion])
    return rows, np.asarray(features, dtype=float).reshape(-1, 41)


# Fitting/development helper: nothing on the predict.py path calls this.
def targets(reference, truth, period, bank, exact):
    ranges = [
        (lo + a, lo + b)
        for lo, hi in _period_bounds(period)
        for a, b in pm.valid_runs(truth[lo:hi])
    ]
    actual, original = (pm.possession_spells(state, ranges) for state in (truth, reference))
    pairs = pm.spell_overlaps(actual, original)
    before = exact["counts"](actual, pairs)
    indices = {spell: i for i, spell in enumerate(original)}
    selected = []
    scores = []
    for i, (lo, hi, a, rival) in enumerate(bank):
        old = indices.get((lo, hi, 1 - rival))
        if old is None:
            continue
        patch = [(lo, a, 1 - rival), (a, hi, rival)]
        overlaps = exact["official_pairs"](actual, pairs, patch, old, old + 1)
        value = (exact["counts"](actual, overlaps) - before).mean() - PRICE
        selected.append(i)
        scores.append((value, value > 0))
    return np.asarray(selected, dtype=int), np.asarray(scores, dtype=float).reshape(-1, 2)


def describe(reference, period, track, streams, edge, movement, cut):
    rows, x = spell_features(reference, period, track, streams, edge, cut)
    additions = []
    for lo, stop, start, side in rows:
        middle = (start + stop) // 2
        spans = ((max(lo, start - 25), start), (start, middle), (middle, stop), (stop, stop + 9))
        vector = []
        for a, b in spans:
            # All four windows lie in the live passage or the opening stoppage.
            sample = movement[a:b].mean(axis=0)
            vector.extend([*(sample[side] - sample[1 - side]), *sample.mean(axis=0)])
        additions.append(np.r_[track[start:stop, 19].mean(), vector])
    return rows, np.column_stack((x, np.asarray(additions, dtype=float).reshape(-1, 73)))


def counts(truth, pairs):
    return np.array(
        [pm.matched_gt_flags(pairs, len(truth), cutoff).sum() for cutoff in pm.SPELL_IOUS]
    )


def official_pairs(truth, original_pairs, new, start, stop):
    delta = len(new) - (stop - start)
    kept = [
        (q, i, j + (delta if j >= stop else 0))
        for q, i, j in original_pairs
        if j < start or j >= stop
    ]
    patched = kept + [(q, i, start + j) for q, i, j in pm.spell_overlaps(truth, new)]
    return sorted(patched, key=lambda row: (-row[0], row[1], row[2]))


def _period_bounds(period: np.ndarray) -> list[tuple[int, int]]:
    changes = (np.flatnonzero(period[1:] != period[:-1]) + 1).tolist()
    edges = [0, *changes, len(period)]
    return list(zip(edges[:-1], edges[1:]))


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


# Fitting/development helper: nothing on the predict.py path calls this.
def fit_parameters(groups):
    design, target = (np.concatenate(parts) for parts in zip(*groups, strict=True))
    weights = np.concatenate([np.full(len(y), 1 / len(y) / len(groups)) for _, y in groups])
    center = weights @ design
    scale = np.sqrt(weights @ ((design - center) ** 2)).clip(0.05)
    x = np.column_stack((np.ones(len(design)), ((design - center) / scale).clip(-6, 6)))
    penalty = 0.12 * np.eye(x.shape[1])
    penalty[0, 0] = 1e-6
    credit = np.linalg.solve((x.T * weights) @ x + penalty, x.T @ (weights * target[:, 0]))
    odds = logistic(x, weights, target[:, 1], penalty * 0.6)
    return dict(
        center=center.tolist(), scale=scale.tolist(), credit=credit.tolist(), odds=odds.tolist()
    )


def predictor(parameters, feature_count=114):
    center, scale, credit, odds = (
        np.asarray(parameters[name], dtype=float) for name in ("center", "scale", "credit", "odds")
    )
    if (
        center.shape != (feature_count,)
        or scale.shape != center.shape
        or credit.shape != (feature_count + 1,)
        or odds.shape != credit.shape
        or not all(np.isfinite(value).all() for value in (center, scale, credit, odds))
        or np.any(scale <= 0)
    ):
        raise ValueError("Invalid stoppage correction weights")

    def predict(features):
        values = np.column_stack(
            (np.ones(len(features)), ((features - center) / scale).clip(-6, 6))
        )
        confidence = 1 / (1 + np.exp(-(values @ odds).clip(-18, 18)))
        return np.column_stack((values @ credit, confidence))

    return predict


def apply(reference, bank, features, model, tactics):
    revised = reference.copy()
    score = model(features)
    odds = np.log(tactics[:, 0].clip(1e-7)) - np.log(tactics[:, 1].clip(1e-7))
    best = {}
    for (start, stop, proposed, team), (credit, confidence), row in zip(
        bank, score, features, strict=True
    ):
        if credit < -0.13 or confidence < 0.15 or proposed - start < 19 or row[41] < 0.60:
            continue
        sign = 1 - 2 * team
        if (
            sign * odds[proposed:stop].mean() < 0.10
            or sign * odds[proposed - 10 : proposed].mean() > 0.12
        ):
            continue
        candidate = float(credit + 0.10 * confidence), proposed, stop, team
        if candidate > best.get(stop, (-np.inf, 0, 0, 0)):
            best[stop] = candidate
    edits = sorted(best.values(), reverse=True)[:2]
    for _, a, b, team in edits:
        assert np.all(reference[a:b] == 1 - team)
        revised[a:b] = team
    np.testing.assert_array_equal(revised == pm.D, reference == pm.D)
    return revised, len(edits)


def predict_suffix(reference, period, tracking, members, edges, movement, tactics, correction):
    """Apply the final takeover correction to a frame-aligned incoming timeline."""
    n = len(period)
    if (
        reference.shape != (n,)
        or tracking.shape != (n, 25)
        or members.shape != (3, n, 3)
        or edges.shape != (n, 6)
        or movement.shape != (n, 2, 9)
        or tactics.shape != (n, 3)
    ):
        raise ValueError(
            "The incoming timeline, tracking features and forecasts must share a frame grid"
        )

    def cuts(labels, periods):
        changes = (labels[1:] != labels[:-1]) | (periods[1:] != periods[:-1])
        return np.r_[0, np.flatnonzero(changes) + 1, len(labels)]

    rows, features = describe(reference, period, tracking, members, edges, movement, cuts)
    prediction, edits = apply(reference, rows, features, correction, tactics)
    if len(cuts(prediction, period)) - len(cuts(reference, period)) != edits:
        raise ValueError("The correction produced an inconsistent number of possession spells")
    return prediction
