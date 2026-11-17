"""Remove short false takeovers with observed support for the original side."""

from bisect import bisect_left

import numpy as np

from datalake.evaluate import pm

from .stoppage import _period_bounds, counts, official_pairs


def nonoverlapping(paths, weights, indices):
    """Select positive-weight intervals with no overlap or shared endpoint."""
    order = sorted((i for i in indices if weights[i] > 1e-7), key=lambda i: paths[i][1])
    ends = [paths[i][1] for i in order]
    maxima = [0.0]
    backtrack = []
    for cursor, i in enumerate(order):
        previous = bisect_left(ends, paths[i][0], hi=cursor)
        utility = weights[i] + maxima[previous]
        wins = utility > maxima[-1]
        backtrack.append((wins, previous))
        maxima.append(utility if wins else maxima[-1])
    selected = []
    cursor = len(order)
    while cursor:
        wins, previous = backtrack[cursor - 1]
        if wins:
            selected.append(order[cursor - 1])
        cursor = previous if wins else cursor - 1
    return selected


def proposals(reference, period, tracking, members):
    if len(reference) == 0:
        return [], np.empty((0, 30))
    changes = (reference[1:] != reference[:-1]) | (period[1:] != period[:-1])
    stops = np.r_[0, np.flatnonzero(changes) + 1, len(reference)]
    codes = reference[stops[:-1]]
    odds = np.log(members[:, :, 0].clip(1e-7)) - np.log(members[:, :, 1].clip(1e-7))
    # tracking columns follow features.FEATURES: 19 ball_missing, 21/22 nearest_home_m/
    # nearest_away_m, 23/24 nearest_home_missing/nearest_away_missing. distance is away minus
    # home, so it is positive when the home side is nearer the ball.
    visible = (tracking[:, 19] == 0) & (tracking[:, 23:25] == 0).all(axis=1)
    distance = tracking[:, 22] - tracking[:, 21]
    rows, features = [], []
    for i in range(1, len(codes) - 1):
        lo, start, stop, hi = map(int, stops[i - 1:i + 3])
        rival = int(codes[i])
        if rival >= pm.D or not 9 <= stop - start <= 105 or period[lo] != period[hi - 1]:
            continue
        host = 1 - rival
        lead, trail = int(codes[i - 1]), int(codes[i + 1])
        if lead not in (host, pm.D) or trail not in (host, pm.D) or lead == trail == pm.D:
            continue
        first = start if lead == pm.D else lo
        last = stop if trail == pm.D else hi
        prior_dead = i >= 2 and codes[i - 2] == pm.D and start - lo < 110
        next_dead = i + 2 < len(codes) and codes[i + 2] == pm.D and hi - stop < 110
        near_stoppage = lead == pm.D or trail == pm.D or prior_dead or next_dead
        replaced = []
        if lead == host:
            replaced.append((lo, start, host))
        replaced.append((start, stop, rival))
        if trail == host:
            replaced.append((stop, hi, host))
        if any(b - a < 7 for a, b, _ in replaced):
            continue
        sign = 1 - 2 * host
        middle = (start + stop) // 2
        spans = ((max(lo, start - 10), start), (start, middle),
                 (middle, stop), (stop, min(hi, stop + 10)))
        vector = [np.log1p(stop - start), np.log1p(start - first), np.log1p(last - stop),
                  lead == pm.D, trail == pm.D, near_stoppage]
        for a, b in spans:
            active = visible[a:b] & (period[a:b] == period[start])
            rival_distance, host_distance = tracking[a:b, 21 + rival], tracking[a:b, 21 + host]
            contact = active & (rival_distance < 1.6) & (host_distance > rival_distance + .30)
            vector.extend((sign * odds[:, a:b].mean(axis=1)).tolist())
            vector.extend([float(active.mean()), float(np.mean(sign * distance[a:b] * active)),
                           float(contact.mean())])
        rows.append((first, last, start, stop, host, replaced, near_stoppage))
        features.append(vector)
    return rows, np.asarray(features, dtype=float).reshape(-1, 30)


# Fitting/development helper: nothing on the predict.py path calls this.
def fit_targets(reference, truth, period, rows):
    ranges = [(lo + a, lo + b)
              for lo, hi in _period_bounds(period) for a, b in pm.valid_runs(truth[lo:hi])]
    gold, predicted = (pm.possession_spells(labels, ranges) for labels in (truth, reference))
    overlaps = pm.spell_overlaps(gold, predicted)
    initial = counts(gold, overlaps)
    positions = {tuple(spell): i for i, spell in enumerate(predicted)}
    examples = []
    for i, (lo, hi, _, _, host, replaced, _) in enumerate(rows):
        position = positions.get(replaced[0])
        if position is None or any(positions.get(spell) != position + j for j, spell in enumerate(replaced)):
            continue
        updated = official_pairs(gold, overlaps, [(lo, hi, host)], position, position + len(replaced))
        # One spell replaces len(replaced) predicted spells, so len(replaced) - 1 spells are
        # removed; each removal is credited .40, the same magnitude stoppage.PRICE charges for
        # adding one spell.
        credit = (counts(gold, updated) - initial).mean() + .40 * (len(replaced) - 1)
        examples.append((i, credit, credit > 0))
    return np.asarray(examples, dtype=float).reshape(-1, 3)


def predict_interruption(reference, period, tracking, members, tactics, models):
    """Apply the strict, short-visible-interruption policy to the incoming timeline."""
    n = len(period)
    if (reference.shape != (n,) or tracking.shape != (n, 25)
            or members.shape != (3, n, 3) or tactics.shape != (n, 3)):
        raise ValueError("The incoming timeline, tracking features and forecasts must share a frame grid")
    rows, features = proposals(reference, period, tracking, members)
    result = reference.copy()
    if not rows:
        return result
    natural, mixed = models
    utility, confidence = np.minimum(natural(features), mixed(features)).T
    odds = np.log(tactics[:, 0].clip(1e-7)) - np.log(tactics[:, 1].clip(1e-7))
    accepted = []
    for i in np.flatnonzero((utility > .08) & (confidence > .43)):
        _, _, start, stop, host, _, near_stoppage = rows[i]
        if not near_stoppage or stop - start > 46:
            continue
        if (1 - 2 * host) * odds[start:stop].mean() < .18:
            continue
        if max(features[i, 17], features[i, 23]) > .035 or min(features[i, 15], features[i, 21]) < .35:
            continue
        accepted.append(i)
    for i in nonoverlapping(rows, utility - .08, accepted):
        _, _, start, stop, host, _, _ = rows[i]
        result[start:stop] = host
    return result
