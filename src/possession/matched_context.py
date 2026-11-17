"""Recover missed possession spells by comparing proposals with external training examples."""

from pathlib import Path

import numpy as np

from datalake.core import D


# Columns of describe()'s 26-wide rows compared in the neighbour search: the three log lengths
# (0-2) and the 4 spans x 5 series block (6-25). Left out: 3 = ending flag (used separately by
# _neighbours), 4 = scaled log rank, 5 = proposal credit.
CONTEXT_COLUMNS = [0, 1, 2, *range(6, 26)]


def describe(reference, period, tracking, proposals, forecast, direct):
    """Describe hidden-ball proposals within the current possession timeline."""
    changes = (reference[1:] != reference[:-1]) | (period[1:] != period[:-1])
    cuts = np.r_[0, np.flatnonzero(changes) + 1, len(reference)]
    base_odds = (np.log(forecast[:, 0].clip(1e-7)) -
                 np.log(forecast[:, 1].clip(1e-7))).clip(-6, 6)
    direct_odds = (np.log(direct[:, 0].clip(1e-7)) -
                   np.log(direct[:, 1].clip(1e-7))).clip(-6, 6)
    # tracking columns as named in features.FEATURES: 19 = ball_missing,
    # 21/22 = nearest_home_m/nearest_away_m, 23/24 = nearest_home_missing/nearest_away_missing.
    visible = tracking[:, 19] == 0
    observed = visible & (tracking[:, 23:25] == 0).all(axis=1)
    distance = np.where(observed, tracking[:, 22] - tracking[:, 21], 0.0)
    series = np.column_stack((base_odds, direct_odds, forecast[:, 2], visible, distance))
    prefix = np.pad(series.cumsum(axis=0, dtype=float), ((1, 0), (0, 0)))
    rows, features = [], []
    for rank, (start, stop, team, credit) in enumerate(proposals[:850]):
        a, b, side = map(int, (start, stop, team))
        if (a < 12 or b + 12 >= len(period) or not 10 <= b - a <= 115
                or period[a] != period[b - 1] or reference[a] != 1 - side):
            continue
        position = cuts.searchsorted(a, side="right") - 1
        lo, hi = map(int, cuts[position:position + 2])
        if a - lo < 14 or hi - b < -11:
            continue
        ending = hi < len(reference) and reference[hi] == D and abs(b - hi) <= 12
        b = hi if ending else b
        if (not ending and hi - b < 14) or b - a < 10 or visible[a:b].mean() > 0.56:
            continue
        middle = (a + b) // 2
        spans = ((a - 10, a), (a, middle), (middle, b), (b, hi if ending else b + 10))
        block = np.zeros((4, 5))
        for i, (u, v) in enumerate(spans):
            if u < v:
                block[i] = (prefix[v] - prefix[u]) / (v - u)
        block[:, [0, 1, 4]] *= 1 - 2 * side
        rows.append((lo, hi, a, b, side))
        features.append([np.log1p(b - a), np.log1p(a - lo), np.log1p(hi - b),
                         float(ending), np.log1p(rank) / 7.0, credit, *block.ravel()])
    return np.asarray(rows, dtype=int).reshape(-1, 5), np.asarray(features, dtype=float).reshape(-1, 26)


def _transform(features, center, spread):
    values = ((features[:, CONTEXT_COLUMNS] - center) / spread).clip(-4, 4)
    values[:, :3] *= 0.8
    return values


# Fitting/development helper: nothing on the predict.py path calls this.
def build_index(batches):
    """Prepare an example index with host identities scoped to each fitting match."""
    descriptors = np.concatenate([features[:, CONTEXT_COLUMNS] for _, features, _ in batches])
    center = np.median(descriptors, axis=0)
    spread = np.maximum(np.quantile(descriptors, 0.80, axis=0) -
                        np.quantile(descriptors, 0.20, axis=0), 0.12)
    hosts = [np.unique(rows[:, :2], axis=0, return_inverse=True)[1] for rows, _, _ in batches]
    return {
        "center": center,
        "spread": spread,
        "offsets": np.r_[0, np.cumsum([len(rows) for rows, _, _ in batches])],
        "host": np.concatenate(hosts),
        "vectors": np.concatenate([_transform(features, center, spread) for _, features, _ in batches]),
        "gain": np.concatenate([gain for _, _, gain in batches]),
        "ending": np.concatenate([features[:, 3].astype(bool) for _, features, _ in batches]),
    }


def load_index(path: Path):
    """Load a numeric example index exported from external development matches."""
    with np.load(path, allow_pickle=False) as model:
        index = {name: model[name] for name in
                 ("center", "spread", "offsets", "host", "vectors", "gain", "ending")}
    center, spread, offsets, host, vectors, gain, ending = index.values()
    size = len(host)
    width = len(CONTEXT_COLUMNS)
    if (center.shape != (width,) or spread.shape != center.shape or np.any(spread <= 0)
            or vectors.shape != (size, width) or host.shape != (size,) or gain.shape != host.shape
            or ending.shape != host.shape or offsets.ndim != 1 or len(offsets) < 2
            or offsets.dtype.kind not in "iu" or offsets[0] != 0 or offsets[-1] != size
            or np.any(np.diff(offsets) < 0)
            or not all(np.isfinite(array).all() for array in (center, spread, vectors, gain))):
        raise ValueError("Invalid fitted context index")
    return index


def _neighbours(test, ending, vectors, host_ids, gain, example_endings):
    """Return expected gain, positive support and nearest distance for each fitting match."""
    distances = ((test[:, None] - vectors[None, :]) ** 2).mean(axis=2)
    result = np.full((len(test), 3), np.nan)
    for i, distance in enumerate(distances):
        # Try the matching ending first, then fill from the remaining hosts.
        order = np.argsort(np.where(ending[i] == example_endings, distance, np.inf), kind="stable")
        used = set()
        neighbours = []
        for j in order:
            if not np.isfinite(distance[j]):
                break
            if host_ids[j] in used:
                continue
            used.add(host_ids[j])
            neighbours.append(j)
            if len(neighbours) == 5:
                break
        if len(neighbours) >= 3:
            weights = np.exp(-np.minimum(distance[neighbours], 4.0) * 3)
            observed = gain[neighbours]
            result[i] = np.average(observed, weights=weights), np.average(observed > 0, weights=weights), distance[neighbours[0]]
    return result


def retrieve(features, index):
    test = _transform(features, index["center"], index["spread"])
    offsets = index["offsets"]
    votes = []
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        segment = slice(start, stop)
        votes.append(_neighbours(test, features[:, 3], index["vectors"][segment], index["host"][segment],
                                index["gain"][segment], index["ending"][segment]))
    results = np.stack(votes)
    expected, positive, closest = results.transpose(2, 0, 1)
    available = np.isfinite(expected).sum(axis=0)
    divisor = available.clip(1)
    return np.column_stack((np.nansum(expected, axis=0) / divisor,
                            np.nansum(positive, axis=0) / divisor,
                            np.nansum(closest, axis=0) / divisor,
                            (expected > 0).sum(axis=0), available))


def _repeated(candidate, other_rows):
    _, _, start, stop, side = candidate
    bounds, team = other_rows[:, 2:4], other_rows[:, -1]
    intersection = np.minimum(bounds[:, 1], stop) - np.maximum(bounds[:, 0], start)
    union = np.maximum(bounds[:, 1], stop) - np.minimum(bounds[:, 0], start)
    return np.any((team == side) & (intersection >= 0.60 * union))


def apply(reference, groups, index):
    """Apply the repeated-context selection to candidate banks on the input timeline."""
    offers = {}
    banks = [rows for rows, _ in groups]
    for rows, features in groups:
        scores = retrieve(features, index)
        for row, (credit, positive, distance, _, available) in zip(rows, scores, strict=True):
            if available < 3 or credit <= -0.12 or distance > 0.65:
                continue
            repeated = sum(_repeated(row, other) for other in banks)
            if repeated < 2:
                continue
            lo, hi, a, b, side = map(int, row)
            value = credit + 0.12 * positive - 0.075 * distance + 0.022 * repeated
            offers.setdefault((lo, hi), []).append((value, a, b, side))
    chosen = sorted((max(options) for options in offers.values()), reverse=True)[:2]
    revised = reference.copy()
    for _, start, stop, side in chosen:
        if not np.all(reference[start:stop] == 1 - side):
            raise ValueError("Context proposal does not lie within one opposing possession")
        revised[start:stop] = side
    return revised, len(chosen)
