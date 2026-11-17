"""Use model agreement to restore live play inside a short stoppage."""

import numpy as np

from datalake.core import D


def apply(reference, period, forecasts, tactics, cuts):
    """Return the forecast-control timeline.

    Forecasts are ordered by family, member, frame and H/A/D probability.
    """
    if (forecasts.ndim != 4 or forecasts.shape[0] != 3 or forecasts.shape[1] == 0
            or forecasts.shape[2:] != (len(reference), 3)):
        raise ValueError("Expected three aligned families of possession forecasts")
    result = reference.copy()
    if not len(result):
        return result
    members = forecasts.reshape(-1, len(reference), 3)
    bounds = cuts(reference, period)
    sides = reference[bounds[:-1]]
    live = np.mean(members[:, :, :2].sum(axis=2), axis=0)
    odds = np.log(members[:, :, 0].clip(1e-6)) - np.log(members[:, :, 1].clip(1e-6))
    tactics_odds = np.log(tactics[:, 0].clip(1e-6)) - np.log(tactics[:, 1].clip(1e-6))
    state = np.mean(odds, axis=0) + 0.35 * tactics_odds
    for i in range(1, len(sides) - 1):
        before, dead, after = map(int, sides[i - 1:i + 2])
        if dead != D or max(before, after) >= D:
            continue
        lo, start, stop, hi = map(int, bounds[i - 1:i + 3])
        # Both neighbouring live spells must last at least 6 frames, and the stoppage 8 to 100.
        if min(start - lo, hi - stop) < 6 or not 8 <= stop - start <= 100:
            continue
        # stop is exclusive, so these are the 5th frame before and the 5th frame after the stoppage;
        # skip when they fall in different periods.
        if period[start - 5] != period[stop + 4]:
            continue
        segment = slice(start, stop)
        support = forecasts[:, :, segment, :2].mean(axis=2).sum(axis=2).mean(axis=1)
        alive = live[segment].mean()
        if support.min() < 0.48 or alive < 0.60 or tactics[segment, :2].sum(axis=1).mean() < 0.50:
            continue
        dominant = int(state[segment].mean() < 0)
        if before == after and dominant == before:
            result[segment] = before
        if before != after and alive >= 0.66:
            ownership = (1 - 2 * before) * state[segment]
            switch = start + int(np.argmax(np.r_[0.0, np.cumsum(ownership)]))
            result[start:switch] = before
            result[switch:stop] = after
    return result
