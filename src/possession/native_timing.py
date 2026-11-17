"""Shift close home-away boundaries using a fitted match-level timing table."""

import json
from pathlib import Path

import numpy as np

from datalake.core import D


def sites(reference, period, tracking):
    changes = (reference[1:] != reference[:-1]) | (period[1:] != period[:-1])
    cuts = np.r_[0, np.flatnonzero(changes) + 1, len(reference)]
    teams = reference[cuts[:-1]]
    for i in range(1, len(teams)):
        lo, at, hi = map(int, cuts[i - 1:i + 2])
        old, new = map(int, teams[i - 1:i + 1])
        short, long = sorted((at - lo, hi - at))
        if max(old, new) >= D or not 8 <= short <= 160 or period[lo] != period[hi - 1]:
            continue
        # Column 19 of the feature table is ball_missing (see features.FEATURES), so this is the
        # share of frames within 10 of the boundary, inside the two spells, with the ball observed.
        visibility = (tracking[max(lo, at - 10):min(hi, at + 10), 19] == 0).mean()
        # Ball-visibility group: 0 = at least 65% of frames, 1 = 25% up to 65%, 2 = under 25%.
        visibility_group = 0 if visibility >= 0.65 else 1 if visibility >= 0.25 else 2
        # Spell-length group: 1 = both adjacent spells under 65 frames; otherwise 0 = the earlier
        # spell is the shorter one, 2 = the later spell is shorter or the two are equal.
        duration_group = 1 if long < 65 else 0 if at - lo < hi - at else 2
        yield at, old, new, (visibility_group, duration_group)


def load_table(path: Path):
    table = {}
    for visibility, duration, gain, shift, agreement in json.loads(path.read_text())["groups"]:
        if (visibility not in (0, 1, 2) or duration not in (0, 1, 2)
                or not np.isfinite(gain) or shift not in range(-4, 5) or agreement not in range(6)):
            raise ValueError("Invalid fitted boundary timing group")
        table[visibility, duration] = gain, shift, agreement
    return table


def apply(reference, period, tracking, table):
    """Keep a small boundary shift when three fitting matches support its direction."""
    revised = reference.copy()
    if not len(reference):
        return revised, 0
    proposals = []
    for at, old, new, group in sites(reference, period, tracking):
        gain, shift, agreement = table.get(group, (0.0, 0, 0))
        if gain <= 0.008 or not shift or agreement < 3:
            continue
        start, stop = sorted((at, at + shift))
        proposals.append((gain, start, stop, new if shift < 0 else old))
    selected = []
    for _, start, stop, side in sorted(proposals, reverse=True):
        if any(start < right + 2 and stop > left - 2 for left, right in selected):
            continue
        revised[start:stop] = side
        selected.append((start, stop))
    return revised, len(selected)
