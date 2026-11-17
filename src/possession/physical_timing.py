"""Correct short ownership around observed ball contact and restarts."""

import numpy as np

from datalake.core import D
from datalake.remap_touch import _contact, _restart
from .interruption import nonoverlapping
from .stoppage_kernel import contact_evidence, receiver_motion


def _cuts(reference, period):
    changes = (reference[1:] != reference[:-1]) | (period[1:] != period[:-1])
    return np.r_[0, np.flatnonzero(changes) + 1, len(reference)]


def _slots(evidence, lo, hi, side):
    near = (evidence["dist"][side][lo:hi] < 2.5) & evidence["seen"][side][lo:hi]
    return set(evidence["slot"][side][lo:hi][near])


def _contact_gate(evidence, lo, hi, side):
    valid = evidence["seen"][side][lo:hi] | (evidence["dist"][side][lo:hi] > 1.5)
    if not valid.all():
        return False
    dwell, distinct = _contact(evidence, lo, hi, side)
    return (
        dwell < 13
        and distinct <= 1
        and receiver_motion(evidence, lo, hi, side)[0] < 0.18
    )


def _strict_splits(reference, period, evidence):
    cuts = _cuts(reference, period)
    teams = reference[cuts[:-1]]
    for i, (host, side, after) in enumerate(
        zip(teams[:-2], teams[1:-1], teams[2:], strict=True)
    ):
        left, lo, hi, right = cuts[i : i + 4]
        if (
            side >= D
            or host != 1 - side
            or after != host
            or period[left] != period[right - 1]
        ):
            continue
        if not 8 <= hi - lo < 50 or min(lo - left, right - hi) < 8:
            continue
        if not evidence["ball_seen"][lo:hi].all():
            continue
        observed = evidence["seen"][side][lo:hi] & evidence["seen"][host][lo:hi]
        if observed.mean() < 0.7 or not _contact_gate(evidence, lo, hi, side):
            continue
        inside = _slots(evidence, lo, hi, host)
        shoulders = _slots(evidence, max(left, lo - 20), lo, host) | _slots(
            evidence, hi, min(right, hi + 20), host
        )
        if inside & shoulders:
            yield (int(left), int(right), int(lo), int(hi), int(side))


def _restart_splits(reference, period, evidence):
    cuts = _cuts(reference, period)
    teams = reference[cuts[:-1]]
    for i in range(len(teams) - 3):
        host, side, stopped, after = teams[i : i + 4]
        left, lo, hi, resume, right = cuts[i : i + 5]
        if (
            side >= D
            or host != 1 - side
            or stopped != D
            or after != host
            or period[left] != period[right - 1]
        ):
            continue
        if not 8 <= hi - lo < 50 or min(lo - left, right - resume) < 8:
            continue
        if (
            not evidence["ball_seen"][lo:hi].all()
            or _restart(evidence, resume) != "other"
        ):
            continue
        restart_near = evidence["seen"][host][resume : resume + 5] & (
            evidence["dist"][host][resume : resume + 5] < 2.5
        )
        if restart_near.sum() < 3 or not _contact_gate(evidence, lo, hi, side):
            continue
        if _slots(evidence, max(left, lo - 20), lo, host) & _slots(
            evidence, lo, hi, host
        ):
            yield (int(left), int(resume), int(lo), int(hi), int(side))


def _pick(candidates):
    chosen = nonoverlapping(candidates, [1.0] * len(candidates), range(len(candidates)))
    return [candidates[index] for index in chosen]


def apply_restart_context(reference, match, *, evidence=None):
    evidence = contact_evidence(match) if evidence is None else evidence
    period = match["period"]
    strict = _pick(list(_strict_splits(reference, period, evidence)))
    restart = [
        row
        for row in _restart_splits(reference, period, evidence)
        if not any(row[0] <= chosen[1] and chosen[0] <= row[1] for chosen in strict)
    ]
    result = reference.copy()
    for _, _, lo, hi, side in [*strict, *_pick(restart)]:
        result[lo:hi] = 1 - side
    return result


def _contact_controls(evidence):
    ready, close = [], []
    for side in (0, 1):
        mine = evidence["seen"][side] & evidence["ball_seen"]
        opponent = evidence["seen"][1 - side]
        distance = evidence["dist"][side]
        rival = evidence["dist"][1 - side]
        close.append(mine & (distance < 1.5))
        ready.append(mine & opponent & (distance < 1.5) & (distance + 0.35 < rival))
    return ready, close


def _events(evidence, room, ready, close):
    lo, boundary, hi, before, after = room
    seen = evidence["ball_seen"]
    slots = evidence["slot"][after]
    options = []
    for i in range(max(lo + 10, boundary - 20), min(hi - 13, boundary + 20) + 1):
        if not ready[after][i] or ready[after][i - 3 : i].sum() > 1:
            continue
        # Start one frame earlier when team `after` was already ready on the previous frame.
        first = i - int(ready[after][i - 1])
        if first < lo + 10 or not seen[first - 10 : first + 13].all():
            continue
        stable = (
            ready[after][first : first + 13] & (slots[first : first + 13] == slots[i])
        ).sum()
        if stable < 10 or close[before][first : first + 5].sum() > 2:
            continue
        host = close[before][max(lo, first - 14) : first].sum()
        if host >= 6:
            options.append((int(first), int(stable), int(host)))
    return options


def apply_first_contact(reference, match, *, evidence=None):
    evidence = contact_evidence(match) if evidence is None else evidence
    period = match["period"]
    cuts = _cuts(reference, period)
    teams = reference[cuts[:-1]]
    ready, close = _contact_controls(evidence)
    result = reference.copy()
    for i, (before, after) in enumerate(zip(teams[:-1], teams[1:], strict=True)):
        lo, boundary, hi = cuts[i : i + 3]
        if max(before, after) >= D or before == after or period[lo] != period[hi - 1]:
            continue
        if min(boundary - lo, hi - boundary) < 16:
            continue
        options = _events(evidence, (lo, boundary, hi, before, after), ready, close)
        if not options:
            continue
        first, _, _ = max(
            options,
            key=lambda x: (
                x[1] + 0.30 * x[2] - 0.10 * abs(x[0] - boundary),
                -abs(x[0] - boundary),
            ),
        )
        at = int(
            np.clip(first, max(lo + 14, boundary - 12), min(hi - 14, boundary + 12))
        )
        if boundary - at >= 3:
            result[at:boundary] = after
        if at - boundary >= 3:
            result[boundary:at] = before
    return result
