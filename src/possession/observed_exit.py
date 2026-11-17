"""Adjust possession near observed ball exits."""

import numpy as np

from datalake.core import D
from evaluation import possession_metrics as metrics
from .decoding import decode


def _inventory(labels, period):
    changes = (
        np.flatnonzero((labels[1:] != labels[:-1]) | (period[1:] != period[:-1])) + 1
    )
    cuts = np.r_[0, changes, len(labels)]
    return cuts[:-1], cuts[1:], labels[cuts[:-1]]


def _clip_live(spells, start, stop):
    lo, hi, sides = spells
    a = hi.searchsorted(start, side="right")
    b = lo.searchsorted(stop, side="left")
    return [
        (max(start, int(u)), min(stop, int(v)), int(side))
        for u, v, side in zip(lo[a:b], hi[a:b], sides[a:b], strict=True)
        if side < D
    ]


def _match_count(reference, proposal):
    overlap = metrics.spell_overlaps(reference, proposal)
    return np.mean(
        [
            metrics.matched_gt_flags(overlap, len(reference), iou).sum()
            for iou in metrics.SPELL_IOUS
        ]
    )


def _candidate_intervals(spells, home_start, dead_start, dead_stop, host):
    lo, hi, teams = spells
    entry = lo[1:-1]
    valid = (
        (teams[:-2] == host)
        & (teams[1:-1] == 1 - host)
        & (teams[2:] == D)
        & (entry >= max(home_start + 14, dead_start - 105))
        & (entry <= dead_start + 8)
    )
    return [
        (int(lo[j]), int(hi[j]))
        for j in np.flatnonzero(valid) + 1
        if 10 <= hi[j] - lo[j] <= 105
        and abs(hi[j] - dead_start) <= 23
        and dead_stop - hi[j] >= 12
    ]


def _pitch_geometry(track):
    scale = np.array([track["pitch_length_m"], track["pitch_width_m"]], dtype=float)
    ball = track["ball_xy"] * scale
    present = track["ball_observed"] & np.isfinite(ball).all(1)
    sides = np.column_stack(
        (ball[:, 0], scale[0] - ball[:, 0], ball[:, 1], scale[1] - ball[:, 1])
    )
    outside = present & (np.min(sides, axis=1) <= -0.10)
    return ball, sides, present, outside


# Returns (events, outside). events has one int32 row per ball exit that passes the checks in
# the loop: the first observed frame at least 0.10 m outside the pitch, the estimated crossing
# frame, and the number of consecutive observed frames, starting with that first frame and
# capped at 8, that stay outside, no more than 11 frames after it and in the same period.
# outside is the per-frame flag for an observed ball at least 0.10 m outside the pitch.
def exit_events(track):
    period = track["period"]
    ball, sides, present, outside = _pitch_geometry(track)
    numbered = np.flatnonzero(present)
    if not len(numbered):
        return np.empty((0, 3), dtype=np.int32), outside
    inside = present & (sides.min(axis=1) >= 0)
    last_inside = np.maximum.accumulate(np.where(inside, np.arange(len(period)), -1))
    starts = np.flatnonzero(
        outside[numbered]
        & np.r_[
            True,
            ~outside[numbered[:-1]] | (period[numbered[:-1]] != period[numbered[1:]]),
        ]
    )
    events = []
    for seq in starts:
        a = int(numbered[seq])
        if a == 0:
            continue
        last = int(last_inside[a - 1])
        if last < 0 or a - last > 25 or period[a] != period[last]:
            continue
        speed = np.linalg.norm(ball[a] - ball[last]) * 25.0 / (a - last)
        if speed > 55.0:
            continue
        candidates = numbered[seq:][:8]
        keep = (
            (candidates <= a + 11)
            & (period[candidates] == period[a])
            & outside[candidates]
        )
        stopped = np.flatnonzero(~keep)
        persistence = int(stopped[0]) if len(stopped) else len(keep)
        if persistence < 2:
            continue
        side = int(sides[a].argmin())
        dr = sides[last, side] - sides[a, side]
        exact = (
            a
            if a - last > 10
            else int(np.ceil(last + (a - last) * sides[last, side] / dr))
        )
        events.append((a, exact, persistence))
    return np.asarray(events, dtype=np.int32).reshape(-1, 3), outside


# Fitting/development helper: nothing on the predict.py path calls this.
def fit_exit_delays(training):
    observed_offsets, crossing_offsets = [], []
    for events, period, labels in training:
        starts = (
            np.flatnonzero(
                (labels[1:] == 2) & (labels[:-1] < 2) & (period[1:] == period[:-1])
            )
            + 1
        )
        last = -1
        for first, cross, _ in events:
            nearby = starts[
                (starts >= first - 2)
                & (starts <= first + 20)
                & (period[starts] == period[first])
            ]
            if len(nearby) != 1 or nearby[0] == last:
                continue
            stop = int(nearby[0])
            if np.any(labels[min(first, stop) : max(first, stop)] >= 3):
                continue
            observed_offsets.append(stop - int(first))
            crossing_offsets.append(stop - int(cross))
            last = stop
    return {
        name: int(np.clip(np.median(offsets + [0] * 4), 0, 10))
        for name, offsets in (
            ("observed", observed_offsets),
            ("crossing", crossing_offsets),
        )
    }


# Fitting/development helper: nothing on the predict.py path calls this.
def fit_crossing_delay(training):
    return fit_exit_delays(training)["crossing"]


def _nearest_players(match):
    pitch = np.array(
        [match["pitch_length_m"], match["pitch_width_m"]], dtype=np.float32
    )
    ball = match["ball_xy"] * pitch
    observed_ball = match["ball_observed"] & np.isfinite(ball).all(axis=1)
    nearest = []
    for side in ("home", "away"):
        players = match[f"{side}_xy"] * pitch
        distances = np.linalg.norm(players - ball[:, None], axis=-1)
        valid = (
            match[f"{side}_observed"] & observed_ball[:, None] & np.isfinite(distances)
        )
        nearest.append(np.where(valid, distances, np.inf).min(axis=1))
    return observed_ball, np.stack(nearest, axis=1)


def apply_observed_exit(reference, match, member_probs, tactics):
    """Apply the fixed-stoppage policy selected on the development folds.

    Forecast members are grouped by model family, with four held contexts per family.
    """
    period = match["period"]
    if np.shape(member_probs) != (12, len(period), 3):
        raise ValueError(
            "Expected 12 frame probability streams, grouped by model family."
        )
    members = [
        _inventory(decode(np.log(p.clip(1e-7)), period, 7, 6), period)
        for p in member_probs
    ]
    exits = exit_events(match)[0][:, 1]
    visible, nearest = _nearest_players(match)
    stop_odds = np.log(tactics[:, 2].clip(1e-7)) - np.log(
        tactics[:, :2].sum(axis=1).clip(1e-7)
    )
    team_odds = np.log(tactics[:, 0].clip(1e-7)) - np.log(tactics[:, 1].clip(1e-7))
    starts, stops, states = _inventory(reference, period)
    result = reference.copy()
    for j in np.flatnonzero((states[:-1] < D) & (states[1:] == D)):
        lo, hi, end = int(starts[j]), int(stops[j]), int(stops[j + 1])
        if hi - lo < 29 or end - hi < 23 or period[lo] != period[end - 1]:
            continue
        host = int(states[j])
        rival = 1 - host
        sources = [
            _candidate_intervals(member, lo, hi, end, host) for member in members
        ]
        proposals = {(a, b) for source in sources for a, b in source if b != hi}
        if not proposals:
            continue
        local = [_clip_live(member, lo, min(end, hi + 23)) for member in members]
        prior = [_match_count(draft, [(lo, hi, host)]) for draft in local]
        ranked = []
        for a, b in proposals:
            if not np.any((exits >= max(a, hi - 8)) & (np.abs(exits - b) <= 6)):
                continue
            votes = sum(
                any(
                    abs(start - a) <= 8 and abs(stop - b) <= 6 for start, stop in source
                )
                for source in sources
            )
            if (
                votes < 3
                or (1 - 2 * rival) * team_odds[a:b].mean() < 0.12
                or stop_odds[b : b + 8].mean() < -0.22
            ):
                continue
            contact = (
                visible[a:b]
                & (nearest[a:b, rival] < 1.65)
                & (nearest[a:b, host] > nearest[a:b, rival] + 0.42)
            )
            if contact.sum() < max(4, 0.18 * (b - a)):
                continue
            proposal = [(lo, a, host), (a, b, rival)]
            gain = (
                (np.array([_match_count(draft, proposal) for draft in local]) - prior)
                .reshape(3, 4)
                .mean(1)
            )
            value = (
                0.6 * gain.mean()
                + 0.4 * np.sort(gain)[-2:].mean()
                - 0.40
                - 0.003 * abs(b - hi)
            )
            if value >= 0.12:
                ranked.append((float(value), a, b))
        if ranked:
            _, a, _ = max(ranked)
            result[a:hi] = rival
    return result


def apply_early_exit(reference, match, crossing_delay, *, events=None):
    """Advance a stoppage across a short apparent turnover after the ball leaves the pitch."""
    period = match["period"]
    if events is None:
        events, _ = exit_events(match)
    _, _, seen, outside = _pitch_geometry(match)
    starts, stops, teams = _inventory(reference, period)
    result = reference.copy()
    for i in range(2, len(teams)):
        previous, rival, dead = teams[i - 2 : i + 1]
        lo, start, stop, end = starts[i - 2], starts[i - 1], starts[i], stops[i]
        if (
            max(previous, rival) >= D
            or dead != D
            or previous == rival
            or period[lo] != period[end - 1]
        ):
            continue
        if not 5 <= stop - start <= 48 or start - lo < 12:
            continue
        window = (
            (events[:, 0] >= max(lo + 8, start - 35))
            & (events[:, 0] <= start + 1)
            & (events[:, 0] < stop)
        )
        candidates = []
        for first, cross, persistence in events[window]:
            visible = seen[first:stop]
            if np.any(visible & ~outside[first:stop]) or period[first] != period[stop]:
                continue
            observations = np.flatnonzero(visible)
            if len(observations) < 3 or stop - (first + observations[-1]) > 22:
                continue
            candidates.append((int(cross), int(persistence)))
        if len(candidates) != 1:
            continue
        cross, persistence = candidates[0]
        estimate = cross + crossing_delay
        if (
            persistence >= 3
            and stop - estimate <= 60
            and lo + 8 < estimate <= start + 3
        ):
            result[min(start, estimate) : stop] = D
    return result


def apply_persistent_exit(reference, match, observed_delay, *, events=None):
    """Move an existing stoppage onset when at least five ball observations are outside."""
    period = match["period"]
    if events is None:
        events, _ = exit_events(match)
    events = events[events[:, 2] >= 5]
    starts, _, teams = _inventory(reference, period)
    result = reference.copy()
    for i in np.flatnonzero((teams[:-1] < D) & (teams[1:] == D)):
        floor, stop = starts[i : i + 2]
        if period[stop] != period[stop - 1]:
            continue
        first = events[:, 0]
        within = (first < stop) & (first > floor) & (first >= stop - 48)
        within &= period[first] == period[stop]
        measured = first[within]
        if len(measured) != 1:
            continue
        estimate = int(measured[0]) + observed_delay
        if (
            floor < estimate < stop
            and stop - estimate <= 48
            and period[estimate - 1] == period[estimate]
        ):
            result[estimate:stop] = D
    return result


def add_persistent_exits(contact, exit_reference, persistent, period):
    """Transfer persistent exit changes inside a live possession with stable ownership."""
    changed = exit_reference != persistent
    boundaries = np.flatnonzero(np.diff(np.r_[False, changed, False]))
    result = contact.copy()
    for a, b in boundaries.reshape(-1, 2):
        if not np.all(persistent[a:b] == D) or not np.all(exit_reference[a:b] < D):
            raise ValueError(
                "Persistent exit adjustments may only extend dead-ball spans"
            )
        if a < 8 or b >= len(contact) or period[a - 8] != period[b]:
            continue
        if (
            np.all(contact[a - 8 : b] == contact[a])
            and contact[b] == exit_reference[b] == D
            and np.array_equal(contact[a:b], exit_reference[a:b])
        ):
            result[a:b] = D
    return result


def combine_receiver_exits(receiver, contact, contact_with_exits, period):
    """Apply compatible ball-exit changes to the coupled receiver timeline."""
    starts, stops, _ = _inventory(contact, period)
    bounds = np.r_[starts, stops[-1]]
    changed = contact_with_exits != contact
    period_breaks = period[1:] != period[:-1]
    begins = np.flatnonzero(changed & np.r_[True, ~changed[:-1] | period_breaks])
    ends = np.flatnonzero(changed & np.r_[~changed[1:] | period_breaks, True]) + 1
    expanded = []
    for lo, hi in zip(begins, ends, strict=True):
        left, right = bounds.searchsorted([lo, hi - 1], side="right") - 1
        if left > 0 and period[bounds[left] - 1] == period[lo]:
            left -= 1
        if right + 2 < len(bounds) and period[bounds[right + 1]] == period[lo]:
            right += 1
        expanded.append((int(bounds[left]), int(bounds[right + 1])))
    blocks = []
    for start, stop in sorted(expanded):
        if blocks and start < blocks[-1][1]:
            blocks[-1] = blocks[-1][0], max(stop, blocks[-1][1])
        else:
            blocks.append((start, stop))
    result = receiver.copy()
    for a, b in blocks:
        if np.array_equal(contact[a:b], result[a:b]):
            result[a:b] = contact_with_exits[a:b]
    return result
