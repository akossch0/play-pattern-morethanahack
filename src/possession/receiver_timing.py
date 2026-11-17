"""Retime a live turnover using observed ball convergence and receiver motion."""

import numpy as np


def cuts(codes, period):
    return np.r_[
        0, 1 + np.flatnonzero((codes[1:] != codes[:-1]) | (period[1:] != period[:-1])), len(codes)
    ]


def logodds(prob):
    logged = np.log(prob[:, :2].clip(1e-7))
    return logged[:, 0] - logged[:, 1]


def apply(base, period, events, policy):
    selected = []
    if policy is None:
        return base.copy(), selected
    kind, frame_gate, edge_gate, quorum = policy
    for center, before, after, choices in events:
        ranked = []
        for pos, brake, redirect, physical, gru, tcn, edge in choices:
            if kind == "brake" and not brake or kind == "redirect" and not redirect:
                continue
            vote = tcn >= frame_gate
            if edge_gate:
                vote &= edge >= edge_gate
            if gru < frame_gate or vote.sum() < quorum:
                continue
            value = (
                physical * 0.09
                + gru * 0.45
                + tcn.mean() * 0.45
                + edge.mean() * 0.13
                - 0.03 * abs(pos - center)
            )
            ranked.append((value, pos))
        if ranked:
            value, pos = max(ranked)
            selected.append(
                (
                    value,
                    min(pos, center),
                    max(pos, center),
                    before if pos > center else after,
                    center,
                    pos,
                )
            )
    result = base.copy()
    used = []
    for _, start, stop, label, center, pos in sorted(selected, reverse=True):
        if any(start <= high and stop >= low for low, high in used):
            continue
        result[start:stop] = label
        used.append((start, stop))
    np.testing.assert_array_equal(base == 2, result == 2)
    np.testing.assert_array_equal(base[cuts(base, period)[:-1]], result[cuts(result, period)[:-1]])
    return result, used


# For each candidate frame in rows, look at the 25 frames around it (12 before to 12 after) and
# test whether the ball converges on the nearest `actor`-side player and then moves with that
# player. Returns (contact flags, contact flags that also need the rival to fall away, score).
def receive(track, rows, actor, legacy):
    field = np.array([track["pitch_length_m"], track["pitch_width_m"]], np.float32)
    first, second = ("home", "away")[actor], ("home", "away")[legacy]
    ball_at = track["ball_xy"][rows[:, None] + np.arange(-12, 13)] * field
    own = track[f"{first}_xy"][rows[:, None] + np.arange(-12, 13)] * field
    other = track[f"{second}_xy"][rows[:, None] + np.arange(-12, 13)] * field
    observed = track["ball_observed"][rows[:, None] + np.arange(-12, 13)].all(1)
    own_range = np.nan_to_num(np.linalg.norm(own - ball_at[:, :, None], axis=-1), nan=25.0)
    other_range = np.nan_to_num(np.linalg.norm(other - ball_at[:, :, None], axis=-1), nan=25.0)
    selected = own_range[:, 11:14].mean(1).argmin(1)
    own_range = own_range[np.arange(len(rows)), :, selected]
    own_path = own[np.arange(len(rows)), :, selected]
    # Incoming ball must converge on the same player who is closest at control.
    # The ball and receiver then stay coupled, without requiring a velocity kink.
    before = own_range[:, :8].mean(1)
    local = own_range[:, 11:14].mean(1)
    after = own_range[:, 13:].mean(1)
    own_move = own_path[:, 22] - own_path[:, 13]
    ball_move = ball_at[:, 22] - ball_at[:, 13]
    together = np.linalg.norm(own_move - ball_move, axis=1)
    rival = other_range.min(2)
    advance = np.linalg.norm(ball_at[:, 12] - ball_at[:, 3], axis=1)
    separates = rival[:, 13:20].mean(1) - rival[:, 6:12].mean(1)
    contact = (
        observed
        & (local <= 1.30)
        & (after < 1.70)
        & (before - local > 1.15)
        & (together < 0.75)
        & (advance > 1)
    )
    clean = contact & (separates > 0.25) & (rival[:, 13:20].mean(1) > 1.8)
    defined = np.nan_to_num(
        (before - local) * 0.75 + (1.7 - after) * 1.4 + separates * 0.40 - together * 0.40, nan=-100
    )
    return contact, clean, defined


def events(track, period, base, previous, gru, forecasts):
    border = cuts(base, period)
    sides = base[border[:-1]]
    gru_odds = logodds(gru)
    logits = np.stack([logodds(p) for p, _ in forecasts])
    edge = np.stack([e[:, :2] for _, e in forecasts])
    edge = np.log(edge.clip(0.025, 0.975)) - np.log1p(-edge.clip(0.025, 0.975))
    result = []
    for i in range(1, len(sides)):
        lo, old, hi = map(int, border[i - 1 : i + 2])
        before, after = map(int, sides[i - 1 : i + 1])
        if before >= 2 or after >= 2 or before == after or period[lo] != period[hi - 1]:
            continue
        if not 100 <= min(old - lo, hi - old) <= 625:
            continue
        if not np.array_equal(base[old - 4 : old + 4], previous[old - 4 : old + 4]):
            continue
        rows = np.r_[np.arange(old - 25, old - 3), np.arange(old + 4, old + 26)]
        contact, clean, physical = receive(track, rows, after, before)
        choices = []
        for j in np.flatnonzero(contact):
            pos = int(rows[j])
            a, b = sorted((old, pos))
            # sign makes the home-minus-away log-odds count as support for the team that would
            # gain the frames between old and pos.
            sign = (1 - 2 * after) * (1 if pos < old else -1)
            frame = sign * logits[:, a:b].mean(1)
            support = sign * gru_odds[a:b].mean()
            directed = edge[:, pos - 4 : pos + 5, before].max(1) - edge[
                :, old - 4 : old + 5, before
            ].max(1)
            choices.append(
                (pos, True, bool(clean[j]), float(physical[j]), support, frame, directed)
            )
        if choices:
            result.append((old, before, after, choices))
    return result


def predict_receiver_timing(track, contact_timeline, before_contacts, tactics, forecasts):
    """Use tracking contacts and typed TCN agreement to refine the decoded timeline."""
    period = track["period"]
    candidates = events(track, period, contact_timeline, before_contacts, tactics, forecasts)
    labels, _ = apply(contact_timeline, period, candidates, ("any", 0.04, 0.04, 3))
    return labels
