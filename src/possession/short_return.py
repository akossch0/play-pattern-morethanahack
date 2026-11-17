"""Detect a short opponent possession from a partially observed pass."""

import numpy as np

from datalake import core


def _windows(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return edges.reshape(-1, 2)


def _contacts(dist, side, period):
    near = dist[:, side].min(axis=1)
    other = dist[:, 1 - side].min(axis=1)
    slot = dist[:, side].argmin(axis=1)
    ident = np.where((near <= 1.65) & (other >= near + 0.35), slot + 32 * period, -1)
    ends = np.r_[0, np.flatnonzero(ident[1:] != ident[:-1]) + 1, len(ident)]
    return [(int(a), int(b), int(slot[a])) for a, b in zip(ends[:-1], ends[1:], strict=True)
            if b - a >= 3 and ident[a] >= 0]


def _corridor(ball, seen, release, receive):
    # The ball must progress along a short, fast chord. Requiring agreement on
    # observed samples on both lips rejects a gap that hides an arbitrary turn.
    travel = receive - release
    chord = ball[receive] - ball[release]
    length = np.linalg.norm(chord)
    # Chord of 2 to 14 m covered at a mean 3 to 30 m/s; travel is the flight time in frames.
    if not (2 <= length <= 14 and 3 <= length * core.FPS / travel <= 30):
        return None
    middle = np.arange(release, receive + 1)
    samples = middle[seen[middle]]
    if len(samples) < 5 or not ((samples < (release + receive) / 2).sum() >= 2
                                and (samples > (release + receive) / 2).sum() >= 2):
        return None
    offset = ball[samples] - ball[release]
    direction = chord / length
    progress = offset @ direction
    cross = np.linalg.norm(offset - progress[:, None] * direction, axis=1)
    if cross.max() > 0.95 or np.diff(progress).min() < -.50:
        return None
    if np.max(np.abs(progress - length * (samples - release) / travel)) > 1.15:
        return None
    return float(length), float(np.mean(cross))


def _receiver_trap(ball, seen, player, release, receive, end):
    flank = np.arange(receive + 1, min(end, receive + 9))
    flank = flank[seen[flank] & np.isfinite(player[flank]).all(axis=1)]
    if len(flank) < 3:
        return False
    time = flank[-1] - receive
    incoming = ball[receive] - ball[release]
    outgoing = ball[flank[-1]] - ball[receive]
    incoming_rate = np.linalg.norm(incoming) / (receive - release)
    continuation = (outgoing @ incoming) / (np.linalg.norm(incoming) * time)
    relative = (ball[flank] - ball[receive]) - (player[flank] - player[receive])
    return bool(continuation <= .55 * incoming_rate and
                np.median(np.linalg.norm(relative, axis=1)) < .8)


def _candidates(track, incumbent):
    scale = np.array([track["pitch_length_m"], track["pitch_width_m"]], dtype=np.float32)
    ball = track["ball_xy"] * scale
    seen = track["ball_observed"]
    period = track["period"]
    players = np.stack((track["home_xy"], track["away_xy"]), axis=1) * scale
    observed = np.stack((track["home_observed"], track["away_observed"]), axis=1)
    players[~observed] = np.nan
    dist = np.linalg.norm(players - ball[:, None, None], axis=-1)
    dist[~np.isfinite(dist) | ~seen[:, None, None]] = np.inf
    team_dist = dist.min(axis=2)
    for rival in (0, 1):
        host = 1 - rival
        rival_contacts = _contacts(dist, rival, period)
        host_contact = seen & (team_dist[:, host] <= 1.8) & (team_dist[:, rival] > team_dist[:, host])
        for first, second in zip(rival_contacts[:-1], rival_contacts[1:], strict=True):
            a, b, giver = first
            c, d, receiver = second
            if giver == receiver or period[a] != period[d - 1] or not 2 <= c - b <= 27:
                continue
            release, receive = b - 1, c
            if not np.all(incumbent[max(0, a - 3):d + 3] == host):
                continue
            flight = _corridor(ball, seen, release, receive)
            if flight is None:
                continue
            rival_track = observed[release:receive + 1, rival]
            pair_occluded = (~rival_track[:, [giver, receiver]]).any(axis=1)
            ball_occluded = ~seen[release:receive + 1]
            if not (ball_occluded.any() or pair_occluded.any()):
                continue
            if any(y - x > 14 for x, y in _windows(ball_occluded)):
                continue
            before = np.flatnonzero(host_contact[max(0, a - 33):a]) + max(0, a - 33)
            after = np.flatnonzero(host_contact[d:d + 33]) + d
            if not len(before) or not len(after):
                continue
            left, right = int(before[-1]), int(after[0])
            if period[left] != period[right] or not np.all(incumbent[left:right + 1] == host):
                continue
            # Exclude observed intervening host touches and require both opponent
            # actors to beat the nearest visible host at their respective contact.
            if host_contact[release:receive + 1].any():
                continue
            margin = min(team_dist[release, host] - dist[release, rival, giver],
                         team_dist[receive, host] - dist[receive, rival, receiver])
            if margin < .65 or not np.isfinite(margin):
                continue
            start, stop = (left + a + 1) // 2, (d + right) // 2
            if not 12 <= stop - start <= 85:
                continue
            trap = _receiver_trap(ball, seen, players[:, rival, receiver], release, receive, d + 3)
            yield {
                "start": start, "stop": stop, "rival": rival, "release": release,
                "receive": receive, "length_m": flight[0], "mean_cross_error_m": flight[1],
                "margin_m": float(margin), "ball_missing_frames": int(ball_occluded.sum()),
                "player_missing_frames": int(pair_occluded.sum()), "receiver_trap": trap,
            }


def apply_short_return(reference, tracking):
    """Apply the receiver-trap policy without changing dead-ball frames."""
    result = reference.copy()
    stop_of_previous = 0
    proposals = sorted(_candidates(tracking, reference), key=lambda row: row["start"])
    for row in proposals:
        start, stop, rival = row["start"], row["stop"], row["rival"]
        if start <= stop_of_previous or not row["receiver_trap"]:
            continue
        if not np.all(reference[start - 1:stop + 1] == 1 - rival):
            raise ValueError("A proposed short return extends outside the original possession")
        result[start:stop] = rival
        stop_of_previous = stop
    return result


def _cuts(labels, period):
    return np.r_[0, np.flatnonzero((labels[1:] != labels[:-1]) | (period[1:] != period[:-1])) + 1, len(labels)]


def merge_timing_branches(reference, period, flank_timing, credit_timing):
    """Combine compatible retimings without changing the possession order."""
    base, left, right = [_cuts(labels, period) for labels in (reference, flank_timing, credit_timing)]
    if base.shape != left.shape or base.shape != right.shape:
        raise ValueError("Expected each timing branch to preserve the number of spells")
    dl, dr = left - base, right - base
    agrees = dl * dr > 0
    shorter = np.where(np.abs(dl) <= np.abs(dr), left, right)
    new = np.where(dl == 0, right, np.where(dr == 0, left, np.where(agrees, shorter, base)))
    while (np.diff(new) <= 0).any():
        clash = np.flatnonzero(np.diff(new) <= 0)
        affected = np.unique(np.r_[clash, clash + 1])
        new[affected] = base[affected]
    result = reference.copy()
    for i in np.flatnonzero(new != base):
        start, stop = sorted((new[i], base[i]))
        result[start:stop] = reference[base[i]] if new[i] < base[i] else reference[base[i] - 1]
    if not np.array_equal(result == core.D, reference == core.D) or not np.array_equal(_cuts(result, period), new):
        raise ValueError("A timing edit changed the possession order or a dead-ball frame")
    return result
