"""Simulate tracking gaps and coordinate noise on external matches.

The transform uses a seed and leaves labels unchanged. Default observation rates come from
the organizer's published tracking notes; the remaining parameters control the simulation.
"""

import argparse
import json

import numpy as np

from . import core

FPS = core.FPS
VERSION = "degrade_v1"
MAX_BALL_SPEED = 40.0           # m/s. Above this the ball speed is a tracking artefact


def _runs(mask):
    """(start, end) pairs, end exclusive, of the true runs in a 1-d boolean array."""
    edge = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.column_stack([np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)])


def _zscore(v):
    """Standardised and clipped, so one extreme frame cannot own the sampling weights."""
    sd = v.std()
    return np.clip((v - v.mean()) / sd, -2.0, 3.0) if sd > 0 else np.zeros_like(v)


def ball_gap_weights(m, crowd_weight, crowd_radius_m, speed_weight):
    """Relative chance, per frame, that a ball gap is centred there: higher with more players
    within `crowd_radius_m` of the ball and with a faster ball. Sums to 1."""
    scale = np.array([m["pitch_length_m"], m["pitch_width_m"]])
    ball = m["ball_xy"].astype(np.float64) * scale
    near = np.zeros(len(ball))
    for side in ("home_xy", "away_xy"):
        delta = m[side].astype(np.float64) * scale - ball[:, None, :]
        with np.errstate(invalid="ignore"):
            near += np.sum(np.hypot(delta[..., 0], delta[..., 1]) < crowd_radius_m, axis=1)
    speed = core.derive_speed(m["ball_xy"], m["pitch_length_m"], m["pitch_width_m"])
    speed = np.nan_to_num(np.minimum(speed.astype(np.float64), MAX_BALL_SPEED))
    w = np.exp(crowd_weight * _zscore(near) + speed_weight * _zscore(speed))
    return w / w.sum()


def gap_lengths(rng, n, mean_gap_s, gap_sigma):
    """n gap lengths in frames, lognormal with the given mean: heavy-tailed, at least one frame."""
    mu = np.log(mean_gap_s * FPS) - gap_sigma ** 2 / 2
    return np.maximum(1, np.round(rng.lognormal(mu, gap_sigma, n))).astype(np.int64)


def _counted(label):
    """The frames a target is measured on: the labelled ones, or all when none is labelled."""
    counted = label != core.X
    return counted if counted.any() else np.ones(len(label), dtype=bool)


def _period_bounds(period):
    """Row indices where a period starts, plus the total length."""
    cut = np.flatnonzero(np.diff(period.astype(int)) != 0) + 1
    return np.concatenate([[0], cut, [len(period)]]).astype(np.int64)


def ball_dropout(m, rng, ball_missing, mean_gap_s, gap_sigma, weights):
    """Boolean (T,) mask of ball frames to remove, in contiguous gaps, so that the missing share
    over the labelled frames (counting frames already missing in the source) is `ball_missing`.
    Gaps fall on unlabelled frames too, but those do not count towards the target. No drawn gap
    crosses half-time (two gaps can still meet there)."""
    if not 0.0 <= ball_missing <= 1.0:
        raise ValueError("ball_missing must be within 0..1, got {}".format(ball_missing))
    missing = np.isnan(m["ball_xy"][:, 0])
    t = len(missing)
    counted = _counted(m["label"])
    need = int(round(ball_missing * counted.sum())) - int(missing[counted].sum())
    if need <= 0:
        return np.zeros(t, dtype=bool)
    bounds = _period_bounds(m["period"])
    gone = missing.copy()
    while need > 0:
        batch = max(16, int(2 * need / (mean_gap_s * FPS)))
        centres = rng.choice(t, size=batch, p=weights)
        lengths = gap_lengths(rng, batch, mean_gap_s, gap_sigma)
        for c, n in zip(centres, lengths):
            i = np.searchsorted(bounds, c, side="right") - 1
            a = max(bounds[i], c - n // 2)
            b = min(bounds[i + 1], a + n)
            fresh = np.flatnonzero(~gone[a:b] & counted[a:b])
            if len(fresh) > need:
                b = a + fresh[need]     # trim the last gap so the target is hit exactly
                fresh = fresh[:need]
            if b <= a:
                continue
            gone[a:b] = True
            need -= len(fresh)
            if need == 0:
                break
    return gone & ~missing


def _segments(present, rng, track_break_rate, break_gap_s):
    """Cut each slot's runs of presence at random breaks. Returns (start, end, slot, track,
    after_break) tuples; the frames between two pieces of a track are the time it is lost."""
    out = []
    track = 0
    for slot in range(present.shape[1]):
        for a, b in _runs(present[:, slot]):
            n = rng.poisson(track_break_rate * (b - a) / FPS / 60.0)
            breaks = np.sort(rng.integers(a + 1, max(a + 2, b), size=n))
            gaps = 1 + rng.geometric(min(1.0, 1.0 / max(1.0, break_gap_s * FPS)), size=n)
            cur, after = int(a), False
            for at, gap in zip(breaks, gaps):
                if at <= cur or at >= b:
                    continue
                out.append((cur, int(at), slot, track, after))
                cur, after = int(at + gap), True
            if cur < b:
                out.append((cur, int(b), slot, track, after))
            track += 1
    out.sort()
    return out


def fragment_tracks(xy, speed, observed, rng, track_break_rate, break_gap_s):
    """Break one team's tracks and let each reappear in a different free slot.

    A slot takes a new track only after standing empty for at least one frame, as in the contract.
    A track that was not broken keeps its slot unless a broken one got there first. Returns new
    (xy, speed, observed) and the number of breaks applied."""
    present = np.isfinite(xy[..., 0])
    new_xy = np.full_like(xy, np.nan)
    new_speed = np.full_like(speed, np.nan)
    new_observed = np.zeros_like(observed)
    slots = xy.shape[1]
    free_from = np.zeros(slots, dtype=np.int64)     # first frame at which a slot may start a track
    last_slot = {}
    breaks = 0
    for a, b, src, track, after in _segments(present, rng, track_break_rate, break_gap_s):
        free = np.flatnonzero(free_from <= a)
        if not after and src in free:
            dst = src
        else:
            other = free[free != last_slot.get(track, -1)]
            if len(other):
                dst = int(rng.choice(other))
            elif len(free):
                dst = int(free[0])      # only its old slot is free: the break shows as a gap
            else:
                continue                # more simultaneous tracks than slots: the piece is lost
        breaks += int(after)
        new_xy[a:b, dst] = xy[a:b, src]
        new_speed[a:b, dst] = speed[a:b, src]
        new_observed[a:b, dst] = observed[a:b, src]
        free_from[dst] = b + 1
        last_slot[track] = dst
    return new_xy, new_speed, new_observed, breaks


def _run_mask(u, lengths, q):
    """Runs starting where u < q and lasting `lengths` frames, as a boolean mask like u."""
    t = len(u)
    rows, cols = np.nonzero(u < q)
    step = np.zeros((t + 1, u.shape[1]), dtype=np.int32)
    np.add.at(step, (rows, cols), 1)
    np.add.at(step, (np.minimum(rows + lengths[rows, cols], t), cols), -1)
    return np.cumsum(step[:-1], axis=0) > 0


def player_dropout(present, counted, rng, target_players, player_drop, drop_run_s):
    """Boolean mask like `present` (T, slots of both teams) of player positions to remove, in
    short runs, so the mean player count over the `counted` frames reaches `target_players`
    (or falls by the fraction `player_drop`, when that is given). The start rate is found by
    bisection on one fixed random field, so the result is deterministic and the target is met."""
    now = present[counted].sum(1).mean()
    target = now * (1.0 - player_drop) if player_drop is not None else target_players
    u = rng.random(present.shape, dtype=np.float32)
    lengths = rng.geometric(min(1.0, 1.0 / max(1.0, drop_run_s * FPS)), present.shape).astype(np.int32)
    if target is None or now <= target:
        return np.zeros_like(present)
    lo, hi = 0.0, 1.0
    for _ in range(20):
        q = (lo + hi) / 2
        left = (present & ~_run_mask(u, lengths, q))[counted].sum(1).mean()
        if left > target:
            lo = q
        else:
            hi = q
    return present & _run_mask(u, lengths, hi)


def add_noise(xy, rng, noise_m, length, width):
    """Independent Gaussian noise of `noise_m` metres per axis; NaN stays NaN."""
    if noise_m <= 0:
        return xy
    sigma = (noise_m / np.array([length, width])).astype(np.float32)
    return (xy + rng.standard_normal(xy.shape, dtype=np.float32) * sigma).astype(np.float32)


def degrade(match, seed, ball_missing=0.41, mean_gap_s=2.0, gap_sigma=1.0, crowd_weight=0.5,
            crowd_radius_m=5.0, speed_weight=0.5, track_break_rate=1.0, break_gap_s=1.0,
            target_players=20.5, player_drop=None, drop_run_s=0.4, noise_m=0.10):
    """A degraded copy of a harmonised match, in the same contract shape. `match` is not modified.

    ball_missing      target share of labelled (non-X) frames without a ball, source gaps
                      included. None: off
    mean_gap_s        mean length of one drawn ball gap (gaps that overlap merge into longer ones)
    gap_sigma         lognormal sigma of the gap length
    crowd_weight      how strongly players within crowd_radius_m of the ball attract a gap
    speed_weight      how strongly ball speed attracts a gap
    track_break_rate  breaks per player-minute. 0: slots are left exactly as they are
    break_gap_s       mean time a broken track is lost before it reappears in another slot
    target_players    mean players per labelled frame after dropout. None: off
    player_drop       instead of target_players: fraction of player-frames to drop
    drop_run_s        mean length of one player dropout
    noise_m           Gaussian position noise per axis in metres, players and ball. 0: off

    Steps, each with its own random stream so that changing one knob does not reshuffle the
    others: ball gaps (chosen on the clean data), track breaks, player dropout, noise. Speeds are
    the source's: they move with their track and are removed with it, but carry no noise.
    """
    params = dict(seed=seed, ball_missing=ball_missing, mean_gap_s=mean_gap_s, gap_sigma=gap_sigma,
                  crowd_weight=crowd_weight, crowd_radius_m=crowd_radius_m,
                  speed_weight=speed_weight, track_break_rate=track_break_rate,
                  break_gap_s=break_gap_s, target_players=target_players, player_drop=player_drop,
                  drop_run_s=drop_run_s, noise_m=noise_m)
    r_ball, r_break, r_drop, r_noise = [np.random.default_rng(s)
                                        for s in np.random.SeedSequence(seed).spawn(4)]
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in match.items()}
    length, width = float(match["pitch_length_m"]), float(match["pitch_width_m"])

    if ball_missing is not None:
        weights = ball_gap_weights(match, crowd_weight, crowd_radius_m, speed_weight)
        gone = ball_dropout(match, r_ball, ball_missing, mean_gap_s, gap_sigma, weights)
        out["ball_xy"][gone] = np.nan
        out["ball_observed"][gone] = False

    breaks = 0
    if track_break_rate > 0:
        for side in ("home", "away"):
            xy, speed, observed, n = fragment_tracks(
                out[side + "_xy"], out[side + "_speed"], out[side + "_observed"], r_break,
                track_break_rate, break_gap_s)
            out[side + "_xy"], out[side + "_speed"], out[side + "_observed"] = xy, speed, observed
            breaks += n

    if target_players is not None or player_drop is not None:
        slots = out["home_xy"].shape[1]
        present = np.isfinite(np.concatenate([out["home_xy"], out["away_xy"]], axis=1)[..., 0])
        drop = player_dropout(present, _counted(out["label"]), r_drop, target_players,
                              player_drop, drop_run_s)
        for side, part in (("home", drop[:, :slots]), ("away", drop[:, slots:])):
            out[side + "_xy"][part] = np.nan
            out[side + "_speed"][part] = np.nan
            out[side + "_observed"][part] = False

    for name in ("home_xy", "away_xy", "ball_xy"):
        out[name] = add_noise(out[name], r_noise, noise_m, length, width)

    # default=: a seed or knob taken from a numpy array is a numpy scalar, which json rejects.
    out["degradation"] = json.dumps(dict(version=VERSION, track_breaks=breaks, **params),
                                    sort_keys=True, default=lambda v: v.item())
    return out


def measure(m):
    """The numbers to compare with DATA.md: ball presence, ball gap lengths, player count, and how
    the 46 slots are used."""
    ball = np.isfinite(m["ball_xy"][:, 0])
    graded = m["label"] != core.X
    # Per period, so that two gaps meeting at half-time are not read as one long gap.
    bounds = _period_bounds(m["period"])
    gaps = np.concatenate([_runs(~ball[a:b]) + a for a, b in zip(bounds[:-1], bounds[1:])])
    gap_s = (gaps[:, 1] - gaps[:, 0]) / FPS if len(gaps) else np.zeros(1)
    present = np.isfinite(np.concatenate([m["home_xy"], m["away_xy"]], axis=1)[..., 0])
    track_s = np.concatenate([np.diff(_runs(present[:, s]), axis=1).ravel()
                              for s in range(present.shape[1])] + [np.zeros(0)]) / FPS
    q = lambda v, p: round(float(np.percentile(v, p)), 2)
    return {
        "ball_present_pct": round(100 * float(ball.mean()), 1),
        "ball_present_graded_pct": round(100 * float(ball[graded].mean()), 1) if graded.any() else 0.0,
        "ball_gaps": int(len(gaps)),
        "ball_gap_s_mean": round(float(gap_s.mean()), 2),
        "ball_gap_s_p50": q(gap_s, 50), "ball_gap_s_p90": q(gap_s, 90),
        "ball_gap_s_p99": q(gap_s, 99), "ball_gap_s_max": round(float(gap_s.max()), 1),
        "mean_players": round(float(present.sum(1).mean()), 2),
        "mean_players_graded": round(float(present[graded].sum(1).mean()), 2) if graded.any() else 0.0,
        "slots_used": int(present.any(0).sum()),
        "slot_occupancy_pct": [int(round(100 * v)) for v in present.mean(0)],
        "track_runs": int(len(track_s)),
        "track_run_s_p50": q(track_s, 50) if len(track_s) else 0.0,
        "track_run_s_mean": round(float(track_s.mean()), 1) if len(track_s) else 0.0,
    }


def same(a, b):
    """True when two matches hold identical arrays (NaN equal to NaN)."""
    return all(np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f") for k in core.ARRAYS)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("keys", nargs="+")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for key in args.keys:
        clean = core.load(key)
        out = degrade(clean, args.seed)
        core.validate(out)
        before, after = measure(clean), measure(out)
        print(key, json.loads(out["degradation"]))
        for name in before:
            print("  {:26s} {} -> {}".format(name, before[name], after[name]))
        print("  same seed, same output:    {}".format(same(out, degrade(clean, args.seed))))
        print("  other seed differs:        {}".format(not same(out, degrade(clean, args.seed + 1))))
        print("  labels unchanged:          {}".format(np.array_equal(clean["label"], out["label"])))


if __name__ == "__main__":
    main()
