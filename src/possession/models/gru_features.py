"""Player sets, team shape and optional ball features for the tactics GRU.

Each period is sampled independently. Home and away views orient the
respective team towards positive x and keep player slots unordered.
Coordinates and centroid offsets use half-pitch units. Velocities
use centred differences in metres per second, scaled by 8 m/s.
"""

import numpy as np

from . import team_shape as tc

FPS = 25
STEP = 5                        # 25 fps -> 5 fps
N_FEAT = 7
SLOTS = 23
VEL_SCALE = 8.0                 # m/s
H, A, D, X = 0, 1, 2, 3


def _velocity(xy_m, span=4):
    """(T, P, 2) metres at 25 fps -> (T, P, 2) m/s by a centred difference over `span` frames
    (even). NaN wherever either end is missing, so it never spans a gap's edge."""
    half = span // 2
    v = np.full_like(xy_m, np.nan)
    if len(xy_m) > span:
        v[half:-half] = (xy_m[span:] - xy_m[:-span]) / (span / FPS)
    return v


THAW = False                    # hide a ball that does not move at all (see frozen_ball)


def frozen_ball(ball, min_frames=10):
    """(T,) bool: frames where the ball position repeats exactly for at least min_frames in a row.
    DFL writes the last position on every dead-ball frame, so there the ball is a label in
    disguise, not a measurement. Real tracking jitters, so it is never exactly still."""
    same = np.r_[False, np.all(ball[1:] == ball[:-1], axis=1) & np.isfinite(ball[1:, 0])]
    edge = np.flatnonzero(np.diff(np.r_[0, same.astype(np.int8), 0]))
    out = np.zeros(len(ball), dtype=bool)
    for a, b in zip(edge[::2], edge[1::2]):
        if b - a >= min_frames:
            out[max(a - 1, 0):b] = True
    return out


def attacks_right(home_xy, away_xy):
    """True if home attacks towards +x in this period.

    Each team's whole-period centroid sits on the side of the goal it defends: when a team
    attacks, its own back line stays deep while the defending team packs in front of its goal.
    Comparing the two centroids over a whole period is robust to set pieces and to the window
    before a kick-off.
    """
    hx = np.nanmean(home_xy[..., 0])
    ax = np.nanmean(away_xy[..., 0])
    return bool(hx < ax)


def _tokens(xy, v, length, width, flip):
    """Token features for one set of players, in a frame that is rotated 180 degrees if `flip`."""
    x, y = xy[..., 0], xy[..., 1]
    vx, vy = v[..., 0], v[..., 1]
    if flip:
        x, y, vx, vy = 1.0 - x, 1.0 - y, -vx, -vy
    xs = (x - 0.5) * 2.0                       # -1 .. 1 along the length
    ys = (y - 0.5) * 2.0
    with np.errstate(invalid="ignore"):
        cx = np.nanmean(np.where(np.isfinite(xs), xs, np.nan), axis=-1, keepdims=True)
        cy = np.nanmean(np.where(np.isfinite(ys), ys, np.nan), axis=-1, keepdims=True)
    speed = np.hypot(vx, vy)
    out = np.stack([xs, ys, xs - cx, ys - cy, vx / VEL_SCALE, vy / VEL_SCALE, speed / VEL_SCALE], -1)
    present = np.isfinite(xs)
    out[~present] = 0.0
    out[~np.isfinite(out)] = 0.0              # velocity unknown at a track's first and last frames
    return out.astype(np.float32), present


def _ball(xy, v, flip):
    """(T, 2) ball position and (T, 2) m/s velocity -> (T, 4) features in the frame, and a present mask."""
    x, y, vx, vy = xy[:, 0], xy[:, 1], v[:, 0], v[:, 1]
    if flip:
        x, y, vx, vy = 1.0 - x, 1.0 - y, -vx, -vy
    out = np.stack([(x - 0.5) * 2.0, (y - 0.5) * 2.0, vx / (3 * VEL_SCALE), vy / (3 * VEL_SCALE)], -1)
    present = np.isfinite(out[:, 0])
    out[~present] = 0.0
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32), present


N_TOUCH = 7
TOUCH = False                   # append touch features to ball_h / ball_a (see touch_features)


def _since(event):
    """(T,) seconds since the last True in `event`, inf before the first."""
    idx = np.where(event, np.arange(len(event)), -1)
    last = np.maximum.accumulate(idx)
    return np.where(last >= 0, (np.arange(len(event)) - last) / float(FPS), np.inf)


def touch_features(ball_m, home_m, away_m, pick, half=2):
    """Who is playing the ball, read at the full 25 fps and summarised per picked row.

    ball_m (T, 2), home_m / away_m (T, P, 2) in metres. A frame is a touch by a team when its
    player is the nearest to the ball and either the ball's velocity changes sharply within 2 m of
    him (a kick, a deflection) or he has it within 1 m with no opponent inside 1.5 m (carrying).
    Returns (home view, away view), each (len(pick), N_TOUCH):

        share of the row's frames with an own / an opponent touch
        exp(-seconds since the last own / opponent touch / 2)
        who touched last: +1 own, -1 opponent, 0 nobody in the last 10 s
        who is nearer the ball over the row's frames: +1 own .. -1 opponent
        the sharpest change of ball velocity in the row's frames, / 30 m/s

    A 5 fps row cannot see a one-touch interception; these can. All zero where the row has no ball.
    """
    T = len(ball_m)
    here = np.isfinite(ball_m[:, 0])

    def nearest(players):
        d = np.linalg.norm(players - ball_m[:, None], axis=-1)
        return np.where(np.isfinite(d), d, np.inf).min(1)

    dh, da = nearest(home_m), nearest(away_m)
    nearer_team = (da > dh).astype(float) - (dh > da)
    near = np.where(here & (np.minimum(dh, da) < 3.0), nearer_team, 0.0)
    v = _velocity(ball_m[:, None], 2)[:, 0]
    dv = np.zeros(T)
    dv[1:-1] = np.linalg.norm(v[2:] - v[:-2], axis=-1)
    dv = np.where(np.isfinite(dv), dv, 0.0)
    kick = (dv > 4.0) & (np.minimum(dh, da) < 2.0)
    touch_h = here & (near > 0) & (kick | ((dh < 1.0) & (da > 1.5)))
    touch_a = here & (near < 0) & (kick | ((da < 1.0) & (dh > 1.5)))
    sh, sa = _since(touch_h), _since(touch_a)
    last_touch_team = (sa > sh).astype(float) - (sh > sa)
    last = np.where(np.minimum(sh, sa) > 10.0, 0.0, last_touch_team)

    def window(x, fn):
        pad = np.pad(x.astype(np.float64), half, mode="edge")
        return fn(np.stack([pad[k:k + T] for k in range(2 * half + 1)]), 0)[pick]

    cols = [window(touch_h, np.mean), window(touch_a, np.mean), np.exp(-sh / 2.0)[pick], np.exp(-sa / 2.0)[pick],
            last[pick], window(near, np.mean), np.clip(window(dv, np.max) / 30.0, 0, 1)]
    home = np.stack(cols, -1)
    home[~here[pick]] = 0.0
    away = home[:, [1, 0, 3, 2, 4, 5, 6]].copy()
    away[:, [4, 5]] *= -1
    return home.astype(np.float32), away.astype(np.float32)


def _tactics(home, away, vh, va, scale, right, length, width, delta_rows=None):
    """Tactical features for both views from one pass: the away view is the home view with the
    two teams swapped and the pitch rotated half a turn."""
    sign = 1.0 if right else -1.0
    hp, ap = (home - 0.5) * scale * sign, (away - 0.5) * scale * sign
    g, e_h, e_a = tc.view(hp, vh * sign, ap, va * sign, length, width, delta_rows=delta_rows)
    n = tc.N_SIDE
    own, opp, shared = g[:, :n].copy(), g[:, n:2 * n].copy(), g[:, 2 * n:].copy()
    own_a, opp_a = opp.copy(), own.copy()
    for part in (own_a, opp_a):
        part[:, -2:] *= -1
    sh_a = shared.copy()
    sh_a[:, :2] *= -1
    sh_a[:, [6, 7]] = shared[:, [7, 6]]
    return g, np.concatenate([own_a, opp_a, sh_a], 1), e_h, e_a


def match_features(match, step=STEP, vel_span=4, with_ball=False, target=None, tactics=False, *,
                   phase=0, thaw=None, touch=None, delta_rows=None):
    """{"periods": [per-period dicts]} with, per period of T rows:

        own_h, opp_h   (T, 23, 7) float32   home's view: home players, away players
        own_a, opp_a   (T, 23, 7) float32   away's view: away players, home players
        pres_h, pres_a (T, 23) bool         which home / away slots hold a player
        label          (T,) uint8           the match's label at that frame (whatever view it was loaded with)
        frames         (T,) int             the harmonised frame index of each row
        home_right     bool                 home attacks +x in this period
    """
    thaw = THAW if thaw is None else thaw
    touch = TOUCH if touch is None else touch
    length, width = float(match["pitch_length_m"]), float(match["pitch_width_m"])
    scale = np.array([length, width], dtype=np.float64)
    out = []
    for number in np.unique(match["period"]):
        rows = np.flatnonzero(match["period"] == number)
        home = match["home_xy"][rows].astype(np.float64)
        away = match["away_xy"][rows].astype(np.float64)
        vh = _velocity(home * scale, vel_span)
        va = _velocity(away * scale, vel_span)
        # Start at row `phase`, clamped to the last row, so a period with no more than `phase`
        # rows still yields one sample instead of an empty pick.
        pick = np.arange(min(phase, len(rows)-1), len(rows), step)
        right = attacks_right(home, away)
        own_h, pres_h = _tokens(home[pick], vh[pick], length, width, flip=not right)
        opp_h, _ = _tokens(away[pick], va[pick], length, width, flip=not right)
        own_a, pres_a = _tokens(away[pick], va[pick], length, width, flip=right)
        opp_a, _ = _tokens(home[pick], vh[pick], length, width, flip=right)
        extra = {}
        if with_ball:
            ball = match["ball_xy"][rows].astype(np.float64)
            src = (target if target is not None else match)["ball_xy"][rows].astype(np.float64)
            if thaw:
                ball[frozen_ball(ball)] = np.nan
                src = src.copy()
                src[frozen_ball(src)] = np.nan
            vb = _velocity(ball * scale, 2)
            extra["ball_h"], extra["ball_pres"] = _ball(ball[pick], vb[pick], flip=not right)
            extra["ball_a"], _ = _ball(ball[pick], vb[pick], flip=right)
            if touch:
                t_h, t_a = touch_features(ball * scale, home * scale, away * scale, pick, half=max(1, step // 2))
                extra["ball_h"] = np.concatenate([extra["ball_h"], t_h], 1)
                extra["ball_a"] = np.concatenate([extra["ball_a"], t_a], 1)
            # Auxiliary target: the true ball, from `target` (the clean match) when given.
            vt = _velocity(src * scale, 2)
            extra["tgt_h"], extra["tgt_pres"] = _ball(src[pick], vt[pick], flip=not right)
            extra["tgt_a"], _ = _ball(src[pick], vt[pick], flip=right)
            # Possessor proxy: the tracked player nearest the true ball, as an index into
            # [own players, opponents] of each view; -1 where there is no ball or no player.
            allp = np.concatenate([home[pick], away[pick]], 1) * scale          # (T, 46, 2)
            dist = np.linalg.norm(allp - (src[pick] * scale)[:, None], axis=-1)
            dist = np.where(np.isfinite(dist), dist, np.inf)
            near = dist.argmin(1)
            ok = np.isfinite(dist.min(1)) & (dist.min(1) < 3.0)
            P = home.shape[1]
            extra["near_h"] = np.where(ok, near, -1).astype(np.int64)
            extra["near_a"] = np.where(ok, (near + P) % (2 * P), -1).astype(np.int64)
        if tactics:
            g_h, g_a, e_h, e_a = _tactics(home[pick], away[pick], vh[pick], va[pick], scale, right, length, width, delta_rows)
            extra["glob_h"], extra["glob_a"] = g_h, g_a
            own_h, opp_h = np.concatenate([own_h, e_h], -1), np.concatenate([opp_h, e_a], -1)
            own_a, opp_a = np.concatenate([own_a, e_a], -1), np.concatenate([opp_a, e_h], -1)
        out.append({
            **extra,
            "own_h": own_h, "opp_h": opp_h, "own_a": own_a, "opp_a": opp_a,
            "pres_h": pres_h, "pres_a": pres_a,
            "label": match["label"][rows[pick]].astype(np.uint8),
            "frames": rows[pick], "home_right": right,
        })
    return {"periods": out, "n_frames": len(match["label"]), "key": str(match["match_key"])}


# Fitting/development helper: nothing on the predict.py path calls this.
def trainable(label, pres_h, pres_a):
    """Rows a loss may use: labelled, and with at least one player of each team tracked. Rows with
    no tracking teach "no tracking means dead ball" (A-League), which the competition will not
    reward."""
    return (label != X) & pres_h.any(-1) & pres_a.any(-1)
