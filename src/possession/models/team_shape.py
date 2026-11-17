"""Tactics-informed features from player positions only.

Football gives away who has the ball in how the two teams shape up and move:

- the team in possession spreads (longer, wider), the other stays compact (Moura 2012);
- both teams drift towards the goal being attacked, and on a turnover the sprint directions and
  the back line's movement flip within a second or two;
- defenders close down the carrier, so the pressed player's team has the ball;
- players run towards the ball, so their velocity lines meet near it (a ball proxy);
- a dead ball shows as walking pace, a player on the touchline (throw-in), a crowd in a penalty
  area with a player at the corner, or both teams in their own halves (kick-off).

Slots are anonymous tracklets, so everything here is a count, an order statistic (GK = deepest,
back line = second deepest, ...) or a nearest-neighbour quantity. None depends on slot order or on
a slot being one person.

    view(own_p, own_v, opp_p, opp_v, length, width) -> glob (T, N_GLOB), ext_own, ext_opp (T, P, N_EXT)

Inputs are in the viewing team's frame: metres, centred on the pitch, the viewing team attacking
+x. NaN marks a missing player.
"""

import warnings

import numpy as np

N_EXT = 5
DELTA_ROWS = 5          # rows per second: the 1 s deltas look this far each way
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16


def _kth(sorted_vals, k):
    """Column k of an ascending sort with NaN last; NaN where fewer than k + 1 values."""
    return sorted_vals[:, k]


def _nanmean(a, axis=-1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _nanmedian(a, axis=-1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def convergence_point(p, v, min_speed=1.5):
    """Where the moving players' velocity lines meet, by weighted least squares.

    p, v (T, N, 2). Returns c (T, 2), residual (T,) in metres (RMS distance of c to the lines),
    valid (T,). Players run towards the ball, so c is a rough ball proxy."""
    s = np.hypot(v[..., 0], v[..., 1])
    ok = np.isfinite(s) & np.isfinite(p[..., 0]) & (s > min_speed)
    w = np.where(ok, s, 0.0)
    u = np.where(ok[..., None], v / np.where(s > 0, s, 1.0)[..., None], 0.0)
    pp = np.where(ok[..., None], p, 0.0)
    # M_i = I - u u^T ;  A = sum w M_i ;  b = sum w M_i p_i
    m00 = 1 - u[..., 0] ** 2
    m01 = -u[..., 0] * u[..., 1]
    m11 = 1 - u[..., 1] ** 2
    a00, a01, a11 = (w * m00).sum(1), (w * m01).sum(1), (w * m11).sum(1)
    b0 = (w * (m00 * pp[..., 0] + m01 * pp[..., 1])).sum(1)
    b1 = (w * (m01 * pp[..., 0] + m11 * pp[..., 1])).sum(1)
    det = a00 * a11 - a01 ** 2
    valid = (ok.sum(1) >= 3) & (det > 1e-3 * np.maximum(a00 * a11, 1e-9))
    det = np.where(valid, det, 1.0)
    c = np.stack([(a11 * b0 - a01 * b1) / det, (a00 * b1 - a01 * b0) / det], -1)
    d = c[:, None] - pp
    perp = d - (d * u).sum(-1, keepdims=True) * u
    res = np.sqrt((w * (perp ** 2).sum(-1)).sum(1) / np.maximum(w.sum(1), 1e-9))
    # Also discard a point outside |x| < 70 m, |y| < 50 m of the pitch centre
    # (a 140 x 100 m box, in the centred metres view() passes in).
    valid &= (np.abs(c[:, 0]) < 70) & (np.abs(c[:, 1]) < 50)
    c = np.where(valid[:, None], c, 0.0)
    return c, np.where(valid, res, 0.0), valid


def _pairs(a, b):
    """(T, Na, Nb) distances, inf where either is missing."""
    d = np.linalg.norm(a[:, :, None].astype(np.float32) - b[:, None].astype(np.float32), axis=-1)
    return np.where(np.isfinite(d), d, np.inf)


def _delta(x, k=None):
    """x(t + k) - x(t - k), zero at the ends and across NaN."""
    k = DELTA_ROWS if k is None else k
    out = np.zeros_like(x)
    if len(x) > 2 * k:
        out[k:-k] = x[2 * k:] - x[:-2 * k]
    return np.where(np.isfinite(out), out, 0.0)


def _side(p, v, sign, d_opp, c, c_ok, length, width, delta_rows):
    """Scalars for one team. `sign` is +1 if it attacks +x. depth = sign * x grows towards the
    goal it attacks; its own goal is at depth -length / 2."""
    depth, y = sign * p[..., 0], p[..., 1]
    vd, vy = sign * v[..., 0], v[..., 1]
    s = np.hypot(vd, vy)
    here = np.isfinite(depth)
    n = here.sum(1)
    cd, cy = _nanmean(depth), _nanmean(y)
    stretch = _nanmean(np.hypot(depth - cd[:, None], y - cy[:, None]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd, sy = np.nanstd(depth, axis=1), np.nanstd(y, axis=1)
    up = np.sort(depth, axis=1)                         # deepest first
    down = -np.sort(-depth, axis=1)                     # most advanced first
    ys, yd = np.sort(y, axis=1), -np.sort(-y, axis=1)
    gk, line, third = _kth(up, 0), _kth(up, 1), _kth(up, 2)
    front, front2 = _kth(down, 0), _kth(down, 1)
    out_y = np.where(here, np.abs(y) - width / 2, -np.inf)
    idx = out_y.argmax(1)
    rows = np.arange(len(p))
    touch = np.clip(out_y[rows, idx], -20, 5)
    touch_speed = s[rows, idx]
    near_opp = d_opp.min(2)                             # (T, P)
    near_opp = np.where(here & np.isfinite(near_opp), near_opp, np.nan)
    # Pressure on each player: every opponent closer than 4 m adds 1 - distance / 4
    # (metres; 0 for a missing player).
    press = np.where(here[..., None], np.clip(1 - d_opp / 4.0, 0, None), 0.0).sum(2)
    dc = np.where(here, np.hypot(p[..., 0] - c[:, None, 0], y - c[:, None, 1]), np.inf)
    dc_sorted = np.sort(dc, axis=1)
    dc3 = np.where(np.isfinite(dc_sorted[:, :3]), dc_sorted[:, :3], np.nan)
    to_c = c[:, None] - p
    cosang = (to_c[..., 0] * v[..., 0] + to_c[..., 1] * v[..., 1]) / np.maximum(s * np.hypot(to_c[..., 0], to_c[..., 1]), 1e-6)
    swarm = (here & (s > 3) & (cosang > 0.8) & (dc < 20)).sum(1)
    vmean = np.hypot(_nanmean(vd), _nanmean(vy))
    feats = [
        cd / 50, sd / 20, sy / 20, stretch / 20,
        gk / 50, line / 50, third / 50, _nanmedian(depth) / 50, front2 / 50, front / 50,
        (front2 - line) / 50, (_kth(yd, 1) - _kth(ys, 1)) / 50,
        _nanmean(vd) / 8, _nanmean(s) / 8, _nanmedian(s) / 8,
        _nanmean(np.where(here, s < 1.0, np.nan)), (vd > 4).sum(1) / 11, (vd < -4).sum(1) / 11,
        vmean / np.maximum(_nanmean(s), 0.1),
        touch / 5, touch_speed / 8, n / 11, (depth > 0).sum(1) / 11,
        press.max(1), press.sum(1) / np.maximum(n, 1), _nanmedian(near_opp) / 10,
        np.where(c_ok, dc_sorted[:, 0], 0) / 20, np.where(c_ok, _nanmean(dc3), 0) / 20,
        np.where(c_ok, (depth < sign * c[:, None, 0]).sum(1), 0) / 11, np.where(c_ok, swarm, 0) / 11,
        _delta(cd, delta_rows) / 10, _delta(stretch, delta_rows) / 10,
        _delta(line, delta_rows) / 10, _delta(_nanmean(vd), delta_rows) / 8,
        # y-signed (mirrored by the y flip): keep these two last
        cy / 34, _nanmean(vy) / 8,
    ]
    feats = np.stack([np.where(np.isfinite(f), f, 0.0) for f in feats], -1)
    rank = np.argsort(np.argsort(np.where(here, depth, np.inf), axis=1), axis=1) / np.maximum(n, 1)[:, None]
    near_mate = _pairs(p, p)
    near_mate[:, np.arange(p.shape[1]), np.arange(p.shape[1])] = np.inf
    near_mate = near_mate.min(2)
    ext = np.stack([
        np.clip(np.where(np.isfinite(near_opp), near_opp, 30.0) / 10, 0, 3),
        press,
        np.clip(np.where(np.isfinite(near_mate), near_mate, 30.0) / 10, 0, 3),
        np.where(c_ok[:, None], np.clip(np.where(np.isfinite(dc), dc, 60.0) / 20, 0, 3), 0.0),
        rank,
    ], -1)
    ext[~here] = 0.0
    return feats.astype(np.float32), ext.astype(np.float32)


N_SIDE = 36
N_SHARED = 10
N_GLOB = 2 * N_SIDE + N_SHARED
# Indices of glob that change sign when the pitch is mirrored in y.
Y_SIGNED = (N_SIDE - 2, N_SIDE - 1, 2 * N_SIDE - 2, 2 * N_SIDE - 1, 2 * N_SIDE + 1)


def view(own_p, own_v, opp_p, opp_v, length, width, *, delta_rows=None):
    """Global and per-player tactical features in the viewing team's frame."""
    allp = np.concatenate([own_p, opp_p], 1)
    allv = np.concatenate([own_v, opp_v], 1)
    c, res, c_ok = convergence_point(allp, allv)
    d = _pairs(own_p, opp_p)
    g_own, e_own = _side(own_p, own_v, +1, d, c, c_ok, length, width, delta_rows)
    g_opp, e_opp = _side(opp_p, opp_v, -1, d.transpose(0, 2, 1), c, c_ok, length, width, delta_rows)
    s = np.hypot(allv[..., 0], allv[..., 1])
    here = np.isfinite(allp[..., 0])
    x, y = allp[..., 0], allp[..., 1]
    in_width = np.abs(y) < BOX_HALF_WIDTH
    own_box = (here & in_width & (x < -length / 2 + BOX_DEPTH)).sum(1)
    opp_box = (here & in_width & (x > length / 2 - BOX_DEPTH)).sum(1)
    corner = np.where(here, np.hypot(np.abs(x) - length / 2, np.abs(y) - width / 2), np.inf).min(1)
    kick = ((own_p[..., 0] <= 1.0) | ~np.isfinite(own_p[..., 0])).all(1) & \
           ((opp_p[..., 0] >= -1.0) | ~np.isfinite(opp_p[..., 0])).all(1)
    shared = np.stack([
        c[:, 0] / 50, c[:, 1] / 34, np.clip(res, 0, 30) / 10, c_ok.astype(np.float64),
        _nanmedian(s) / 8, _nanmean(np.where(here, s < 1.0, np.nan)),
        own_box / 22, opp_box / 22, np.clip(np.where(np.isfinite(corner), corner, 40.0), 0, 40) / 20,
        kick.astype(np.float64),
    ], -1)
    shared = np.where(np.isfinite(shared), shared, 0.0).astype(np.float32)
    return np.concatenate([g_own, g_opp, shared], 1), e_own, e_opp
