"""Shared pieces of the harmoniser: the output shape, resampling, speed and validation.

A source loader returns a "raw" dict on the provider's own frame grid; finalise() turns that into
the contract shape (see CONTRACT.md) and validate() checks it. Loaders never write files.
"""

import os

import numpy as np

from .matches import BY_KEY, CONTRACT_VERSION
from .paths import EXTERNAL, HARMONISED, ROOT

FPS = 25
SLOTS = 23                      # player slots per team
MAX_SPEED = 12.0                # m/s. Faster than any player; anything above is set to NaN
# Label codes (CONTRACT.md): H home, A away, D dead ball, X unlabelled.
H, A, D, X = 0, 1, 2, 3         # same codes as evaluation/possession_metrics.py
LABELS = "HADX"

ARRAYS = ("frame", "period", "time_s", "home_xy", "away_xy", "home_speed", "away_speed",
          "ball_xy", "home_observed", "away_observed", "ball_observed", "label", "interpolated")
SCALARS = ("match_key", "source", "label_source", "split", "pitch_length_m", "pitch_width_m",
           "fps", "native_fps", "contract_version")


def centre_metres_to_norm(x, y, length, width, y_up=True):
    """Centre-origin metres -> 0-1 with the origin top-left (the Metrica convention)."""
    xn = np.asarray(x, dtype=np.float64) / length + 0.5
    yn = np.asarray(y, dtype=np.float64) / width
    return xn, (0.5 - yn) if y_up else (0.5 + yn)


class SlotAllocator:
    """Gives each track a column. A column is reused once its track has ended, which is how the
    competition tracking behaves: slots are not people."""

    def __init__(self, reuse):
        self.reuse = reuse
        self.slot_of = {}
        self.last_seen = {}
        self.owner = [None] * SLOTS

    def slot(self, track, t):
        if track in self.slot_of:
            self.last_seen[track] = t
            return self.slot_of[track]
        for i in range(SLOTS):
            holder = self.owner[i]
            # A slot is handed on only after it has stood empty for a native frame. Otherwise the
            # resampler would interpolate between two different players, and the speed would be
            # differenced across the handover.
            if holder is None or (self.reuse and self.last_seen[holder] < t - 1):
                if holder is not None:
                    del self.slot_of[holder]
                self.owner[i] = track
                self.slot_of[track] = i
                self.last_seen[track] = t
                return i
        return None     # more simultaneous tracks than slots: the caller counts the drop


def _resample(values, src_pos):
    """Linear interpolation of `values` (N, ...) at fractional row positions. A result is NaN
    unless both neighbours are present, so gaps are never bridged."""
    lo = np.floor(src_pos).astype(np.int64)
    hi = np.minimum(lo + 1, len(values) - 1)
    w = (src_pos - lo).astype(np.float32)
    w = w.reshape((-1,) + (1,) * (values.ndim - 1))
    exact = w == 0
    with np.errstate(invalid="ignore"):
        mixed = (1 - w) * values[lo] + w * values[hi]
    return np.where(exact, values[lo], mixed).astype(np.float32)


def derive_speed(xy, length, width):
    """Speed in m/s from normalised positions at 25 fps, by a centred difference over 0.16 s."""
    metres = xy.astype(np.float64) * np.array([length, width])
    out = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    step = 2    # frames either side of the centre; 2 * step / FPS is the docstring's 0.16 s
    if len(xy) > 2 * step:
        delta = metres[2 * step:] - metres[:-2 * step]
        out[step:-step] = np.hypot(delta[..., 0], delta[..., 1]) / (2 * step / FPS)
    out[np.isnan(xy[..., 0])] = np.nan
    return out


def clean_speed(speed, xy):
    """Speed is unknown where the position is, and above MAX_SPEED it is a tracking artefact."""
    speed = speed.astype(np.float32).copy()
    speed[np.isnan(xy[..., 0])] = np.nan
    with np.errstate(invalid="ignore"):
        speed[speed > MAX_SPEED] = np.nan
    return speed


def finalise(key, raw):
    """Raw loader output -> contract arrays.

    raw: native_fps, pitch (length_m, width_m), and one entry per period in raw["periods"], each
    with home_xy (N, SLOTS, 2), away_xy, ball_xy (N, 2), label (N,), and optionally home_speed /
    away_speed (N, SLOTS) in m/s and home_observed / away_observed (N, SLOTS) / ball_observed (N,)
    booleans. Rows are consecutive native frames; missing data is NaN / X.

    Nothing here is a modelling choice: units, frame rate and layout change, values do not. The
    label is the provider's, untouched.
    """
    entry = BY_KEY[key]
    native = raw["native_fps"]
    length, width = raw["pitch"]
    parts = {name: [] for name in ARRAYS}
    for number, p in raw["periods"]:
        n = len(p["label"])
        if native == FPS:
            pos = np.arange(n, dtype=np.float64)
        else:
            duration = (n - 1) / native
            pos = np.arange(int(np.floor(duration * FPS + 1e-9)) + 1) / FPS * native
            pos = np.minimum(np.round(pos, 6), n - 1)
        # An absolute test: np.isclose is relative, and late in a half it would call every frame
        # "on the native grid".
        on_grid = np.abs(pos - np.round(pos)) < 1e-6
        t = len(pos)
        home = _resample(p["home_xy"], pos)
        away = _resample(p["away_xy"], pos)
        ball = _resample(p["ball_xy"], pos)
        lo = np.floor(pos + 1e-6).astype(np.int64)
        hi = np.where(on_grid, lo, np.minimum(lo + 1, n - 1))
        label = p["label"][lo]
        observed = {}
        for name, xy in (("home", p["home_xy"]), ("away", p["away_xy"]), ("ball", p["ball_xy"])):
            seen = p.get(name + "_observed")
            seen = np.isfinite(xy[..., 0]) if seen is None else (seen & np.isfinite(xy[..., 0]))
            # A resampled frame counts as observed only if both of its native neighbours were.
            observed[name] = seen[lo] & seen[hi]
        if "home_speed" in p and native == FPS:
            home_speed, away_speed = p["home_speed"], p["away_speed"]
        else:
            home_speed = derive_speed(home, length, width)
            away_speed = derive_speed(away, length, width)
        parts["period"].append(np.full(t, number, dtype=np.int8))
        parts["time_s"].append((np.arange(t) / FPS).astype(np.float32))
        parts["home_xy"].append(home)
        parts["away_xy"].append(away)
        parts["home_speed"].append(clean_speed(home_speed, home))
        parts["away_speed"].append(clean_speed(away_speed, away))
        parts["ball_xy"].append(ball)
        for name in ("home", "away", "ball"):
            parts[name + "_observed"].append(observed[name])
        parts["label"].append(label.astype(np.uint8))
        parts["interpolated"].append(~on_grid)
    out = {name: np.concatenate(parts[name]) for name in ARRAYS if name != "frame"}
    out["frame"] = np.arange(len(out["label"]), dtype=np.int32)
    out.update(match_key=key, source=entry["source"], label_source=entry["label_source"],
               split=entry["split"], pitch_length_m=float(length), pitch_width_m=float(width),
               fps=FPS, native_fps=float(native), contract_version=CONTRACT_VERSION)
    return out


def validate(m):
    """Raise AssertionError with a readable reason if `m` breaks the contract."""
    t = len(m["label"])
    shapes = {"frame": (t,), "period": (t,), "time_s": (t,), "home_xy": (t, SLOTS, 2),
              "away_xy": (t, SLOTS, 2), "home_speed": (t, SLOTS), "away_speed": (t, SLOTS),
              "ball_xy": (t, 2), "home_observed": (t, SLOTS), "away_observed": (t, SLOTS),
              "ball_observed": (t,), "label": (t,), "interpolated": (t,)}
    dtypes = {"frame": np.int32, "period": np.int8, "time_s": np.float32, "home_xy": np.float32,
              "away_xy": np.float32, "home_speed": np.float32, "away_speed": np.float32,
              "ball_xy": np.float32, "home_observed": np.bool_, "away_observed": np.bool_,
              "ball_observed": np.bool_, "label": np.uint8, "interpolated": np.bool_}
    for name in ARRAYS:
        assert name in m, "missing array {}".format(name)
        assert m[name].shape == shapes[name], "{} has shape {}, expected {}".format(
            name, m[name].shape, shapes[name])
        assert m[name].dtype == dtypes[name], "{} has dtype {}".format(name, m[name].dtype)
    for name in SCALARS:
        assert name in m, "missing scalar {}".format(name)
    assert t > 0, "no frames"
    assert np.array_equal(m["frame"], np.arange(t)), "frame is not 0..T-1"
    assert set(np.unique(m["period"])) <= {1, 2}, "period outside {1, 2}"
    assert np.all(np.diff(m["period"].astype(int)) >= 0), "periods out of order"
    assert m["label"].max() <= X, "label code above 3"
    for name in ("home", "away", "ball"):
        assert not np.any(m[name + "_observed"] & np.isnan(m[name + "_xy"][..., 0])), \
            "{} marked observed where there is no position".format(name)
    assert int(m["fps"]) == FPS
    for name in ("home_xy", "away_xy", "ball_xy"):
        v = m[name]
        finite = v[np.isfinite(v)]
        # Tracks legitimately run a little off the pitch (throw-ins, the technical area).
        assert finite.size == 0 or (finite.min() > -0.5 and finite.max() < 1.5), \
            "{} is not normalised (range {:.2f}..{:.2f})".format(name, finite.min(), finite.max())
        assert np.array_equal(np.isnan(v[..., 0]), np.isnan(v[..., 1])), \
            "{} has x without y".format(name)
    for side in ("home", "away"):
        speed = m[side + "_speed"]
        assert not np.any(np.isfinite(speed) & np.isnan(m[side + "_xy"][..., 0])), \
            "{} speed without a position".format(side)
        finite = speed[np.isfinite(speed)]
        assert finite.size == 0 or finite.min() >= 0, "negative speed"
        assert finite.size == 0 or finite.max() <= MAX_SPEED, \
            "{} speed above {} m/s ({:.0f})".format(side, MAX_SPEED, finite.max())
    labels = BY_KEY[str(m["match_key"])]["labels"]
    if "D" not in labels:
        assert not np.any(m["label"] == D), "D label in a source that has none"
    return True


def summary(m):
    """Numbers worth reading before trusting a match."""
    label = m["label"]
    t = len(label)
    graded = label != X
    alive = (label == H) | (label == A)
    change = np.flatnonzero(np.diff(label.astype(int)) != 0) + 1
    starts = np.concatenate([[0], change])
    spells = int(np.sum(alive[starts]))
    minutes = graded.sum() / FPS / 60
    players = np.isfinite(m["home_xy"][..., 0]).sum(1) + np.isfinite(m["away_xy"][..., 0]).sum(1)
    seen = m["home_observed"].sum(1) + m["away_observed"].sum(1)
    return {
        "frames": t,
        "minutes": round(t / FPS / 60, 1),
        "labelled_pct": round(100 * graded.mean(), 1),
        "H_pct": round(100 * np.mean(label == H), 1),
        "A_pct": round(100 * np.mean(label == A), 1),
        "D_pct": round(100 * np.mean(label == D), 1),
        "X_pct": round(100 * np.mean(label == X), 1),
        "possessions": spells,
        "possessions_per_min": round(spells / minutes, 2) if minutes else 0.0,
        "mean_possession_s": round(alive.sum() / FPS / spells, 1) if spells else 0.0,
        "ball_present_pct": round(100 * np.mean(np.isfinite(m["ball_xy"][:, 0])), 1),
        "ball_observed_pct": round(100 * float(m["ball_observed"].mean()), 1),
        "mean_players": round(float(players.mean()), 1),
        "mean_players_observed": round(float(seen.mean()), 1),
        "interpolated_pct": round(100 * m["interpolated"].mean(), 1),
    }


def save(m, directory=None):
    directory = directory or HARMONISED
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, str(m["match_key"]) + ".npz")
    np.savez_compressed(path, **m)
    return path


def load(key, directory=None, labels="provider"):
    """One harmonised match. `labels` picks a label view (see datalake/remap.py): "provider" is
    what is stored; any other view is computed on load and never written back."""
    path = os.path.join(directory or HARMONISED, key + ".npz")
    with np.load(path, allow_pickle=False) as data:
        out = {name: data[name] for name in ARRAYS}
        for name in SCALARS:
            value = data[name]
            out[name] = value.item() if value.ndim == 0 else value
    from . import remap
    out["label"] = remap.apply(labels, out["label"], out["period"], match=out)
    out["label_view"] = labels
    return out


def empty_period(n):
    """Blank native-grid arrays for one period."""
    return {
        "home_xy": np.full((n, SLOTS, 2), np.nan, dtype=np.float32),
        "away_xy": np.full((n, SLOTS, 2), np.nan, dtype=np.float32),
        "ball_xy": np.full((n, 2), np.nan, dtype=np.float32),
        "label": np.full(n, X, dtype=np.uint8),
    }
