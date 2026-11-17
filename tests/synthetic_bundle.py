"""Write a synthetic tactical tracking bundle with the organizer's file layout and frame count.

The tracks are seeded random motion, so the output says nothing about model quality. The
bundle only exercises loading, inference, validation and byte-for-byte repeatability.
"""

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import games

SLOTS_PER_TEAM = 23
SEED = 20261117


def _smooth_walk(rng, frames, columns, scale, window):
    steps = rng.normal(0.0, scale, (frames + window, columns)).astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / window
    drift = np.stack([np.convolve(steps[:, c], kernel, mode="valid")[:frames] for c in range(columns)], axis=1)
    return np.cumsum(drift, axis=0)


def simulate(frames, rng):
    """Normalized player positions (frames, 46, 2) and ball positions (frames, 2), NaN if missing."""
    slots = 2 * SLOTS_PER_TEAM
    anchors = np.column_stack((rng.uniform(0.1, 0.9, slots), rng.uniform(0.1, 0.9, slots)))
    walk = _smooth_walk(rng, frames, slots * 2, 0.004, 25).reshape(frames, slots, 2)
    # Subtract a linear ramp that reaches the walk's final value at the last frame, so the
    # walk ends at zero.
    walk -= np.linspace(0, 1, frames)[:, None, None] * walk[-1]
    players = (anchors + 0.6 * walk).clip(0.02, 0.98).astype(np.float32)
    active = np.zeros((frames, slots), dtype=bool)
    active[:, :11] = active[:, SLOTS_PER_TEAM:SLOTS_PER_TEAM + 11] = True
    for slot in range(slots):
        for start in rng.integers(0, frames, 12):
            active[start:start + int(rng.integers(50, 4000)), slot] ^= True
    players[~active] = np.nan

    ball = np.empty((frames, 2), dtype=np.float32)
    carriers = np.flatnonzero(active.mean(axis=0) > 0.5)
    frame, position = 0, np.array([0.5, 0.5], dtype=np.float32)
    while frame < frames:
        hold = int(rng.integers(15, 220))
        travel = int(rng.integers(8, 40))
        stop = min(frames, frame + travel)
        if rng.random() < 0.06:
            target = np.array([rng.choice([-0.03, 1.03]), rng.uniform(0, 1)], dtype=np.float32)
            ball[frame:stop] = position + np.linspace(0, 1, stop - frame)[:, None] * (target - position)
            end = min(frames, stop + int(rng.integers(100, 500)))
            ball[stop:end] = target
            frame, position = end, target
            continue
        carrier = int(rng.choice(carriers))
        path = np.nan_to_num(players[:, carrier], nan=0.5)
        target = path[min(stop, frames - 1)]
        ball[frame:stop] = position + np.linspace(0, 1, stop - frame)[:, None] * (target - position)
        end = min(frames, stop + hold)
        ball[stop:end] = path[stop:end] + rng.normal(0, 0.002, (end - stop, 2))
        frame, position = end, ball[end - 1].copy()
    hidden = np.zeros(frames, dtype=bool)
    for start in rng.integers(0, frames, frames // 180):
        hidden[start:start + int(rng.integers(5, 220))] = True
    ball[hidden] = np.nan
    return players, ball


def write_bundle(directory, stem=None, seed=SEED):
    """Write the metadata and raw tracking files; ``stem`` replaces the organizer's file stem."""
    frames = games.source_frames("tactical", games.GAME)
    stem = stem or games.SOURCES["tactical"]["stem"]
    folder = Path(directory) / "tactical"
    meta = folder / (stem + games.SUFFIXES["atd_meta"])
    raw = folder / (stem + games.SUFFIXES["atd_raw"])
    meta.parent.mkdir(parents=True, exist_ok=True)
    players, ball = simulate(frames, np.random.default_rng(seed))
    speed = np.zeros(players.shape[:2], dtype=np.float32)
    pitch = np.asarray(games.GAMES[games.GAME]["pitch"], dtype=np.float32)
    speed[1:] = np.linalg.norm((players[1:] - players[:-1]) * pitch, axis=-1) * 25
    entries = "".join(
        f'<Player id="P{slot}" teamId="T{slot // SLOTS_PER_TEAM}"/>' for slot in range(players.shape[1])
    )
    meta.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<MatchParameters><Teams><Team id="T0"/><Team id="T1"/></Teams>'
        f"<Players>{entries}</Players></MatchParameters>\n"
    )
    with raw.open("w", encoding="ascii", newline="\n") as handle:
        for frame in range(frames):
            groups = ";".join(
                "NaN,NaN,NaN" if np.isnan(x) else f"{x:.5f},{y:.5f},{min(v, 11.9):.2f}"
                for (x, y), v in zip(players[frame].tolist(), np.nan_to_num(speed[frame]).tolist())
            )
            x, y = ball[frame].tolist()
            tail = "NaN,NaN" if np.isnan(x) else f"{x:.5f},{y:.5f}"
            handle.write(f"{frame + 1}:{groups}:{tail}\n")
    return meta, raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Bundle root to create (tactical/ is written inside)")
    parser.add_argument("--stem", help="File stem, if not the organizer's")
    args = parser.parse_args()
    for path in write_bundle(args.directory, args.stem):
        print(path)
