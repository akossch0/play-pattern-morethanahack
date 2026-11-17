"""Training inputs: clean harmonised matches made to look like broadcast tracking.

Each development match is degraded with a fixed per-match seed, then its ball is hidden with
gap lengths resampled from the 20 A-League profile matches. The frame models and the span
detector train on these inputs; the tactics model builds its own copies (``tactics_gru``).
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np

from datalake import core, degrade, remap

LABEL_VIEW = "controlled_v2_touch"
GAP_SEED = 0
DEGRADE_SEED = 1
SPLIT = Path(__file__).resolve().parents[1] / "docs" / "training-split.json"


def split():
    return json.loads(SPLIT.read_text())


def development_matches():
    return [f"dfl_{name}" for name in split()["training_pool"]["dfl"]]


def profile_matches():
    return [f"scal_{name}" for name in split()["training_pool"]["skillcorner_a_league"]]


def leave_out_pairs(keys):
    """Every pair of development matches, with the three matches a model for that pair trains on."""
    keys = sorted(keys)
    return [(pair, [key for key in keys if key not in pair]) for pair in combinations(keys, 2)]


def period_bounds(period):
    changes = (np.flatnonzero(period[1:] != period[:-1]) + 1).tolist()
    edges = [0, *changes, len(period)]
    return list(zip(edges[:-1], edges[1:]))


def run_lengths(state):
    bounds = np.r_[0, np.flatnonzero(state[:-1] != state[1:]) + 1, len(state)]
    return state[bounds[:-1]], np.diff(bounds)


def gap_profile(keys=None, directory=None):
    """Lengths of the ball-visible and ball-missing runs in the profile matches."""
    observed_runs, missing_runs = [], []
    for key in keys or profile_matches():
        match = core.load(key, directory=directory)
        observed, period = match["ball_observed"], match["period"]
        for lo, hi in period_bounds(period):
            state, lengths = run_lengths(observed[lo:hi])
            observed_runs.append(lengths[state])
            missing_runs.append(lengths[~state])
    return np.concatenate(observed_runs), np.concatenate(missing_runs)


def sample_gaps(frames, key, period, observed, missing):
    """A missing-ball mask for one period: alternate visible and hidden runs, then crop."""
    seed = np.random.SeedSequence([GAP_SEED, period, *key.encode("ascii")])
    rng = np.random.Generator(np.random.PCG64(seed))
    lengths = []
    total = 0
    while total < 2 * frames:
        visible = int(rng.choice(observed))
        hidden = int(rng.choice(missing))
        lengths.extend((visible, hidden))
        total += visible + hidden
    offset = int(rng.integers(total - frames + 1))
    stream = np.repeat(np.arange(len(lengths)) % 2 == 1, lengths)
    return stream[offset:offset + frames].copy()


def hide_ball(match, name, observed, missing):
    """Apply ``sample_gaps`` to every period of ``match`` in place."""
    for lo, hi in period_bounds(match["period"]):
        gone = sample_gaps(hi - lo, name, int(match["period"][lo]), observed, missing)
        rows = np.arange(lo, hi)[gone]
        match["ball_xy"][rows] = np.nan
        match["ball_observed"][rows] = False
    return match


def prepared_match(key, profile, directory=None):
    """One development match as the frame models saw it, with the training label view."""
    clean = core.load(key, directory=directory)
    label = remap.apply(LABEL_VIEW, clean["label"], clean["period"], match=clean)
    seed = int(np.random.SeedSequence([DEGRADE_SEED, *key.encode("ascii")]).generate_state(1)[0])
    match = degrade.degrade(clean, seed=seed, ball_missing=None)
    hide_ball(match, key, *profile)
    match["label"] = label
    return match


def prepared_matches(keys=None, directory=None):
    profile = gap_profile(directory=directory)
    return {key: prepared_match(key, profile, directory) for key in keys or development_matches()}
