"""Apply the adapted tactics GRUs to frame-aligned tracking."""

import json
from pathlib import Path

import numpy as np
import torch

from .models import gru_features, team_shape
from .models.gru import TacticNet


PLAYER_TOKENS = ("own_h", "opp_h", "own_a", "opp_a")
MODEL_INPUTS = (*PLAYER_TOKENS, "pres_h", "pres_a", "glob_h", "glob_a",
                "ball_h", "ball_a", "ball_pres")


def _mirror(batch):
    mirrored = batch.copy()
    for keys, fields in (
        (PLAYER_TOKENS, [1, 3, 5]),  # y-position, y-centroid-offset, y-velocity
        (("ball_h", "ball_a"), [1, 3]),
        (("glob_h", "glob_a"), list(team_shape.Y_SIGNED)),
    ):
        for key in keys:
            mirrored[key] = batch[key].clone()
            mirrored[key][..., fields] *= -1
    return mirrored


def _class_probabilities(model, batch):
    team, dead = model(*(batch[key] for key in MODEL_INPUTS))
    dead = torch.sigmoid(dead)
    home = torch.sigmoid(team)
    return torch.stack(((1 - dead) * home, (1 - dead) * (1 - home), dead), dim=-1)


def _predict_period(model, features, step, device):
    n = len(features["frames"])
    # Rows per window: 1500 native frames (60 s at 25 fps).
    length = 1500 // step
    output = np.empty((n, 3), dtype=np.float32)
    best_margin = np.full(n, -1)
    starts = list(range(0, max(n - length, 0) + 1, length // 2))
    if starts[-1] + length < n:
        starts.append(n - length)
    for start in starts:
        stop = min(start + length, n)
        batch = {
            key: torch.from_numpy(np.ascontiguousarray(features[key][None, start:stop])).to(device)
            for key in MODEL_INPUTS
        }
        predictions = [
            _class_probabilities(model, view)[0].float().cpu().numpy()
            for view in (batch, _mirror(batch))
        ]
        probability = np.mean(predictions, axis=0)
        idx = np.arange(start, stop)
        margin = np.minimum(idx - start, stop - 1 - idx)
        take = margin > best_margin[idx]
        output[idx[take]] = probability[take]
        best_margin[idx[take]] = margin[take]
    return output


def _load_members(checkpoint_dir, device, members):
    directories = sorted(p for p in checkpoint_dir.iterdir() if p.is_dir() and (p / "model.pt").is_file())
    if members is not None:
        requested = set(members)
        directories = [p for p in directories if p.name in requested]
        if {p.name for p in directories} != requested:
            raise ValueError("One or more requested tactics checkpoints could not be found.")
    if not directories:
        raise ValueError(f"No adapted tactics checkpoints in {checkpoint_dir}")

    networks = []
    settings = None
    for path in directories:
        adapter = json.loads((path / "config.json").read_text())
        config = json.loads((path / adapter["base_config"]).read_text())
        inputs = (adapter["sample_step"], config["vel_span"], config["thaw"], config["touch"])
        if settings is not None and settings != inputs:
            raise ValueError("Tactics checkpoints disagree on their feature configuration.")
        settings = inputs
        model = TacticNet(
            config["width"], config["hidden"], config["layers"], config["dropout"],
            styles=2 if config["styles"] else 1,
            n_ball=4 + (gru_features.N_TOUCH if config["touch"] else 0),
        ).to(device)
        model.load_state_dict(torch.load(path / "model.pt", map_location=device, weights_only=True))
        networks.append(model.eval())
    return networks, settings


@torch.inference_mode()
def predict_gru(graded_match, weights_dir, device="cpu", *, members=None):
    """Return one row of H, A, D probabilities per in-play-period frame."""
    checkpoint_dir = Path(weights_dir) / "tactics-gru"
    networks, settings = _load_members(checkpoint_dir, device, members)
    step, vel_span, thaw, touch = settings
    period = graded_match["period"]
    result = np.zeros((len(period), 3), dtype=np.float32)
    for phase in range(step):
        prepared = gru_features.match_features(
            graded_match, step=step, vel_span=vel_span, with_ball=True, tactics=True,
            phase=phase, thaw=thaw, touch=touch, delta_rows=max(1, round(25 / step)),
        )
        for features in prepared["periods"]:
            samples = features["frames"]
            rows = np.flatnonzero(period == period[samples[0]])
            nxt = np.searchsorted(samples, rows).clip(0, len(samples) - 1)
            prev = (nxt - 1).clip(0)
            nearest = np.where(rows - samples[prev] <= samples[nxt] - rows, prev, nxt)
            probabilities = [
                _predict_period(model, features, step, device)[nearest] for model in networks
            ]
            result[rows] += np.mean(probabilities, axis=0)
    result /= step
    np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=2e-6)
    return result
