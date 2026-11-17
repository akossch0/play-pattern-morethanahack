"""Run checkpoint ensembles on the two graded halves at 25 Hz."""

from pathlib import Path

import numpy as np
import torch

from .models import geometry
from .models.tcn import TransferTCN, TransitionTCN

CHUNK_FRAMES = 8192
CHECKPOINT_BANKS = (
    ("typed", "model.pt"),
    ("tcn", "base.pt"),
    ("tcn-ensemble", "base.pt"),
)
# One pair of state codes per transition column of the typed model, with 0 = home, 1 = away,
# 2 = dead ball. This module only uses the count. credit_timing.apply documents the first two
# columns as the home-to-away and away-to-home scores.
EDGE_PAIRS = ((0, 1), (1, 0), (2, 0), (2, 1), (0, 2), (1, 2))


def _period_bounds(period):
    breaks = np.r_[0, np.flatnonzero(period[1:] != period[:-1]) + 1, len(period)]
    return tuple(zip(breaks[:-1], breaks[1:]))


@torch.inference_mode()
def _infer_checkpoint(features, bounds, checkpoint_path, typed, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    channels = checkpoint["channels"]
    center, scale = (checkpoint[key].numpy() for key in ("center", "scale"))
    if features.shape[1] != channels * 2 or center.shape != (channels * 2,) or scale.shape != center.shape:
        raise ValueError(f"Invalid feature dimensions for {checkpoint_path}")
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError(f"Invalid feature normalization for {checkpoint_path}")
    model = TransitionTCN(channels) if typed else TransferTCN(channels)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    probabilities = np.empty((len(features), 3), dtype=np.float32)
    edges = np.empty((len(features), len(EDGE_PAIRS)), dtype=np.float32) if typed else None
    for lo, hi in bounds:
        for start in range(lo, hi, CHUNK_FRAMES):
            end = min(hi, start + CHUNK_FRAMES)
            left, right = max(lo, start - model.radius), min(hi, end + model.radius)
            normalized = (features[left:right] - center) / scale
            values = torch.from_numpy(np.ascontiguousarray(normalized.T[None])).to(device)
            scores = model(values)[0, :, start - left:end - left].T
            # Columns 0-2 are normalised log-probabilities (home, away, dead ball), hence exp;
            # typed checkpoints add raw transition logits in columns 3+, hence sigmoid.
            probabilities[start:end] = scores[:, :3].exp().cpu().numpy()
            if edges is not None:
                edges[start:end] = scores[:, 3:].sigmoid().cpu().numpy()
    return probabilities, edges


def predict_pair_streams(graded_match, weights_dir, pairs=None, device="cpu"):
    """Return each pair's three class forecasts and six typed transition streams."""
    period = np.asarray(graded_match["period"])
    if period.ndim != 1 or not len(period) or np.any(period[1:] < period[:-1]):
        raise ValueError("Expected nonempty tracking in period order")
    bounds = _period_bounds(period)
    features = geometry.extract(graded_match)
    weights_dir = Path(weights_dir)
    pair_paths = tuple(
        {tuple(file.relative_to(weights_dir / bank).parts[:2]): file
         for file in (weights_dir / bank).glob(f"*/*/{filename}")}
        for bank, filename in CHECKPOINT_BANKS
    )
    available = set(pair_paths[0])
    if len(available) != 10 or any(set(family) != available for family in pair_paths):
        raise ValueError("The TCN checkpoint banks must contain the same ten excluded pairs")
    requested = available if pairs is None else {tuple(sorted(pair)) for pair in pairs}
    if requested - available:
        raise ValueError("One or more requested checkpoint pairs are unavailable")

    outputs = {}
    for pair in sorted(requested):
        typed, edges = _infer_checkpoint(features, bounds, pair_paths[0][pair], True, device)
        other = [_infer_checkpoint(features, bounds, bank[pair], False, device)[0]
                 for bank in pair_paths[1:]]
        outputs[pair] = np.stack([typed, *other]), edges
    return outputs
