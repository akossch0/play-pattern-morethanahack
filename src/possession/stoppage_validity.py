"""Estimate whether short possession spells survive overlap matching."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .interruption import nonoverlapping
from .stoppage_kernel import feature_inputs, restart_edge_features

REPEAT_SEEDS = (20270721, 20270722)


def _network(feature_count):
    return nn.Sequential(
        nn.Linear(feature_count, 44), nn.LayerNorm(44), nn.GELU(),
        nn.Dropout(.16), nn.Linear(44, 22), nn.GELU(), nn.Linear(22, 2),
    )


# Fitting/development helper: nothing on the predict.py path calls this.
def fit_validity(parts, seeds):
    """Fit credit and validity heads using labeled external candidate rows."""
    features, validity, rewards = (
        np.concatenate(items) for items in zip(*parts, strict=True)
    )
    center, scale = features.mean(0), features.std(0).clip(.04)
    standardized = torch.from_numpy(((features - center) / scale).clip(-6, 6))
    targets = torch.from_numpy(validity)
    rewards = torch.from_numpy(rewards)
    members = []
    for seed in seeds:
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        model = _network(features.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0011, weight_decay=.07)
        for _ in range(420):
            batch = torch.randint(len(features), (140,), generator=generator)
            predicted = model(standardized[batch])
            objective = F.smooth_l1_loss(predicted[:, 0], rewards[batch], beta=.28)
            objective += .45 * F.binary_cross_entropy_with_logits(predicted[:, 1], targets[batch])
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
        members.append(model.eval())
    return members, center, scale


# Fitting/development helper: nothing on the predict.py path calls this.
def encode_validity(trained):
    """Convert fitted members and standardization statistics to checkpoint tensors."""
    members, center, scale = trained
    return {
        "center": torch.from_numpy(center),
        "scale": torch.from_numpy(scale),
        "members": [network.state_dict() for network in members],
    }


def decode_validity(record):
    center = record["center"].numpy()
    scale = record["scale"].numpy()
    if center.ndim != 1 or scale.shape != center.shape or np.any(scale <= 0):
        raise ValueError("Invalid stoppage validity feature scaling")
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or not record["members"]:
        raise ValueError("Missing or nonfinite stoppage validity coefficients")
    members = []
    for weights in record["members"]:
        model = _network(len(center))
        model.load_state_dict(weights, strict=True)
        members.append(model.eval())
    return members, center, scale


@torch.inference_mode()
def score_model(trained, features):
    members, center, scale = trained
    standardized = torch.from_numpy(((features - center) / scale).clip(-6, 6))
    scores = torch.stack([member(standardized) for member in members]).mean(0)
    return scores[:, 0].numpy(), scores[:, 1].sigmoid().numpy()


def validity_candidates(reference, period, evidence, pressure, forecast_pairs):
    """Measure short live spells immediately before or after a stoppage."""
    probability = np.mean([member[0] for member in forecast_pairs], axis=0)
    edges = np.mean([member[1] for member in forecast_pairs], axis=0)
    changes = (reference[1:] != reference[:-1]) | (period[1:] != period[:-1])
    boundaries = np.r_[0, np.flatnonzero(changes) + 1, len(reference)]
    states = reference[boundaries[:-1]]
    candidates = []
    for i in range(1, len(states) - 1):
        before, side, after = map(int, states[i - 1:i + 2])
        lo, a, b, hi = map(int, boundaries[i - 1:i + 3])
        # [a, b) is the candidate spell, with neighbours [lo, a) and [b, hi). Keep it only if it
        # is 7-70 frames long, team-owned (codes 0/1; 2 is dead ball, 3 unlabelled) and all
        # three spells lie in one period.
        if not 7 <= b - a <= 70 or side >= 2 or period[lo] != period[hi - 1]:
            continue
        if before == 2 and after == 1 - side and hi - b >= 10:
            row = a, a, b, hi, side
        elif after == 2 and before == 1 - side and a - lo >= 10:
            row = lo, a, b, b, side
        else:
            continue
        features = restart_edge_features(evidence, pressure, probability, edges, row)
        candidates.append((row, features))
    return candidates


def select_validity(reference, candidates, tactics, joint_estimate, multi_estimate):
    """Accept an edit when both fitted draft families and tactics support removal."""
    prediction = reference.copy()
    if not candidates:
        return prediction
    design = np.asarray([features for _, features in candidates], dtype=np.float32)
    gain, validity = multi_estimate(design)
    joint_gain, joint_validity = joint_estimate(design)
    accepted = (gain > .065) & (validity <= .31) & (joint_gain > .065) & (joint_validity <= .31)
    options, weight = [], []
    for i, ((lo, a, b, hi, side), _) in enumerate(candidates):
        if accepted[i] and tactics[a:b, 1 - side].mean() > tactics[a:b, side].mean() + .06:
            options.append((lo, hi, a, b, side))
            weight.append(float(gain[i] - .065))
    for i in nonoverlapping(options, weight, range(len(options))):
        _, _, a, b, side = options[i]
        prediction[a:b] = 1 - side
    return prediction


def load_repeat(path: Path):
    record = torch.load(path, map_location="cpu", weights_only=True)
    return decode_validity(record["joint"]), decode_validity(record["multi"])


def predict_repeat(reference, match, forecasts, tactics, models, *, prepared=None):
    """Apply the independently fitted pair of stoppage-edge ownership models."""
    evidence, pressure = feature_inputs(match) if prepared is None else prepared
    bank = validity_candidates(reference, match["period"], evidence, pressure, forecasts)
    joint, multi = models
    return select_validity(reference, bank, tactics, lambda x: score_model(joint, x),
                           lambda x: score_model(multi, x))
