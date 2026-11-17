"""Score short stoppage-edge possessions with role-specific validity heads."""

from pathlib import Path

import torch

from .stoppage_kernel import feature_inputs
from .stoppage_validity import (
    decode_validity,
    encode_validity,
    fit_validity,
    score_model,
    select_validity,
    validity_candidates,
)


ROLE_SEEDS = (20270713, 20270714)
# Weight of the role-specific fit in the blend; the shared fit gets 1 - ROLE_WEIGHT.
ROLE_WEIGHT = 0.65


# Fitting/development helper: nothing on the predict.py path calls this.
def _fit_pooled(parts):
    total = fit_validity(parts, ROLE_SEEDS)
    by_role = []
    for role in (0, 1):
        selected = []
        for features, validity, gain in parts:
            keep = features[:, 0] == role
            selected.append((features[keep], validity[keep], gain[keep]))
        if sum(len(gain) for _, _, gain in selected) < 20:
            raise ValueError(
                "Too few external training examples for a stoppage ownership role"
            )
        by_role.append(fit_validity(selected, ROLE_SEEDS))
    return total, tuple(by_role)


# Fitting/development helper: nothing on the predict.py path calls this.
def fit_roles(original, alternate):
    """Fit natural-draft and combined-draft heads, each with shared and role fits."""
    return _fit_pooled(original), _fit_pooled(original + alternate)


def _pooled(model, features):
    total, by_role = model
    gain, validity = score_model(total, features)
    for role, fitted in enumerate(by_role):
        # Column 0 is restart_edge_features' `a == lo` flag: role 1 is a spell that directly
        # follows a stoppage, role 0 one that directly precedes it.
        keep = features[:, 0] == role
        if keep.any():
            role_gain, role_validity = score_model(fitted, features[keep])
            gain[keep] = (1 - ROLE_WEIGHT) * gain[keep] + ROLE_WEIGHT * role_gain
            validity[keep] = (1 - ROLE_WEIGHT) * validity[
                keep
            ] + ROLE_WEIGHT * role_validity
    return gain, validity


# Fitting/development helper: nothing on the predict.py path calls this.
def encode_roles(models):
    return [
        {
            "shared": encode_validity(total),
            "roles": [encode_validity(model) for model in roles],
        }
        for total, roles in models
    ]


def load_roles(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return tuple(
        (
            decode_validity(branch["shared"]),
            tuple(decode_validity(role) for role in branch["roles"]),
        )
        for branch in checkpoint["models"]
    )


def select_roles(reference, bank, tactics, models):
    joint, multi = models
    return select_validity(
        reference,
        bank,
        tactics,
        lambda f: _pooled(joint, f),
        lambda f: _pooled(multi, f),
    )


def predict_roles(reference, match, forecasts, tactics, models, *, prepared=None):
    """Remove short possessions when both stoppage-edge fits agree."""
    evidence, pressure = feature_inputs(match) if prepared is None else prepared
    bank = validity_candidates(
        reference, match["period"], evidence, pressure, forecasts
    )
    return select_roles(reference, bank, tactics, models)
