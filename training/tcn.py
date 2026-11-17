"""Fit the frame-level temporal convolutional networks: the two standard fits and the typed fit.

All three take prepared training data and return a checkpoint dictionary; none reads or
writes files.

``data`` maps a match key to ``(features, labels, bounds)``:

    features  (frames, 2 * channels) float32   the home view's channels, then the away view's
    labels    (frames,) integer                0 home, 1 away, 2 dead ball, 3 not labelled
    bounds    [(first, end), ...]              the frame range of each period, end exclusive

Every match in ``data`` is trained on, so leave the held-out matches out of it. Windows are
drawn with ``windows.training_plan``: eight windows of 256 supervised frames a step, 30 % of
them redrawn around a turnover or a restart.

Standard fit (``fit_standard``): ``TransferTCN``, class loss plus the within-spell smoothness
term. It was run in two settings, ``SHORT`` (420 steps, seed 20261112) and ``LONG`` (1500
steps, seed 20261117).

Typed fit (``fit_typed``): ``TransitionTCN``, the same class loss plus a weighted loss on six
transition channels, 1960 steps, seed 20261112.

The fits ran on Apple MPS with two threads; another device or thread count gives weights that
differ in the last digits.
"""

import numpy as np
import torch
from torch.nn import functional as F

from . import windows
from possession.models.tcn import TransferTCN, TransitionTCN

BOUNDARY_FRACTION = .30
LEARNING_RATE = .001
SMOOTHNESS = .02
SHORT = {"steps": 420, "seed": 20261112}
LONG = {"steps": 1500, "seed": 20261117}
TYPED = {"steps": 1960, "seed": 20261112}
TRANSITION_WEIGHT = .4
# (state before, state after) of each transition channel: 0 home, 1 away, 2 dead ball.
PAIRS = ((0, 1), (1, 0), (2, 0), (2, 1), (0, 2), (1, 2))
PAIR_ID = np.full((3, 3), -1, dtype=int)
for _channel, (_old, _new) in enumerate(PAIRS):
    PAIR_ID[_old, _new] = _channel


def _optimizer(model, steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=LEARNING_RATE / 8)
    return optimizer, schedule


def _weights(model):
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def fit_standard(data, steps=SHORT["steps"], seed=SHORT["seed"], device="cpu"):
    """Fit a ``TransferTCN`` on every match in ``data``.

    Returns ``{"state_dict", "center", "scale", "channels", "training_matches", "steps",
    "seed", "sampling", "calibration"}`` (``calibration`` is always False); ``center`` and ``scale`` are the input normalisation the weights
    expect. Pass ``**SHORT`` or ``**LONG`` for the two settings that were run.
    """
    device = torch.device(device)
    center, scale = windows.normalization(data)
    torch.manual_seed(seed)
    model = TransferTCN(len(center) // 2).to(device)
    optimizer, schedule = _optimizer(model, steps)
    plan, sampling = windows.training_plan(data, np.random.default_rng(seed), steps, BOUNDARY_FRACTION)
    mean = torch.from_numpy(center[None, :, None]).to(device)
    std = torch.from_numpy(scale[None, :, None]).to(device)
    radius = model.radius
    for step in range(steps):
        start = step * windows.BATCH_SIZE
        batch = plan[start:start + windows.BATCH_SIZE]
        values, labels, mask = windows.batch(data, batch, radius, len(center), device)
        optimizer.zero_grad(set_to_none=True)
        scores = model((values - mean) / std, mask)[:, :, radius:radius + windows.CORE_FRAMES]
        loss = F.nll_loss(scores, labels, ignore_index=windows.IGNORED_LABEL)
        loss = loss + SMOOTHNESS * windows.spell_smoothness(scores, labels)
        loss.backward()
        optimizer.step()
        schedule.step()
    if not np.isfinite(loss.item()):
        raise ValueError("Nonfinite training loss")
    return {"state_dict": _weights(model), "center": torch.from_numpy(center),
            "scale": torch.from_numpy(scale), "channels": len(center) // 2,
            "training_matches": list(data), "steps": steps, "seed": seed, "sampling": sampling,
            "calibration": False}


def transition_targets(labels, bounds):
    """(frames, 6) float32: a Gaussian bump (sigma 2.5 frames, 8 frames each way) on the channel
    of each change between two known states, in the order of ``PAIRS``."""
    out = np.zeros((len(labels), len(PAIRS)), np.float32)
    offset = np.arange(-8, 9)
    shape = np.exp(-.5 * (offset / 2.5) ** 2)
    for lo, hi in bounds:
        y = labels[lo:hi]
        changes = np.flatnonzero((y[1:] != y[:-1]) & (y[1:] < 3) & (y[:-1] < 3)) + 1
        for t in changes:
            rows = t + offset
            valid = (rows >= 0) & (rows < len(y))
            channel = PAIR_ID[y[t - 1], y[t]]
            rows = lo + rows[valid]
            out[rows, channel] = np.maximum(out[rows, channel], shape[valid])
    return out


def transition_loss(scores, labels, targets, batch, device):
    values = np.zeros((len(batch), labels.shape[1], 6), dtype=np.float32)
    for row, (key, _, _, start, end) in enumerate(batch):
        values[row, :end - start] = targets[key][start:end]
    target = torch.from_numpy(values.transpose(0, 2, 1)).to(device)
    valid = (labels != windows.IGNORED_LABEL).unsqueeze(1)
    weight = (1 + 14 * target) * valid
    return (F.binary_cross_entropy_with_logits(scores, target, reduction="none") * weight).sum() / weight.sum()


def fit_typed(data, steps=TYPED["steps"], seed=TYPED["seed"], device="cpu"):
    """Fit a ``TransitionTCN`` on every match in ``data``.

    Unlike the standard fit, the features are normalised before the windows are cut (so a
    window that runs off its period is padded with zeros in normalised units) and the window
    plan is drawn before the weights are initialised.

    Returns ``{"state_dict", "center", "scale", "channels", "training_matches", "seed",
    "steps"}``.
    """
    device = torch.device(device)
    center, scale = windows.normalization(data)
    channels = len(center) // 2
    training = {key: ((features - center) / scale, labels, bounds)
                for key, (features, labels, bounds) in data.items()}
    plan, _ = windows.training_plan(training, np.random.default_rng(seed), steps, BOUNDARY_FRACTION)
    torch.manual_seed(seed)
    targets = {key: transition_targets(labels, bounds) for key, (_, labels, bounds) in data.items()}
    model = TransitionTCN(channels).to(device)
    optimizer, schedule = _optimizer(model, steps)
    losses = torch.empty(steps, device=device)
    radius = model.radius
    for step in range(steps):
        start = step * windows.BATCH_SIZE
        batch = plan[start:start + windows.BATCH_SIZE]
        values, labels, mask = windows.batch(training, batch, radius, channels * 2, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(values, mask)[:, :, radius:radius + windows.CORE_FRAMES]
        classes = logits[:, :3]
        loss = F.nll_loss(classes, labels, ignore_index=windows.IGNORED_LABEL)
        loss = loss + SMOOTHNESS * windows.spell_smoothness(classes, labels)
        loss = loss + TRANSITION_WEIGHT * transition_loss(logits[:, 3:9], labels, targets, batch, device)
        loss.backward()
        optimizer.step()
        schedule.step()
        losses[step] = loss.detach()
    if not np.isfinite(losses.cpu().numpy()).all():
        raise RuntimeError("Nonfinite loss")
    return {"state_dict": _weights(model), "center": torch.from_numpy(center),
            "scale": torch.from_numpy(scale), "channels": channels,
            "training_matches": list(data), "seed": seed, "steps": steps}
