"""Learn short-possession intervals from labelled external tracking windows."""

import numpy as np
import torch
from torch.nn import functional as F
from .inputs import leave_out_pairs, period_bounds
from possession.features import extract_features
from possession.span_inference import _interactions
from evaluation import possession_metrics as pm
from possession.models.span_detector import (
    CENTERS,
    LEVEL,
    NATIVE,
    STEP,
    HALO,
    TAPS,
    Detector,
)


def ordered_assignment(cost):
    """Minimum-cost matching between ordered query centers and distinct ordered spells."""
    proposals, spells = cost.shape
    assert proposals >= spells
    if not spells:
        return np.empty(0, dtype=int)
    score = cost[:, 0]
    back = []
    indices = np.arange(proposals)
    for column in range(1, spells):
        low = np.minimum.accumulate(score)
        chosen = np.maximum.accumulate(np.where(score <= low, indices, 0))
        back.append(np.r_[0, chosen[:-1]])
        score = cost[:, column] + np.r_[np.inf, low[:-1]]
    pair = [int(score.argmin())]
    for previous in reversed(back):
        pair.append(int(previous[pair[-1]]))
    return np.array(pair[::-1])


def iou_numpy(prediction, target):
    a, b = prediction.T
    x, y = target.T
    shared = (np.minimum(b[:, None], y) - np.maximum(a[:, None], x)).clip(0)
    return shared / (np.maximum(b[:, None], y) - np.minimum(a[:, None], x)).clip(1e-6)


def assignments(spans, logits, truth, one_to_one):
    mask = np.full(len(CENTERS), -1, dtype=np.int64)
    if not len(truth):
        return mask
    start, stop, side = truth.T
    width = stop - start
    center = (start + stop) / 2
    shift = abs(CENTERS[:, None] - center)
    scale_penalty = np.where(
        LEVEL[:, None] == 0, (width[None, :] > 85), width[None, :] < 36
    )
    if one_to_one:
        overlaps = iou_numpy(spans, truth[:, :2])
        log_prob = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
        cost = (
            1.7 * (1 - overlaps)
            - 0.38 * log_prob[:, side]
            + 0.42 * shift / width.clip(12)
            + 0.48 * scale_penalty
        )
        mask[ordered_assignment(cost)] = np.arange(len(truth))
    else:
        within = (shift <= np.maximum(4, np.minimum(24, 0.28 * width))) & ~scale_penalty
        distance = np.where(within, shift / width.clip(10), np.inf)
        chosen = distance.argmin(axis=1)
        keep = within.any(axis=1)
        mask[keep] = chosen[keep]
    return mask


def label_spans(item):
    y, period = item["label"], item["period"]
    edges = np.r_[
        0, np.flatnonzero((y[1:] != y[:-1]) | (period[1:] != period[:-1])) + 1, len(y)
    ]
    left, right = edges[:-1], edges[1:]
    keep = (y[left] < 2) & (right - left >= 6) & (right - left <= 250)
    return np.column_stack((left[keep], right[keep], y[left[keep]]))


def clip_batch(data, rng, count):
    values, labels, spans = [], [], []
    for i in range(count):
        x, item, truth, bounds, short = data[rng.integers(len(data))]
        start, end = bounds[rng.integers(len(bounds))]
        first = int(rng.integers(start, end - NATIVE + 1))
        if i % 3 == 0 and len(short):
            chosen = short[rng.integers(len(short))]
            start, end = bounds[np.searchsorted(bounds[:, 1], chosen, side="right")]
            first = int(
                np.clip(chosen - rng.integers(100, NATIVE - 100), start, end - NATIVE)
            )
        rows = first + TAPS
        values.append(x[rows])
        labels.append(item["label"][rows].clip(max=3))
        inside = (
            (truth[:, :2].mean(1) >= first + HALO)
            & (truth[:, :2].mean(1) < first + NATIVE - HALO)
            & (truth[:, 0] >= first)
            & (truth[:, 1] <= first + NATIVE)
        )
        spans.append(truth[inside] - np.array((first, first, 0)))
    return np.stack(values), np.stack(labels), spans


def loss_terms(output, frame_labels, truth, one_to_one, device):
    frame, classification, spans, quality = output
    detached = spans.detach().cpu().numpy()
    raw = classification.detach().cpu().numpy()
    target = np.full(classification.shape[:2], 2, dtype=np.int64)
    boxes = np.zeros(spans.shape, dtype=np.float32)
    selected = np.zeros(target.shape, dtype=bool)
    for i, parts in enumerate(truth):
        index = assignments(detached[i], raw[i], parts, one_to_one)
        selected[i] = index >= 0
        target[i, selected[i]] = parts[index[selected[i]], 2]
        boxes[i, selected[i]] = parts[index[selected[i]], :2]
    target = torch.from_numpy(target).to(device)
    boxes = torch.from_numpy(boxes).to(device)
    pos = torch.from_numpy(selected).to(device)
    class_error = F.cross_entropy(
        classification.transpose(1, 2), target, reduction="none"
    )
    weight = 0.18 + 0.82 * pos
    classification_loss = (class_error * weight).sum() / weight.sum()
    intersection = (
        torch.minimum(spans[:, :, 1], boxes[:, :, 1])
        - torch.maximum(spans[:, :, 0], boxes[:, :, 0])
    ).clamp(min=0)
    union = (
        torch.maximum(spans[:, :, 1], boxes[:, :, 1])
        - torch.minimum(spans[:, :, 0], boxes[:, :, 0])
    ).clamp(min=1)
    iou = intersection / union
    scale = (boxes[:, :, 1] - boxes[:, :, 0]).clamp(min=10)
    offsets = F.smooth_l1_loss(
        spans / scale[:, :, None], boxes / scale[:, :, None], reduction="none"
    ).mean(2)
    box_loss = ((1 - iou + offsets) * pos).sum() / pos.sum().clamp(min=1)
    thresholds = torch.tensor(pm.SPELL_IOUS, device=device)
    confidence = ((iou.detach()[:, :, None] >= thresholds).float().mean(2)) * pos
    quality_loss = (
        F.binary_cross_entropy_with_logits(quality, confidence, reduction="none")
        * weight
    ).sum() / weight.sum()
    frame_loss = F.cross_entropy(
        frame[:, :, HALO // STEP : -HALO // STEP],
        frame_labels[:, HALO // STEP : -HALO // STEP],
        ignore_index=3,
    )
    return 0.85 * frame_loss + classification_loss + 1.5 * box_loss + 0.5 * quality_loss


def fit(data, device, steps, mode, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Detector(data[0][0].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.015)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=0.00025
    )
    for step in range(steps):
        x, y, truth = clip_batch(data, rng, 8)
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y.astype(np.int64)).to(device)
        loss = loss_terms(model(x), y, truth, mode == "one-to-one", device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
        schedule.step()
        if (step + 1) % 200 == 0 or step + 1 == steps:
            print(mode, step + 1, round(float(loss.detach().cpu()), 4), flush=True)
    return model.eval()


def train_checkpoint(matches, excluded, development_keys, destination, *, steps=420):
    """Fit one detector from its three prepared external matches and save its weights."""
    pair = tuple(sorted(excluded))
    pairs = [held for held, _ in leave_out_pairs(development_keys)]
    training_keys = sorted(set(development_keys) - set(pair))
    if (
        set(matches) != set(training_keys)
        or pair not in pairs
        or len(training_keys) != 3
    ):
        raise ValueError("Pass the three training matches for one held-out pair.")
    raw = {}
    for key in training_keys:
        match = matches[key]
        interactions = _interactions(match).reshape(len(match["period"]), -1)
        raw[key] = np.column_stack((extract_features(match), interactions))
    sample = np.concatenate([raw[key][::13] for key in training_keys])
    mean, scale = sample.mean(0), sample.std(0).clip(min=0.035)
    samples = []
    for key in training_keys:
        item = matches[key]
        features = ((raw[key] - mean) / scale).clip(-8, 8).astype(np.float32)
        truth = label_spans(item)
        short = truth[truth[:, 1] - truth[:, 0] <= 125, :2].mean(1).astype(int)
        samples.append(
            (features, item, truth, np.asarray(period_bounds(item["period"])), short)
        )
    seed = 20261128 + pairs.index(pair)
    model = fit(samples, "cpu", steps, "one-to-one", seed)
    torch.save(
        {
            "state": model.state_dict(),
            "training_matches": training_keys,
            "excluded_matches": list(pair),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "seed": seed,
            "steps": steps,
        },
        destination,
    )
