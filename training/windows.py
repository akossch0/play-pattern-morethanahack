"""Window sampling, batching and the shared loss term of the three frame-model families.

``data`` maps a match key to ``(features, labels, period bounds)``. A training step sees eight
windows of 256 supervised frames, each with the model's receptive field of real frames on
either side where the period has them.
"""

from collections import Counter

import numpy as np
import torch

BATCH_SIZE = 8
CORE_FRAMES = 256
IGNORED_LABEL = 3


def normalization(data):
    """Per-channel mean and scale over both team views of the training matches."""
    channels = next(iter(data.values()))[0].shape[1] // 2
    count = 0
    total = np.zeros(channels, dtype=np.float64)
    squares = np.zeros_like(total)
    for features, _, _ in data.values():
        count += len(features) * 2
        for view in np.split(features, 2, axis=1):
            total += view.sum(axis=0, dtype=np.float64)
            squares += np.einsum("ij,ij->j", view, view, dtype=np.float64)
    mean = total / count
    scale = np.sqrt(np.maximum(squares / count - mean ** 2, 1e-4))
    return np.tile(mean, 2).astype(np.float32), np.tile(scale, 2).astype(np.float32)


def training_plan(training, rng, steps, boundary_fraction=0.0):
    """Every window of the run, drawn up front: uniform over window starts with at least one
    labelled frame, then ``boundary_fraction`` of them redrawn from the starts whose middle
    half holds a turnover or a restart."""
    runs = []
    boundary_starts = []
    total_starts = 0
    for key, (_, labels, bounds) in training.items():
        for lo, hi in bounds:
            core_length = min(CORE_FRAMES, hi - lo)
            valid = labels[lo:hi] != IGNORED_LABEL
            prefix = np.pad(np.cumsum(valid), (1, 0))
            counts = prefix[core_length:] - prefix[:-core_length]
            offsets = np.flatnonzero(counts)
            if len(offsets):
                runs.append((key, lo, hi, core_length, offsets, counts))
                if boundary_fraction:
                    before, after = labels[lo:hi - 1], labels[lo + 1:hi]
                    boundaries = np.pad(
                        (before != after) & (before < IGNORED_LABEL) & (after < 2), (1, 0)
                    )
                    cumulative = np.pad(np.cumsum(boundaries), (1, 0))
                    left = offsets + core_length // 4
                    right = offsets + core_length - core_length // 4
                    near_boundary = cumulative[right] > cumulative[left]
                    boundary_starts.append(total_starts + np.flatnonzero(near_boundary))
                total_starts += len(offsets)
    if not runs:
        raise ValueError("training matches have no supervised frames")
    ends = np.cumsum([len(run[4]) for run in runs])
    samples = rng.integers(int(ends[-1]), size=steps * BATCH_SIZE)
    targeted = 0
    if boundary_fraction:
        eligible = np.concatenate(boundary_starts)
        if not len(eligible):
            raise ValueError("training matches contain no eligible turnover or restart windows")
        targeted = round(len(samples) * boundary_fraction)
        locations = rng.choice(len(samples), size=targeted, replace=False)
        samples[locations] = rng.choice(eligible, size=targeted)
    run_indices = np.searchsorted(ends, samples, side="right")
    offsets_in_run = samples - np.concatenate(([0], ends[:-1]))[run_indices]
    plan = []
    supervised = 0
    for run_index, offset_in_run in zip(run_indices, offsets_in_run, strict=True):
        key, lo, hi, core_length, offsets, counts = runs[run_index]
        offset = int(offsets[offset_in_run])
        start = lo + offset
        plan.append((key, lo, hi, start, start + core_length))
        supervised += int(counts[offset])
    return plan, {
        "sampled_windows": len(plan),
        "sampled_core_frames": sum(end - start for _, _, _, start, end in plan),
        "supervised_frames": supervised,
        "sampled_windows_by_match": dict(Counter(window[0] for window in plan)),
        "targeted_boundary_windows": targeted,
    }


def batch(training, windows, radius, n_features, device):
    """Inputs, labels and (when a window runs off its period) a padding mask."""
    total = CORE_FRAMES + 2 * radius
    values = np.zeros((BATCH_SIZE, n_features, total), dtype=np.float32)
    labels = np.full((BATCH_SIZE, CORE_FRAMES), IGNORED_LABEL, dtype=np.int64)
    mask = np.ones((BATCH_SIZE, 1, total), dtype=np.float32)
    padded = False
    for index, (key, lo, hi, start, end) in enumerate(windows):
        features, target, _ = training[key]
        window_start = start - radius
        left = max(lo, window_start)
        right = min(hi, window_start + total)
        input_start = left - window_start
        input_end = right - window_start
        values[index, :, input_start:input_end] = features[left:right].T
        labels[index, :end - start] = target[start:end]
        if input_start or input_end != total:
            padded = True
            mask[index, 0, :input_start] = 0
            mask[index, 0, input_end:] = 0
    return (
        torch.from_numpy(values).to(device),
        torch.from_numpy(labels).to(device),
        torch.from_numpy(mask).to(device) if padded else None,
    )


def spell_smoothness(logits, labels):
    """Mean squared change in class probability between neighbouring frames of one spell."""
    same_label = (labels[:, :-1] == labels[:, 1:]) & (labels[:, :-1] != IGNORED_LABEL)
    probabilities = logits.softmax(dim=1)
    delta = probabilities[:, :, 1:] - probabilities[:, :, :-1]
    return (delta.square().sum(dim=1) * same_label).sum() / same_label.sum().clamp_min(1)
