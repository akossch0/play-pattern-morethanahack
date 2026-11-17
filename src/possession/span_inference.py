"""Detect short possession spans in four overlapping inference views."""

from pathlib import Path

import numpy as np
import torch

from .features import extract_features
from .models.span_detector import CENTERS, CORE, HALO, NATIVE, STEP, TAPS, Detector

# Frame offsets of the window grid, one per inference view: each view starts its
# CORE-stride windows this many frames earlier (see _infer).
SHIFTS = (0, 97, 194, 291)


def _period_bounds(period):
    edges = np.r_[0, np.flatnonzero(period[1:] != period[:-1]) + 1, len(period)]
    return tuple(zip(edges[:-1], edges[1:]))


def _interactions(tracking):
    period = tracking["period"]
    scale = np.array(
        [tracking["pitch_length_m"], tracking["pitch_width_m"]], dtype=np.float32
    )
    positions = [
        np.where(tracking[f"{side}_observed"][..., None], tracking[f"{side}_xy"] * scale, np.nan)
        for side in ("home", "away")
    ]
    output = np.zeros((len(period), 2, 14), dtype=np.float32)
    for lo, hi in _period_bounds(period):
        for side, (team, opposition) in enumerate(
            zip(positions, positions[::-1], strict=True)
        ):
            for start in range(lo, hi, 2048):
                at = np.arange(start, min(start + 2048, hi))
                before = np.maximum(at - 10, lo)
                after = np.minimum(at + 10, hi - 1)
                own, other = team[at], opposition[at]
                own_vel = (team[after] - team[before]) * (
                    25 / (after - before).clip(1)
                )[:, None, None]
                other_vel = (opposition[after] - opposition[before]) * (
                    25 / (after - before).clip(1)
                )[:, None, None]
                relative = other[:, None] - own[:, :, None]
                distance = np.linalg.norm(relative, axis=-1)
                distance = np.where(np.isfinite(distance), distance, 100.0)
                nearest = distance.argmin(axis=2)
                row = np.arange(len(at))[:, None]
                player = np.arange(own.shape[1])
                separation = distance[row, player, nearest]
                direction = relative[row, player, nearest]
                difference = other_vel[row, nearest] - own_vel
                closing = -np.sum(difference * direction, axis=-1) / separation.clip(
                    0.25
                )
                closing = np.where(
                    np.isfinite(closing) & (separation < 100.0), closing, 0.0
                )
                speed = np.nan_to_num(np.linalg.norm(own_vel, axis=-1), nan=0.0)
                acceleration = (team[after] - own) - (own - team[before])
                change = np.nan_to_num(np.linalg.norm(acceleration, axis=-1), nan=0.0)
                pressure = np.exp(-separation / 4.0)
                centroid = (
                    np.nansum(own, axis=1)
                    / np.isfinite(own[..., 0]).sum(axis=1).clip(1)[:, None]
                )
                spread = np.nan_to_num(
                    np.linalg.norm(own - centroid[:, None], axis=-1), nan=0.0
                )
                weight = pressure.sum(axis=1).clip(0.01)
                values = (
                    np.log1p(separation.min(1)),
                    np.log1p(np.sort(separation, axis=1)[:, 2]),
                    (separation < 2.0).sum(1),
                    (separation < 5.0).sum(1),
                    (pressure * closing).sum(1) / weight,
                    (pressure * closing.clip(0)).max(1),
                    (pressure * speed).sum(1) / weight,
                    (pressure * change).max(1),
                    speed.mean(1),
                    speed.max(1),
                    change.mean(1),
                    change.max(1),
                    spread.mean(1),
                    spread.max(1),
                )
                output[at, side] = np.column_stack(values)
    return output


def _infer(model, data, period, shift):
    frame = np.zeros((len(period), 3), dtype=np.float32)
    proposals = []
    for lo, hi in _period_bounds(period):
        starts = np.arange(lo - HALO - shift, hi - HALO, CORE)
        # Run the detector on near-equal batches of at most 24 windows.
        for batch in np.array_split(starts, max(1, (len(starts) + 23) // 24)):
            taps = (batch[:, None] + TAPS).clip(lo, hi - 1)
            raw, labels, boxes, quality = model(torch.from_numpy(data[taps]))
            frame_p = raw.softmax(1).transpose(1, 2).numpy()
            teams = labels.softmax(-1)[:, :, :2].numpy()
            confidence = quality.sigmoid().numpy()
            boxes = np.rint(boxes.numpy()).astype(int)
            for j, start in enumerate(batch):
                begin, end = max(lo, start + HALO), min(hi, start + NATIVE - HALO)
                rows = np.arange(begin, end)
                frame[rows] = frame_p[j, (rows - start) // STEP]
                a, b = (boxes[j] + start).clip(lo, hi).T
                centers = CENTERS + start
                valid = (lo <= centers) & (centers < hi) & (b - a >= 6) & (b - a <= 250)
                team = teams[j].argmax(1)
                score = teams[j].max(1) * confidence[j]
                proposals.append(
                    np.column_stack((a[valid], b[valid], team[valid], score[valid]))
                )
    np.testing.assert_allclose(frame.sum(1), 1.0, atol=1e-6)
    return frame, np.concatenate(proposals)


def _fuse_views(banks):
    rows = np.concatenate(banks)
    view = np.repeat(np.arange(len(banks)), [len(bank) for bank in banks])
    # Each source window phase contributes at most one vote to each hypothesis.
    buckets = {}
    span = []
    for i, (a, b, side, _) in enumerate(rows):
        keys = [(int(side), t) for t in range(int(a) // 64, int(b - 1) // 64 + 1)]
        span.append(keys)
        for key in keys:
            buckets.setdefault(key, []).append(i)
    proposals = []
    for i, (a, b, side, _) in enumerate(rows):
        ids = np.asarray(sorted({j for key in span[i] for j in buckets[key]}))
        x, y, _, score = rows[ids].T
        overlap = (np.minimum(b, y) - np.maximum(a, x)) / (
            np.maximum(b, y) - np.minimum(a, x)
        )
        match = ids[overlap >= 0.60]
        order = match[np.argsort(-rows[match, 3], kind="stable")]
        _, first = np.unique(view[order], return_index=True)
        votes = rows[order[first]]
        if len(votes) < 2:
            continue
        centers = np.rint(
            np.average(votes[:, :2], weights=votes[:, 3].clip(1e-6), axis=0)
        )
        if centers[1] - centers[0] >= 6:
            proposals.append(
                [*centers, side, float(votes[:, 3].mean() * (0.60 + 0.10 * len(votes)))]
            )
    return np.asarray(proposals).reshape(-1, 4)


def _compact(bank, limit=1500):
    rows = bank[np.argsort(-bank[:, 3], kind="stable")[:18000]]
    kept = []
    buckets = {}
    # Spatial bins restrict suppression comparisons to intervals that can overlap.
    for a, b, k, score in rows:
        bins = range(int(a) // 64, int(b - 1) // 64 + 1)
        keys = [(k, tile) for tile in bins]
        nearby = {i for key in keys for i in buckets.get(key, ())}
        if any(
            (min(b, kept[i][1]) - max(a, kept[i][0]))
            >= 0.65 * (max(b, kept[i][1]) - min(a, kept[i][0]))
            for i in nearby
        ):
            continue
        for key in keys:
            buckets.setdefault(key, []).append(len(kept))
        kept.append((a, b, k, score))
        if len(kept) == limit:
            break
    return np.asarray(kept).reshape(-1, 4)


@torch.inference_mode()
def predict_spans(graded_match, weights_dir, *, pairs=None):
    """Return each detector member’s span bank and frame probabilities.

    Keys are pairs of excluded development match IDs. Span rows contain start
    frame, end frame (exclusive), team and confidence. Frame probabilities
    follow home, away, dead-ball order.
    """
    period = np.asarray(graded_match["period"])
    interactions = _interactions(graded_match).reshape(len(period), -1)
    features = np.column_stack((extract_features(graded_match), interactions))
    directory = Path(weights_dir) / "span-detector"
    if pairs is None:
        paths = sorted(directory.glob("*/*/checkpoint.pt"))
        if len(paths) != 10:
            raise ValueError(
                f"Expected ten span detector checkpoints under {directory}"
            )
    else:
        paths = [directory.joinpath(*sorted(pair), "checkpoint.pt") for pair in pairs]
    forecasts = {}
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        mean = np.asarray(checkpoint["mean"], dtype=np.float32)
        scale = np.asarray(checkpoint["scale"], dtype=np.float32)
        if (
            mean.shape != (features.shape[1],)
            or scale.shape != mean.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
        ):
            raise ValueError(f"Invalid span detector normalization for {path}")
        data = ((features - mean) / scale).clip(-8, 8).astype(np.float32, copy=False)
        model = Detector(features.shape[1])
        model.load_state_dict(checkpoint["state"])
        model.eval()
        views = [_infer(model, data, period, shift) for shift in SHIFTS]
        bank = _compact(
            _fuse_views([_compact(items, 2200) for _, items in views]), 1500
        )
        probability = np.mean([frame for frame, _ in views], axis=0)
        forecasts[path.parent.parent.name, path.parent.name] = bank, probability
    return forecasts
