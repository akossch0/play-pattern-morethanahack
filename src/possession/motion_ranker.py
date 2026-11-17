"""Recover short possessions immediately before a stoppage using player motion."""

import numpy as np
import torch
from torch import nn

from .stoppage import _period_bounds, spell_features

FEATURE_WIDTH = 68
CANDIDATES_PER_SITE = 8


def smooth_tracks(xy, period):
    """Fit a local linear trajectory while retaining the observation mask."""
    result = xy.copy()
    radius = 4
    for lo, hi in _period_bounds(period):
        sample = xy[lo:hi]
        padded = np.pad(sample, ((radius, radius), (0, 0), (0, 0)), constant_values=np.nan)
        valid = np.isfinite(sample).all(-1)
        weight = np.zeros(valid.shape, dtype=np.float32)
        time = np.zeros_like(weight)
        time_sq = np.zeros_like(weight)
        points = np.zeros_like(sample)
        point_time = np.zeros_like(sample)
        for delta in range(-radius, radius + 1):
            offset = padded[radius + delta:radius + delta + len(sample)]
            available = np.isfinite(offset).all(-1)
            coefficients = available * np.float32(np.exp(-0.5 * (delta / 2.5) ** 2))
            point = np.where(available[..., None], offset, 0)
            weight += coefficients
            time += coefficients * delta
            time_sq += coefficients * delta * delta
            points += coefficients[..., None] * point
            point_time += coefficients[..., None] * delta * point
        determinant = weight * time_sq - time * time
        centered = (
            points * time_sq[..., None] - point_time * time[..., None]
        ) / np.maximum(0.001, determinant[..., None])
        result[lo:hi] = np.where((valid & (determinant >= 0.45))[..., None], centered, sample)
    return result


def _shift(values, frames):
    shifted = np.empty_like(values)
    shifted[:frames] = values[:1]
    shifted[frames:] = values[:-frames]
    return shifted


def _team_motion(positions, rivals, period):
    old = _shift(positions, 6)
    with np.errstate(invalid="ignore"):
        velocity = (positions - old) * (25 / 6)
        speed = np.linalg.norm(velocity, axis=2)
    same = period == _shift(period, 6)
    velocity[~same] = np.nan
    speed[~same] = np.nan
    before = _shift(speed, 6)
    output = []
    for lo in range(0, len(period), 4096):
        hi = min(lo + 4096, len(period))
        xy, other = positions[lo:hi], rivals[lo:hi]
        displacement = other[:, None, :, :] - xy[:, :, None, :]
        distance = np.linalg.norm(displacement, axis=3)
        nearest = np.where(np.isfinite(distance), distance, np.inf).argmin(axis=2)
        at = np.arange(hi - lo)[:, None]
        players = np.arange(xy.shape[1])[None, :]
        vector = displacement[at, players, nearest]
        best = distance[at, players, nearest]
        radial = (vector * velocity[lo:hi]).sum(2) / np.maximum(best, 0.15)
        active = np.isfinite(speed[lo:hi])
        count = active.sum(1).clip(1)
        changes = speed[lo:hi] - before[lo:hi]
        changing = np.isfinite(changes).sum(1).clip(1)
        output.append(np.column_stack((
            active.mean(1),
            np.nansum(speed[lo:hi], axis=1) / count,
            np.count_nonzero(speed[lo:hi] > 4.3, axis=1) / count,
            np.count_nonzero(changes > 0.65, axis=1) / changing,
            np.count_nonzero(changes < -0.65, axis=1) / changing,
            np.nansum(np.clip(radial, -5, 5), axis=1) / np.isfinite(radial).sum(1).clip(1),
            np.count_nonzero((best < 4) & (radial > 1.0), axis=1) / count,
            np.count_nonzero(best < 2.2, axis=1) / np.isfinite(best).sum(1).clip(1),
        )))
    return np.vstack(output)


def motion_prefix(home, away, period):
    """Build a cumulative table of player motion for each team."""
    home, away = (smooth_tracks(team, period) for team in (home, away))
    values = np.stack((_team_motion(home, away, period), _team_motion(away, home, period)), axis=1)
    return np.concatenate((np.zeros_like(values[:1]), values.cumsum(0)))


def candidates(reference, period, tracking, forecasts, edges, prefix, cuts):
    bank, features = spell_features(reference, period, tracking, forecasts, edges, cuts)
    motion = []
    hidden = []
    for _, stop, start, rival in bank:
        middle = (start + stop) // 2
        # Bank rows are (spell start, stoppage onset, proposed takeover, rival). Windows: 12 frames
        # before the takeover, the two halves of [start, stop), then the first 9 stoppage frames.
        spans = ((start - 12, start), (start, middle), (middle, stop), (stop, stop + 9))
        windows = [(prefix[b] - prefix[a]) / (b - a) for a, b in spans]
        motion.append(np.concatenate([np.r_[value[rival] - value[1 - rival], value.mean(axis=0)]
                                      for value in windows]))
        # Column 19 of the feature table is ball_missing (see features.FEATURES): the share of
        # frames in [start, stop) with no ball observation.
        hidden.append(tracking[start:stop, 19].mean())
    design = np.column_stack((features[:, :4], np.asarray(motion).reshape(-1, 64)))
    return bank, design, np.asarray(hidden, dtype=float)


def _pack(features, sites):
    order = list(dict.fromkeys(sites.tolist()))
    batch = np.zeros((len(order), CANDIDATES_PER_SITE, features.shape[-1]), dtype=np.float32)
    mask = np.zeros((len(order), CANDIDATES_PER_SITE), dtype=bool)
    indices = np.full((len(order), CANDIDATES_PER_SITE), -1, dtype=int)
    slots = {site: i for i, site in enumerate(order)}
    for j, site in enumerate(sites):
        row = slots[site]
        column = mask[row].sum()
        batch[row, column] = features[j]
        mask[row, column] = True
        indices[row, column] = j
    return batch, mask, indices


class MotionRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.candidate = nn.Sequential(
            nn.Linear(FEATURE_WIDTH, 28), nn.LayerNorm(28), nn.GELU(),
            nn.Dropout(0.15), nn.Linear(28, 16), nn.GELU(),
        )
        self.local = nn.Linear(16, 2)
        self.gate = nn.Sequential(nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, data, active):
        feature = self.candidate(data)
        local = self.local(feature)
        features = feature.masked_fill(~active[..., None], 0)
        average = features.sum(dim=1) / active.sum(dim=1, keepdim=True).clamp_min(1)
        maximum = feature.masked_fill(~active[..., None], -1e3).max(dim=1).values
        propensity = self.gate(torch.cat((average, maximum), dim=1)).squeeze(-1)
        return propensity, local[..., 0].masked_fill(~active, -1e3)


def load_models(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    mean, scale = (checkpoint[name].numpy() for name in ("mean", "scale"))
    models = []
    for state in checkpoint["state_dicts"]:
        model = MotionRanker()
        model.load_state_dict(state)
        models.append(model.eval())
    return mean, scale, models


@torch.inference_mode()
def predict(features, sites, weights):
    mean, scale, models = weights
    normalized = ((features - mean) / scale).clip(-6, 6)
    data, active, indices = _pack(normalized, sites)
    inputs, mask = torch.from_numpy(data), torch.from_numpy(active)
    output = [model(inputs, mask) for model in models]
    probability = torch.stack([item[0].sigmoid() for item in output]).mean(0)
    ranking = torch.stack([item[1].softmax(dim=1) for item in output]).mean(0)
    evidence = torch.column_stack((probability[:, None].broadcast_to(mask.shape)[mask],
                                  ranking[mask])).numpy()
    result = np.empty_like(evidence)
    result[indices[active]] = evidence
    return result


def apply(reference, bank, features, hidden, weights):
    result = reference.copy()
    if not bank:
        return result
    sites = np.asarray([stop for _, stop, _, _ in bank])
    probability, rank = predict(features, sites, weights).T
    options = []
    for index in np.flatnonzero((hidden >= 0.65) & (probability >= 0.72)):
        _, stop, start, rival = bank[index]
        options.append((float(probability[index] * (0.7 + 0.3 * rank[index])), start, stop, rival))
    used = set()
    for _, start, stop, rival in sorted(options, reverse=True):
        if stop in used:
            continue
        result[start:stop] = rival
        used.add(stop)
        if len(used) == 2:
            break
    return result
