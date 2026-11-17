import numpy as np

from datalake import core, degrade


FEATURES = tuple(
    f"{side}_{stat}"
    for side in ("home", "away")
    for stat in (
        "centroid_x", "centroid_y", "std_x", "std_y", "observed_fraction",
        "mean_speed_mps", "max_speed_mps", "speed_observed_fraction",
    )
) + (
    "ball_x", "ball_y", "ball_speed_mps", "ball_missing", "ball_speed_missing",
    "nearest_home_m", "nearest_away_m", "nearest_home_missing", "nearest_away_missing",
)


def period_slices(period: np.ndarray) -> tuple[slice, slice]:
    cut = int(np.searchsorted(period, 2))
    if not 0 < cut < len(period) or not np.all(period[:cut] == 1) or not np.all(period[cut:] == 2):
        raise ValueError("Expected two nonempty periods in play order")
    return slice(0, cut), slice(cut, len(period))


def team_features(
    xy: np.ndarray, speed: np.ndarray, observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = observed & np.isfinite(xy).all(axis=-1)
    count = valid.sum(axis=1)
    xy_filled = np.where(valid[..., None], xy, 0)
    mean = xy_filled.sum(axis=1, dtype=np.float64) / np.maximum(count, 1)[:, None]
    delta = np.where(valid[..., None], xy_filled - mean[:, None, :], 0)
    spread = np.sqrt((delta ** 2).sum(axis=1) / np.maximum(count, 1)[:, None])

    speed_valid = valid & np.isfinite(speed)
    speed_filled = np.where(speed_valid, speed, 0)
    speed_count = speed_valid.sum(axis=1)
    mean_speed = speed_filled.sum(axis=1, dtype=np.float64) / np.maximum(speed_count, 1)
    slots = xy.shape[1]
    features = np.column_stack((
        mean, spread, count / slots, mean_speed, speed_filled.max(axis=1), speed_count / slots,
    ))
    return features, valid


def extract_features(match: dict) -> np.ndarray:
    length, width = match["pitch_length_m"], match["pitch_width_m"]
    ball = match["ball_xy"]
    ball_valid = match["ball_observed"] & np.isfinite(ball).all(axis=1)
    ball = np.where(ball_valid[:, None], ball, np.nan)

    ball_speed = np.full(len(ball), np.nan, dtype=np.float32)
    for rows in period_slices(match["period"]):
        ball_speed[rows] = core.derive_speed(ball[rows], length, width)
        valid = ball_valid[rows]
        # The centered estimate needs all five samples, including those inside the stencil.
        speed = ball_speed[rows]
        speed[2:-2][~(valid[1:-3] & valid[3:-1])] = np.nan
    ball_speed_valid = np.isfinite(ball_speed)
    ball_features = np.column_stack((
        np.where(ball_valid[:, None], ball, 0),
        np.where(ball_speed_valid, np.minimum(ball_speed, degrade.MAX_BALL_SPEED), 0),
        ~ball_valid, ~ball_speed_valid,
    ))

    players, nearest, proximity_missing = [], [], []
    for side in ("home", "away"):
        xy = match[f"{side}_xy"]
        stats, valid = team_features(xy, match[f"{side}_speed"], match[f"{side}_observed"])
        players.append(stats)
        dx = (xy[..., 0] - ball[:, None, 0]) * length
        dy = (xy[..., 1] - ball[:, None, 1]) * width
        dist = np.where(valid & ball_valid[:, None], np.hypot(dx, dy), np.inf).min(axis=1)
        missing = ~np.isfinite(dist)
        nearest.append(np.where(missing, 0, dist))
        proximity_missing.append(missing)

    return np.column_stack((*players, ball_features, *nearest, *proximity_missing)).astype(np.float32)


def geometry(players, opponents, period, lag=12):
    n, slots, _ = players.shape
    index = np.arange(n)
    previous = (index - lag).clip(0, n-1)
    ahead = (index + lag).clip(0, n-1)
    valid_lag = (period[previous] == period) & (period[ahead] == period)
    self_velocity = (players[ahead] - players[previous]) * (25/(2*lag))  # 25 fps frame rate
    rival_velocity = (opponents[ahead] - opponents[previous]) * (25/(2*lag))
    columns = []
    closest = np.empty((n, slots), dtype=int)
    distances = np.empty((n, slots), dtype=float)
    for low in range(0, n, 4096):
        high = min(n, low+4096)
        first, second = players[low:high], opponents[low:high]
        delta = second[:, None, :, :] - first[:, :, None, :]
        distance = np.linalg.norm(delta, axis=-1)
        nearest = np.nan_to_num(distance, nan=np.inf).argmin(axis=2)
        frame = np.arange(high-low)[:, None]
        actor = np.arange(slots)[None, :]
        target = distance[frame, actor, nearest]
        order = np.partition(np.nan_to_num(distance, nan=np.inf), 1, axis=2)
        pressure = ((distance <= 5.).sum(axis=2).clip(max=4) / 4.)
        closest[low:high], distances[low:high] = nearest, target
        # Floor the divisor at 0.15 m so zero separation gives 0 rather than 0/0.
        direction = delta[frame, actor, nearest] / np.maximum(target[..., None], .15)
        own = self_velocity[low:high]
        rival = rival_velocity[low:high][frame, nearest]
        chase = np.sum(own * direction, axis=-1)
        retreat = np.sum(rival * direction, axis=-1)
        usable = np.isfinite(chase) & np.isfinite(retreat) & valid_lag[low:high, None]
        near = usable & (target < 8.)
        denominator = near.sum(1).clip(1)
        separation = np.zeros_like(target)
        finite = np.isfinite(target)&np.isfinite(order[:, :, 1])
        np.subtract(order[:, :, 1], target, out=separation, where=finite)
        columns.append(np.column_stack((
            np.isfinite(target).mean(1),
            np.mean(target < 3.5, axis=1),
            np.mean(pressure, axis=1),
            np.sum(np.where(near, chase.clip(-6, 6), 0.), axis=1) / denominator,
            np.sum(np.where(near, retreat.clip(-6, 6), 0.), axis=1) / denominator,
            np.sum(np.where(near, (chase-retreat).clip(-6, 6), 0.), axis=1) / denominator,
            np.mean(near & (separation < 1.25), axis=1),
        )))
    supported = (distances[previous] < 8.) & (distances[ahead] < 8.) & valid_lag[:, None]
    swapped = (closest[previous] != closest[ahead]) & supported
    sustained = (closest[previous] == closest) & (closest == closest[ahead]) & supported
    return np.column_stack((np.vstack(columns), swapped.mean(1), sustained.mean(1)))
