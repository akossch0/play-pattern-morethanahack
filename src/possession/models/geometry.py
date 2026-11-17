"""Tracking-only touch evidence and team-oriented shape features."""

import numpy as np

from .gru_features import _tactics, _velocity, attacks_right, touch_features

FPS = 25


def history(event, reverse=False):
    if reverse:
        return history(event[::-1])[::-1]
    frames = np.arange(len(event))
    last = np.maximum.accumulate(np.where(event, frames, -1))
    return np.where(last >= 0, (frames - last) / FPS, np.inf)


def rolling_mean(values, radius):
    total = np.concatenate(([0], np.cumsum(values, dtype=np.float64)))
    rows = np.arange(len(values))
    left, right = (rows - radius).clip(0), (rows + radius + 1).clip(max=len(rows))
    return (total[right] - total[left]) / (right - left)


def period_features(home, away, ball, observed, pitch):
    count = len(ball)
    frames = np.arange(count)
    here = np.isfinite(ball).all(axis=-1)
    ball_m = ball * pitch
    vh, va = (_velocity(side * pitch, 8) for side in (home, away))
    right = attacks_right(home, away)
    sign = 1 if right else -1
    # Difference estimates require an uninterrupted observed neighborhood.
    v_ball = _velocity(ball_m, 4)
    full_velocity = np.convolve(here, np.ones(5), mode="full")[2:count + 2] == 5
    v_ball[~full_velocity] = np.nan
    speed = np.linalg.norm(v_ball, axis=1)
    impulse = np.zeros(count, dtype=np.float32)
    impulse[2:-2] = np.linalg.norm(v_ball[4:] - v_ball[:-4], axis=1)
    impulse = np.nan_to_num(impulse)
    impulse_peak = np.lib.stride_tricks.sliding_window_view(np.pad(impulse, 2), 5).max(axis=1)
    nearest_distances = []
    team_columns = []
    for side, velocity in ((home, vh), (away, va)):
        delta = side * pitch - ball_m[:, None]
        distance = np.linalg.norm(delta, axis=-1)
        ordered = np.sort(np.where(np.isfinite(distance), distance, np.inf), axis=-1)
        nearest = np.argmin(np.where(np.isfinite(distance), distance, np.inf), axis=-1)
        nearest_distances.append(ordered[:, 0])
        delta_v = velocity[frames, nearest] - v_ball
        approach = (delta[frames, nearest] * delta_v).sum(1) / np.maximum(ordered[:, 0], .25)
        # Per team: three nearest player-ball distances (m), number of players within 3 m of the
        # ball, the nearest player's speed relative to the ball (m/s) and the rate of change of
        # its distance to the ball (divisor floored at 0.25 m). Each is scaled; all except the
        # player count are clipped first.
        team_columns.append([
            *ordered[:, :3].clip(0, 24).T / 12,
            (distance < 3).sum(1) / 4,
            np.linalg.norm(delta_v, axis=1).clip(0, 25) / 12,
            approach.clip(-15, 15) / 8,
        ])
    with np.errstate(invalid="ignore"):
        touch = touch_features(ball_m, home * pitch, away * pitch, frames, half=2)
    for idx, (own, opp) in enumerate(zip(nearest_distances, nearest_distances[::-1])):
        # Local proximity and motion cues complement point estimates from a sharp impulse.
        near = own + .25 < opp
        kick = here & observed & near & (own < 2.3) & (impulse_peak > 4)
        carry = here & observed & near & (own < 1.25) & (opp > 1.6) & (speed < 12)
        for event in (kick, carry):
            for radius in (4, 15):
                team_columns[idx].append(rolling_mean(event, radius))
            for reverse in (False, True):
                age = history(event, reverse)
                for horizon in (.5, 2, 6):
                    team_columns[idx].append(np.exp(-age / horizon))
    touchline = np.minimum(ball_m[:, 1], pitch[1] - ball_m[:, 1])
    goal_line = np.minimum(ball_m[:, 0], pitch[0] - ball_m[:, 0])
    shared = [
        here, observed, full_velocity, speed.clip(0, 38) / 20,
        impulse_peak.clip(0, 28) / 14,
        touchline.clip(-8, 20) / 12, goal_line.clip(-8, 20) / 12,
        np.exp(-history(here) / 2), np.exp(-history(here, True) / 2),
        rolling_mean(here, 15),
    ]
    sampled = frames[::5]
    tactics = _tactics(
        home[sampled], away[sampled], vh[sampled], va[sampled], pitch, right, *pitch,
    )[:2]
    previous = frames // 5
    following = (previous + 1).clip(max=len(sampled) - 1)
    weight = ((frames % 5) / 5)[:, None]
    output = []
    for idx in (0, 1):
        orientation = sign * (1 - 2 * idx)
        shape = tactics[idx][previous] * (1 - weight) + tactics[idx][following] * weight
        columns = [
            *shared, *team_columns[idx], *team_columns[1 - idx], *touch[idx].T,
            *((ball - .5) * 2 * orientation).T, *(v_ball * orientation / 20).clip(-2, 2).T,
            *shape.T,
        ]
        output.append(np.nan_to_num(np.column_stack(columns), nan=0, posinf=0, neginf=0).astype(np.float32))
    return np.concatenate(output, axis=1)


def extract(tracking):
    pitch = np.array([tracking["pitch_length_m"], tracking["pitch_width_m"]], dtype=np.float32)
    columns = ("home_xy", "away_xy", "ball_xy", "ball_observed")
    period = tracking["period"]
    output = None
    for number in np.unique(period):
        rows = np.flatnonzero(period == number)
        result = period_features(*(tracking[name][rows] for name in columns), pitch)
        if output is None:
            output = np.empty((len(period), result.shape[1]), dtype=np.float32)
        output[rows] = result
    return output
