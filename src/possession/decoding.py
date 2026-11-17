"""Decode possession scores with a transition cost and report short-spell recall counts."""

from __future__ import annotations

from math import isfinite

import numpy as np


def _decode_period(scores: np.ndarray, team_penalty: float, dead_penalty: float) -> np.ndarray:
    # Use Python floats and byte backpointers for the three-state scalar recurrence.
    home, away, dead = scores[0].tolist()
    back_home, back_away, back_dead = (bytearray(len(scores)) for _ in range(3))
    home_scores, away_scores, dead_scores = (column.tolist() for column in scores[1:].T)
    for frame, (emit_home, emit_away, emit_dead) in enumerate(
        zip(home_scores, away_scores, dead_scores), start=1,
    ):
        # Each destination considers all three predecessor states. Ties go to the first.
        from_dead = dead - dead_penalty
        best_home, prev_home = home, 0
        from_away = away - team_penalty
        if from_away > best_home:
            best_home, prev_home = from_away, 1
        if from_dead > best_home:
            best_home, prev_home = from_dead, 2

        best_away, prev_away = home - team_penalty, 0
        if away > best_away:
            best_away, prev_away = away, 1
        if from_dead > best_away:
            best_away, prev_away = from_dead, 2

        # Into dead: both live predecessors pay the same dead_penalty, so the better one is picked
        # first (home on a tie) and charged once; staying dead is free and needs a strict win.
        best_dead, prev_dead = (away, 1) if away > home else (home, 0)
        best_dead -= dead_penalty
        if dead > best_dead:
            best_dead, prev_dead = dead, 2

        back_home[frame], back_away[frame], back_dead[frame] = prev_home, prev_away, prev_dead
        home, away, dead = best_home + emit_home, best_away + emit_away, best_dead + emit_dead

    state = int(np.argmax((home, away, dead)))
    back = (back_home, back_away, back_dead)
    path = bytearray(len(scores))
    for frame in range(len(path) - 1, -1, -1):
        path[frame] = state
        state = back[state][frame]
    return np.frombuffer(path, dtype=np.uint8)


def decode(
    scores: np.ndarray, period: np.ndarray, team_penalty: float, dead_penalty: float,
) -> np.ndarray:
    """Maximize emission scores minus symmetric transition costs within each period.

    Scores have one row per frame and three columns in H/A/D order. A team change costs
    ``team_penalty`` and a change between live play and dead ball costs ``dead_penalty``.
    With both costs zero, the result is exactly the frame-wise argmax, including ties.
    """
    scores, period = np.asarray(scores), np.asarray(period)
    if scores.ndim != 2 or scores.shape[1] != 3 or period.shape != scores.shape[:1]:
        raise ValueError("expected scores of shape (T, 3) and period of shape (T,)")
    if scores.dtype.kind not in "fiu" or not np.isfinite(scores).all():
        raise ValueError("scores must be finite and real")
    team_penalty, dead_penalty = float(team_penalty), float(dead_penalty)
    if any(not isfinite(cost) or cost < 0 for cost in (team_penalty, dead_penalty)):
        raise ValueError("transition penalties must be finite and nonnegative")
    if len(scores) == 0 or team_penalty == dead_penalty == 0:
        return scores.argmax(axis=1).astype(np.uint8)

    changes = (np.flatnonzero(period[1:] != period[:-1]) + 1).tolist()
    bounds = [0, *changes, len(period)]
    path = np.empty(len(scores), dtype=np.uint8)
    for start, stop in zip(bounds, bounds[1:]):
        path[start:stop] = _decode_period(scores[start:stop], team_penalty, dead_penalty)
    return path


# Fitting/development helper: nothing on the predict.py path calls this.
def short_spell_counts(
    truth: np.ndarray, predictions: np.ndarray, period: np.ndarray,
) -> tuple[int, int]:
    """Return (matched, total) for true H/A spells shorter than two seconds at 0.5 IoU."""
    # Keep the official scorer's pandas dependency on the scoring path only.
    from datalake.evaluate import pm, with_period_gap

    truth, predictions, period = map(np.asarray, (truth, predictions, period))
    if truth.ndim != 1 or truth.shape != predictions.shape or truth.shape != period.shape:
        raise ValueError("expected aligned one-dimensional labels, predictions and periods")
    truth = with_period_gap(truth, period)
    predictions = with_period_gap(predictions, period)
    runs = pm.valid_runs(truth)
    filled = pm.fill_abstentions(predictions, runs)
    true_spells = pm.possession_spells(truth, runs)
    pred_spells = pm.possession_spells(filled, runs)
    overlaps = pm.spell_overlaps(true_spells, pred_spells)
    matched = pm.matched_gt_flags(overlaps, len(true_spells), pm.SPELL_IOU_HEADLINE)
    short = np.asarray([stop - start < 2 * pm.FPS for start, stop, _ in true_spells], dtype=bool)
    return int(np.count_nonzero(matched & short)), int(np.count_nonzero(short))
