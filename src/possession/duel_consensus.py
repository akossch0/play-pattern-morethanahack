"""Keep live turnover shifts supported by both draft-based timing estimates."""

import numpy as np


def _boundaries(labels, period):
    changes = (labels[1:] != labels[:-1]) | (period[1:] != period[:-1])
    return np.r_[0, np.flatnonzero(changes) + 1, len(labels)]


def merge_duel_consensus(reference, period, first, second):
    """Return the reference timeline with shifts that both estimates agree on."""
    inputs = (reference, first, second)
    if period.ndim != 1 or any(labels.shape != period.shape for labels in inputs):
        raise ValueError("Timeline and period arrays must share one-dimensional frame indices")
    if not len(reference):
        return reference.copy()
    base, left, right = [_boundaries(labels, period) for labels in inputs]
    if base.shape != left.shape or base.shape != right.shape:
        raise ValueError("Each estimate must preserve the number of spells")
    if any(not np.array_equal(reference[base[:-1]], labels[bounds[:-1]])
           for labels, bounds in zip(inputs[1:], (left, right), strict=True)):
        raise ValueError("Timing estimates must preserve the order of possession states")
    updated = np.where(left == right, left, base)
    # Retain the reference positions if adjacent edits would reverse their order.
    while np.any(np.diff(updated) <= 0):
        collisions = np.flatnonzero(np.diff(updated) <= 0)
        affected = np.unique(np.r_[collisions, collisions + 1])
        updated[affected] = base[affected]
    result = np.repeat(reference[base[:-1]], np.diff(updated))
    if not np.array_equal(result == 2, reference == 2):
        raise ValueError("The timing edits must preserve dead-ball frames")
    if not np.array_equal(_boundaries(result, period), updated):
        raise ValueError("The timing edits must preserve the period boundaries")
    return result
