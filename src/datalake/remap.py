"""Label views: optional, named, versioned ways of reading the provider's labels.

The stored dataset is not opinionated: `label` is exactly what the provider says. Providers switch
possession on any touch, while the competition switches it only on a CONTROLLED gain (official
CHALLENGE.md). Closing that gap is a modelling choice, so it lives here, outside the data:

    from datalake import core
    m = core.load("dfl_J03WMX")                            # provider labels
    m = core.load("dfl_J03WMX", labels="controlled_v1")    # a view, computed on load

Rules for adding a view:
  - give it a new name; never change what an existing name does, or old scores stop meaning
    what they said;
  - it may only move frames between H and A (dead ball and unlabelled frames stay);
  - it must be deterministic, and must not be fitted to the competition's public five minutes;
  - Record the label view alongside each score.
"""

import numpy as np

FPS = 25
H, A, D, X = 0, 1, 2, 3  # Home, Away, Dead ball, unlabelled


def _runs(label):
    cut = np.flatnonzero(np.diff(label.astype(np.int16)) != 0) + 1
    start = np.concatenate([[0], cut])
    stop = np.concatenate([cut, [len(label)]])
    return start, stop, label[start]


def absorb_short_spells(label, min_frames, before_dead_ball):
    """For one period: a spell of team B shorter than `min_frames` that directly follows a spell
    of team A goes back to A when the next thing is A again (a touch inside A's possession) or,
    if `before_dead_ball`, a dead ball (B's touch put it out). Repeated until nothing changes.

    A short spell that opens play after a dead ball or an unlabelled gap is kept: a restart has a
    clear possessor.
    """
    label = label.copy()
    while True:
        start, stop, value = _runs(label)
        changed = False
        for i in range(1, len(value)):
            mine, before = value[i], value[i - 1]
            if mine > A or before > A or mine == before or stop[i] - start[i] >= min_frames:
                continue
            after = value[i + 1] if i + 1 < len(value) else X
            if after == before or (before_dead_ball and after == D):
                label[start[i]:stop[i]] = before
                changed = True
                break
        if not changed:
            return label


VIEWS = {
    # Exactly what is stored.
    "provider": None,
    # Duration only, 2.0 s, both clauses. The first rule the team discussed; kept for reference.
    "controlled_v1": lambda label: absorb_short_spells(label, int(2.0 * FPS), before_dead_ball=True),
    # The same without the before-dead-ball clause, so a clearance out of play stays a possession.
    "controlled_v1_inplay": lambda label: absorb_short_spells(label, int(2.0 * FPS), before_dead_ball=False),
}


def _touch(match):
    from . import remap_touch       # imported here: the view needs the match arrays, not just the label
    return remap_touch.apply_touch_view(match)


# Views that read the tracking as well as the label. Each takes the match dict with provider labels
# and returns the whole label array; it handles periods itself.
MATCH_VIEWS = {
    # A short spell goes back only when tracking shows no player of that team kept the ball.
    # See remap_touch.py for the control tests.
    "controlled_v2_touch": _touch,
}
NAMES = tuple(sorted(set(VIEWS) | set(MATCH_VIEWS)))


def apply(view, label, period, match=None):
    """`label` under a view, computed period by period. Views in MATCH_VIEWS also need `match`,
    the dict of contract arrays with the provider's labels."""
    if view in MATCH_VIEWS:
        if match is None:
            raise ValueError("label view {!r} reads the tracking: pass match=".format(view))
        out = MATCH_VIEWS[view](match)
        changed = out != label
        assert np.all((out[changed] <= A) & (label[changed] <= A)), "a view moved a dead or unlabelled frame"
        return out
    if view not in VIEWS:
        raise KeyError("unknown label view {!r}; have {}".format(view, ", ".join(NAMES)))
    rule = VIEWS[view]
    if rule is None:
        return label
    out = label.copy()
    for number in np.unique(period):
        rows = period == number
        out[rows] = rule(label[rows])
    changed = out != label
    assert np.all((out[changed] <= A) & (label[changed] <= A)), "a view moved a dead or unlabelled frame"
    return out
