"""The "controlled_v2_touch" label view: a short spell goes back to the previous team only when the
tracking shows that the team never kept the ball.

`controlled_v1` (datalake/remap.py) hands back every spell under 2.0 s. About a quarter of those
show a player on the ball for half a second or two team-mates touching it. This view keeps v1's
contexts and 2.0 s cap and adds evidence, so it needs the match arrays and not just the label:

    from datalake import core, remap_touch
    m = core.load("dfl_J03WMX")                    # provider labels
    label = remap_touch.apply_touch_view(m)        # uint8, same length; never written back

    python -m datalake.remap_touch                 # the DFL table: provider / controlled_v1 / this

The rule, per period, scanning left to right and re-evaluating after each flip. A spell of team B
goes back to team A only if ALL hold:
  - context: it directly follows a live spell of A, and is followed by A again (clause 1) or by a
    dead ball after which A takes the restart (clause 2a). A spell that opens play after D / X, a
    spell before a dead ball where B restarts, and a spell at the end of a half are never touched;
  - cap: it is shorter than 2.0 s (50 frames);
  - single-touch gate: the longest run of frames with the same nearest B player within 1.5 m of
    the ball is under 0.5 s (fewer than 13 frames), and at most one distinct B player is ever
    the nearest one within 1.5 m. Only the nearest B player is looked at on each frame, so a
    second B player who is close but never the nearest does not count;
  - restart exclusion: clause 2a does not fire when A's restart is a kick-off (ball within 2 m of
    the centre spot; the 2 m is this module's choice) or a goal kick (ball inside a six-yard
    area). Those follow B's goal or B's shot, not B putting the ball out;
  - missing-data guard: no flip unless, on every frame of the spell, the ball is observed and the
    nearest B player's position is observed; for clause 2a the ball must also be observed on the
    first live frame of the restart. Without the guard a spell with no ball has no contact and
    would flip by default. The guard cannot see a B player who has no position at all: if the
    player who touched the ball is missing while other B players are tracked, the spell shows no
    contact and flips. DFL has no such frames; broadcast tracking does.
D and X frames never change. `match` must carry the provider's labels: a match loaded through
another view is refused, because the result would be a labelling with no name.

Three things the wording leaves open, fixed here:
  - "The same B player" is a slot index, compared only within one spell. Slots carry no identity
    (CONTRACT.md). Where a source reuses a slot for a new track (SkillCorner 2019/20) two players
    can share an index within a spell, and one player can change index; both are rare inside 2 s,
    and outside DFL this view is unvalidated.
  - Distance is 2-D. Ball height is not in the contract, so a ball passing overhead counts as
    close. That only ever blocks a flip.
  - The goal-kick test accepts either six-yard area, because direction of play is not stored. A
    restart by A inside B's six-yard area does not occur in practice (a free kick there is moved
    out to the area's edge), and the 2 m and six-yard tests also catch a free kick taken on those
    spots. Both only ever block a flip.

Measured on the 7 DFL matches (668.4 min, by `python -m datalake.remap_touch`): provider 3061
possessions, 4.58/min; controlled_v1 2017, 3.02/min, 643 flips; this view 2361, 3.53/min, 423
flips (277 clause 1, 146 clause 2a), 1.02% of all frames and 1.84% of live frames changed. 756 of
the changed frames are ones controlled_v1 leaves alone, so this is not a subset of v1. Known
Known limits: deliberate one-touch passes and clearances
still flip, and the gate values are judgement, not fitted.
"""

import numpy as np

FPS = 25
H, A, D, X = 0, 1, 2, 3

CAP_FRAMES = int(2.0 * FPS)             # a spell this long or longer is a real possession
RADIUS_M = 1.5                          # "within reach of the ball", 2-D
DWELL_FRAMES = 13                       # 0.5 s rounded up: this many close frames is keeping it
MAX_PLAYERS = 1                         # two B players close is a B-to-B exchange
KICK_OFF_RADIUS_M = 2.0
GOAL_AREA_DEPTH_M = 5.5
GOAL_AREA_HALF_WIDTH_M = 9.16

FLIPS = ("clause_1", "clause_2a")
KEPT = ("kept_by_gate", "kept_by_guard", "kept_by_restart")


def _runs(label):
    """Maximal runs of equal values in `label`: arrays (start, stop, value), stop exclusive."""
    cut = np.flatnonzero(np.diff(label.astype(np.int16)) != 0) + 1
    start = np.concatenate([[0], cut])
    stop = np.concatenate([cut, [len(label)]])
    return start, stop, label[start]


def _nearest(xy_m, observed, ball_m):
    """Per frame: distance in metres from the ball to the nearest present player, that player's
    slot, and whether that position was observed. Distance is inf where there is no ball or no
    player."""
    if xy_m.shape[1] == 0:
        nobody = np.zeros(len(ball_m), dtype=np.int64)
        return np.full(len(ball_m), np.inf), nobody, nobody.astype(bool)
    with np.errstate(invalid="ignore"):
        dist = np.hypot(xy_m[:, :, 0] - ball_m[:, None, 0], xy_m[:, :, 1] - ball_m[:, None, 1])
    dist = np.where(np.isnan(dist), np.inf, dist)
    slot = dist.argmin(axis=1)
    rows = np.arange(len(slot))
    best = dist[rows, slot]
    return best, slot, observed[rows, slot] & np.isfinite(best)


def _evidence(match):
    """What the gate, the guard and the restart test read, indexed by team code (H = 0, A = 1)."""
    size = np.array([float(match["pitch_length_m"]), float(match["pitch_width_m"])])
    ball_m = match["ball_xy"].astype(np.float64) * size
    teams = [_nearest(match[side + "_xy"].astype(np.float64) * size, match[side + "_observed"], ball_m)
             for side in ("home", "away")]
    return {
        "size": size,
        "ball_m": ball_m,
        "ball_seen": match["ball_observed"] & np.isfinite(ball_m[:, 0]),
        "dist": [t[0] for t in teams],
        "slot": [t[1] for t in teams],
        "seen": [t[2] for t in teams],
    }


def _rows(ev, rows):
    """The evidence on the frames `rows` only, so that one period is indexed from 0."""
    out = {"size": ev["size"], "ball_m": ev["ball_m"][rows], "ball_seen": ev["ball_seen"][rows]}
    for name in ("dist", "slot", "seen"):
        out[name] = [values[rows] for values in ev[name]]
    return out


def _contact(ev, start, stop, team):
    """(longest run of frames with the same nearest `team` slot within RADIUS_M, number of
    distinct `team` slots ever within RADIUS_M) over frames start..stop."""
    close = ev["dist"][team][start:stop] <= RADIUS_M
    slot = ev["slot"][team][start:stop]
    longest = current = 0
    previous = -1
    for k in range(stop - start):
        if close[k]:
            current = current + 1 if slot[k] == previous else 1
            previous = slot[k]
            longest = max(longest, current)
        else:
            current, previous = 0, -1
    return longest, len(set(slot[close].tolist()))


def _measured(ev, start, stop, team):
    return bool(np.all(ev["ball_seen"][start:stop]) and np.all(ev["seen"][team][start:stop]))


def _restart(ev, frame):
    """What kind of restart begins at `frame`, the first live frame after a dead ball: "kick_off",
    "goal_kick", "other", or "unknown" when the ball is not observed there."""
    if not ev["ball_seen"][frame]:
        return "unknown"
    length, width = ev["size"]
    x, y = ev["ball_m"][frame]
    if np.hypot(x - length / 2, y - width / 2) <= KICK_OFF_RADIUS_M:
        return "kick_off"
    if min(x, length - x) <= GOAL_AREA_DEPTH_M and abs(y - width / 2) <= GOAL_AREA_HALF_WIDTH_M:
        return "goal_kick"
    return "other"


def _clause(value, i, duration_only):
    """Which context run `i` is in: "clause_1", "clause_2a" or None. Under `duration_only` any
    dead ball counts, which is controlled_v1's second clause."""
    mine, before = value[i], value[i - 1]
    if mine > A or before > A or mine == before:
        return None
    after = value[i + 1] if i + 1 < len(value) else X
    if after == before:
        return "clause_1"
    if after != D:
        return None
    restarter = value[i + 2] if i + 2 < len(value) else X
    return "clause_2a" if duration_only or restarter == before else None


def _decide(ev, start, stop, value, i, duration_only):
    """("clause_1" | "clause_2a" | None, why a candidate was kept or None) for run `i`."""
    clause = _clause(value, i, duration_only)
    if clause is None or stop[i] - start[i] >= CAP_FRAMES:
        return None, None
    if duration_only:
        return clause, None
    first, last, team = start[i], stop[i], int(value[i])
    if not _measured(ev, first, last, team):
        return None, "kept_by_guard"
    if clause == "clause_2a":
        kind = _restart(ev, start[i + 2])
        if kind == "unknown":
            return None, "kept_by_guard"
        if kind != "other":
            return None, "kept_by_restart"
    longest, players = _contact(ev, first, last, team)
    if longest >= DWELL_FRAMES or players > MAX_PLAYERS:
        return None, "kept_by_gate"
    return clause, None


def _scan(label, ev, counts, kept, duration_only):
    """One period; `ev` is the evidence on that period's frames. After a flip the merged run is
    looked at again, which is the same as restarting from the left: nothing before it has
    changed. `kept` collects spans in period-local frames."""
    label = label.copy()
    start, stop, value = _runs(label)
    i = 1
    while i < len(value):
        clause, why = _decide(ev, start, stop, value, i, duration_only)
        if why:
            kept[why].add((int(start[i]), int(stop[i])))
        if clause is None:
            i += 1
            continue
        label[start[i]:stop[i]] = value[i - 1]
        counts[clause] += 1
        start, stop, value = _runs(label)
        # The flipped run has merged into run i - 1, so that is the run to look at again; run 0
        # has no predecessor and is never a candidate, hence the floor of 1.
        i = max(1, i - 1)
    return label


def _view(match, duration_only=False):
    """(label, report). `duration_only` switches off the gate, the guard and both restart
    conditions; what is left is controlled_v1, and the tests hold it to that."""
    if match.get("label_view", "provider") != "provider":
        raise ValueError("controlled_v2_touch needs the provider's labels, not label view {!r}".format(
            match["label_view"]))
    label, period = match["label"], match["period"]
    ev = _evidence(match)
    out = label.copy()
    counts = {name: 0 for name in FLIPS}
    kept = {name: 0 for name in KEPT}
    for number in np.unique(period):
        # A period is taken by mask, as remap.apply does, so the two agree on any period array.
        rows = period == number
        spans = {name: set() for name in KEPT}
        out[rows] = _scan(label[rows], _rows(ev, rows), counts, spans, duration_only)
        for name in KEPT:
            kept[name] += len(spans[name])
    changed = out != label
    assert np.all((out[changed] <= A) & (label[changed] <= A)), "the view moved a dead or unlabelled frame"
    report = dict(counts)
    report["flips"] = sum(counts.values())
    report["frames_changed"] = int(changed.sum())
    # A candidate kept at one point of the scan can still be absorbed by a neighbour's flip later.
    report.update(kept)
    return out.astype(np.uint8), report


def apply_touch_view(match):
    """`match["label"]` under controlled_v2_touch: uint8, same length.

    `match` is the dict from core.load with the provider's labels (labels="provider"); the arrays
    read are label, period, ball_xy, home_xy, away_xy, the three *_observed arrays and the pitch
    size. Nothing in `match` is modified.
    """
    return _view(match)[0]


def touch_view_report(match):
    """(label, report). report counts executed flips by clause ("clause_1", "clause_2a", "flips"),
    "frames_changed", and the candidate spells left alone, by the first test that kept them
    ("kept_by_guard", "kept_by_restart", "kept_by_gate")."""
    return _view(match)


def possessions(label, period):
    """Number of H / A spells, counted per period."""
    total = 0
    for number in np.unique(period):
        value = _runs(label[period == number])[2]
        total += int(np.sum(value <= A))
    return total


def main():
    from . import core, remap
    from .matches import by_source

    names = ("provider", "controlled_v1", "controlled_v2_touch")
    pooled = {name: {"possessions": 0, "frames_changed": 0} for name in names}
    flips = {name: 0 for name in FLIPS + KEPT}
    frames = live = 0
    print("{:<12} {:>8} {:>8} {:>8}   {:>5} {:>4} {:>4}   {}".format(
        "match", "provider", "v1", "v2_touch", "flips", "c1", "c2a", "kept: gate / guard / restart"))
    for entry in by_source("dfl"):
        m = core.load(entry["key"])
        touch, report = touch_view_report(m)
        labels = {"provider": m["label"], "controlled_v1": remap.apply("controlled_v1", m["label"], m["period"]),
                  "controlled_v2_touch": touch}
        minutes = len(m["label"]) / FPS / 60
        row = []
        for name in names:
            count = possessions(labels[name], m["period"])
            pooled[name]["possessions"] += count
            pooled[name]["frames_changed"] += int(np.sum(labels[name] != m["label"]))
            row.append(count / minutes)
        for name in flips:
            flips[name] += report[name]
        frames += len(m["label"])
        live += int(np.sum(m["label"] <= A))
        print("{:<12} {:>8.2f} {:>8.2f} {:>8.2f}   {:>5} {:>4} {:>4}   {} / {} / {}".format(
            entry["key"], row[0], row[1], row[2], report["flips"], report["clause_1"], report["clause_2a"],
            report["kept_by_gate"], report["kept_by_guard"], report["kept_by_restart"]))
    minutes = frames / FPS / 60
    print("\npooled, {:.1f} min ({:.1f} min live)".format(minutes, live / FPS / 60))
    for name in names:
        p = pooled[name]
        print("  {:<20} {:>5} possessions  {:.2f}/min  frames changed {:>6} = {:.2f}% of all, {:.2f}% of live".format(
            name, p["possessions"], p["possessions"] / minutes, p["frames_changed"],
            100.0 * p["frames_changed"] / frames, 100.0 * p["frames_changed"] / live))
    print("  controlled_v2_touch flips: {} ({} clause 1, {} clause 2a); kept by gate {}, guard {}, restart {}".format(
        flips["clause_1"] + flips["clause_2a"], flips["clause_1"], flips["clause_2a"],
        flips["kept_by_gate"], flips["kept_by_guard"], flips["kept_by_restart"]))


if __name__ == "__main__":
    main()
