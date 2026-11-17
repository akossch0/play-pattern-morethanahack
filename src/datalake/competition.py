"""The competition match in the lake's shape, and the writer for the submission file.

Models are trained on the lake (CONTRACT.md). This module is what lets the same feature code run
on the real match, and turns what a model says back into the CSV the organisers score.

    from datalake import competition
    m = competition.load_competition()            # every video frame, 0..199318
    g = competition.graded(m)                     # only the two halves, lake-shaped
    codes = my_model(g)                           # one H/A/D/X code per row of g
    competition.write_submission("out.csv", competition.predictions_to_submission(codes))

    python -m datalake.competition                # load the real match and print the report

It reads the tactical ATD only (Metrica FIFA/EPTS: metadata XML + raw txt, found through
games.asset_paths). The match files are gated and never leave the machine; nothing here writes
under data/, and nothing here reads the ground truth.

Where the competition match differs from a lake match
-----------------------------------------------------
Each of these can break a model without raising. Read them before reusing lake feature code.

- **Rows are video frames, not play.** load_competition() returns one row per video frame, so
  row i is submission row i. `period` is 1 or 2 inside the graded halves and **0** everywhere
  else (warm-up, halftime, after the whistle): 40,462 of 199,319 rows. A lake match has no
  period 0, and core.validate() rejects one. Code that loops over np.unique(period), or indexes
  by period, will treat period 0 as a third half unless it goes through graded().
- **`time_s` is NaN where period is 0**, and inside a half it counts from the first graded frame
  of that half. That is not always the kickoff: by games.py the second graded block opens about
  117 s before the second-half kickoff anchor, so time_s is not "seconds since kickoff" there.
- **`label` is all X.** There is no truth here. Anything that masks on label != X sees nothing.
- **Slots are tracklets, not players.** The ATD has 46 track slots shared by both teams; a slot
  appears, vanishes and changes person as the tracker loses and re-acquires people, and a player
  is sometimes tracked twice. Each ATD slot is packed, in file order, into a column of its team.
  In the lake a DFL column is one player within a half, and a SkillCorner 2019/20 column follows
  short track runs, so no source makes a column a player for the whole match. Slot order means
  nothing anywhere, so models that treat players as an unordered set are safe.
- **More than 23 slots in one team.** The contract has 23 columns per team. A team's first 23
  ATD slots each keep a fixed column. Any further slot ("overflow") has no column of its own: on
  each frame where it is tracked it borrows a column that no other track uses on that frame, the
  one before or the one after, and keeps it while that stays true. If no column is free the
  observation is dropped.
  Both are counted in m["report"] (overflow_slots, dropped_observations) and a warning is raised
  when any observation is dropped, so the loss is never silent.
- **Speed is the provider's**, in m/s, computed by Metrica with unknown smoothing. In the lake
  only DFL carries provider speed; every other source is a centred difference over 0.16 s. As in
  the lake, speed above core.MAX_SPEED (12 m/s) is stored as NaN and the position is kept.
- **Pitch size comes from games.py** (105.3 x 68.0 m), not from the ATD metadata (105 x 68).
  Lake matches carry their own size; never hard-code one when converting to metres.
- **Positions are not range-checked.** core.validate() requires every lake coordinate to lie
  strictly inside -0.5..1.5. Here nothing is clipped or removed; positions outside that range are
  counted in m["report"]["positions_outside_lake_range"].
- **The tracking is automatic and dirty.** The ball is missing on about 43% of frames, players
  drop out, and around 20.5 objects are tracked on a graded frame. Lake tracking is far cleaner;
  degrading it to match is a separate transform, not part of the contract.
- **`*_observed` equals "has a position"** and `interpolated` is all false: the ATD is native
  25 fps and marks nothing as extrapolated.
- **Home is the first team in the metadata**, following the organizer's starter parser.
  Player groups use the order of the metadata's Player entries; a line with a different number of
  groups raises, because the team of every slot would be wrong.
"""

import math
import os
import sys
import warnings
import xml.etree.ElementTree as ET

import numpy as np

from . import core
from .matches import CONTRACT_VERSION

if core.ROOT not in sys.path:
    sys.path.insert(0, core.ROOT)
import games  # noqa: E402  (vendored in src/, next to this package)

MATCH_KEY = "competition_" + games.GAME
SOURCE = "competition"
FIELD_SEP, GROUP_SEP, VALUE_SEP = ":", ";", ","


def blocks_for(blocks=None):
    """[(period, first_video_frame, last_video_frame)]: the graded halves, inclusive."""
    return list(blocks) if blocks is not None else games.half_video_blocks(games.GAME)


def period_of_frames(n_frames, blocks=None):
    """int8 (n_frames,): 1 or 2 inside a graded half, 0 elsewhere."""
    period = np.zeros(n_frames, dtype=np.int8)
    for number, first, last in blocks_for(blocks):
        # A slice would quietly shorten a half that does not fit, and every length check
        # downstream would then agree with the wrong number.
        if not 0 <= first <= last < n_frames:
            raise ValueError("period {} covers video frames {}..{}, outside 0..{}".format(
                number, first, last, n_frames - 1))
        period[first:last + 1] = number
    return period


def slot_teams(meta_path):
    """Team of each ATD slot, in file order: 0 home (first team in the file), 1 away."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    unknown = sorted(set(slots) - set(team_ids), key=str)
    if unknown:
        raise ValueError("{}: Player entries with unknown teamId {}".format(meta_path, unknown))
    return np.array([0 if team == team_ids[0] else 1 for team in slots], dtype=np.int8)


def _number(text):
    try:
        value = float(text)
    except ValueError:
        return np.nan
    return value if math.isfinite(value) else np.nan


def _group(text):
    """One "x,y,speed" group -> (x, y, speed), NaN for anything absent or unreadable."""
    fields = [_number(f) for f in text.split(VALUE_SEP)] + [np.nan] * 3
    return fields[:3]


def read_raw(raw_path, n_slots, n_frames):
    """Stream the raw ATD into (tracks (n_frames, n_slots, 3), ball (n_frames, 2), counts).

    A line is "<frame_id>:<x,y,speed>;...:<ball>" and frame_id counts from 1, so it lands on row
    frame_id - 1. Rows with no line stay NaN.
    """
    tracks = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    seen = np.zeros(n_frames, dtype=bool)
    counts = {"malformed_lines": 0, "records_out_of_range": 0, "duplicate_records": 0}
    with open(raw_path, encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(FIELD_SEP)
            if len(parts) != 3 or not parts[0].strip().isdigit():
                counts["malformed_lines"] += 1
                continue
            row = int(parts[0]) - 1
            if not 0 <= row < n_frames:
                counts["records_out_of_range"] += 1
                continue
            groups = parts[1].split(GROUP_SEP)
            if len(groups) != n_slots:
                raise ValueError(
                    "{} line {}: {} player groups, but the metadata lists {} slots. The slot to "
                    "team mapping would be wrong, so nothing is loaded.".format(
                        raw_path, number, len(groups), n_slots))
            counts["duplicate_records"] += int(seen[row])
            seen[row] = True
            tracks[row] = [_group(g) for g in groups]
            ball[row] = _group(parts[2])[:2]
    counts["n_records"] = int(seen.sum())
    counts["missing_frames"] = int(n_frames - seen.sum())
    return tracks, ball, counts


def pack_team(tracks):
    """One team's ATD slots (T, S, 3) -> (xy (T, 23, 2), speed (T, 23), dropped observations).

    The first 23 slots keep a fixed column. A later slot borrows a column that no other track
    uses on this frame, the previous one or the next one (so a column never passes straight from
    one track to another in either direction, the lake's rule) and holds it while that stays
    true; with no such column the observation is dropped and counted.
    """
    t = len(tracks)
    out = np.full((t, core.SLOTS, 3), np.nan, dtype=np.float32)
    fixed = min(tracks.shape[1], core.SLOTS)
    out[:, :fixed] = tracks[:, :fixed]
    dropped = 0
    for slot in range(core.SLOTS, tracks.shape[1]):
        # Cells held by other tracks. Taken before this slot writes, so its own cells never
        # block it.
        used = np.isfinite(out[..., 0])
        column, last = None, None
        for row in np.flatnonzero(np.isfinite(tracks[:, slot, 0])):
            free = ~used[row] & ~used[row + 1] if row + 1 < t else ~used[row]
            if column is None or last != row - 1 or not free[column]:
                if row:
                    free = free & ~used[row - 1]
                choices = np.flatnonzero(free)
                column = int(choices[0]) if len(choices) else None
            if column is None:
                dropped += 1
                continue
            out[row, column] = tracks[row, slot]
            last = row
    return out[..., :2], out[..., 2], dropped


def _clean_positions(tracks):
    """A position needs both x and y; a half-present one is treated as missing."""
    partial = np.isnan(tracks[..., 0]) != np.isnan(tracks[..., 1])
    tracks[partial] = np.nan
    return int(partial.sum())


def _outside(xy):
    """Positions core.validate() would reject as not normalised (outside -0.5..1.5)."""
    with np.errstate(invalid="ignore"):
        return int(np.sum(np.any((xy <= -0.5) | (xy >= 1.5), axis=-1)))


def load_competition(data_dir=None, n_frames=None, blocks=None):
    """The competition match as contract-v2 arrays over every video frame 0..N-1.

    Same arrays and scalars as a lake match, with the differences listed in the module docstring:
    period is 0 outside the graded halves, time_s is NaN there, label is all X. m["report"] holds
    the counts worth reading before trusting the load.

    data_dir: the bundle root (default games.DATA_DIR / MTAH_DATA_DIR). n_frames and blocks
    default to the real match from games.py and exist so tests can use a small fixture.
    """
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], games.GAME, data_dir)
    n = int(n_frames) if n_frames is not None else games.source_frames("tactical", games.GAME)
    team = slot_teams(meta_path)
    tracks, ball, report = read_raw(raw_path, len(team), n)
    report["partial_positions"] = _clean_positions(tracks) + _clean_positions(ball)
    report["positions_outside_lake_range"] = _outside(tracks[..., :2]) + _outside(ball)

    length, width = games.GAMES[games.GAME]["pitch"]
    period = period_of_frames(n, blocks)
    time_s = np.full(n, np.nan, dtype=np.float32)
    for number, first, last in blocks_for(blocks):
        time_s[first:last + 1] = np.arange(last - first + 1) / float(core.FPS)

    m = {"frame": np.arange(n, dtype=np.int32), "period": period, "time_s": time_s,
         "ball_xy": ball, "ball_observed": np.isfinite(ball[:, 0]),
         "label": np.full(n, core.X, dtype=np.uint8), "interpolated": np.zeros(n, dtype=bool)}
    report.update(slots={}, overflow_slots={}, dropped_observations={}, speed_above_max={})
    for code, side in ((0, "home"), (1, "away")):
        xy, speed, dropped = pack_team(tracks[:, team == code])
        with np.errstate(invalid="ignore"):
            report["speed_above_max"][side] = int(np.sum(speed > core.MAX_SPEED))
        m[side + "_xy"] = xy
        m[side + "_speed"] = core.clean_speed(speed, xy)
        m[side + "_observed"] = np.isfinite(xy[..., 0])
        slots = int(np.sum(team == code))
        report["slots"][side] = slots
        report["overflow_slots"][side] = max(0, slots - core.SLOTS)
        report["dropped_observations"][side] = dropped
    if any(report["dropped_observations"].values()):
        warnings.warn("competition ATD: {} player observations dropped: a team has more than {} "
                      "slots and no column was free for them".format(
                          sum(report["dropped_observations"].values()), core.SLOTS))
    m.update(match_key=MATCH_KEY, source=SOURCE, label_source="none", split="competition",
             pitch_length_m=float(length), pitch_width_m=float(width), fps=core.FPS,
             native_fps=float(core.FPS), contract_version=CONTRACT_VERSION,
             label_view="provider", report=report)
    validate(m)
    return m


def validate(m):
    """The contract checks that still apply here. core.validate() cannot be used: it requires
    period in {1, 2} and a match listed in matches.py, and this match is neither."""
    t = len(m["label"])
    shapes = {"home_xy": (t, core.SLOTS, 2), "away_xy": (t, core.SLOTS, 2), "ball_xy": (t, 2),
              "home_speed": (t, core.SLOTS), "away_speed": (t, core.SLOTS),
              "home_observed": (t, core.SLOTS), "away_observed": (t, core.SLOTS)}
    for name in core.ARRAYS:
        assert name in m, "missing array {}".format(name)
        expected = shapes.get(name, (t,))
        assert m[name].shape == expected, "{} has shape {}, expected {}".format(
            name, m[name].shape, expected)
    for name in core.SCALARS:
        assert name in m, "missing scalar {}".format(name)
    assert np.array_equal(m["frame"], np.arange(t)), "frame is not 0..T-1"
    assert set(np.unique(m["period"])) <= {0, 1, 2}, "period outside {0, 1, 2}"
    assert np.array_equal(np.isnan(m["time_s"]), m["period"] == 0), "time_s / period disagree"
    for name in ("home", "away", "ball"):
        xy = m[name + "_xy"]
        assert np.array_equal(np.isnan(xy[..., 0]), np.isnan(xy[..., 1])), name + " x without y"
        assert np.array_equal(m[name + "_observed"], np.isfinite(xy[..., 0])), \
            name + "_observed is not 'has a position'"
    for side in ("home", "away"):
        speed = m[side + "_speed"]
        assert not np.any(np.isfinite(speed) & np.isnan(m[side + "_xy"][..., 0])), \
            "{} speed without a position".format(side)
        finite = speed[np.isfinite(speed)]
        assert finite.size == 0 or finite.max() <= core.MAX_SPEED, side + " speed above the cap"
    return True


def graded(m):
    """Only the graded halves, in the lake's shape: period is 1 or 2, frame is 0..T-1 again, and
    `video_frame` says which submission row each row is. Feed this to lake feature code."""
    rows = m["period"] > 0
    out = {name: (m[name][rows] if name in core.ARRAYS else m[name]) for name in m}
    out["video_frame"] = m["frame"][rows]
    out["frame"] = np.arange(int(rows.sum()), dtype=np.int32)
    return out


def to_codes(labels):
    """H/A/D/X letters or 0..3 codes (whole-number floats included) -> uint8 codes, 1-D.
    Anything else raises ValueError."""
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("labels must be one-dimensional, got shape {}".format(values.shape))
    if values.dtype.kind == "f":
        if not np.all(np.isfinite(values) & (values == np.round(values))):
            raise ValueError("label codes must be whole numbers 0..3")
        values = values.astype(np.int64)
    if values.dtype.kind in "iu":
        if values.size and (values.min() < 0 or values.max() > core.X):
            raise ValueError("label codes outside 0..3")
        return values.astype(np.uint8)
    text = np.char.upper(np.char.strip(values.astype(str)))
    codes = np.full(len(text), 255, dtype=np.uint8)
    for code, letter in enumerate(core.LABELS):
        codes[text == letter] = code
    if np.any(codes == 255):
        raise ValueError("labels outside H/A/D/X: {}".format(sorted(set(text[codes == 255]))[:5]))
    return codes


def predictions_to_submission(pred, n_frames=None, blocks=None):
    """Model output -> uint8 codes for every video frame, X outside the graded halves.

    pred is one of:
      - {period: codes}, each as long as that graded half;
      - codes for every video frame (length n_frames);
      - codes for the graded rows only, in graded() order.
    """
    n = int(n_frames) if n_frames is not None else games.source_frames("tactical", games.GAME)
    blocks = blocks_for(blocks)
    period = period_of_frames(n, blocks)
    out = np.full(n, core.X, dtype=np.uint8)
    if isinstance(pred, dict):
        missing = sorted(set(number for number, _, _ in blocks) - set(pred))
        if missing:
            raise ValueError("no predictions for period {}".format(missing))
        extra = sorted(set(pred) - set(number for number, _, _ in blocks), key=str)
        if extra:
            raise ValueError("predictions for unknown period {}".format(extra))
        for number, first, last in blocks:
            codes = to_codes(pred[number])
            if len(codes) != last - first + 1:
                raise ValueError("period {}: {} predictions, the half has {} frames".format(
                    number, len(codes), last - first + 1))
            out[first:last + 1] = codes
        return out
    codes = to_codes(pred)
    n_graded = int(np.sum(period > 0))
    if len(codes) == n:
        out[period > 0] = codes[period > 0]
    elif len(codes) == n_graded:
        out[period > 0] = codes
    else:
        raise ValueError("{} predictions: expected {} (every video frame) or {} (graded frames)"
                         .format(len(codes), n, n_graded))
    return out


def write_submission(path, labels, n_frames=None, blocks=None):
    """Write the competition CSV: header "frame,label", one row per video frame, H/A/D/X.

    labels must already cover every video frame (see predictions_to_submission); a wrong length
    raises instead of writing a file the scorer would reject. Frames outside the graded halves
    are written as X whatever was passed: they are never scored.
    """
    n = int(n_frames) if n_frames is not None else games.source_frames("tactical", games.GAME)
    codes = to_codes(labels)
    if len(codes) != n:
        raise ValueError("{} labels, the submission needs exactly {} rows".format(len(codes), n))
    codes = np.where(period_of_frames(n, blocks) > 0, codes, core.X)
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="ascii", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, core.LABELS[c]) for i, c in enumerate(codes))
    return path


if __name__ == "__main__":
    match = load_competition()
    print("{}: {} video frames, pitch {} x {} m".format(
        match["match_key"], len(match["label"]), match["pitch_length_m"], match["pitch_width_m"]))
    for number in (0, 1, 2):
        print("  period {}: {} frames".format(number, int(np.sum(match["period"] == number))))
    print("  ball present on {:.1f}% of frames".format(100 * match["ball_observed"].mean()))
    for key in sorted(match["report"]):
        print("  {}: {}".format(key, match["report"][key]))
