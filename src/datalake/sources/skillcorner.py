"""SkillCorner open data -> raw arrays. Two releases, two formats, both broadcast tracking at 10 fps
in metres with the origin at the pitch centre.

2019/20 release (9 matches, MIT, at commit 1b99d40): only players visible on the broadcast are
present (about 14 of 22), nothing during replays. Most observations (93-99%) do carry a player ID
(`trackable_object`), but slots here follow the provider's short track runs (`track_id`), not the
ID, so a player passes through many slots in a match. That fragmentation is close to how the
competition tracking behaves. Labels: `possession.group` gives H / A; there is
no dead-ball signal, so every other frame is X.

A-League 2024/25 release (20 matches, MIT): a frame has either all 22 players or none (19-34% of
frames per match have no tracking at all), with `is_detected` false where a position was
extrapolated off-screen. Every position is kept, and the *_observed
arrays say which ones the provider actually saw, so dropping the extrapolated ones is the user's
choice and not ours. Labels: `possession.group` gives
H / A; a frame that no phase-of-play row covers is D ("phases of play are only defined when the
ball is in play"); a frame in play with no possessing team is X.
"""

import csv
import json
import os

import numpy as np

from ..core import A, D, EXTERNAL, H, SLOTS, SlotAllocator, X, centre_metres_to_norm, empty_period

NATIVE_FPS = 10
GROUP_LABEL = {"home team": H, "away team": A}


def _norm(x, y, pitch):
    xn, yn = centre_metres_to_norm(x, y, pitch[0], pitch[1], y_up=True)
    return float(xn), float(yn)


def _period_rows(frames):
    """{period: (first row, last row)} over a list of frame dicts."""
    spans = {}
    for row, f in enumerate(frames):
        p = f.get("period")
        if p in (1, 2):
            first, _ = spans.get(p, (row, row))
            spans[p] = (first, row)
    return spans


def load_2019(match_id):
    folder = os.path.join(EXTERNAL, "skillcorner_2019", str(match_id))
    with open(os.path.join(folder, "match_data.json")) as handle:
        meta = json.load(handle)
    with open(os.path.join(folder, "structured_data.json")) as handle:
        frames = json.load(handle)
    pitch = (float(meta["pitch_length"]), float(meta["pitch_width"]))
    home_id = meta["home_team"]["id"]
    side_of = {p["trackable_object"]: ("home" if p["team_id"] == home_id else "away")
               for p in meta["players"]}
    ball_id = meta["ball"]["trackable_object"]
    group_side = {"home team": "home", "away team": "away",
                  "home goalkeeper": "home", "away goalkeeper": "away"}

    known = set(group_side) | {"referee", "ball", "balls"}
    periods, dropped, unknown = [], 0, {}
    for number, (first, last) in sorted(_period_rows(frames).items()):
        n = last - first + 1
        out = empty_period(n)
        slots = {"home": SlotAllocator(reuse=True), "away": SlotAllocator(reuse=True)}
        for row in range(first, last + 1):
            f = frames[row]
            if f.get("period") != number:
                continue
            i = row - first
            out["label"][i] = GROUP_LABEL.get((f.get("possession") or {}).get("group"), X)
            for obj in f["data"]:
                ident = obj.get("trackable_object")
                group = obj.get("group_name")
                # The provider writes the ball as "ball" and, in a few frames of match 3442, "balls".
                if ident == ball_id or group in ("ball", "balls"):
                    out["ball_xy"][i] = _norm(obj["x"], obj["y"], pitch)
                    continue
                side = side_of.get(ident) if ident is not None else group_side.get(obj.get("group_name"))
                if side is None:
                    if group is not None and group not in known:
                        unknown[group] = unknown.get(group, 0) + 1
                    continue                          # referee, or an object we do not recognise
                slot = slots[side].slot(obj["track_id"], i)
                if slot is None:
                    dropped += 1
                    continue
                out[side + "_xy"][i, slot] = _norm(obj["x"], obj["y"], pitch)
        periods.append((number, out))
    if unknown:
        raise ValueError("unrecognised group_name values {}: decide how to read them".format(unknown))
    return {"native_fps": NATIVE_FPS, "pitch": pitch, "periods": periods, "dropped_tracks": dropped}


def load_aleague(match_id):
    folder = os.path.join(EXTERNAL, "skillcorner_aleague", str(match_id))
    stem = os.path.join(folder, str(match_id))
    with open(stem + "_match.json") as handle:
        meta = json.load(handle)
    pitch = (float(meta["pitch_length"]), float(meta["pitch_width"]))
    home_id = meta["home_team"]["id"]
    side_of = {p["id"]: ("home" if p["team_id"] == home_id else "away") for p in meta["players"]}

    frames = []
    with open(stem + "_tracking_extrapolated.jsonl") as handle:
        for line in handle:
            if line.strip():
                frames.append(json.loads(line))
    row_of_frame = {f["frame"]: row for row, f in enumerate(frames)}

    in_play = np.zeros(len(frames), dtype=bool)
    with open(stem + "_phases_of_play.csv", newline="") as handle:
        for phase in csv.DictReader(handle):
            start, end = int(float(phase["frame_start"])), int(float(phase["frame_end"]))
            rows = [row_of_frame[f] for f in (start, end) if f in row_of_frame]
            if len(rows) == 2:
                in_play[rows[0]:rows[1] + 1] = True

    periods = []
    for number, (first, last) in sorted(_period_rows(frames).items()):
        base = frames[first]["frame"]
        n = frames[last]["frame"] - base + 1
        out = empty_period(n)
        out["home_observed"] = np.zeros((n, SLOTS), dtype=bool)
        out["away_observed"] = np.zeros((n, SLOTS), dtype=bool)
        out["ball_observed"] = np.zeros(n, dtype=bool)
        slots = {"home": SlotAllocator(reuse=False), "away": SlotAllocator(reuse=False)}
        for row in range(first, last + 1):
            f = frames[row]
            if f.get("period") != number:
                continue
            i = f["frame"] - base
            if not in_play[row]:
                out["label"][i] = D
            else:
                out["label"][i] = GROUP_LABEL.get((f.get("possession") or {}).get("group"), X)
            ball = f.get("ball_data") or {}
            if ball.get("x") is not None:
                out["ball_xy"][i] = _norm(ball["x"], ball["y"], pitch)
                out["ball_observed"][i] = bool(ball.get("is_detected"))
            for obj in f.get("player_data") or []:
                side = side_of.get(obj["player_id"])
                if side is None or obj.get("x") is None:
                    continue
                slot = slots[side].slot(obj["player_id"], i)
                if slot is None:
                    raise ValueError("more players than slots")
                out[side + "_xy"][i, slot] = _norm(obj["x"], obj["y"], pitch)
                out[side + "_observed"][i, slot] = bool(obj.get("is_detected"))
        periods.append((number, out))
    return {"native_fps": NATIVE_FPS, "pitch": pitch, "periods": periods}
