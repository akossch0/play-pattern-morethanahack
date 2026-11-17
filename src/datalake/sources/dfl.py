"""DFL / IDSSE open data (Bassek et al. 2025, CC-BY 4.0) -> raw arrays.

Positions are 25 Hz, in metres with the origin at the pitch centre and y pointing up. The ball
carries the two flags the labels come from:

    BallStatus      1 = ball in play, 0 = play interrupted
    BallPossession  1 = home team, 2 = away team   (stays set while the ball is dead)

The label is D when BallStatus is 0. When BallStatus is 1, BallPossession determines H or A.
A frame with no ball record or an unknown live possession is X.
"""

import glob
import os
import xml.etree.ElementTree as ET

import numpy as np

from ..core import A, D, EXTERNAL, H, SLOTS, X, centre_metres_to_norm

SECTIONS = (("firstHalf", 1), ("secondHalf", 2))
KMH_TO_MS = 1 / 3.6


def _one(pattern):
    hits = glob.glob(os.path.join(EXTERNAL, "dfl", pattern))
    if len(hits) != 1:
        error = FileNotFoundError("expected one DFL file")
        error.filename = os.path.join(EXTERNAL, "dfl", pattern)
        raise error
    return hits[0]


def match_info(match_id):
    root = ET.parse(_one("DFL_02_01_matchinformation_*_DFL-MAT-{}.xml".format(match_id))).getroot()
    env = root.find(".//Environment")
    teams = {t.get("Role").lower(): t.get("TeamId") for t in root.iter("Team")}
    return {"pitch": (float(env.get("PitchX")), float(env.get("PitchY"))),
            "home": teams["home"], "away": teams.get("guest") or teams.get("away")}


def load(match_id):
    info = match_info(match_id)
    length, width = info["pitch"]
    path = _one("DFL_04_03_positions_raw_observed_*_DFL-MAT-{}.xml".format(match_id))

    # One pass over the file. Each FrameSet is one object (a player or the ball) in one half.
    sets = []                       # (section, team id, {column: array})
    current, rows = None, None
    for event, element in ET.iterparse(path, events=("start", "end")):
        if event == "start" and element.tag == "FrameSet":
            current = (element.get("GameSection"), element.get("TeamId"))
            rows = []
        elif event == "end" and element.tag == "Frame":
            get = element.get
            rows.append((int(get("N")), float(get("X")), float(get("Y")), float(get("S") or "nan"),
                         int(get("BallStatus") or -1), int(get("BallPossession") or -1)))
            element.clear()
        elif event == "end" and element.tag == "FrameSet":
            sets.append((current[0], current[1], np.array(rows, dtype=np.float64).reshape(-1, 6)))
            element.clear()

    periods = []
    for section, number in SECTIONS:
        mine = [s for s in sets if s[0] == section and len(s[2])]
        balls = [s for s in mine if s[1].upper() == "BALL"]
        if not balls:
            continue
        first = int(min(s[2][:, 0].min() for s in mine))
        n = int(max(s[2][:, 0].max() for s in mine)) - first + 1
        home = np.full((n, SLOTS, 2), np.nan, dtype=np.float32)
        away = np.full((n, SLOTS, 2), np.nan, dtype=np.float32)
        home_speed = np.full((n, SLOTS), np.nan, dtype=np.float32)
        away_speed = np.full((n, SLOTS), np.nan, dtype=np.float32)
        ball = np.full((n, 2), np.nan, dtype=np.float32)
        label = np.full(n, X, dtype=np.uint8)
        used = {"home": 0, "away": 0}
        for _, team, data in mine:
            index = data[:, 0].astype(np.int64) - first
            xn, yn = centre_metres_to_norm(data[:, 1], data[:, 2], length, width, y_up=True)
            if team.upper() == "BALL":
                ball[index, 0], ball[index, 1] = xn, yn
                status, owner = data[:, 4], data[:, 5]
                value = np.full(len(data), X, dtype=np.uint8)
                value[(status == 1) & (owner == 1)] = H
                value[(status == 1) & (owner == 2)] = A
                value[status == 0] = D
                label[index] = value
                continue
            side = "home" if team == info["home"] else "away" if team == info["away"] else None
            if side is None:
                continue                              # referees, if present
            slot = used[side]
            if slot >= SLOTS:
                raise ValueError("more than {} {} players in {}".format(SLOTS, side, section))
            used[side] += 1
            xy, speed = (home, home_speed) if side == "home" else (away, away_speed)
            xy[index, slot, 0], xy[index, slot, 1] = xn, yn
            speed[index, slot] = data[:, 3] * KMH_TO_MS
        periods.append((number, {"home_xy": home, "away_xy": away, "ball_xy": ball, "label": label,
                                 "home_speed": home_speed, "away_speed": away_speed}))
    if not periods:
        raise ValueError("no ball frames found in {}".format(os.path.basename(path)))
    return {"native_fps": 25, "pitch": info["pitch"], "periods": periods}
