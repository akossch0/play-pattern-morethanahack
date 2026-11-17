"""Convert downloaded external tracking into validated 25 Hz NumPy archives.

    PYTHONPATH=src python -m datalake.harmonise --source dfl
    PYTHONPATH=src python -m datalake.harmonise dfl_J03WMX dfl_J03WN1
"""

import argparse
import sys
import time

from . import core
from .matches import BY_KEY, MATCHES
from .sources import dfl, skillcorner


LOADERS = {
    "dfl": dfl.load,
    "skillcorner_aleague": skillcorner.load_aleague,
    "skillcorner_2019": skillcorner.load_2019,
}


def harmonise(key):
    entry = BY_KEY[key]
    raw = LOADERS[entry["source"]](entry["native_id"])
    match = core.finalise(key, raw)
    core.validate(match)
    return match


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("keys", nargs="*", help="match keys; default is every match")
    parser.add_argument("--source", choices=LOADERS, help="only matches from this source")
    parser.add_argument("--out", help="output directory (default src/data/team-dataset)")
    args = parser.parse_args(argv)

    keys = args.keys or [m["key"] for m in MATCHES if not args.source or m["source"] == args.source]
    unknown = [k for k in keys if k not in BY_KEY]
    if unknown:
        sys.exit("unknown match key(s): {}".format(", ".join(unknown)))

    failed = 0
    for key in keys:
        started = time.time()
        try:
            match = harmonise(key)
        except FileNotFoundError as error:
            print("{:<16} MISSING  {}".format(key, error.filename), flush=True)
            failed += 1
            continue
        except (AssertionError, ValueError) as error:
            print("{:<16} FAILED   {}".format(key, error), flush=True)
            failed += 1
            continue
        core.save(match, args.out)
        s = core.summary(match)
        print("{:<16} ok  {:>6} fr  H {:>4}  A {:>4}  D {:>4}  X {:>4}  poss {:>3} ({:.1f}/min, "
              "{:.1f} s)  ball {:>5}%  players {:>4}  {:.0f}s".format(
                  key, s["frames"], s["H_pct"], s["A_pct"], s["D_pct"], s["X_pct"],
                  s["possessions"], s["possessions_per_min"], s["mean_possession_s"],
                  s["ball_present_pct"], s["mean_players"], time.time() - started), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
