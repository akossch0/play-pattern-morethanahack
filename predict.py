"""Generate and validate the competition CSV from the organizer's tactical tracking bundle."""

import argparse
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import warnings

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from datalake import competition, core
from evaluation import possession_metrics as pm
from possession.pipeline import predict_tracking

THREADS = 4
LOAD_WARNINGS = ("malformed_lines", "records_out_of_range", "duplicate_records", "missing_frames")


def check_csv(path, period):
    """Reject anything but the exact submission layout with X only outside the graded halves."""
    lines = Path(path).read_bytes().split(b"\n")
    if lines[0] != b"frame,label" or lines[-1] != b"" or len(lines) != len(period) + 2:
        raise ValueError("The CSV needs the frame,label header and one row per video frame")
    expected = {True: {b"H", b"A", b"D"}, False: {b"X"}}
    for frame, (line, graded) in enumerate(zip(lines[1:-1], (period > 0).tolist())):
        index, _, label = line.partition(b",")
        if index != str(frame).encode() or label not in expected[graded]:
            raise ValueError(f"Invalid row for video frame {frame}")
    issue = pm.validate_submission(pm.read_label_csv(path), len(period))
    if issue:
        raise ValueError(issue)


def generate(data_dir, output, *, weights=ROOT / "weights"):
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    match = competition.load_competition(data_dir)
    for field in LOAD_WARNINGS:
        if match["report"][field]:
            warnings.warn(f"Tactical tracking: {field.replace('_', ' ')} = {match['report'][field]}")
    with torch.inference_mode():
        codes = predict_tracking(competition.graded(match), weights)
    period = match["period"]
    if codes.shape != (np.count_nonzero(period),) or not np.isin(codes, (core.H, core.A, core.D)).all():
        raise ValueError("Expected one H, A or D prediction for each graded frame")
    labels = competition.predictions_to_submission(codes, n_frames=len(period))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".prediction-", dir=output.parent) as folder:
        path = Path(folder) / output.name
        competition.write_submission(path, labels, n_frames=len(period))
        check_csv(path, period)
        os.replace(path, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Root of the unpacked organizer bundle (contains tactical/)")
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV")
    args = parser.parse_args()
    generate(args.data_dir, args.output)
    print(f"Wrote and validated {args.output}")


if __name__ == "__main__":
    main()
