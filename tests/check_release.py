"""Run the documented command twice and require a valid, byte-identical CSV.

    python tests/check_release.py                       # synthetic bundle in a temporary folder
    python tests/check_release.py --data-dir /path/to/organizer-data --keep /private/review

With ``--data-dir`` the two runs use the real bundle and the exact README command. The CSV
is kept only when ``--keep`` names a folder; this script prints its SHA-256, never labels.
``--stem`` renames the synthetic files for sandboxes that refuse the organizer's file names.
"""

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import games
from synthetic_bundle import write_bundle

# One-liner that sets the tactical file stem in the vendored games module before running
# predict.py through runpy; run() uses it only when the requested stem differs from the
# registry's, which main() allows only with the synthetic bundle.
RENAMED = (
    "import runpy, sys; sys.path.insert(0, {src!r}); import games; "
    "games.SOURCES['tactical']['stem'] = {stem!r}; sys.argv = {argv!r}; "
    "runpy.run_path({script!r}, run_name='__main__')"
)


def strict_check(path):
    """Independent of predict.py: exact bytes, frame order, and X exactly outside the halves."""
    frames = games.source_frames("tactical", games.GAME)
    graded = [False] * frames
    for _, first, last in games.half_video_blocks(games.GAME):
        graded[first:last + 1] = [True] * (last - first + 1)
    data = path.read_bytes()
    assert data.isascii() and b"\r" not in data and data.endswith(b"\n"), "not plain LF-terminated ASCII"
    lines = data[:-1].split(b"\n")
    assert lines[0] == b"frame,label", "wrong header"
    assert len(lines) == frames + 1, f"{len(lines) - 1} rows, expected {frames}"
    for frame, line in enumerate(lines[1:]):
        index, _, label = line.partition(b",")
        assert index == b"%d" % frame, f"row {frame} has frame {index!r}"
        assert label in ((b"H", b"A", b"D") if graded[frame] else (b"X",)), f"row {frame} has label {label!r}"
    return frames, sum(graded)


def run(data_dir, output, stem):
    argv = [str(ROOT / "predict.py"), "--data-dir", str(data_dir), "--output", str(output)]
    if stem == games.SOURCES["tactical"]["stem"]:
        command = [sys.executable, *argv]
    else:
        command = [sys.executable, "-c", RENAMED.format(
            src=str(ROOT / "src"), stem=stem, argv=argv, script=argv[0])]
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    subprocess.run(command, check=True, cwd=output.parent, env=environment)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, help="Organizer bundle; omit to use a synthetic one")
    parser.add_argument("--stem", default=games.SOURCES["tactical"]["stem"], help="Synthetic file stem")
    parser.add_argument("--keep", type=Path, help="Folder that receives the verified CSV")
    args = parser.parse_args()
    if args.data_dir is not None and args.stem != games.SOURCES["tactical"]["stem"]:
        parser.error("--stem applies to the synthetic bundle only")
    with TemporaryDirectory(prefix="release-check-") as scratch:
        scratch = Path(scratch)
        data_dir = args.data_dir
        if data_dir is None:
            data_dir = scratch / "bundle"
            write_bundle(data_dir, args.stem)
        digests = []
        for name in ("first", "second"):
            output = scratch / name / "possession.csv"
            output.parent.mkdir()
            run(data_dir, output, args.stem)
            frames, graded = strict_check(output)
            digests.append(hashlib.sha256(output.read_bytes()).hexdigest())
        assert digests[0] == digests[1], "the two runs wrote different bytes"
        if args.keep is not None:
            args.keep.mkdir(parents=True, exist_ok=True)
            (args.keep / "possession.csv").write_bytes(output.read_bytes())
    source = "organizer bundle" if args.data_dir is not None else "synthetic bundle"
    print(f"{source}: {frames} rows, {graded} graded, strict format valid, two runs byte-identical")
    print(f"sha256 {digests[0]}")


if __name__ == "__main__":
    main()
