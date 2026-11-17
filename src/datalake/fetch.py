"""Download or verify the external files listed in the SHA-256 manifest."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import urllib.request

from .paths import EXTERNAL

MANIFEST = Path(__file__).with_name("manifest.json")


def _check(handle, entry, destination=None):
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: handle.read(1 << 20), b""):
        size += len(chunk)
        digest.update(chunk)
        if destination is not None:
            destination.write(chunk)
    if size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
        raise ValueError("size or SHA-256 differs from the manifest")


def _download(path, entry):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    request = urllib.request.Request(entry["url"], headers={"User-Agent": "play-pattern-fetch"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as output:
                temporary = Path(output.name)
                _check(response, entry, output)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sources = sorted({key.split("/", 1)[0] for key in manifest})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="*", metavar="SOURCE", help="choose from " + ", ".join(sources))
    parser.add_argument("--verify", action="store_true", help="verify local files without downloading")
    args = parser.parse_args(argv)
    selected = set(args.sources or sources)
    if selected - set(sources):
        parser.error("unknown source: " + ", ".join(sorted(selected - set(sources))))

    failures = 0
    for key, entry in manifest.items():
        if key.split("/", 1)[0] not in selected:
            continue
        path = Path(EXTERNAL) / key
        try:
            if args.verify or path.exists():
                with path.open("rb") as handle:
                    _check(handle, entry)
            else:
                _download(path, entry)
        except (OSError, ValueError) as error:
            print(f"{key}: {error}", file=sys.stderr, flush=True)
            failures += 1
        else:
            print(f"{key}: verified", flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
