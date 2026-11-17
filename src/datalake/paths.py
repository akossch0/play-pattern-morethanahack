"""Default locations for downloaded and harmonised external matches."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL = os.environ.get("PLAY_PATTERN_EXTERNAL_DIR") or os.path.join(ROOT, "data", "external")
HARMONISED = os.environ.get("PLAY_PATTERN_HARMONISED_DIR") or os.path.join(ROOT, "data", "team-dataset")
