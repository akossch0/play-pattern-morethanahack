# Third-party sources

## Metrica Sports

`src/games.py`, `src/evaluation/possession_metrics.py` and `src/evaluation/validate_submission.py` are unmodified copies from the [More Than A Hack 2027 starter repository](https://github.com/metrica-sports/more-than-a-hack-2027) (MIT License, Copyright (c) 2026-2027 Metrica Sports). [LICENSE-METRICA](LICENSE-METRICA) is that repository's licence file and applies to those three files. The tactical-metadata parser `slot_teams()` in `src/datalake/competition.py` follows the parser in the same repository's `starter/baseline.py` and is covered by the same licence.

SHA-256 of the three copies:

- `src/games.py`: `3b2122794e9e0476e2b3a871d14880e5fa47c4feaeacf99caef054c2309076ce`
- `src/evaluation/possession_metrics.py`: `341bad48904b489d6e875397b27687f5d6a4d22a397f3dc8af3a00d48f16874d`
- `src/evaluation/validate_submission.py`: `07ac29499acbd337e1dc1de1832d63f6e1fa1d981e22f6ad08212529e2bbee14`

The organizer distributes the challenge match separately to enrolled teams. Obtain it directly from the organizer.

## External training data

- **DFL / IDSSE.** Manuel Bassek, Robert Rein, Hendrik Weber and Daniel Memmert (2025), [An integrated dataset of spatiotemporal and event data in elite soccer](https://doi.org/10.6084/m9.figshare.28196177.v1), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The downloader (`src/datalake/fetch.py`) uses the [PySport mirror](https://huggingface.co/datasets/pysport/idsse-data/tree/a715a38dfbaf5f58e431727c2b78d174101a703c). The preparation code (`src/datalake/harmonise.py` and `src/datalake/sources/`) converts coordinates and frame grids and derives labels from the published ball flags.
- **SkillCorner.** Two releases from the SkillCorner open-data repository: the [2024/25 A-League release](https://github.com/SkillCorner/opendata/tree/4340d274572876239c154c90bc507a9b3250a656) and a separate [2019/20 public sample](https://github.com/SkillCorner/opendata/tree/1b99d401ddc82e366c5c20d6860878668fd5858f), [MIT](https://github.com/SkillCorner/opendata/blob/4340d274572876239c154c90bc507a9b3250a656/LICENSE), © 2020 SkillCorner.

The download manifest `src/datalake/manifest.json` records the source URL, size and SHA-256 of each file.

The checkpoints and fitted parameter files under `weights/` were fitted on matches from these sources (`docs/training-split.json` lists the matches). `weights/matched-context-index.npz` was built from the five DFL development matches. In its loader's words it is "a numeric example index exported from external development matches" (`src/possession/matched_context.py`). For each of its 1,123 candidate examples it holds 23 standardized summary values (log spell lengths, and window averages of model outputs, ball visibility and the difference between the two teams' nearest-player distances to the ball), a host-spell number, a fitted gain and a flag for candidates that end at a stoppage. It holds no tracking coordinates, frame indices or per-frame labels.
