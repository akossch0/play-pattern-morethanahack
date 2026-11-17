# Play Pattern

Possession labels (home, away, dead ball) for every frame of the tactical video, computed from the organizer's tactical player and ball tracking.

## Run

Use Python 3.13. Unpack the organizer's bundle so that `tactical/BAR-ATH_tactical_atd.xml` and `tactical/BAR-ATH_tactical_atd.txt` sit under one folder, then run:

```sh
python -m pip install -r requirements.txt
python predict.py --data-dir /path/to/organizer-data --output output/possession.csv
```

The command reads only the two tactical tracking files. It writes the 199,319-row `frame,label` CSV after checking the header, row order, labels and the `X` rows outside the two graded halves. It runs on CPU with a fixed thread count and takes several minutes.

## Check

```sh
python tests/check_release.py
python tests/check_release.py --data-dir /path/to/organizer-data
```

The first form builds a synthetic full-length bundle from seeded random motion. The second uses the real bundle. Both run the command above twice in separate processes, validate each CSV strictly and require identical bytes. The script prints the row counts and the file's SHA-256, and keeps the CSV only if you add `--keep <folder>`.

The organizer's own format check can be run on a finished file with `python src/evaluation/validate_submission.py output/possession.csv`.

## How it works

The frame models use player movement and ball position to estimate home possession, away possession or dead ball. A decoder joins those estimates into spells. Ball exits, player contacts and a short-possession detector help place the boundaries. See the [architecture diagram](docs/architecture.pdf) and [technical note](TECHNICAL-NOTE.md).

On five external DFL development matches with simulated tracking gaps, the model scored **73.98% spell F1** and **92.70% macro F1**.

## Files

| Folder | Contents |
|---|---|
| `src/` | Tracking loader, features, models, possession decoder and organizer’s validator. |
| `weights/` | Trained models and fitted decoder parameters used for inference. |
| `training/` | Fitting routines for the frame models, tactics GRU and short-possession detector. See the [training splits and settings](docs/training-provenance.md). |
| `docs/` | Architecture diagram, training settings and external match inventory. |
| `tests/` | Synthetic tracking fixture and the reproducibility check. |

See [third-party sources](NOTICE.md) for data attribution and licences.
