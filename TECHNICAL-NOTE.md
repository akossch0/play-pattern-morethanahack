# Play Pattern possession model

## Approach

For each frame of tactical tracking, the model estimates home possession, away possession or dead ball. It writes `X` outside the graded halves.

1. Three temporal convolutional networks read nearby frames for ball movement, player distance and contact. A tactics model reads how both teams move and spread out, which provides evidence when the ball disappears.
2. The decoder gives the convolutional and tactics predictions equal weight and joins successive frames into possessions. A loose ball stays with the current team until there is enough evidence of a change.
3. A visible ball exit helps place the start of a stoppage. Sustained contact with the receiving player helps time a turnover. A separate detector proposes short possessions that the first timeline may have merged. The final rules check these proposals against ball motion, player movement and surrounding play.

The [architecture diagram](docs/architecture.pdf) shows the full path and the three kinds of model.

## Data and training

Inference uses the organizer’s tactical tracking metadata and positions. We considered Smart Tagging for restarts, but the external training matches lacked comparable tags.

Training used five external DFL matches with provider possession and ball-status labels converted to controlled possession. The tactics model also used 27 SkillCorner matches with provider team-possession labels. Gaps between phases provided a weak dead-ball label for the 20 A-League matches. Those matches also supplied unlabeled ball-visibility patterns for simulating gaps in the DFL tracking. The [training table](docs/training-provenance.md) gives the splits and settings. The [source list](NOTICE.md) identifies providers and licences.

## Evaluation and limitations

On five DFL development matches with simulated tracking gaps, the model scored **73.98% spell F1** and **92.70% macro F1**. Each fold excluded its scored match from fitting. Corrections raised spell F1 from 73.39% to 73.98% on the same matches used to select them. That gain is too small to establish an improvement on a new match. Long gaps still hide brief possessions. The model recovered 10 of 77 true spells shorter than one second. We would test improvements on additional external matches with long tracking gaps.

After model selection, the public five-minute pipeline check scored 72.50% spell F1 and 92.88% macro F1. Two machines independently generated the same full-match CSV.

## Reproduction

Install the pinned dependencies, then run this command on the organizer’s bundle. The output contains 199,319 frame labels. The [README](README.md) includes the validator and repeat check.

```sh
python predict.py --data-dir /path/to/organizer-data --output output/possession.csv
```
