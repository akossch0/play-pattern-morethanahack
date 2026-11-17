# Training the possession models

The five development matches are DFL J03WMX, J03WN1, J03WOH, J03WOY and J03WPY. For each pair of held-out matches, the frame models fit on the remaining three. This gives ten checkpoints per model family. Each checkpoint records its training matches. The [match inventory](training-split.json) lists the external data.

| Model | Training data per checkpoint | Settings |
|---|---|---|
| [Frame TCN](../training/tcn.py) | Three DFL matches | 420 updates, seed 20261112, AdamW, batch of 8 windows |
| Longer-trained frame TCN | Same three matches | Same architecture and loss, 1,500 updates, seed 20261117 |
| Transition TCN | Same three matches | 1,960 updates, seed 20261112, six transition outputs alongside the frame labels |
| [Short-possession detector](../training/short_possessions.py) | Same three matches | 420 updates, seeds 20261128–20261137 in sorted order of held-out pairs |
| [Tactics GRU, initial fit](../training/tactics_gru.py) | Four DFL matches, 20 A-League matches and seven other SkillCorner matches | 3,000 updates, seed 0, batch 32, learning rate 0.002, half the training sample drawn from DFL |
| Tactics GRU, final fit | The same four DFL matches | 420 updates, seed 20261113, batch 8, learning rate 0.0002, one sample every two frames |

For a development score, each contributing model excluded the match being scored. For the submission the draft averages all ten TCN pairs and all five GRUs. The [technical note](../TECHNICAL-NOTE.md) describes the later possession corrections and how the full-match configuration differs from evaluation. The shipped weights regenerate the submission directly.

The GRU fitting routines take prepared feature dictionaries. Their module documents the clean and degraded copies used in each phase. For each fold the saved base configuration gives its source match IDs and hyperparameters. The submission uses the bundled weights directly.
