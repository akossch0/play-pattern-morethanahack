# Tracking data format (v2)

The external match loader stores each match as a 25 Hz NumPy archive. The two halves are concatenated in play order. Label views and simulated tracking gaps operate on loaded copies.

## Arrays

`T` is the number of frames. Each team has 23 tracking slots (`P`).

| Field | Type and shape | Meaning |
|---|---|---|
| `frame` | int32 · T | Sequential row index from zero |
| `period` | int8 · T | First or second half |
| `time_s` | float32 · T | Seconds from the start of the half |
| `home_xy`, `away_xy` | float32 · T × P × 2 | Player positions |
| `home_speed`, `away_speed` | float32 · T × P | Speed in metres per second |
| `ball_xy` | float32 · T × 2 | Ball position |
| `home_observed`, `away_observed` | bool · T × P | The source measured the player position |
| `ball_observed` | bool · T | The source measured the ball position |
| `label` | uint8 · T | 0 home, 1 away, 2 dead ball, 3 unlabelled |
| `interpolated` | bool · T | The output frame falls between native samples |

The archive also stores `match_key`, `source`, `label_source`, `split`, `pitch_length_m`, `pitch_width_m`, `fps`, `native_fps` and `contract_version`.

## Coordinates and timing

- Coordinates are normalized to pitch length and width with the origin at the top left. Off-pitch values can fall outside [0, 1]. Teams retain their original direction of play in each half.
- Missing positions and speeds use `NaN`. A-League files can supply extrapolated positions with the corresponding observation flag set to false.
- A slot represents a track. Slot order has no tactical meaning, and a slot can be reused after a track ends.
- Positions are interpolated between available native samples. A position stays missing when either interpolation endpoint is missing. Labels use the earlier native sample.
- The loader uses native speed when available at 25 Hz and otherwise estimates it from a centred 0.16-second difference. Speeds above 12 m/s are set to `NaN`.

DFL tracking can freeze the ball during stoppages. A-League tracking can lose all players and the ball during replays and close-ups. Both are source-specific cues. The data loader preserves them, and the training pipeline handles them through masking or degradation.

## Labels

| Label source | Live possession | Dead ball |
|---|---|---|
| DFL | Provider possession flag | Provider ball-status flag |
| SkillCorner A-League | Provider possession estimate | Gaps between play phases, used as a proxy |
| SkillCorner 2019/20 | Provider possession estimate | Unlabelled |

The training loss and scorer exclude unlabelled ground-truth frames. Evaluation reports label sources separately.

| Label view | Rule |
|---|---|
| `provider` | Uses the stored labels. |
| `controlled_v1` | Returns an opposing spell under two seconds to the prior team when that team resumes or play stops. |
| `controlled_v1_inplay` | Returns a short spell only when the prior team resumes during live play. |
| `controlled_v2_touch` | Checks short opposing spells for sustained control or a second player’s touch. Preserves spells when the required ball or player observation is missing. For reassignment before a stoppage, the prior team must also take the restart, and the restart must fall outside the kickoff and goal-kick zones. |

A label view changes only home and away ownership. The selected development result uses `controlled_v2_touch`.

## Match assignment and competition frames

Match files carry a `train` or `test` split. Development holds out whole matches from the training split. Resampled and degraded copies keep the same assignment.

The competition adapter reads the organizer’s asset bundle separately. It uses video-frame indices, marks frames outside the two graded halves with period 0, and starts with all labels unlabelled. For inference, the adapter compacts the two halves and keeps each row’s video-frame index for CSV export.
