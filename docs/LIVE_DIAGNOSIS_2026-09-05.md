# Live perception and learning audit

## Verified faults and changes

- `cr_bot.py` ran upstream unit inference before replacing the units with YOLO.
  A ready YOLO pipeline now skips that redundant inference; upstream card,
  screen and number detection still run. Failed YOLO initialization retains
  the upstream unit detector.
- Idle frames incurred an unconditional 150 ms sleep after capture and
  inference. Idle polling now targets a minimum 50 ms iteration, including
  processing time. Actual card tap delays are preserved.
- `LATENCY` logs every 60 battle ticks separate capture/UI detection,
  board/guard processing, and policy inference. These exclude tap execution.
- Enemy cycle tracking consumed raw first sightings while elixir accounting
  deduplicated them. Both now consume the same accepted purchase events.
- Tracks expired after matching, allowing a long-unseen unit to be revived
  without a new purchase event. Expiration now precedes association, and
  association compares velocity-predicted positions.
- Learning paired names in detection order with IDs in insertion order.
  Recording now pairs each name with its actual track ID, even when the
  detector changes output order.

## Limits that remain

Opponent elixir is inferred, not measured. This pipeline excludes spell
effects, can miss deployments, and cannot reliably distinguish every spawned
child from a purchased card. Name-based deduplication intentionally misses a
second same-card purchase while an earlier copy remains visible. A long
occlusion on the enemy side can still be mistaken for a new deployment after
the deduplication window. This is not an exact elixir or card-cycle oracle.

The tracker retains identity through brief missing detections, but the board
observation still exposes current detections only. Prolonged track loss can
also be mistaken for a kill by the bandit reward model. Reliable kill credit
requires stronger event evidence than absence alone; old learned files were
not erased or presented as repaired by these changes.

Default `run.ps1` mode is rules plus contextual-bandit adjustments. Live
`-Brain rl` is frozen inference; it does not update network weights between
games. Simulator PPO training is a separate path. A few live losses therefore
do not establish neural training collapse.

## Validation

133 focused tests passed after adding regression cases for purchase
deduplication, short gaps, expiration, name/ID ordering and YOLO fallback.
The initial broader run passed 148 tests and failed one checkpoint behavior
test: `test_every_live_decision_it_does_make_is_legal` selected no plays on
30 boards. Re-running that test with the original HEAD versions of economy,
tracker and policy reproduced the failure. It is not introduced by this patch.

No live matches or retraining were launched, and no win-rate improvement has
been measured. Identify the user's exact run/checkpoint before changing PPO
hyperparameters. Compare frozen checkpoints on identical evaluation opponents
and seeds; keep held-out evaluation separate from training returns. Use the
new timing logs to locate remaining latency before changing capture or taps.
