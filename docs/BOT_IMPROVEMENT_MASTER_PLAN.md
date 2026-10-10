# HastyCR performance improvement master plan

Prepared 2026-09-05. This is a proposed engineering and research roadmap, not a
claim that these capabilities are implemented or that any win-rate gain is
guaranteed. Start with one supported deck and the actual live configuration.
Broaden only after that path is reliable.

## Objective and working diagnosis

Improve held-out match performance, defensive response time, and stability
across successive games. Treat those as separate measurements. A higher
training return, faster simulator, or better detector mAP alone is insufficient.

Verified in the inspected code:

- Live rules mode uses contextual-bandit adjustments; live RL is frozen
  checkpoint inference. Simulator PPO is a separate learning path.
- The previous patch removed duplicate upstream troop inference under ready
  YOLO, reduced idle waiting, aligned cycle and spending events, and corrected
  learning name/ID pairing. These still need live performance validation.
- The current spatial representation has counts, HP, air and building planes,
  but no explicit troop identity, attack target, range, or velocity fields.
- The live bridge supplies nominal full troop HP. It does not measure current
  troop HP, unlike the simulator observation.
- Opponent elixir and cycle inference are not input scalars to the RL network.
- Track identity survives brief gaps, but the policy board currently exposes
  current detections only. Missing tracks can be interpreted as kills by the
  rules-mode reward model.
- Simulator decisions are spaced at 500 ms. Live capture, inference, and taps
  have a different and variable timing distribution.
- An existing checkpoint test produces no plays on 30 boards, including
  affordable states, and reproduces with the original code.

Historical reports are dated evidence, not current truth. In particular, a high
correlation between total return and wins does not rule out shaping problems:
the terminal win reward contributes directly to that correlation.

## Priority and dependency map

| Phase | Main output | Priority | Depends on |
|---|---|---|---|
| 0 | Reproducible baseline and release protocol | P0 | Nothing |
| 1 | Timestamped logs and replayable traces | P0 | Baseline IDs |
| 2 | Persistent unit and game-event tracking | P0 | Trace fixtures |
| 3 | Uncertain elixir, cycle and health estimates | P0 | Tracking/events |
| 4 | Fast, acknowledged action pipeline | P0 | Timing telemetry |
| 5 | Versioned shared observation contract | P0 | State estimates |
| 6 | Training correctness and reward audit | P0 | Baseline/evaluation |
| 7 | Tactical curriculum and imitation data | P1 | Correct environment |
| 8 | Recurrent policy and richer actions | P1 | Shared observations |
| 9 | Diverse opponent league | P1 | Stable learning |
| 10 | Simulator calibration and transfer | P0/P1 | Replay evidence |
| 11 | Controlled candidate deployment | P0 | Evaluation gates |
| 12 | Short-horizon planning and scaling | P2 | Reliable model/state |

Phases overlap where practical, but train the integrated policy only after its
observation, timing, and reward contracts are stable. Do not batch every idea
into one experimental run.

## Phase 0 — Establish what is actually running

1. Record the launcher, rules/RL mode, checkpoint hash, detector hash, side
   classifier, config, git revision, emulator resolution, and device.
2. Snapshot existing learned tables and validated checkpoints. Keep them
   immutable as baselines; do not silently clear or overwrite them.
3. Separate runs into rules-frozen, rules-learning, frozen RL, and simulator
   training. A learning label must describe a real update operation.
4. Compare successive match blocks while controlling opponent mix and game
   mode. Few-game win-rate changes are diagnostic prompts, not proof of drift.
5. Build a matrix of existing candidate checkpoints versus frozen opponents.
   Include both seats, fixed decks, and seed lists.
6. Keep development scenarios, selection evaluation, and a final holdout
   separate. Repeatedly using a seed set for selection makes it validation,
   not an untouched test set.
7. Report wins, losses, draws, crown difference, opponent breakdown, and
   uncertainty. Use paired comparisons on common scenarios; aggregate across
   independent training seeds for substantial training claims.

Exit: one reproducible baseline command and one comparison command, with no
ambiguity about which model/config generated a result.

## Phase 1 — Instrument the complete decision chain

Create a structured trace with monotonic timestamps and frame/action IDs:
capture start/end, source frame time when available, detection, tracking,
state assembly, policy start/end, chosen action, taps, and game acknowledgement.
Distinguish capture completion time from actual frame age.

Record why no play occurred: deliberate hold, insufficient elixir, uncertain
hand, invalid placement, pending deployment, missing screen state, inference
error, or stale input. Track the hold probability and best playable alternative
for RL, plus value estimates and legal action counts.

Log p50/p95/p99 frame age, stage latency, and action acknowledgement latency;
elixir leakage while capped; tower damage during unanswered threats; failed,
duplicated, or wrong-slot plays; perception gaps; cycle and spending events.

Add a replay viewer overlay with raw detections, tracked identity, confidence,
predicted location, purchase/death events, selected placement, and reasons.
Retain short frame buffers around failures, rather than unlimited video for
every experiment. Configure storage limits and trace sampling explicitly.

Exit: each long silence can be attributed to a measured stage or policy hold.

## Phase 2 — Persistent troop tracking and deployment events

1. Track both sides, retaining class probabilities, side probabilities,
   location, velocity, age, last observation, and identity.
2. Introduce tentative, confirmed, occluded, and retired states. Never declare
   death from one missing detection.
3. Compare the custom tracker with ByteTrack/BoT-SORT on the same recordings.
   They are baselines, not guaranteed solutions for overlapping identical
   sprites. Persistent identity and retained lost tracks are supported by
   Ultralytics: https://docs.ultralytics.com/modes/track
4. Use time-aware motion gating and assignment, not only detection iteration
   order. Account for variable frame spacing and stationary buildings.
5. Associate lower-confidence detections with existing plausible tracks while
   requiring stronger evidence to create new tracks.
6. Stabilize troop class and side over time. An ambiguous side prediction
   should not instantly convert an attacking enemy into an ally.
7. Expose short-lived predicted tracks with confidence/age, separately from
   freshly observed tracks. Tune ghost lifetime using measured death and
   occlusion errors rather than retaining everything for several seconds.
8. Infer card deployment events separately from unit instances. Group swarms;
   distinguish second purchases from returning tracks; recognize spawned
   children, transformations, revivals, and spawner buildings.
9. Add spell deployment/event recognition. Spells need duration/area events,
   not permanent troop tracks.
10. Add explicit tests for overlapping copies, rapid cycles, swarm spawning,
    crossing the bridge, death spawns, spells, and class/side flicker.

Metrics: ID switches, fragmentation, side-flip rate, deployment precision and
recall, double-charge rate, delayed first detection, and ghost lifetime.
Split evaluation by whole match/session to avoid adjacent-frame leakage.

Exit: annotated clips replay deterministically through the tracker, and the
candidate improves event errors without increasing missed urgent threats.

## Phase 3 — A game-state estimate that knows its uncertainty

### Opponent elixir

Maintain a small distribution or weighted hypotheses over possible elixir,
instead of treating a single floating-point estimate as observed truth.
Regenerate hypotheses using the verified game-mode clock and subtract costs
for plausible deployment events. Handle special elixir sources/sinks where
supported. A purchase implies sufficient pre-purchase resources, but delayed
detection and generated units must be considered before using that constraint.

For uncertain events, preserve alternate hypotheses. Report mean, plausible
range, age, and confidence. Keep hidden-spend uncertainty explicit, especially
for missed spells. A missing detection is never a refund or second purchase.
Test against simulator truth behind a noisy observation layer and synchronized
two-perspective friendly recordings when available.

### Opponent hand and cycle

Represent probabilities that relevant answers are available, plus observed
deck, confirmed plays, possible unseen plays, and applicable special cycle
rules. Exact cycle claims require complete supporting observations. Unknown
cards and long visibility gaps must reduce confidence.

### Health and tactical state

Estimate troop HP where visible; otherwise use intervals/confidence and
limited damage-history estimates. Do not encode unknown as healthy. Track
building age and plausible expiry, active spell areas, deploy delays, estimated
target, time to tower contact, and friendly defensive coverage. Handle tower
HP loss separately from temporary bar occlusion and maintain crown evidence.

Exit: uncertainty is calibrated on labeled data. For example, a declared 90%
elixir interval should cover truth near 90% of the time on the tested slice;
report interval width so a useless always-0-to-10 estimate cannot pass alone.

## Phase 4 — Faster reactions and verified execution

Use latest-frame capture with bounded buffers. Drop stale frames instead of
building an inference queue. Keep one authoritative state writer and one
serialized action executor. Run recording/rendering outside the critical path.

Benchmark screenshot and stream capture on the actual emulator, including
frame freshness, decoder buffering and recovery. Benchmark CPU/GPU inference
under concurrent emulator/training load. Validate supported exported/precision
variants on recorded frames before adopting a speed optimization.

Schedule expensive readers at suitable rates: hand after a likely rotation,
tower HP at a validated cadence, screen navigation differently from combat.
Do not skip the detector for long intervals just to report higher loop FPS.

Use an action lifecycle: proposed -> validated -> sent -> acknowledged,
rejected or timed out. Confirm with hand rotation, elixir change, or expected
deployment evidence. Reconcile uncertain execution instead of repeatedly
tapping the same card. Reset pending actions at match boundaries.

Revalidate action freshness and obvious legality before sending, without
returning to multiple complete detections per tap. Bound retries and prevent
an old target from being applied to a newly rotated hand slot.

Provisional target, subject to hardware baseline: p95 fresh-frame-to-command
submission under 250 ms during combat. Measure game acknowledgement separately;
this target is not a promise about game deployment animation/network delay.

Exit: lower p95 latency at unchanged or better successful-deployment rate.

## Phase 5 — One observation contract for simulator and live play

Create a versioned schema for entities, public game state, estimated hidden
state, masks, temporal information, and confidence. Share encoding code.
Checkpoints declare the schema version, supported deck/action space, and
normalization; incompatible checkpoints fail clearly.

Add explicit unit identity or embeddings, combat attributes, velocity,
observation age, health confidence, active spells, building age, previous
action/acknowledgement, elapsed time since decision, opponent elixir beliefs,
and answer availability. Stage these feature groups as separate ablations.

Route simulator truth through a live-like sensor/estimator before actor
encoding. Perfect hidden information can be an oracle diagnostic or optional
training-only critic input, never an accidental live actor dependency.

Compare full-health approximation, health estimates, and unknown-health
encoding; do not assume a noisy HP reader is automatically better.

Exit: equivalent public-state fixtures encode identically in both paths, and
each new feature is obtainable during live play or explicitly marked unknown.

## Phase 6 — Audit training and reward before scaling it

Audit terminal versus truncation bootstrapping, GAE, reset boundaries, masks,
rollout action/log-probability alignment, reward units, finite gradients,
minibatch normalization, optimizer restoration, and value warmup. Existing
KL stopping, entropy scheduling, anchoring and warmup should be tested, not
duplicated under new names.

Log policy/value loss, explained variance, advantage distribution, gradient
norm, clipping fraction, KL, legal-action-conditioned entropy, action-family
frequencies, hold behavior, and evaluation outcomes. Investigate non-finite or
extreme updates before allowing long training to continue.

For rules-mode learning, do not count lost tracks as confirmed kills. Associate
credit with plausible engagements; prevent duplicate reward across overlapping
episodes and multiple swarm bodies. Record observation confidence and exclude
or down-weight unknowable outcomes. Version learned tables by reward schema.

For PPO, decompose terminal, crown, tower and other shaping terms. Test whether
reward favors damage without tower conversion, useless cycling, or waiting.
High total-return/win correlation alone is insufficient evidence. Compare
terminal-only, current shaping, and carefully bounded alternatives; use
potential-based shaping where its assumptions fit. Do not add a global
penalty for waiting: saving elixir is often the correct action.

Audit time semantics before changing decision frequency. Discounting, reward
accumulation and rollout horizons must refer consistently to physical time.

Exit: deterministic training-contract tests pass, learning health is finite,
and at least several independent seeds distinguish promising changes from
one lucky run. Training return never selects the live release by itself.

## Phase 7 — Tactical curriculum and imitation warm start

Build resettable situations with explicit success metrics: building pulls,
air defense, ranged support, swarm responses, spell placement, tank/support
separation, counterpushes, answer-out-of-cycle attacks, lane commitment,
spell finishers, and conserving a lead. Include situations where waiting is
correct and situations where delaying loses a tower.

Randomize timing, positions, hand, remaining HP, elixir and opposing options.
Retain held-out variants to detect memorization. Mix tactical starts with
full matches to avoid losing long-horizon behavior.

Warm-start from demonstrably useful rule decisions and annotated examples.
Record alternatives and outcomes; teacher actions are not automatically
optimal. Use replay/VOD imitation only where hand, timing, side and action
labels are reliable. Quarantine ambiguous labels.

Exit: tactical improvements transfer to held-out full matches, with no new
regressions in previously solved defenses.

## Phase 8 — Give the policy memory and appropriate choices

First compare the existing feed-forward model with recent-frame stacking.
Then test a modest GRU/LSTM policy with previous action and elapsed-time input.
Memory must reset at match boundaries and be masked correctly during batched
training. Use sequence rollouts; do not shuffle independent frames as if
recurrent hidden state were irrelevant.

If feature ablations justify it, add an entity encoder with pooled/attention
features, keeping the spatial board for placement. Benchmark inference before
growing the architecture.

Compare the flat action head with a conditional choice: wait/play/ability,
then card, then legal tile. Preserve joint action probabilities and masks in
PPO. Add a coarse-to-fine placement experiment only after parity tests.
Support champion/ability actions end to end, or mask unsupported actions in
both the supported training configuration and live policy.

Explore event-triggered decisions and explicit wait durations only after
matching training and live time semantics. Retrain affected policies.

Exit: richer memory/actions win controlled ablations at acceptable latency.
Larger models alone are not an acceptance criterion.

## Phase 9 — Diverse opponents and protection against forgetting

Extend the existing opponent/self-play infrastructure into a reproducible
mixture of rules, scripted tactics, historical snapshots, current peers,
different archetypes and targeted exploiters. Preserve opponent hashes.

Avoid only training against the latest copy, and avoid selecting by self-play
win rate alone. Use held-out opponents and cross-play matrices to detect
non-transitive strengths and forgetting. Track results by archetype and seat.
Tune difficulty gradually while retaining easy anchors and old failure cases.

Exit: gains extend beyond the training opponent, without catastrophic losses
against earlier benchmark styles.

## Phase 10 — Close simulator-to-live gaps using evidence

Prioritize interactions encountered by the supported deck: movement, first
hit timing, targeting, building pulls, collision, deployment delay, projectile
impact, damage, tower behavior, knockback and card rotation. Extend existing
calibration/replay tooling rather than rewriting the simulator wholesale.

Reproduce observed commands and compare trajectories, contact times, damage,
deaths and target switches. Correct consistent bias before randomizing it.
Use empirical ranges for genuinely uncertain parameters; excessive arbitrary
randomization can make useful tactics unlearnable.

Add correlated sensor corruption: burst dropout, class/side mistakes,
position noise, unknown health, delayed frames, hand ambiguity, delayed/failed
actions, and missing spells. Derive distributions from live traces. Ensure
robustness tests actually alter observations/execution, as historical reports
mention ineffective perturbation paths.

Maintain clean-sim and corrupted-sim scores. A policy that wins only with exact
HP and zero latency has not met the transfer objective.

Exit: improved performance on recorded failure fixtures and calibrated noisy
simulation, followed by bounded live validation.

## Phase 11 — Controlled evaluation and release

Use four levels: fast regression scenarios; broad simulator opponent suite;
recorded perception/shadow replay; and bounded live/friendly validation.
Shadow disagreement is diagnostic, not counterfactual proof of a better
action because replay does not change in response to the proposed action.

Retain a champion checkpoint. Promote a candidate only after paired held-out
evaluation, uncertainty reporting, acceptable worst-slice results, execution
reliability and latency. Choose sample sizes from baseline variance and the
minimum gain worth detecting. A 20-game screening run is not a precise release
win-rate estimate. Use fresh final holdout data after repeated selection.

Write checkpoint manifests and atomic promotion pointers. Never let two
trainers write the same checkpoint/log directory. Keep a frozen fallback and
an explicit rollback reason. Trigger rollback on repeated integrity failures
or meaningful benchmark regressions, not a few random live losses.

Exit: any released model can be reproduced, compared and reverted.

## Phase 12 — Optional advanced research

Only after earlier gates pass:

- Short-horizon simulation of a small candidate action set, across several
  plausible hidden states and opponent responses, under a fixed latency budget.
- Learned value ranking for tactical alternatives and beam search rather than
  exhaustive branching over every card/tile sequence.
- Distillation of slower planners/ensembles into a fast live model.
- Auxiliary prediction of next events, threats and unit survival, without
  leaking hidden simulator state to the deployed actor.
- Active-learning queues for detector failure clips and data-coverage reports.
- Broader deck support with card embeddings, card-specific masks and separate
  evaluation slices, after the initial deck works reliably.
- Throughput engineering only after learning efficiency is measured: vector
  simulation, batched opponents and hardware contention control.

These are experiments with explicit stopping conditions. Abandon a planner
that exploits simulator inaccuracies or loses its gains under plausible
hidden states. Do not put an LLM request in the combat decision path.

## First implementation sequence

1. Freeze and identify the actual live/training baseline.
2. Add complete timing, hold-reason and acknowledgement traces.
3. Build annotated dropout/deployment/reordering fixtures from recordings.
4. Add temporal tracking and deployment-event tests on both sides.
5. Add uncertainty-aware elixir/cycle estimation and verified action state.
6. Create the shared versioned observation encoder and live-like simulator
   sensor layer.
7. Audit reward/training contracts and reproduce checkpoint hold failures.
8. Run controlled feature, timing and memory ablations with frozen opponents.
9. Improve the largest measured tactical and simulator-transfer failures.
10. Promote only through held-out and bounded live evaluation.

Initial planning windows, not commitments: approximately 1–2 weeks for
baseline/telemetry/execution work, 2–4 for tracking and state estimation,
2–4 for observation/training experiments, and 2–4 for transfer and evaluation.
Data labeling, compute and simulator mismatches may dominate these estimates.

## Evidence and external references

- Local audit: `docs/LIVE_DIAGNOSIS_2026-09-05.md`.
- Observation and time contracts: `sim/env.py`,
  `scripts/brain/rl_policy.py`.
- Historical evidence: `docs/RL_SPRINT4_STATE.md` and
  `reports/rl_sprint4/`; dates and scope must accompany quoted results.
- Ultralytics persistent tracking and tracker settings:
  https://docs.ultralytics.com/modes/track
- Separate evaluation and RL experimental guidance:
  https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
- Recurrent sequence learning considerations, not a proposal to replace PPO
  with this algorithm: https://openreview.net/pdf?id=r1lyTjAqYX
