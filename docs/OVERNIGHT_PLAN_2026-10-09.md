# Plan, 2026-10-09: the replay corpus calibrates our engine, and our engine plans

Supersedes the first draft of this file written the same day, which rebuilt
vegetableleaf/ClashAI's pipeline in his order. The owner rejected that: the bot
should be ours. This version keeps only generic public resources - the
IL_Replay dataset (252,238 anonymised ladder matches, a third party's, not
his) and standard techniques - and builds the parts that make it different on
our own engine.

## The idea in four lines

1. **Re-drive** every replay through `sim/` and compare the engine's ending to
   the recorded one. 252k recorded endings become a fidelity benchmark for our
   engine, and the cards whose matches diverge become the fix list. This is
   the owner's "use the replays to make the sim 1:1" idea, made measurable.
2. **Train on what the replays did, weighted by what it achieved.** Not plain
   imitation: each recorded play is scored by what our engine says happened
   in the seconds after it (tower damage traded, elixir traded), and only up
   to the point where our engine still agreed with the recording.
3. **Track the opponent's cycle** from public plays: which cards they can be
   holding, when their win condition is back. Fed to the model as features.
4. **Plan at decision time.** The learned policy proposes a few plays; the
   engine rolls each forward a few seconds against cycle-consistent opponent
   replies and the bot picks the best. A reactive policy cannot do this; a
   deterministic integer engine can.

Steps 1-2 are the training twist. Steps 3-4 are what the live bot does that a
pure imitation bot does not.

## What is generic and taken, what is ours

| part | source |
|---|---|
| Replay corpus | IL_Replay on Hugging Face (public, third party) |
| Downloader, loader, card mapping | ours (`tools/il_replay_download.py`, `sim/replay_corpus.py`) |
| Engine | ours (`sim/`), calibrated by step 1 |
| Re-drive, fidelity benchmark, divergence cut | ours |
| Outcome-weighted training targets | ours (the idea is advantage-weighted regression, a standard technique) |
| Cycle tracker | ours |
| Decision-time search | ours |
| Perception | open: own vision upgrade or own memory mapping - decided separately |

RoyaleSim stays a reference we may compare against, not the engine.

## Ground rules

Same as CLAUDE.md: never start the emulator, only instance 3, measure before
concluding, guessed constants stay marked. Plus:

- Opponent hand, next card and elixir are never model inputs. Opponent elixir
  is estimated from public plays.
- Held-out splits are by match, never by row.
- Every phase ends in a number written to `reports/`.

---

## Phase A - Corpus (running)

- `tools/il_replay_download.py`: 104 parquet files, 1.9 GB, SHA-256 checked,
  resumable. Running since 2026-10-09.
- `sim/replay_corpus.py`: parquet -> `Match` records in our own names and
  frame: decks (with evolution slots and levels), plays (tick, side, engine
  card, engine point), result, crowns, final tower HP.
  - Card names: RoyaleAPI slug -> engine name through
    `sim.card_catalog_audit.report()["mapped"]` (120/120 mapped); `-ev1` ->
    `_ev1`, `-hero` -> `_hero` when the engine has that row.
  - Coordinates: dataset native units are 1000 per tile on an 18x32 board.
    Orientation is checked, not assumed: the recorded team's own-half plays
    must land on our side's half for > 95% of troop plays.
- **Gate A:** coverage report `reports/il_replay_coverage.md`: matches whose 16
  cards all load in our engine; same for Hog 2.6 sides; the top unloadable
  cards by number of matches they block.

## Phase B - Re-drive and fidelity benchmark

- `sim/redrive.py`: build the `Match` from the record, step 50 ms ticks, and at
  each recorded tick deploy the recorded card at the recorded point.
  - The engine enforces hand and elixir. A recorded play the engine refuses is
    itself a fidelity signal (our elixir or cycle drifted), so it is counted,
    then forced (card moved into hand, elixir topped up) and flagged.
- Per match: winner agreement, crowns agreement, final tower HP error per
  tower, first tick of a forced play, number of forced plays.
- Divergence cut: the first tick where the engine's tower HP departs from what
  the recorded result allows (a tower the recording keeps alive falls), or the
  first forced play. Rows after it are not trusted for training.
- **Gate B:** `reports/redrive_fidelity.md` over a 2,000-match sample: winner
  agreement, tower HP error distribution, forced-play rate, and the per-card
  ranking of divergence (cards over-represented in diverging matches, with
  counts). The ranking is the engine fix list.

## Phase C - Engine fixes from the ranking

- Take the top cards from Gate B, one at a time. For each: find the mechanic,
  fix it, re-run Gate B on the same 2,000 matches, keep the fix only if
  agreement rises and the Hog control (7 hits unblocked) still holds.
- Ongoing, not a one-off. Every fix is a commit with its before/after numbers.

## Phase D - Training rows with outcome weights

- `sim/replay_rows.py`: during the re-drive, for each side at each play and
  every 0.5 s of waiting, record:
  - observation: entity tokens (card id, side, x, y, hp fraction, air/ground,
    building), own hand and next card, own elixir, time, tower HP, plus the
    cycle tracker's opponent features. No opponent hidden state.
  - action: wait, or (card, tile).
  - outcome over the next 10 s from the engine: enemy tower HP lost minus own,
    and elixir traded. Discounted, clipped.
- Rows after the divergence cut are dropped.
- Weight = exp(advantage / beta) clipped, times a mild bonus for the match
  winner's side. Plain imitation is beta -> infinity, which is the ablation.
- **Gate D:** row counts, weight distribution, and a sanity table: the same
  play type (e.g. Log on Skeleton Army) gets positive advantage more often than
  a known bad one (Fireball on a lone Skeleton).

## Phase E - Model and training

- New entity-token model (`sim/model_tokens.py`): transformer over units +
  hand + scalars, heads for wait/play, card (pointer over hand), tile
  (conditioned on card), and value.
- Stage 1: all decks, weighted. Stage 2: fine-tune on Hog 2.6 sides.
- Ablations, same seeds: plain imitation vs weighted vs weighted + divergence
  cut. If weighting does not beat plain imitation, it is dropped and the
  report says so.
- **Gate E:** held-out pro agreement (card, tile top-1), placement metric
  (`scripts/placement_score.py`), and win rate in our engine against the rule
  engine, meta decks, and a ghost opponent that replays recorded pro commands.

## Phase F - Cycle tracker

- `scripts/brain/cycle_tracker.py`: from public plays only, the set of cards
  the opponent can hold, the next-up cards once their deck is known, and time
  since each card was last played. Unit-tested on re-driven replays where the
  truth is known.
- **Gate F:** on held-out replays, the tracked possible-hand set contains the
  true hand at every play, and the set size shrinks to 4 within the first
  cycle once all 8 cards are seen.

## Phase G - Decision-time search

- At each decision: top-k plays from the policy (plus wait), each rolled
  forward 3-5 s in a copy of the engine, opponent replies sampled from the
  policy restricted to the tracker's possible hand; score with the value head
  at the horizon. Budget measured in ms; CPU only, so it never fights training.
- **Gate G:** search vs no search, same checkpoint, paired seeds in the
  engine, against the ghost opponent. Must also fit the live decision budget.

## Phase H - Live

Perception is decided separately (own vision upgrade or own memory mapping).
The live bridge feeds the same token observation, compensates the measured
decision-to-landing delay, logs tap error against the spawned position, and
runs friendlies before any ladder match, on the owner's word only.

---

## Order and what runs unattended

A (running) -> B -> C loop. D needs B. E needs D. F is independent and can be
built any time. G needs E and F. H needs perception.

Training never runs during a live session on this PC (vegetableleaf measured
decisions going from ~40 ms to 176-347 ms under CPU load).

---

## Status after the first night (2026-10-09/10)

Details and every number: `docs/PROGRESS_2026-10-09.md`.

- Phase A, B done. Phase C started: Elixir Collector income and Lightning's
  radius fixed in our engine; start elixir 6 measured. Then RoyaleSim was
  measured against our engine on the same 2,000 matches and won on every
  axis (winner agreement 62.6% vs 49.5%), so it now generates training rows.
- Phase D, E done on both engines. Best pro agreement: `rs_hog` (RoyaleSim
  rows, Hog fine-tune) - Hog card 65.8%, tile exact 30.1%.
- Outcome weighting (our twist #2) was tried and **did not help**; dropped.
- Phase G: search is the big gain - +24 pp against a reactive opponent with
  a true engine copy - but only ~+7 pp (within noise) when the copy is rebuilt
  from what a player can see, which is the live case.
- Phase H: read-only memory reader built (Python + native C), guarded to the
  pinned build, checked on the home screen. Live-compatible rows and model
  (`rs_live`) built so live inputs match training inputs.

## Next, in order

1. **First friendly match with the owner** (no ladder): `frame.py --watch 20`
   + `validate()`, then `live_bridge.py --ckpt runs/tokens/rs_live_hog/best.pt`
   in dry run. This settles the reader's open items (side, towers, hand
   deck ids, evolution ids) - see `live_bridge.py`'s docstring.
2. **Close the rebuild gap in search.** Measured harness exists
   (`tools/rs_eval.py --search --rebuild`). Ideas, each to be measured: a
   shorter horizon, scoring by the value head alone, carrying our own spells
   in flight by re-issuing them, and deploying-unit timing.
3. **Delay compensation**: decide on the board ~1.1 s ahead (RoyaleSim's
   measured command delay, 21-22 ticks), which the rebuild makes easy -
   step the rebuilt battle forward before choosing.
4. **Tap path** (only after 1-3 and the owner's go): reuse the existing bot's
   tap code and `emulator.verify`; log each tap against the spawned unit's
   position.
5. Engine fix list in our own engine keeps going (`reports/redrive_fidelity.md`),
   but it is no longer on the critical path.
