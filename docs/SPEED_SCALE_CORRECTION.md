# Every unit in the simulator was 20% too slow

**Date:** 2026-08-30
**Change:** `SPEED_SCALE_NUM, SPEED_SCALE_DEN` in `sim/entities.py`, from
`50, 3` (x1000/60) to `20, 1` (x20).

| class | raw `Speed` | was | now |
|---|---|---|---|
| Slow | 45 | 0.75 tiles/s | **0.90** |
| Medium | 60 | 1.00 | **1.20** |
| Fast | 90 | 1.50 | **1.80** |
| Very Fast | 120 | 2.00 | **2.40** |

This is the third time this constant has been wrong. It was `x12.5`, then
`x1000/60`; each pass has moved every unit in the game faster. This is the
first pass with evidence from outside the simulator, so the case is set out in
full below and the thing that would overturn it is stated at the end.

## Why it matters more than 20% sounds

Movement speed is the only quantity in the engine measured against the clock.
Hit speed, load time and projectile flight are all in real milliseconds, so
slowing movement by 20% does not slow the game down uniformly - it changes how
much damage a unit takes per tile walked, by 20%, for every unit and every
interaction. A Hog crossing under two princess towers, a Musketeer trading with
a Knight, whether a block lands before the target is past: all of it shifted.
It is the largest single fidelity error the project has found, and it was
present in every checkpoint trained to date.

## The evidence

**1. The data file's own units.** `Range = 800` is 0.8 tiles. `HitSpeed = 1600`
is 1600 milliseconds. The shipped CSVs are millitiles and milliseconds
throughout. Read `Speed = 120` in those units - millitiles per tick, at the
client's 20Hz - and a Hog Rider walks 2.4 tiles/sec. The "tiles per minute"
reading that gives 2.0 is a community convention; it is not derivable from the
file, and it would be the only quantity in the file denominated in minutes.

**2. An independently written engine.** `Jason-XII/clash-royale-simulator` was
built from the same shipped data by someone else, and converts x20.

`tools/diff_scenarios.py` drives both engines on the same 36 scenarios - three
cards, six placements, both seats - and reports which tower a lone unit walks
to and when it arrives.

* Lane choice agreed in **36 of 36** both before the change and after it. Lane
  commitment is not broken; the earlier apparent lane bug in this harness was
  half a tile of coordinate convention (`arena.tile()` takes a tile index and
  returns its centre, so `tile(8.5, y)` is absolute 9.0 - exactly the mirror
  line).
* Arrival time was **uniformly 20-23% later in ours**, on every card and every
  placement.
* The *detour cost* of an off-centre placement matched to the tick: a Hog
  placed at x=7.0 instead of x=3.5 arrived 0.55s later in both engines.

Same pathing, one scalar apart. After the change the two engines agree within
0.05-0.40s across all 18 comparable rows, and the residual is the difference in
the two harnesses' arrival gates rather than in the engines.

**3. Recorded ladder footage.** `tools/vod/calibrate.py` over 148 tracked
matches. The medium band, which is where the two highest-n cards sit:

| card | n | observed | predicted (old) | predicted (new) |
|---|---|---|---|---|
| Musketeer | 405 | 1.23 | 1.00 | **1.20** |
| Knight | 146 | 1.22 | 1.00 | **1.20** |
| Bomber | 26 | 1.18 | 1.00 | **1.20** |
| Firecracker (long baselines) | 12 | 1.81 | 1.50 | **1.80** |

The medium band moved from +22% error to +2%.

## Why the fast and slow rows disagree, and why that is not a counter-example

The footage measures Skeletons at 1.60 against 1.80 predicted, and a Hog Rider
at 1.41 against 2.40. Those are the two fastest classes, and the disagreement
runs monotonically with speed across all nine measured cards: Giant +15%, Golem
+14%, medium band +2%, Skeletons -11%, Hog -41%.

That is the signature of the measurement, not of the engine:

* Detector jitter adds to apparent displacement without adding to elapsed time,
  so it **inflates slow units**.
* Lost association truncates a fast unit's track, so it **deflates fast ones**.

The middle of the band is where both are small, and it is also where the
highest-n cards happen to sit. A genuine engine error looks different: it
offsets every row by the same proportion and leaves the ordering flat - which
is exactly what the table looked like *before* this change. `calibrate.py` now
prints slowest-first and says so, because the previous ordering (by size of
disagreement) hid the pattern.

## The alternative that was checked and rejected

If the video timebase were 20% fast, every measured speed would be inflated by
one factor and the medium-band agreement above would be an artefact.

Match durations rule it out. A ladder match is 3:00 regulation plus 2:00
overtime, so the tracked span of a full game has a hard ceiling near 300s plus
a few seconds of pre-game arena. Measured across 148 spans, the distribution
peaks sharply at **300-320s** and falls off a cliff after 340s. A 20% fast
clock would put that peak at 360-384s. It does not.

## What would overturn this

Time a Medium unit over a known distance in a real match. Six tiles in five
seconds means 1.20 and this is right; six tiles in six seconds means 1.00 and
it is not. `tests/test_speed_units.py` states the anchor in exactly that form,
alongside the Hog Rider's bridge-to-tower time, so the next person to question
it has something to hold a stopwatch to.

## Consequences

* **Every checkpoint trained before this date was trained on a 20%-slow game.**
  They are not invalid - they are tuned for different physics. Expect behaviour
  to shift, particularly anything timing-sensitive around blocks and tower
  trades.
* The two anchor-coefficient variations running when this landed
  (`anchor_cf005`, `anchor_cf025`) had already loaded the old module and keep
  running the old speeds. Their comparison against `anchored_v3` stays
  internally consistent; none of the three transfers cleanly to the corrected
  engine.
