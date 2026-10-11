"""Adaptive difficulty curriculum for training.

The core problem this solves: if scripted_share stays fixed, the agent can
plateau against easy opponents without being pushed harder. Real ladder players
are not a fixed difficulty; the agent should progressively face harder styles.

`OpponentCurriculum` tracks the rolling win-rate across recent episodes and
raises the difficulty tier when the agent wins too consistently. Each tier
shifts the scripted opponent mix toward faster-reacting, higher-pressure styles.

Tiers (easy → hard):
  0: 40% rule-engine, 60% meta decks, styles: cycle/control/beatdown
  1: 40% rule-engine, 40% meta, 20% aggressive, styles: all styles
  2: 30% rule-engine, 30% meta, 40% aggressive, reaction jitter tightened
  3: 20% rule-engine, 20% meta, 60% aggressive, min reaction = 0.3s

The curriculum promotes when win_rate > PROMOTE_THRESHOLD over the last
WINDOW episodes, and demotes (to prevent giving up on hard tiers) when
win_rate < DEMOTE_THRESHOLD.

Usage::

    curriculum = OpponentCurriculum(vec_env)
    for step in training_loop:
        curriculum.on_episode_end(result == "win")
        ...

This integrates naturally alongside the league: the curriculum controls the
scripted share/style, while the league controls the learned-opponent share.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

PROMOTE_THRESHOLD = 0.60   # promote if win-rate above this over the window
DEMOTE_THRESHOLD  = 0.30   # demote if win-rate below this (avoid giving up)
WINDOW            = 50     # rolling window of episodes to measure win-rate

# Each tier is (scripted_share, brain_share, description).
# scripted_share is the fraction of episodes against meta decks.
# brain_share (within scripted) is the fraction against the rule engine.
TIERS = [
    (0.60, 0.40, "tier-0: 40% rule-engine, 60% meta  [easy]"),
    (0.40, 0.40, "tier-1: 40% rule-engine, 40% meta, 20% aggressive  [medium]"),
    (0.30, 0.30, "tier-2: 30% rule-engine, 30% meta, 40% aggressive  [hard]"),
    (0.20, 0.20, "tier-3: 20% rule-engine, 20% meta, 60% aggressive  [very hard]"),
]


class OpponentCurriculum:
    """Adaptive difficulty wrapper around VecClashEnv.

    Tracks a rolling win-rate and promotes/demotes the difficulty tier
    automatically. On promotion the new scripted_share and brain_share are
    pushed to the running workers via the VecClashEnv's `league_add` channel
    (not currently exposed, so we log the recommendation and the caller can
    restart or act on it).

    In practice, pass this to the training loop and call `on_episode_end` after
    each completed episode. The current tier's parameters are available as
    `self.scripted_share` and `self.brain_share`.
    """

    def __init__(self, log_fn=None, start_tier: int = 0):
        self._tier = max(0, min(start_tier, len(TIERS) - 1))
        self._history: deque[bool] = deque(maxlen=WINDOW)
        self._log = log_fn or (lambda msg: print(msg, flush=True))
        self._episodes_at_tier = 0
        self._log(f"Curriculum start: {TIERS[self._tier][2]}")

    @property
    def tier(self) -> int:
        return self._tier

    @property
    def scripted_share(self) -> float:
        return TIERS[self._tier][0]

    @property
    def brain_share(self) -> float:
        return TIERS[self._tier][1]

    @property
    def win_rate(self) -> Optional[float]:
        if not self._history:
            return None
        return sum(self._history) / len(self._history)

    def on_episode_end(self, won: bool) -> bool:
        """Register the outcome of one episode. Returns True if tier changed."""
        self._history.append(won)
        self._episodes_at_tier += 1
        if len(self._history) < WINDOW // 2:
            return False   # too few samples to decide

        rate = self.win_rate
        changed = False
        if rate > PROMOTE_THRESHOLD and self._tier < len(TIERS) - 1:
            self._tier += 1
            changed = True
            self._log(
                f"Curriculum PROMOTE → {TIERS[self._tier][2]}  "
                f"(win-rate {rate:.0%} over last {len(self._history)} episodes)"
            )
        elif rate < DEMOTE_THRESHOLD and self._tier > 0:
            self._tier -= 1
            changed = True
            self._log(
                f"Curriculum DEMOTE  → {TIERS[self._tier][2]}  "
                f"(win-rate {rate:.0%} over last {len(self._history)} episodes)"
            )
        if changed:
            self._episodes_at_tier = 0
        return changed

    def status(self) -> str:
        rate = self.win_rate
        rate_str = f"{rate:.0%}" if rate is not None else "n/a"
        return (f"curriculum tier={self._tier}  {TIERS[self._tier][2]}  "
                f"win-rate={rate_str}  episodes_at_tier={self._episodes_at_tier}")
