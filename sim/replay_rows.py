"""Training rows from re-driven replays: what a player saw, what they did,
and what our engine says it achieved.

Each recorded match is re-driven through `sim.redrive`. At every tick on which
a player made a recorded play, and once a second while they waited, a row is
taken from that player's point of view - always oriented as the bottom player,
the same frame the live bot sees:

* units: up to `MAX_UNITS` living non-tower entities, nearest to the player's
  own king first: unit id, own/enemy, x, y, hit point fraction, flying,
  building, still deploying.
* towers: hit point fraction of own left / right / king, then the enemy's.
* scalars: own elixir, clock, double and overtime flags.
* own hand (4 card ids) and next card, as the engine holds them.
* public opponent memory: the opponent's distinct cards seen so far and its
  last four plays with how long ago. Never its hand, next card or elixir.
* action: wait, or (card id, tile x, tile y) on the 18 x 32 grid.
* outcome: over the next `HORIZON_TICKS` (10 s), enemy tower hit points lost
  minus own, as fractions of a full tower. This is our engine's verdict on the
  play, the input to outcome-weighted training (`sim.train_tokens`).
* trust: whether the row precedes this match's first serious divergence
  (`RedriveResult.first_forced_tick` / `first_divergence_tick`). Rows after it
  describe a board the real game never had.

The vocabularies are the engine's own sorted card and unit names, so ids are
stable as long as the game data is.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np

from .replay_corpus import OPPONENT, TEAM, ReplayRecord, side_of

MAX_UNITS = 48
UNIT_FEATURES = 7            # own, x, y, hp, flying, building, deploying
HORIZON_TICKS = 200          # 10 s
WAIT_EVERY_TICKS = 20        # one wait sample a second
GRID_W, GRID_H = 18, 32
PAD, UNK = 0, 1


@lru_cache(maxsize=1)
def card_vocab() -> Dict[str, int]:
    from .gamedata import load_gamedata
    names = sorted(load_gamedata(level=11))
    return {n: i + 2 for i, n in enumerate(names)}


@lru_cache(maxsize=1)
def unit_vocab() -> Dict[str, int]:
    from .gamedata import load_characters, load_gamedata, to_snake_case
    names = {to_snake_case(k) for k in load_characters(11)}
    names |= {c.unit.name for c in load_gamedata(11).values() if c.unit}
    return {n: i + 2 for i, n in enumerate(sorted(names))}


def _card_id(name: Optional[str]) -> int:
    return card_vocab().get(name, UNK) if name else PAD


def _orient(x: int, y: int, side: int) -> tuple:
    """World millitiles -> the player's own frame, where it is the bottom side."""
    if side < 0:
        return 18_000 - x, 32_000 - y
    return x, y


def _tile(x: int, y: int, side: int) -> tuple:
    ox, oy = _orient(x, y, side)
    return (min(GRID_W - 1, max(0, ox // 1000)),
            min(GRID_H - 1, max(0, oy // 1000)))


def _towers(match, side: int) -> np.ndarray:
    """Own L, R, king then enemy L, R, king, as hit point fractions.

    Left and right are the player's own: rotating the board for the top player
    swaps which world tower is on its left.
    """
    out = []
    for who in (side, -side):
        towers = match.towers[who]
        lanes = ("left", "right") if side > 0 else ("right", "left")
        for lane in (*lanes, "king"):
            t = towers[lane]
            out.append(t.hitpoints / max(1, t.max_hitpoints) if t.alive else 0.0)
    return np.asarray(out, dtype=np.float32)


@dataclass
class _Mem:
    seen: Dict[int, set]
    last: Dict[int, list]


def cycle_hand(deck: List[str], played: List[str]):
    """The real hand and next card from the play history, or None.

    Clash Royale's queue is first in, first out: a played card goes to the back
    and the front comes into the hand. So once a player has made four plays,
    the queue is exactly their last four plays, oldest first - the next card
    is the fourth most recent - and the hand is the other four cards. That is
    the true hand, where the engine's own is a guess forced to agree with each
    play. Mirror replays a card without cycling it the same way, so a history
    containing Mirror returns None.
    """
    if len(played) < 4 or "mirror" in played[-8:]:
        return None
    queue = played[-4:]
    if len(set(queue)) < 4:
        return None
    hand = [c for c in deck if c not in queue]
    if len(hand) != 4:
        return None
    return hand, queue[0]


class RowCollector:
    """`redrive` observer that samples rows for both players."""

    def __init__(self, rec: ReplayRecord):
        self.rec = rec
        self.mem = _Mem(seen={1: set(), -1: set()}, last={1: [], -1: []})
        self.rows: List[dict] = []
        self.tower_track: List[tuple] = []        # (tick, own_total(+1), own_total(-1))
        self._next_play: Dict[int, List[int]] = {1: [], -1: []}
        self.decks = {side_of(n): [c.base_engine for c in rec.decks[n]] for n in rec.decks}
        self.history: Dict[int, List[str]] = {1: [], -1: []}
        for p in rec.plays:
            if p.kind == "play_card" and p.engine:
                self._next_play[side_of(p.side)].append(p.tick)

    def _units(self, match, side: int):
        ids = np.zeros(MAX_UNITS, dtype=np.int16)
        feats = np.zeros((MAX_UNITS, UNIT_FEATURES), dtype=np.float16)
        king = match.towers[side]["king"].pos
        live = [e for e in match.battle.entities.values()
                if getattr(e, "alive", True) and not e.is_tower]
        live.sort(key=lambda e: (e.pos.x - king.x) ** 2 + (e.pos.y - king.y) ** 2)
        vocab = unit_vocab()
        for k, e in enumerate(live[:MAX_UNITS]):
            ox, oy = _orient(e.pos.x, e.pos.y, side)
            ids[k] = vocab.get(e.name, UNK)
            feats[k] = (1.0 if e.side == side else 0.0, ox / 18_000, oy / 32_000,
                        e.hitpoints / max(1, e.max_hitpoints),
                        1.0 if e.flying else 0.0, 1.0 if e.is_building else 0.0,
                        1.0 if getattr(e, "deploy_remaining_ms", 0) > 0 else 0.0)
        return ids, feats

    def _row(self, match, tick: int, side: int, action) -> dict:
        player = match.players[side]
        ids, feats = self._units(match, side)
        t_ms = match.elapsed_ms
        real = cycle_hand(self.decks[side], self.history[side])
        cards = real[0] if real else player.hand[:4]
        nxt = real[1] if real else player.next_card
        hand = [_card_id(c) for c in cards] + [PAD] * (4 - len(cards))
        opp = -side
        seen = sorted(self.mem.seen[opp])[:8]
        last = self.mem.last[opp][-4:]
        return {
            "tick": tick, "side": side,
            "unit_ids": ids, "unit_feats": feats,
            "towers": _towers(match, side),
            "scalars": np.asarray([player.elixir / 10_000, min(1.0, t_ms / 300_000),
                                   1.0 if t_ms >= 120_000 else 0.0,
                                   1.0 if t_ms >= 180_000 else 0.0], dtype=np.float32),
            "hand": np.asarray(hand, dtype=np.int16),
            "next": _card_id(nxt),
            "hand_exact": real is not None,
            "opp_seen": np.asarray(seen + [PAD] * (8 - len(seen)), dtype=np.int16),
            "opp_last": np.asarray([c for c, _ in last] + [PAD] * (4 - len(last)), dtype=np.int16),
            "opp_last_age": np.asarray([min(1.0, (tick - t) / 400) for _, t in last]
                                       + [1.0] * (4 - len(last)), dtype=np.float16),
            "action": action,
        }

    def __call__(self, match, tick: int, due) -> None:
        own = {1: sum(_towers(match, 1)[:3]), -1: sum(_towers(match, -1)[:3])}
        self.tower_track.append((tick, own[1], own[-1]))
        playing = {1: [], -1: []}
        for p in due:
            if p.kind == "play_card" and p.engine:
                playing[side_of(p.side)].append(p)
        for side in (1, -1):
            for p in playing[side]:
                tx, ty = _tile(p.x, p.y, side)
                self.rows.append(self._row(match, tick, side,
                                           (_card_id(p.engine), tx, ty)))
            if not playing[side] and tick % WAIT_EVERY_TICKS == 0:
                # A wait is only a wait if no play follows within half a second;
                # otherwise the player had already decided and was mid-tap.
                soon = any(tick < t <= tick + 10 for t in self._next_play[side])
                if not soon and match.elapsed_ms > 0:
                    self.rows.append(self._row(match, tick, side, (PAD, 0, 0)))
        for side in (1, -1):
            for p in playing[side]:
                cid = _card_id(p.engine)
                self.mem.seen[side].add(cid)
                self.mem.last[side].append((cid, tick))
                self.history[side].append(p.engine)

    def finish(self, result) -> List[dict]:
        """Attach outcome over the horizon, trust and the recorded result."""
        track = self.tower_track
        ticks = np.asarray([t for t, _, _ in track])
        own = {1: np.asarray([a for _, a, _ in track]), -1: np.asarray([b for _, _, b in track])}
        cut = min(x for x in (result.first_forced_tick, result.first_divergence_tick,
                              10 ** 9) if x is not None)
        winner = result.recorded_winner
        for row in self.rows:
            s = row["side"]
            i = int(np.searchsorted(ticks, row["tick"]))
            j = int(np.searchsorted(ticks, row["tick"] + HORIZON_TICKS))
            i, j = min(i, len(ticks) - 1), min(j, len(ticks) - 1)
            lost_own = own[s][i] - own[s][j]
            lost_enemy = own[-s][i] - own[-s][j]
            row["outcome"] = float(lost_enemy - lost_own)
            row["trusted"] = row["tick"] + HORIZON_TICKS <= cut
            name = TEAM if s == 1 else OPPONENT
            row["won"] = 1.0 if winner == name else (0.5 if winner == "draw" else 0.0)
        return self.rows


def rows_for(rec: ReplayRecord):
    """(rows, RedriveResult) for one recorded match."""
    from .redrive import redrive
    collector = RowCollector(rec)
    result = redrive(rec, observer=collector)
    if not result.ok:
        return [], result
    return collector.finish(result), result


FIELDS = ("tick", "side", "hand_exact", "unit_ids", "unit_feats", "towers", "scalars", "hand",
          "next", "opp_seen", "opp_last", "opp_last_age", "outcome", "trusted", "won")


def stack(rows: List[dict], tags: List[str]) -> Dict[str, np.ndarray]:
    """Rows -> column arrays for one shard. `tags` is the match tag per row."""
    out = {k: np.stack([np.asarray(r[k]) for r in rows]) for k in FIELDS}
    act = np.asarray([r["action"] for r in rows], dtype=np.int16)
    out["act_card"], out["act_x"], out["act_y"] = act[:, 0], act[:, 1], act[:, 2]
    out["match"] = np.asarray(tags)
    return out
