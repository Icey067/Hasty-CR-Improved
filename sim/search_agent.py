"""Decision-time search on top of `TokenAgent`: the model proposes, the
engine checks.

At a decision the token policy has already chosen to play (the gate is
sampled exactly as `TokenAgent` does, so the play rate is unchanged), the
model's top `cards` x `tiles` placements are each tried in a copy of the
match and run `horizon_ticks` forward. Each copy is scored from our side:

    score = (enemy tower HP lost - own tower HP lost) over the horizon
            + value head's 10 s tower-outcome prediction at the horizon
            + prior_weight * log p(card, tile) from the model

and the best is played. The prior keeps the search from trading a sensible
play for one that only looks good because the copy is wrong.

Hidden information stays hidden: the opponent makes no plays inside the
copies, so its real hand is never consulted. That is a deliberately weak
opponent model; a cycle-consistent one is the next step.

Copies share the read-only card and spell tables (`_shared_memo`), which
takes a copy from ~59 ms to ~3 ms.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch

from .arena import Point
from .replay_rows import _orient, _towers
from .token_agent import GRID_W, TokenAgent, vocab_name


def _shared_memo(match) -> dict:
    memo = {}
    for obj in (match.cards, match.spells, match.battle.unit_lookup):
        memo[id(obj)] = obj
    for value in list(match.cards.values()) + list(match.spells.values()):
        memo[id(value)] = value
    return memo


class SearchAgent(TokenAgent):
    def __init__(self, checkpoint, side: int, cards: int = 2, tiles: int = 3,
                 horizon_ticks: int = 60, prior_weight: float = 0.05, **kw):
        super().__init__(checkpoint, side, **kw)
        self.k_cards, self.k_tiles = cards, tiles
        self.horizon = horizon_ticks
        self.prior_weight = prior_weight
        self.searched = 0
        self.changed = 0

    def _score(self, match) -> float:
        b, _ = self._obs(match)
        _, _, value = self.model(b, card_ids=b["hand"][:, 0].clamp(min=0))
        return float(value[0, 0])

    @torch.no_grad()
    def act(self, match) -> Optional[tuple]:
        self._watch_opponent(match)
        b, _ = self._obs(match)
        player = match.players[self.side]
        gate, _, _ = self.model(b, card_ids=b["hand"][:, 0].clamp(min=0))
        gate = gate[0].float().cpu().numpy()
        gate[0] += self.wait_bias
        for slot, card in enumerate(player.hand[:4]):
            if not match.can_play(self.side, card):
                gate[1 + slot] = -1e9
        for slot in range(len(player.hand), 4):
            gate[1 + slot] = -1e9
        p = np.exp(gate - gate.max())
        p /= p.sum()
        if self.rng.random() < float(p[0]) ** (self.decide_ms / 1000.0) or gate[1:].max() < -1e8:
            return None

        slot_logp = np.log(np.maximum(p[1:], 1e-12))
        order = [s for s in np.argsort(-gate[1:]) if gate[1 + s] > -1e8][:self.k_cards]
        candidates = []
        for slot in order:
            card = player.hand[slot]
            _, tile, _ = self.model(b, card_ids=b["hand"][:, slot])
            tile = tile[0].float().cpu().numpy()
            legal = self._legal_tiles(match, card)
            if not legal.any():
                continue
            tile[~legal] = -1e9
            tile_logp = tile - np.logaddexp.reduce(tile[legal])
            for t in np.argsort(-tile)[:self.k_tiles]:
                if tile[t] > -1e8:
                    candidates.append((card, int(t), float(slot_logp[slot] + tile_logp[t])))
        if not candidates:
            return None

        before = _towers(match, self.side)
        best, best_score = None, -1e18
        memo_base = _shared_memo(match)
        for card, t, logp in candidates:
            sim = copy.deepcopy(match, dict(memo_base))
            tx, ty = t % GRID_W, t // GRID_W
            wx, wy = _orient(tx * 1000 + 500, ty * 1000 + 500, self.side)
            if not sim.play_card(self.side, card, Point(wx, wy)):
                continue
            for _ in range(self.horizon):
                if sim.finished:
                    break
                sim.step()
            after = _towers(sim, self.side)
            swing = float((before[3:] - after[3:]).sum() - (before[:3] - after[:3]).sum())
            score = swing + self._score(sim) + self.prior_weight * logp
            if score > best_score:
                best, best_score = (card, tx, ty), score
        if best is None:
            return None
        self.searched += 1
        if best[:3] != (candidates[0][0], candidates[0][1] % GRID_W, candidates[0][1] // GRID_W):
            self.changed += 1
        card, tx, ty = best
        wx, wy = _orient(tx * 1000 + 500, ty * 1000 + 500, self.side)
        if not match.play_card(self.side, card, Point(wx, wy)):
            return None
        self.history.append(vocab_name(card))
        self.plays[card] = self.plays.get(card, 0) + 1
        return card, tx, ty, "search"
