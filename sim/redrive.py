"""Play a recorded match through our engine and measure where they part.

A replay is the inputs of a deterministic game plus its ending. Playing the
inputs through `sim.match.Match` and comparing endings says how close our
engine is to the real one, over as many matches as the corpus holds - which is
the point: 252k recorded endings are a fidelity benchmark nobody has to label.

The engine is not told to agree. Where it refuses a recorded play, that refusal
is the measurement:

* **hand** - the card is not in the engine's hand. The real starting order is
  not in the record, so early on this is expected; once every card has been
  seen it means the engine's cycle disagrees with the game's.
* **elixir** - the engine says the player cannot afford it. The real player
  could, so the engine's elixir is behind (start value, regen, or a cost).
* **place** - the engine refuses the tile. Either our deploy-area rule or the
  frame conversion disagrees with the game.

Each refusal is counted and then forced (card moved into hand, elixir topped
up by the shortfall, tile nudged to the nearest legal one) so the match goes
on and later disagreements are still seen. The first forced play after the
opening cycle, and the first tower the engine kills that the recording kept,
mark where this match stops being trustworthy as training data.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from statistics import mode
from typing import Dict, List, Optional

from .arena import Point, deploy_area_ok, in_arena
from .replay_corpus import OPPONENT, TEAM, ReplayRecord, side_of

TICK_MS = 50
TAIL_TICKS = 200          # keep running 10 s past the last recorded play
MAX_TICKS = 300_000 // TICK_MS


@lru_cache(maxsize=8)
def _cards(level: int) -> dict:
    from .gamedata import load_gamedata
    return load_gamedata(level=level)


@lru_cache(maxsize=8)
def _spells(level: int) -> dict:
    # A Match given no spell table resolves every spell as nothing at all.
    # The first 2,000-match run of this harness did exactly that, and its
    # per-card ranking blamed Rocket, Earthquake and Poison for it.
    from .spells import load_spells
    return load_spells(level=level)


def match_level(rec: ReplayRecord) -> int:
    """One level for the whole match: the engine builds one card table.

    Ladder decks are usually one level throughout; tournament modes cap every
    card at the same level. The most common level of the 16 cards is used and
    the spread is reported, so a match whose levels disagree can be filtered.
    """
    levels = [c.level for deck in rec.decks.values() for c in deck if c.level]
    return max(1, min(16, mode(levels))) if levels else 11


@dataclass
class RedriveResult:
    tag: str
    ok: bool = True
    error: str = ""
    level: int = 0
    level_spread: int = 0
    plays: int = 0
    forced: Counter = field(default_factory=Counter)
    elixir_short_milli: int = 0
    first_forced_tick: Optional[int] = None
    abilities: int = 0
    abilities_refused: int = 0
    engine_crowns: Dict[str, int] = field(default_factory=dict)
    recorded_crowns: Dict[str, int] = field(default_factory=dict)
    engine_winner: str = ""
    recorded_winner: str = ""
    end_tick: int = 0
    first_divergence_tick: Optional[int] = None
    missing_spawns: List[str] = field(default_factory=list)
    # (tick, side, card, shortfall milli-elixir) for every elixir refusal
    elixir_events: List[tuple] = field(default_factory=list)

    @property
    def winner_agrees(self) -> bool:
        return self.engine_winner == self.recorded_winner

    @property
    def crowns_agree(self) -> bool:
        return self.engine_crowns == self.recorded_crowns


def _recorded_winner(rec: ReplayRecord) -> str:
    r = rec.result.lower()
    if r in ("victory", "win"):
        return TEAM
    if r in ("defeat", "loss"):
        return OPPONENT
    t, o = rec.crowns[TEAM], rec.crowns[OPPONENT]
    return TEAM if t > o else OPPONENT if o > t else "draw"


def _nudge(point: Point, side: int, down: list) -> Optional[Point]:
    """Nearest legal troop tile to `point`, searching out to 2 tiles."""
    best = None
    for dy in range(-2000, 2001, 500):
        for dx in range(-2000, 2001, 500):
            p = Point(point.x + dx, point.y + dy)
            if in_arena(p) and deploy_area_ok(p, side, down):
                d = dx * dx + dy * dy
                if best is None or d < best[0]:
                    best = (d, p)
    return best[1] if best else None


def build_match(rec: ReplayRecord, seed: int = 0):
    from .match import Match
    level = match_level(rec)
    cards = dict(_cards(level))
    decks = ([c.base_engine for c in rec.decks[TEAM]],
             [c.base_engine for c in rec.decks[OPPONENT]])
    slots = {}
    for side_name in (TEAM, OPPONENT):
        evo = {c.base_engine: c.engine for c in rec.decks[side_name]
               if c.form != "base" and c.engine}
        if evo:
            slots[side_of(side_name)] = evo
    for mapping in slots.values():
        for name in mapping.values():
            if name not in cards and name in _cards(level):
                cards[name] = _cards(level)[name]
    return Match(cards=cards, decks=decks, seed=seed, level=level,
                 evolution_slots=slots, tower_level=level,
                 spells=dict(_spells(level))), level


def crowns(match, side: int) -> int:
    """Crowns as the game counts them: the king is worth all three."""
    if not match.towers[-side]["king"].alive:
        return 3
    return sum(1 for lane in ("left", "right") if not match.towers[-side][lane].alive)


def _force_hand(player, card: str) -> None:
    """Swap `card` into the hand from wherever it sits in the queue."""
    if card in player.hand:
        return
    if card in player.queue:
        player.queue.remove(card)
    if len(player.hand) >= 4:
        # The card it replaces goes to the front of the queue, so the cycle
        # the engine keeps is as close to the recorded one as it can be.
        player.queue.insert(0, player.hand.pop())
    player.hand.append(card)


def _deploy(match, rec_play, result: RedriveResult, opening_ticks: int) -> None:
    side = side_of(rec_play.side)
    player = match.players[side]
    card = rec_play.engine
    point = Point(rec_play.x, rec_play.y)
    forced = []
    if card not in player.hand:
        forced.append("hand")
        _force_hand(player, card)
    spec = match.cards.get(card)
    cost = (spec.cost if spec else 0) * 1000
    if card == "mirror":
        _, mcost = match.mirrored(side)
        cost = mcost * 1000
    if player.elixir < cost:
        forced.append("elixir")
        result.elixir_short_milli += cost - player.elixir
        result.elixir_events.append((rec_play.tick, rec_play.side, card,
                                     cost - player.elixir))
        player.elixir = cost
    if not match.play_card(side, card, point):
        down = [lane for lane, t in match.towers[-side].items()
                if lane != "king" and not t.alive]
        alt = _nudge(point, side, down)
        if alt is None or not match.play_card(side, card, alt):
            forced.append("dropped")
        else:
            forced.append("place")
    for reason in forced:
        result.forced[reason] += 1
    # Hand refusals come from the unknown starting order and the harness's
    # guess at the queue, not from the engine, so only elixir and placement
    # refusals count toward the trust cut. (`opening_ticks` is kept for the
    # signature; hand refusals are still counted in `forced`.)
    serious = [r for r in forced if r != "hand"]
    if serious and result.first_forced_tick is None:
        result.first_forced_tick = rec_play.tick


def _ability(match, rec_play, result: RedriveResult) -> None:
    from .env import ability_entities
    side = side_of(rec_play.side)
    result.abilities += 1
    for entity in ability_entities(match, side):
        if match.can_activate_ability(side, entity.uid):
            if match.activate_ability(side, entity.uid):
                return
    result.abilities_refused += 1


def redrive(rec: ReplayRecord, seed: int = 0, opening_ticks: int = 400,
            observer=None) -> RedriveResult:
    """Play `rec` through the engine. `opening_ticks` (default 20 s) is how
    long hand refusals are excused as the unknown starting order.

    `observer(match, tick, due)` is called every tick before that tick's
    recorded plays (`due`, a list of Play) are applied - the moment a player
    decided on them - which is where `sim.replay_rows` takes its samples."""
    result = RedriveResult(tag=rec.tag)
    if not rec.loadable():
        result.ok, result.error = False, "unloadable: " + ",".join(rec.missing())
        return result
    try:
        match, level = build_match(rec, seed)
    except Exception as exc:                              # pragma: no cover
        result.ok, result.error = False, f"build: {type(exc).__name__}: {exc}"
        return result
    result.level = level
    levels = [c.level for d in rec.decks.values() for c in d if c.level]
    result.level_spread = (max(levels) - min(levels)) if levels else 0
    result.recorded_crowns = dict(rec.crowns)
    result.recorded_winner = _recorded_winner(rec)

    plays = [p for p in rec.plays if p.engine or p.kind == "activate_ability"]
    last = max((p.tick for p in plays), default=0)
    horizon = min(MAX_TICKS, last + TAIL_TICKS)
    i = 0
    tick = 0
    try:
        while tick <= horizon and not match.finished:
            if observer is not None:
                j = i
                while j < len(plays) and plays[j].tick <= tick:
                    j += 1
                observer(match, tick, plays[i:j])
            while i < len(plays) and plays[i].tick <= tick:
                p = plays[i]
                if p.kind == "activate_ability":
                    _ability(match, p, result)
                else:
                    result.plays += 1
                    _deploy(match, p, result, opening_ticks)
                i += 1
            match.step(TICK_MS)
            tick += 1
            if result.first_divergence_tick is None:
                for name, side in ((TEAM, 1), (OPPONENT, -1)):
                    if crowns(match, side) > rec.crowns[name]:
                        result.first_divergence_tick = tick
                        break
    except Exception as exc:
        result.ok, result.error = False, f"step {tick}: {type(exc).__name__}: {exc}"
    result.end_tick = tick
    result.engine_crowns = {TEAM: crowns(match, 1), OPPONENT: crowns(match, -1)}
    if match.result == "bottom":
        result.engine_winner = TEAM
    elif match.result == "top":
        result.engine_winner = OPPONENT
    else:
        t, o = result.engine_crowns[TEAM], result.engine_crowns[OPPONENT]
        result.engine_winner = TEAM if t > o else OPPONENT if o > t else "draw"
    result.missing_spawns = sorted(getattr(match, "missing_spawns", ()))
    return result
