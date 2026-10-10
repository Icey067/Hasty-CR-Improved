"""The IL_Replay corpus, read into our engine's names and our board's frame.

IL_Replay (a public Hugging Face dataset, `tools/il_replay_download.py`) holds
252,238 anonymised ladder matches. Each has both decks with card levels, every
card play with its 20 Hz tick, side and native board coordinates, the result,
the crowns and the final tower hitpoints. This module turns a match into a
`ReplayRecord` that `sim.redrive` can play through the engine.

Three translations happen here, and each is checked rather than assumed:

* **Card names.** RoyaleAPI slugs (`the-log`, `mini-pekka`) become engine names
  (`log`, `minipekka`) through `sim.card_catalog_audit`, which maps each public
  card onto the client's own `sc_key`. Evolutions (`-ev1`) and hero forms
  (`-hero`) become the engine's `_ev1` / `_hero` rows when the engine has them.
  A card with no engine row maps to None and is counted, never guessed.
* **Board frame.** The dataset's native coordinates are 1000 units per tile on
  the 18 x 32 board, the same units as `sim.arena`. Which end of the board each
  side owns differs between matches, so it is inferred per match from where
  each side puts its troops: a player deploys troops on its own half far more
  often than anywhere else. The match is then rotated so the recorded "team"
  is our side 1, the bottom player (own half y >= 16 tiles). A rotation, not a
  mirror, keeps the board's handedness.
* **Who is who.** "team" is the player whose replay was collected; "opponent"
  is the other. Both are learnable; `side_of` gives the engine side.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, List, Optional

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "il_replay"

BOARD_W, BOARD_H = 18_000, 32_000
HALF_Y = 16_000

TEAM, OPPONENT = "team", "opponent"
SIDE = {TEAM: 1, OPPONENT: -1}


@dataclass
class DeckCard:
    key: str                      # RoyaleAPI slug as recorded, e.g. "cannon-ev1"
    level: int
    engine: Optional[str]         # engine card name, None if the engine lacks it
    base_engine: Optional[str]    # the base card's engine name (evo/hero stripped)
    form: str                     # "base", "evo" or "hero"


@dataclass
class Play:
    tick: int
    side: str                     # TEAM or OPPONENT
    key: str                      # RoyaleAPI slug of the card played
    engine: Optional[str]         # engine name of the BASE card played
    x: int                        # our frame, millitiles
    y: int
    kind: str = "play_card"       # or "activate_ability"


@dataclass
class ReplayRecord:
    tag: str
    game_mode: str
    battle_type: str
    result: str                   # from the team's point of view
    crowns: Dict[str, int]
    decks: Dict[str, List[DeckCard]]
    final_towers: Dict[str, Dict[str, int]]
    plays: List[Play] = field(default_factory=list)
    rotated: bool = False         # True if the native frame was turned 180 deg
    tower_cards: Dict[str, str] = field(default_factory=dict)   # e.g. "tower-princess"

    def loadable(self) -> bool:
        """Every card in both decks has an engine row."""
        return all(c.base_engine for deck in self.decks.values() for c in deck)

    def missing(self) -> List[str]:
        return sorted({c.key for deck in self.decks.values() for c in deck
                       if not c.base_engine})

    def base_deck(self, side: str) -> frozenset:
        return frozenset(c.base_engine or c.key for c in self.decks[side])


# ----------------------------------------------------------------- card names

@lru_cache(maxsize=1)
def _engine_table() -> dict:
    from .gamedata import load_gamedata
    return load_gamedata(level=11)


def _engine_cards() -> frozenset:
    return frozenset(_engine_table())


@lru_cache(maxsize=1)
def _public_map() -> Dict[str, str]:
    from .card_catalog_audit import report
    return dict(report()["mapped"])


def split_form(key: str) -> tuple:
    """`cannon-ev1` -> ("cannon", "evo"); `wizard-hero` -> ("wizard", "hero")."""
    if key.endswith("-hero"):
        return key[:-5], "hero"
    if "-ev" in key and key.rsplit("-ev", 1)[1].isdigit():
        return key.rsplit("-ev", 1)[0], "evo"
    return key, "base"


def _canonical(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


@lru_cache(maxsize=1)
def _engine_by_canonical() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in _engine_cards():
        out.setdefault(_canonical(name), name)
    return out


def _base_engine(base_key: str) -> Optional[str]:
    """Base slug -> engine name.

    The public snapshot's `sc_key` bridge is authoritative where it has the
    card; it is older than the corpus, so cards released after it (Vines,
    Goblinstein, Boss Bandit, ...) fall back to matching the slug against
    engine names with separators and case removed. Only an exact match is
    taken, so a card the engine truly lacks (Void, Rune Giant) stays None.
    """
    mapped = _public_map().get(base_key)
    if mapped:
        return mapped
    return _engine_by_canonical().get(_canonical(base_key))


@lru_cache(maxsize=None)
def engine_name(key: str) -> tuple:
    """RoyaleAPI slug -> (engine name of this form or None, engine base or None)."""
    base_key, form = split_form(key)
    base = _base_engine(base_key)
    if base is None:
        return None, None
    if form == "base":
        return base, base
    suffix = "_ev1" if form == "evo" else "_hero"
    for variant in (base + suffix, base_key.replace("-", "_") + suffix):
        if variant in _engine_cards():
            return variant, base
    return None, base


# ------------------------------------------------------------------ the frame

def _is_troop_like(engine: Optional[str]) -> bool:
    """Cards whose placement says which half a player owns: not spells."""
    spec = _engine_table().get(engine) if engine else None
    return spec is not None and spec.unit is not None


def orient(plays: List[dict]) -> bool:
    """True if the team owns the native LOW-y half, so the match must rotate.

    Uses the median y of each side's non-spell plays. The two medians must sit
    on opposite halves; if they do not (a match with almost no troop plays),
    the team's median alone decides.
    """
    ys = {TEAM: [], OPPONENT: []}
    for p in plays:
        if p["kind"] == "play_card" and _is_troop_like(p["engine"]):
            ys[p["side"]].append(p["ny"])
    team = statistics.median(ys[TEAM]) if ys[TEAM] else None
    opp = statistics.median(ys[OPPONENT]) if ys[OPPONENT] else None
    if team is not None:
        return team < HALF_Y
    if opp is not None:
        return opp >= HALF_Y
    return False


def to_frame(nx: int, ny: int, rotated: bool) -> tuple:
    if rotated:
        nx, ny = BOARD_W - nx, BOARD_H - ny
    return (min(max(nx, 0), BOARD_W - 1), min(max(ny, 0), BOARD_H - 1))


def side_of(side: str) -> int:
    return SIDE[side]


# ------------------------------------------------------------------- reading

def _deck(players: list) -> List[DeckCard]:
    out = []
    for player in players[:1]:                  # 1v1: one player per side
        for card in player.get("deck", []):
            key = card.get("card_key", "")
            engine, base = engine_name(key)
            out.append(DeckCard(key=key, level=int(card.get("level") or 0),
                                engine=engine, base_engine=base,
                                form=split_form(key)[1]))
    return out


def _towers(players: list) -> Dict[str, int]:
    hp = (players[0].get("final_tower_hitpoints") or {}) if players else {}
    return {k: int(hp[k]) for k in ("king", "princess_left", "princess_right")
            if hp.get(k) is not None}


def parts(root: Path = CORPUS) -> List[str]:
    """Part names present in BOTH folders, e.g. 'part-000000.parquet'."""
    have = lambda d: {p.name for p in (root / d).glob("part-*.parquet")}
    return sorted(have("replays") & have("actions"))


def read_part(name: str, root: Path = CORPUS) -> Iterator[ReplayRecord]:
    """Every match in one part, with its plays in tick order."""
    import pyarrow.parquet as pq

    acts = pq.read_table(
        root / "actions" / name,
        columns=["replay_tag", "event_index", "kind", "side", "card_key",
                 "replay_tick_20hz", "native_x", "native_y",
                 "ability_source_candidates"]).to_pylist()
    by_tag: Dict[str, list] = {}
    for a in acts:
        key = a["card_key"] or ((a["ability_source_candidates"] or [""])[0])
        by_tag.setdefault(a["replay_tag"], []).append({
            "i": a["event_index"], "tick": int(a["replay_tick_20hz"] or 0),
            "side": a["side"], "kind": a["kind"], "key": key or "",
            "engine": engine_name(key)[1] if key else None,
            "nx": int(a["native_x"] or 0), "ny": int(a["native_y"] or 0)})

    reps = pq.read_table(root / "replays" / name,
                         columns=["replay_tag", "battle_type", "game_mode",
                                  "result", "payload_json"]).to_pylist()
    for r in reps:
        battle = json.loads(r["payload_json"]).get("battle", {})
        team = battle.get("team", {}).get("players", [])
        opp = battle.get("opponent", {}).get("players", [])
        raw = sorted(by_tag.get(r["replay_tag"], []),
                     key=lambda p: (p["tick"], p["i"]))
        rotated = orient(raw)
        plays = []
        for p in raw:
            x, y = to_frame(p["nx"], p["ny"], rotated)
            plays.append(Play(tick=p["tick"], side=p["side"], key=p["key"],
                              engine=p["engine"], x=x, y=y, kind=p["kind"]))
        yield ReplayRecord(
            tag=r["replay_tag"], game_mode=r["game_mode"] or "",
            battle_type=r["battle_type"] or "", result=r["result"] or "",
            crowns={TEAM: int(battle.get("team", {}).get("crowns") or 0),
                    OPPONENT: int(battle.get("opponent", {}).get("crowns") or 0)},
            decks={TEAM: _deck(team), OPPONENT: _deck(opp)},
            final_towers={TEAM: _towers(team), OPPONENT: _towers(opp)},
            plays=plays, rotated=rotated,
            tower_cards={TEAM: ((team[0].get("tower_card") or {}).get("card_key", "") if team else ""),
                         OPPONENT: ((opp[0].get("tower_card") or {}).get("card_key", "") if opp else "")})


def iter_corpus(root: Path = CORPUS, limit_parts: int = 0) -> Iterator[ReplayRecord]:
    names = parts(root)
    for name in names[:limit_parts] if limit_parts else names:
        yield from read_part(name, root)
