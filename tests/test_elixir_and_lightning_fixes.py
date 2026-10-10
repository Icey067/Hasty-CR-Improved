"""Regression tests for two engine fixes found by re-driving IL_Replay
(docs/PROGRESS_2026-10-09.md): the Elixir Collector never paid elixir, and
Lightning loaded with radius 0 so it never hit a crown tower."""
from __future__ import annotations

from sim.arena import Point
from sim.gamedata import load_gamedata
from sim.match import Match, elixir_table
from sim.spells import load_spells


def test_elixir_table_reads_the_client_values():
    t = elixir_table()
    assert t["elixir_collector"] == {"generate_ms": 13000, "amount_milli": 1000,
                                     "death_milli": 1000, "death_opp_milli": 0}
    assert t["elixir_golem1"]["death_opp_milli"] == 1000


def _match(deck, seed=1):
    cards = load_gamedata(level=11)
    return Match(cards=dict(cards), decks=(deck, deck), seed=seed,
                 spells=dict(load_spells(level=11)))


def test_a_collector_pays_its_owner_over_its_life():
    deck = ["elixir _collector", "knight", "archer", "giant", "minions", "arrows", "fireball", "zap"]
    with_pump, without = _match(deck), _match(deck)
    for m, play in ((with_pump, True), (without, False)):
        p = m.players[1]
        p.hand = ["elixir _collector"] + [c for c in p.hand if c != "elixir _collector"][:3]
        p.elixir = 6000
        if play:
            assert m.play_card(1, "elixir _collector", Point(9000, 28000))
        else:
            p.elixir -= 6000                       # spend the same, place nothing
    gained = {}
    for name, m in (("pump", with_pump), ("none", without)):
        total = 0
        for _ in range(int(40_000 / 50)):
            before = m.players[1].elixir
            m.step(50)
            total += m.players[1].elixir - before
            m.players[1].elixir = 0                # never cap out, so all income counts
        gained[name] = total
    # 40 s of a 13 s generator: three payouts of one elixir each.
    assert gained["pump"] - gained["none"] == 3000


def test_lightning_hits_a_crown_tower_beside_the_tap():
    deck = ["lightning", "knight", "archer", "giant", "minions", "musketeer", "skeletons", "cannon"]
    m = _match(deck)
    p = m.players[1]
    p.hand = ["lightning"] + [c for c in p.hand if c != "lightning"][:3]
    p.elixir = 10000
    tower = m.towers[-1]["left"]
    before = tower.hitpoints
    assert m.play_card(1, "lightning", Point(tower.pos.x + 1500, tower.pos.y + 1500))
    for _ in range(100):
        m.step(50)
    assert before - tower.hitpoints == load_spells(level=11)["lightning"].tower_damage_override
