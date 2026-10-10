"""The IL_Replay -> re-drive -> rows pipeline (sim/replay_corpus, sim/redrive,
sim/replay_rows). Pure-logic tests run anywhere; the data-backed ones skip
when data/il_replay is not downloaded."""
from __future__ import annotations

import pytest

from sim.replay_corpus import (BOARD_H, BOARD_W, CORPUS, OPPONENT, TEAM, engine_name,
                               parts, read_part, split_form, to_frame)
from sim.replay_rows import cycle_hand


def test_split_form_reads_evolution_and_hero_suffixes():
    assert split_form("cannon-ev1") == ("cannon", "evo")
    assert split_form("wizard-hero") == ("wizard", "hero")
    assert split_form("the-log") == ("the-log", "base")
    # A card whose name merely contains "-ev" is not an evolution.
    assert split_form("electro-spirit")[1] == "base"


def test_engine_names_cover_the_hog_deck_and_newer_cards():
    for slug, engine in [("hog-rider", "hog_rider"), ("the-log", "log"),
                         ("ice-golem", "ice_golemite"), ("mini-pekka", "minipekka")]:
        assert engine_name(slug)[1] == engine
    # Newer than the public snapshot, matched by name instead.
    assert engine_name("vines")[1] == "vines"
    assert engine_name("berserker-hero")[0] == "berserker_hero"
    # Truly absent from the engine: stays None rather than a guess.
    assert engine_name("void") == (None, None)


def test_rotation_is_a_half_turn_not_a_mirror():
    assert to_frame(1000, 2000, rotated=True) == (BOARD_W - 1000, BOARD_H - 2000)
    assert to_frame(1000, 2000, rotated=False) == (1000, 2000)


def test_cycle_hand_is_the_deck_minus_the_last_four_plays():
    deck = list("ABCDEFGH")
    assert cycle_hand(deck, list("ABC")) is None            # fewer than four plays
    hand, nxt = cycle_hand(deck, list("ABCD"))
    assert sorted(hand) == list("EFGH") and nxt == "A"
    hand, nxt = cycle_hand(deck, list("ABCDEF"))
    assert sorted(hand) == list("ABGH") and nxt == "C"
    assert cycle_hand(deck, list("ABCD") + ["mirror"]) is None


needs_data = pytest.mark.skipif(not parts(CORPUS), reason="IL_Replay not downloaded")


@needs_data
def test_team_troops_land_on_the_team_half_after_rotation():
    own = total = 0
    for k, rec in enumerate(read_part(parts(CORPUS)[0])):
        for p in rec.plays:
            if p.kind == "play_card" and p.engine and p.engine in ("hog_rider", "knight", "giant",
                                                                    "musketeer", "valkyrie"):
                total += 1
                own += (p.y >= 16_000) == (p.side == TEAM)
        if k > 300:
            break
    assert total > 200 and own / total > 0.9


@needs_data
def test_redrive_runs_and_the_played_card_is_in_the_reconstructed_hand():
    from sim.replay_rows import rows_for
    rec = next(r for r in read_part(parts(CORPUS)[0]) if r.loadable() and r.plays)
    rows, result = rows_for(rec)
    assert result.ok and rows
    exact = [r for r in rows if r["hand_exact"] and r["action"][0] > 0]
    assert exact and all(r["action"][0] in r["hand"] for r in exact)
    assert {r["side"] for r in rows} == {1, -1}
