"""Every card the detector can name must resolve to a unit the brain knows.

A class the model detects and `units.json` cannot resolve is dropped inside
`YoloVision.detect`, and until now it was dropped in silence. That is not a
degraded detection - the unit is simply absent from the board as far as the
policy is concerned, with no log line, no counter and no test.

It cost a match on 2026-08-31. A Goblin Giant was detected at 0.92 confidence
standing next to our own Cannon and discarded, because `goblin-giant` had no
entry. Seventy-eight of the model's 201 classes were in that state. The policy
saw an empty lane, held its elixir, and the push went through - which read from
outside as "the bot reacts too slowly" and would have been chased into the
policy, where there was nothing wrong.

So the coverage is asserted here rather than inferred from a win rate. A class
is allowed to resolve to nothing only by being *named* as scenery, interface,
a spell, or dataset padding. Falling through is not allowed.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "vendor" / "ClashRoyaleBuildABot")):
    if path not in sys.path:
        sys.path.insert(0, path)

# Two separate tables, and the bug lived in the gap between them. `units.json`
# carries the *stats* the brain reasons with; the vendored `NAME2UNIT` decides
# whether a detection survives `YoloVision.detect` at all. A card needs to be
# in both, and being in one is exactly what "invisible" looked like.
UNITS = json.loads((ROOT / "scripts" / "brain" / "units.json")
                   .read_text(encoding="utf-8"))


def _resolvable() -> dict:
    """What `YoloVision` resolves a class against: the vendor table plus ours."""
    from brain.vision import _extra_units, _key
    from clashroyalebuildabot.namespaces.units import NAME2UNIT
    merged = dict(NAME2UNIT)
    merged.update(_extra_units())
    return {_key(unit.name): unit for unit in merged.values()}


def _stats(card_key: str):
    """`units.json` is keyed by card name; the detector path is normalised."""
    from brain.vision import _key
    return next((v for k, v in UNITS.items() if _key(k) == card_key), None)


def _classes() -> list[str]:
    """The detector's class names, read from the weights without loading torch."""
    from brain.vision import resolve_weights
    weights = resolve_weights()
    if weights is None or not weights.exists():
        pytest.skip("detector weights not on disk")
    pytest.importorskip("ultralytics")
    from ultralytics import YOLO
    return sorted(YOLO(str(weights)).names.values())


def test_no_detector_class_falls_through_unnamed():
    from brain.vision import CLASS_ALIASES, NOT_A_UNIT, NOT_A_UNIT_EITHER, _key
    known = _resolvable()
    missing = []
    for raw in _classes():
        if raw in NOT_A_UNIT or raw in NOT_A_UNIT_EITHER:
            continue
        key = _key(raw)
        if known.get(CLASS_ALIASES.get(key, key)) is None:
            missing.append(raw)
    assert not missing, (
        "these classes are detectable and invisible to the brain; give each "
        "one a units.json entry, or name it in NOT_A_UNIT / NOT_A_UNIT_EITHER "
        f"so the omission is deliberate: {missing}")


def test_the_card_that_caused_this_resolves():
    """A named regression, so the general test above cannot be watered down."""
    from brain.vision import _key
    key = _key("goblin-giant")
    assert _resolvable().get(key) is not None, "dropped inside detect()"
    stats = _stats(key)
    assert stats is not None, "resolves, but the brain has no stats for it"
    assert stats["tank"] is True


@pytest.mark.parametrize("alias,card", [
    ("elixir-golem-big", "elixir_golem"),
    ("elixir-golem-mid", "elixir_golem"),
    ("phoenix-big", "phoenix"),
])
def test_a_units_two_stages_resolve_to_the_one_card(alias, card):
    """The detector names split and reformed stages separately."""
    from brain.vision import CLASS_ALIASES, _key
    assert CLASS_ALIASES.get(_key(alias)) == _key(card)
    assert _resolvable().get(_key(card)) is not None
    assert _stats(_key(card)) is not None


def test_every_extra_unit_also_has_stats():
    """Resolving is half of it: the brain reads traits by name afterwards.

    A unit that resolves in `detect` and has no `units.json` entry comes out
    with zero hit points and "not a building", which is how a Cannon once read
    as a troop. Both tables or neither.
    """
    from brain.vision import EXTRA_UNITS, _key
    without = [name for name, *_ in EXTRA_UNITS if _stats(_key(name)) is None]
    assert not without, f"resolve but have no stats: {without}"


def test_the_two_drop_points_agree():
    """`detect` and `apply` each resolve names, and each used to drop them.

    They read different tables - `_units` and a local `by_name` - so a card
    added to one and not the other is dropped one step later, which looks
    identical from the log.
    """
    from brain.vision import _extra_units, _key
    from clashroyalebuildabot.namespaces.units import NAME2UNIT
    detect_side = _resolvable()
    apply_side = dict(NAME2UNIT)
    apply_side.update(_extra_units())
    apply_names = {_key(unit.name) for unit in apply_side.values()}
    assert set(detect_side) == apply_names
