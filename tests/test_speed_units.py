"""Is the movement-speed unit still millitiles per tick?

This is the bug that has cost the project the most, and it has now been wrong
three times. `Speed = 120` for a Hog Rider was first read as x12.5, then as
tiles per minute (x1000/60), and is now read as millitiles per tick at the
client's 20Hz (x20). Each pass moved every unit in the game faster. The full
evidence for the current reading is at `SPEED_SCALE_NUM` in `sim/entities.py`;
what matters here is that the suite pins it.

What made the earlier errors survivable for so long was that no test pinned the
unit to a value a person could check in a real match. These do, in both
directions: a Medium unit covers 1.2 tiles per second, and a Hog Rider crosses
from the bridge to the tower it is attacking in a shade under three seconds.

The field is also reused. `Speed` carries walk speeds in the 40-120 band and
projectile speeds in the 300-9999 band, in the same files, so a parser that
reached into the wrong section would pick up a plausible number and make a
Knight sprint. That is asserted separately.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.arena import MT, Point                                # noqa: E402
from sim.entities import make_unit, speed_to_mt_per_sec        # noqa: E402
from sim.gamedata import load_gamedata                         # noqa: E402

CARDS = load_gamedata(level=11)
GAMEDATA = ROOT / "tmp" / "gamedata" / "csv_logic"

# The game's named movement classes: the raw `Speed` value, and the tiles per
# second it means. A tick is 50ms, so the raw number is millitiles per tick and
# twenty of them pass a second.
BANDS = {"Slow": (45, 0.90), "Medium": (60, 1.20),
         "Fast": (90, 1.80), "Very Fast": (120, 2.40)}

# Nothing that walks exceeds Very Fast. Dashes and charges are separate fields.
FASTEST_WALK = 2.40


@pytest.mark.parametrize("name,raw,tiles_per_second", sorted(
    (k, v[0], v[1]) for k, v in BANDS.items()))
def test_a_named_speed_class_walks_at_its_documented_pace(name, raw,
                                                          tiles_per_second):
    """If Medium reads 1.00 here, the tiles-per-minute error is back."""
    assert speed_to_mt_per_sec(raw) / MT == pytest.approx(tiles_per_second,
                                                          abs=1e-3), name


def test_a_medium_unit_covers_six_tiles_in_five_seconds():
    """The anchor a person can check in a real match.

    Stated as a distance and a duration rather than a rate because that is what
    someone can actually time off a recording: a Knight walking six tiles takes
    five seconds, not six.
    """
    knight = make_unit(1, CARDS["knight"].unit, 1, Point(0, 0))
    seconds = 6.0 * MT / knight.speed_mt_per_sec
    assert seconds == pytest.approx(5.0, abs=0.05)


def test_a_hog_rider_crosses_the_bridge_to_the_tower_in_under_three_seconds():
    """The other end of the band, and the one a Hog player knows by feel."""
    from sim import arena
    hog = make_unit(1, CARDS["hog_rider"].unit, 1, Point(0, 0))
    # Bridge lip to the point it stops, which is its range short of the tower.
    tiles = (arena.RIVER_TOP - arena.ENEMY_PRINCESS["right"].y
             - hog.range_mt) / MT
    seconds = abs(tiles) * MT / hog.speed_mt_per_sec
    assert 2.5 < seconds < 3.1, (tiles, seconds)


def test_every_unit_walks_at_a_plausible_pace():
    """No unit should move at a projectile's speed.

    The fastest thing that walks in this game is Very Fast. Anything above that
    means a projectile-band number reached a character's walk speed.
    """
    offenders = []
    for name, card in CARDS.items():
        unit = card.unit
        if unit is None:
            continue
        entity = make_unit(1, unit, 1, Point(0, 0))
        tiles_per_second = entity.speed_mt_per_sec / MT
        if tiles_per_second > FASTEST_WALK + 1e-6:
            offenders.append((name, tiles_per_second))
    assert not offenders, offenders


def test_the_speed_field_really_is_reused_across_two_bands():
    """Guards the premise of the test above rather than the code.

    If the shipped data ever stopped mixing walk and projectile speeds in one
    field, the check above would be guarding nothing and should be revisited
    instead of quietly passing forever.
    """
    values = set()
    for path in GAMEDATA.rglob("*.toml"):
        for found in re.finditer(r"^\s*Speed\s*=\s*(\d+)\s*$",
                                 path.read_text(errors="ignore"), re.M):
            values.add(int(found.group(1)))
    assert values & {raw for raw, _ in BANDS.values()}, "no walk-band speeds found"
    assert {value for value in values if value >= 300}, "no projectile-band speeds found"
