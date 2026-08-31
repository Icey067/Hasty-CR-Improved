"""A troop must not walk away from its target before walking towards it.

This is the bug the owner reported three times and no test could see, because
every check this project had compared *endpoints*: which tower did the unit
reach. Both engines always agreed on that, so the differential said "no lane
bug" while a Knight placed two tiles left of centre visibly walked right for
three seconds before turning around.

The cause was the cost model. `distance_field` counted steps with eight-way
movement, which is a Chebyshev metric: from (7,23) towards a goal at (3,16)
the distance is max(4,7) = 7, and stepping sideways to (8,23) is max(5,7) = 7
as well. The sideways move is free. Any tile where |dy| > |dx| is a tie, which
is most of a lane, and the tie was broken by the order of `_STEPS` - whose
first entry was (1, 0), right.

Measured against the reference engine on the same placement before the fix:
it walked 7.0 -> 6.4 -> 5.6 -> 4.7 and ours walked 7.0 -> 7.1 -> 7.4 -> 7.5.

So these tests watch the route, not the destination.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim import arena                                          # noqa: E402
from sim.arena import MT, Point                                # noqa: E402
from sim.engine import Battle                                  # noqa: E402
from sim.entities import make_tower, make_unit                 # noqa: E402
from sim.gamedata import load_gamedata                         # noqa: E402

CARDS = load_gamedata(level=11)
TOWER = (3052, 109, 800, 7500)
TICK = 0.05


def _walk(card: str, x: float, y: float, seconds: float = 12.0):
    """Where a lone troop is, twice a second, with nothing in its way."""
    battle = Battle()
    for lane in ("left", "right"):
        battle.add(make_tower(0, -1, arena.ENEMY_PRINCESS[lane], *TOWER))
    king = battle.add(make_tower(0, -1, arena.ENEMY_KING, *TOWER, king=True))
    king.target_only_buildings = True
    unit = battle.add(make_unit(0, CARDS[card].unit, 1,
                                Point(int(x * MT), int(y * MT))))
    trail = []
    for tick in range(int(seconds / TICK)):
        battle.step()
        if not unit.alive:
            break
        if tick % 10 == 0:
            trail.append((unit.pos.x / MT, unit.pos.y / MT))
    return trail


# Placements either side of the centre line, none of them on it: `arena.tile`
# always lands on a half-integer, so exactly 9.0 is not reachable in play.
@pytest.mark.parametrize("x", [5.5, 6.5, 7.5, 8.5])
def test_a_troop_left_of_centre_never_drifts_right(x):
    trail = _walk("knight", x, 23.0)
    assert trail, "the unit never moved"
    worst = max(px for px, _py in trail)
    assert worst <= x + 0.15, (
        f"placed at x={x} and reached x={worst:.2f} before turning back; "
        f"it is walking away from the left bridge. Trail: "
        f"{[(round(a, 1), round(b, 1)) for a, b in trail[:6]]}")


@pytest.mark.parametrize("x", [9.5, 10.5, 11.5, 12.5])
def test_a_troop_right_of_centre_never_drifts_left(x):
    trail = _walk("knight", x, 23.0)
    assert trail, "the unit never moved"
    worst = min(px for px, _py in trail)
    assert worst >= x - 0.15, (
        f"placed at x={x} and reached x={worst:.2f} before turning back; "
        f"it is walking away from the right bridge. Trail: "
        f"{[(round(a, 1), round(b, 1)) for a, b in trail[:6]]}")


def test_the_route_makes_progress_every_second():
    """A path that turns around wastes seconds a real match does not give it.

    Stated as progress towards the bridge rather than as a shape, because the
    shape is allowed to differ from the reference engine's; the loss of tempo
    is what actually costs a game.
    """
    trail = _walk("knight", 7.5, 23.0, seconds=10.0)
    bridge = arena.BRIDGE_X[0] / MT
    gaps = [abs(px - bridge) for px, _py in trail]
    for earlier, later in zip(gaps, gaps[1:]):
        assert later <= earlier + 0.05, (
            f"got further from the bridge: {earlier:.2f} -> {later:.2f}")


def test_the_step_costs_are_not_uniform():
    """Guards the premise rather than the behaviour.

    If every step costs the same again, the ties come back and the tests above
    start passing or failing on the order of a tuple.
    """
    from sim.pathfind import _DIAGONAL, _ORTHOGONAL, _STEPS
    assert _DIAGONAL > _ORTHOGONAL, "a diagonal must cost more than a straight"
    costs = {step for _dx, _dy, step in _STEPS}
    assert costs == {_ORTHOGONAL, _DIAGONAL}
    for dx, dy, step in _STEPS:
        want = _DIAGONAL if dx and dy else _ORTHOGONAL
        assert step == want, (dx, dy, step)
