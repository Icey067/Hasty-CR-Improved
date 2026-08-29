"""Compare our engine against the reference one on behaviours, not just damage.

The single body-block scenario in `differential.py` was not enough to localise
anything: both engines under-model blocking in the same direction, so agreement
told us nothing about which is right. Widening the comparison is the point - a
defect that shows in one scenario and not another is far easier to place than a
single number that disagrees.

Nothing is copied from the reference implementation. It is imported and driven
as a black box and only its outputs are read.

Two things the reference makes easy to get wrong, both found by tracing rather
than by reading:

* Its `speed` is already tiles per second (`gamedata` speed / 60) and `step`
  takes seconds, so 0.05 is a legitimate tick. But a deploy at y=14.5 - legal
  by its own `deploy_card` - lands on the river band and `ensure_walkability`
  teleports the unit eight tiles downfield on the first step. Deploy at y=10.
* Hit points live on `.hp`, not `.hitpoints`, and units have a one second
  deploy delay before they move.
"""
from __future__ import annotations
import contextlib, io, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "_references" / "clash-royale-simulator" / "src" / "clasher_new"
sys.path.insert(0, str(ROOT))
TICK = 0.05

# Both engines put the princess towers on tile centres 3.5 and 14.5, so a
# "lane" means the same thing in each and an x is directly comparable - but
# only in ABSOLUTE tiles. `arena.tile()` takes a tile *index* and returns its
# centre, so `tile(8.5, y)` is absolute 9.0, exactly the mirror line. Comparing
# that against the reference's absolute 8.5 reported six "wrong lane" rows that
# were half a tile of convention and nothing else. Everything below is absolute
# and goes through `abs_point()`.
#
# Travel distance is equalised too. The reference's towers sit at y=25.5 and
# ours at 7.5, so the same start y is a different walk and the arrival times
# were not comparable either.

TRAVEL = 15.5


def abs_point(x: float, y: float):
    from sim.arena import MT, Point
    return Point(int(round(x * MT)), int(round(y * MT)))


def ours(card: str, x: float, side: int, seconds: float = 40.0):
    """Which tower a lone unit walks to, and when it gets in range.

    side +1 is the bottom seat attacking upward, -1 the top seat.
    """
    from sim import arena
    from sim.engine import Battle, distance
    from sim.entities import make_tower, make_unit
    from sim.gamedata import load_gamedata
    C = load_gamedata(level=11); P = (3052, 109, 800, 7500)
    princess = arena.ENEMY_PRINCESS if side > 0 else arena.ALLY_PRINCESS
    king_at = arena.ENEMY_KING if side > 0 else arena.ALLY_KING
    start_y = (princess["left"].y / 1000.0) + (TRAVEL if side > 0 else -TRAVEL)

    b = Battle()
    ids = {lane: b.add(make_tower(0, -side, princess[lane], *P)).uid
           for lane in ("left", "right")}
    k = b.add(make_tower(0, -side, king_at, *P, king=True))
    k.target_only_buildings = True
    u = b.add(make_unit(0, C[card].unit, side, abs_point(x, start_y)))
    for tick in range(int(seconds / TICK)):
        b.step()
        for lane, uid in ids.items():
            t = b.entities[uid]
            if distance(u.pos, t.pos) <= u.range_mt + t.collision_radius_mt:
                return round(tick * TICK, 2), lane
        if not u.alive:
            return None, "died"
    return None, "never"


def ref(card: str, x: float, seconds: float = 40.0):
    if not REF.exists():
        return None, "n/a"
    sys.path.insert(0, str(REF))
    prev = os.getcwd(); os.chdir(REF)
    try:
        from battle import BattleState
        from core import Position
        from player import PlayerState
        deck = ["HogRider", "Skeletons", "Knight", "Musketeer",
                "IceSpirits", "Firecracker", "Golem", "SkeletonKing"]
        names = {"hog_rider": "HogRider", "knight": "Knight",
                 "musketeer": "Musketeer", "skeletons": "Skeletons"}
        with contextlib.redirect_stdout(io.StringIO()):
            st = BattleState(PlayerState(0, list(deck), 10.0),
                             PlayerState(1, list(deck), 10.0))
            # Player 0 defends the top (y=6.5) and attacks downward, so its own
            # half is y < 16 and y=10 is well clear of the river band.
            if not st.deploy_card(0, names[card], Position(x, 10.0)):
                return None, "refused"
            unit = next((e for e in st.entities.values()
                         if type(e).__name__ != "Building"), None)
            towers = {("left" if e.position.x < 9 else "right"): e
                      for e in st.entities.values()
                      if getattr(e, "player", None) == 1
                      and getattr(e, "name", "") != "KingTower"}
            if unit is None or len(towers) != 2:
                return None, "setup"
            for tick in range(int(seconds / TICK)):
                st.step(TICK)
                if not getattr(unit, "is_alive", True):
                    return None, "died"
                for lane, t in towers.items():
                    d = ((unit.position.x - t.position.x) ** 2 +
                         (unit.position.y - t.position.y) ** 2) ** 0.5
                    # A Musketeer halts six tiles out and shoots; a fixed
                    # 2.5 tile gate called that "never arrived" and then
                    # "died", which is a property of the gate, not the engine.
                    if d <= unit.data.range + 1.6:
                        return round(tick * TICK, 2), lane
        return None, "never"
    finally:
        os.chdir(prev)


def ours_aggro(walker: str, bait: str, offset: float, seconds: float = 2.5):
    """Does a unit walking up the lane divert to a bait `offset` tiles aside?

    Returns the tile y at which it stopped advancing, or None if it never did.
    A building-targeter should walk straight past; everything else should
    divert once the bait is inside its sight range plus the bait's radius.
    """
    from sim import arena
    from sim.engine import Battle
    from sim.entities import make_tower, make_unit
    from sim.gamedata import load_gamedata
    C = load_gamedata(level=11); P = (3052, 109, 800, 7500)
    b = Battle()
    towers = set()
    for lane in ("left", "right"):
        towers.add(b.add(make_tower(0, -1, arena.ENEMY_PRINCESS[lane], *P)).uid)
    k = b.add(make_tower(0, -1, arena.ENEMY_KING, *P, king=True))
    k.target_only_buildings = True
    towers.add(k.uid)
    # Same geometry as the reference: the walker four tiles short of the bait,
    # both on their own side of the river.
    u = b.add(make_unit(0, C[walker].unit, 1, abs_point(14.5, 18.0)))
    b.add(make_unit(0, C[bait].unit, -1, abs_point(14.5 - offset, 14.0)))
    for _ in range(int(seconds / TICK)):
        b.step()
        if not u.alive:
            break
        # Any target that is not one of the three crown towers is a divert.
        # Testing `name != "tower"` instead let a Hog locking onto a princess
        # tower read as a divert, and every row came back the same.
        if u.target_uid and u.target_uid not in towers:
            return True
    return False


def ref_aggro(walker: str, bait: str, offset: float, seconds: float = 2.5):
    if not REF.exists():
        return None
    sys.path.insert(0, str(REF))
    prev = os.getcwd(); os.chdir(REF)
    try:
        from battle import BattleState
        from core import Position
        from player import PlayerState
        deck = ["HogRider", "Skeletons", "Knight", "Musketeer",
                "IceSpirits", "Firecracker", "Golem", "SkeletonKing"]
        names = {"hog_rider": "HogRider", "knight": "Knight",
                 "musketeer": "Musketeer", "skeletons": "Skeletons"}
        with contextlib.redirect_stdout(io.StringIO()):
            st = BattleState(PlayerState(0, list(deck), 100.0),
                             PlayerState(1, list(deck), 100.0))
            # Player 0 may deploy up to y=14 and player 1 from y=18, so four
            # tiles is the closest the two can legally start.
            if not st.deploy_card(0, names[walker], Position(14.5, 14.0)):
                return None
            if not st.deploy_card(1, names[bait], Position(14.5 - offset, 18.0)):
                return None
            walkers = [e for e in st.entities.values()
                       if type(e).__name__ != "Building"
                       and getattr(e, "player", None) == 0]
            if not walkers:
                return None
            u = walkers[0]
            for _ in range(int(seconds / TICK)):
                st.step(TICK)
                if not getattr(u, "is_alive", True):
                    break
                t = st.entities.get(getattr(u, "target_id", None))
                if t is not None and type(t).__name__ != "Building":
                    return True
        return False
    finally:
        os.chdir(prev)


def aggro_threshold(fn, walker, bait, seconds):
    """The largest lateral offset at which the bait is still acquired."""
    found = None
    offset = 0.0
    while offset <= 10.0:
        got = fn(walker, bait, offset, seconds)
        if got is None:
            return None
        if got:
            found = offset
        offset += 0.25
    return found


def aggro_table():
    print()
    print("Aggro: how far to the side a bait can sit and still be acquired.")
    print("Bait four tiles up the lane; the horizon is short on purpose, so")
    print("this measures the acquisition radius and not who closes distance.")
    print()
    print(f"{'walker':<11}{'bait':<11}{'sight':>7}{'ours':>8}{'ref':>8}"
          f"{'predicted':>11}")
    print("-" * 56)
    from sim.arena import Point
    from sim.entities import make_unit
    from sim.gamedata import load_gamedata
    C = load_gamedata(level=11)
    for walker, bait in (("knight", "musketeer"), ("hog_rider", "musketeer"),
                         ("musketeer", "knight"), ("skeletons", "knight")):
        w = make_unit(1, C[walker].unit, 1, Point(0, 0))
        t = make_unit(2, C[bait].unit, -1, Point(0, 0))
        sight = w.sight_range_mt / 1000.0
        reach = sight + t.collision_radius_mt / 1000.0
        # Both engines acquire when centre distance <= sight + the target's
        # collision radius, so with four tiles of separation up the lane the
        # largest sideways offset that still reaches is this.
        want = (reach ** 2 - 16.0) ** 0.5 if reach > 4.0 else float("nan")
        a = aggro_threshold(ours_aggro, walker, bait, 2.5)
        b = aggro_threshold(ref_aggro, walker, bait, 2.5)
        fmt = lambda v: "none" if v is None else f"{v:.2f}"
        mark = "" if a == b else "   <-- differs"
        print(f"{walker:<11}{bait:<11}{sight:>7.1f}{fmt(a):>8}{fmt(b):>8}"
              f"{want:>11.2f}{mark}")


if __name__ == "__main__":
    print("Which tower a lone unit walks to, from a given x.")
    print("Towers stand at x=3.5 and x=14.5; the mirror line is x=9.0.\n")
    print(f"{'card':<11}{'x':>6}{'seat':>7}{'ours':>9}{'lane':>8}  |"
          f"{'ref':>8}{'lane':>8}")
    print("-" * 60)
    bad = 0
    for card in ("knight", "hog_rider", "musketeer"):
        for x in (3.5, 7.0, 8.6, 9.4, 11.0, 14.5):
            for seat, label in ((1, "bottom"), (-1, "top")):
                a_t, a_l = ours(card, x, seat)
                b_t, b_l = (ref(card, x) if seat == 1 else (None, ""))
                at = "never" if a_t is None else f"{a_t:.2f}s"
                expect = "left" if x < 9.0 else "right"
                flag = "" if a_l == expect else "   <-- WRONG LANE"
                if flag:
                    bad += 1
                if seat == 1:
                    bt = "never" if b_t is None else f"{b_t:.2f}s"
                    tail = f"  |{bt:>8}{b_l:>8}"
                else:
                    tail = "  |" + " " * 16
                print(f"{card:<11}{x:>6}{label:>7}{at:>9}{a_l:>8}{tail}{flag}")
    print(f"\n{bad} placements walked to the tower on the wrong side.")
