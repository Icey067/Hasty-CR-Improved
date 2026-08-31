"""Sweep every placement across the board and compare which tower is attacked.

`diff_scenarios.py` checked six x positions and found the two engines agreed
36 times out of 36, which was reported as "no lane bug". Six samples on an
empty board cannot support that: the two engines decide lanes by different
mechanisms and a disagreement would show up between the samples, not on them.

Ours commits by a hard split at x=9.0 (`arena.same_lane`) and walks a flow
field. The reference has no lane rule at all - it runs A* over a hand-authored
cost map where a lane cell costs 5, an off-lane cell 8 and the river 800, so
troops funnel into the real corridors because that is the cheap way through.
Those agree in the middle of each half and are free to differ anywhere else.

This walks x in half-tile steps at several depths and prints both answers.
"""
from __future__ import annotations
import contextlib, io, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "_references" / "clash-royale-simulator" / "src" / "clasher_new"
sys.path.insert(0, str(ROOT))
TICK = 0.05


def abs_point(x: float, y: float):
    from sim.arena import MT, Point
    return Point(int(round(x * MT)), int(round(y * MT)))


def ours(card: str, x: float, start_y: float, seconds: float = 45.0):
    from sim import arena
    from sim.engine import Battle, distance
    from sim.entities import make_tower, make_unit
    from sim.gamedata import load_gamedata
    C = load_gamedata(level=11); P = (3052, 109, 800, 7500)
    b = Battle()
    ids = {lane: b.add(make_tower(0, -1, arena.ENEMY_PRINCESS[lane], *P)).uid
           for lane in ("left", "right")}
    k = b.add(make_tower(0, -1, arena.ENEMY_KING, *P, king=True))
    k.target_only_buildings = True
    u = b.add(make_unit(0, C[card].unit, 1, abs_point(x, start_y)))
    for _ in range(int(seconds / TICK)):
        b.step()
        for lane, uid in ids.items():
            t = b.entities[uid]
            if distance(u.pos, t.pos) <= u.range_mt + t.collision_radius_mt:
                return lane
        if not u.alive:
            return "died"
    return "never"


def ref(card: str, x: float, start_y: float, seconds: float = 45.0):
    if not REF.exists():
        return "n/a"
    sys.path.insert(0, str(REF))
    prev = os.getcwd(); os.chdir(REF)
    try:
        from battle import BattleState
        from core import Position
        from player import PlayerState
        deck = ["HogRider", "Skeletons", "Knight", "Musketeer",
                "IceSpirits", "Firecracker", "Golem", "SkeletonKing"]
        names = {"hog_rider": "HogRider", "knight": "Knight",
                 "musketeer": "Musketeer"}
        with contextlib.redirect_stdout(io.StringIO()):
            st = BattleState(PlayerState(0, list(deck), 10.0),
                             PlayerState(1, list(deck), 10.0))
            # Their board is mirrored: player 0 defends the top and attacks
            # downward, so our depth measured from our own towers is theirs
            # measured from the other end.
            if not st.deploy_card(0, names[card], Position(x, 31.0 - start_y)):
                return "refused"
            unit = next((e for e in st.entities.values()
                         if type(e).__name__ != "Building"), None)
            towers = {("left" if e.position.x < 9 else "right"): e
                      for e in st.entities.values()
                      if getattr(e, "player", None) == 1
                      and getattr(e, "name", "") != "KingTower"}
            if unit is None or len(towers) != 2:
                return "setup"
            for _ in range(int(seconds / TICK)):
                st.step(TICK)
                if not getattr(unit, "is_alive", True):
                    return "died"
                for lane, t in towers.items():
                    d = ((unit.position.x - t.position.x) ** 2 +
                         (unit.position.y - t.position.y) ** 2) ** 0.5
                    if d <= unit.data.range + 1.6:
                        return lane
        return "never"
    finally:
        os.chdir(prev)


if __name__ == "__main__":
    card = sys.argv[1] if len(sys.argv) > 1 else "knight"
    print(f"card: {card}.  Towers at x=3.5 and x=14.5; our split is x=9.0.\n")
    disagree = 0
    total = 0
    for start_y in (23.0, 20.0, 17.0):
        print(f"--- deployed at y={start_y} "
              f"({'own back' if start_y > 22 else 'mid' if start_y > 18 else 'at the bridge'})")
        line_a, line_b, xs = [], [], []
        x = 0.5
        while x < 18.0:
            a, b = ours(card, x, start_y), ref(card, x, start_y)
            xs.append(x); line_a.append(a); line_b.append(b)
            total += 1
            if a != b:
                disagree += 1
            x += 0.5
        def row(label, vals):
            return label + " ".join(
                {"left": "L", "right": "R"}.get(v, "?") for v in vals)
        print("      x: " + " ".join(f"{v:>4.1f}"[-1] for v in xs))
        print(row("  ours: ", line_a))
        print(row("   ref: ", line_b))
        print("  diff:  " + " ".join(
            "^" if a != b else " " for a, b in zip(line_a, line_b)))
        print()
    print(f"{disagree} of {total} placements chose a different tower.")


# ---------------------------------------------------------- tower down

def ours_tower_down(card: str, x: float, down: str, start_y: float = 22.0,
                    seconds: float = 60.0):
    """Which target a unit walks to when one enemy princess is already dead.

    This is where a lane rule earns its keep. Straight-line distance sends a
    troop across the arena to the surviving princess; the real game walks it at
    the king. `arena.same_lane` is meant to stop that.
    """
    from sim import arena
    from sim.engine import Battle, distance
    from sim.entities import make_tower, make_unit
    from sim.gamedata import load_gamedata
    C = load_gamedata(level=11); P = (3052, 109, 800, 7500)
    b = Battle()
    ids = {}
    for lane in ("left", "right"):
        t = b.add(make_tower(0, -1, arena.ENEMY_PRINCESS[lane], *P))
        ids[lane] = t.uid
        if lane == down:
            # `alive` is derived from hit points, so zeroing those is how a
            # tower is taken down here.
            t.hitpoints = 0
    king = b.add(make_tower(0, -1, arena.ENEMY_KING, *P, king=True))
    ids["king"] = king.uid
    u = b.add(make_unit(0, C[card].unit, 1, abs_point(x, start_y)))
    for _ in range(int(seconds / TICK)):
        b.step()
        for lane, uid in ids.items():
            # A destroyed tower is removed from the table, not kept as a
            # corpse, so this is a lookup that can miss rather than an
            # `alive` check.
            t = b.entities.get(uid)
            if t is None or not t.alive:
                continue
            if distance(u.pos, t.pos) <= u.range_mt + t.collision_radius_mt:
                return lane
        if not u.alive:
            return "died"
    return "never"


def tower_down_table():
    print("\nOne enemy princess already destroyed. Which target does a lone")
    print("troop reach? Crossing the whole arena to the surviving princess is")
    print("the behaviour `arena.same_lane` exists to prevent.\n")
    print(f"{'card':<11}{'placed x':>9}{'tower down':>12}{'walks to':>11}"
          f"{'   verdict'}")
    print("-" * 58)
    bad = 0
    for card in ("knight", "hog_rider"):
        for down in ("right", "left"):
            for x in (2.0, 5.0, 9.0, 13.0, 16.0):
                got = ours_tower_down(card, x, down)
                side = "left" if x < 9.0 else "right"
                # Legal: its own lane's tower if that one still stands, or the
                # king. Never the far princess.
                far = "left" if side == "right" else "right"
                ok = got != far
                if not ok:
                    bad += 1
                print(f"{card:<11}{x:>9.1f}{down:>12}{got:>11}"
                      f"{'' if ok else '   <-- CROSSED THE ARENA'}")
    print(f"\n{bad} placements crossed to the far princess.")
