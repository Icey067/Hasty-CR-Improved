"""Compare the route a troop walks, not just the tower it ends at.

Every lane comparison so far has read the endpoint: which tower did it reach.
Both engines agree on that in 104 of 105 placements. That says nothing about
the shape of the walk between here and there, and "the pathfinding is off" is a
complaint about the shape.

The two mechanisms differ exactly there. Ours steers a flow field toward the
target. The reference runs A* over a cost map in which the real walking lanes
cost 5, off-lane ground costs 8 and the river 800, so its troops funnel into
the corridors a real match walks and ours are free to cut diagonally.
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


def ours_path(card: str, x: float, start_y: float, seconds: float = 30.0):
    from sim import arena
    from sim.engine import Battle
    from sim.entities import make_tower, make_unit
    from sim.gamedata import load_gamedata
    C = load_gamedata(level=11); P = (3052, 109, 800, 7500)
    b = Battle()
    for lane in ("left", "right"):
        b.add(make_tower(0, -1, arena.ENEMY_PRINCESS[lane], *P))
    k = b.add(make_tower(0, -1, arena.ENEMY_KING, *P, king=True))
    k.target_only_buildings = True
    u = b.add(make_unit(0, C[card].unit, 1, abs_point(x, start_y)))
    out = []
    for tick in range(int(seconds / TICK)):
        b.step()
        if not u.alive:
            break
        if tick % 10 == 0:                      # twice a second
            out.append((u.pos.x / 1000.0, u.pos.y / 1000.0))
    return out


def ref_path(card: str, x: float, start_y: float, seconds: float = 30.0):
    if not REF.exists():
        return []
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
        out = []
        with contextlib.redirect_stdout(io.StringIO()):
            st = BattleState(PlayerState(0, list(deck), 10.0),
                             PlayerState(1, list(deck), 10.0))
            if not st.deploy_card(0, names[card], Position(x, 31.0 - start_y)):
                return []
            u = next((e for e in st.entities.values()
                      if type(e).__name__ != "Building"), None)
            if u is None:
                return []
            for tick in range(int(seconds / TICK)):
                st.step(TICK)
                if not getattr(u, "is_alive", True):
                    break
                if tick % 10 == 0:
                    # Mirrored back into our frame so the two are comparable.
                    out.append((u.position.x, 31.0 - u.position.y))
        return out
    finally:
        os.chdir(prev)


def draw(card="knight", x=7.0, start_y=23.0):
    a = ours_path(card, x, start_y)
    b = ref_path(card, x, start_y)
    print(f"\n{card} placed at x={x}, y={start_y}. "
          f"Bridges at x=3.5 and x=14.5, river at y=16.\n")
    print(f"{'t':>5}{'ours x,y':>16}{'ref x,y':>16}{'x apart':>9}")
    print("-" * 48)
    worst = 0.0
    for i in range(max(len(a), len(b))):
        pa = a[i] if i < len(a) else None
        pb = b[i] if i < len(b) else None
        sa = f"{pa[0]:5.1f},{pa[1]:5.1f}" if pa else "     -"
        sb = f"{pb[0]:5.1f},{pb[1]:5.1f}" if pb else "     -"
        gap = abs(pa[0] - pb[0]) if pa and pb else float("nan")
        if pa and pb:
            worst = max(worst, gap)
        if i % 2 == 0:
            print(f"{i * 0.5:>5.1f}{sa:>16}{sb:>16}{gap:>9.1f}")
    print(f"\nwidest sideways separation: {worst:.1f} tiles")
    return worst


if __name__ == "__main__":
    worst = []
    for x in (7.0, 9.0, 11.0):
        worst.append((x, draw("knight", x, 23.0)))
    print("\n\nsummary, sideways separation by placement:")
    for x, w in worst:
        print(f"  x={x:<5} {w:.1f} tiles apart at the widest")
