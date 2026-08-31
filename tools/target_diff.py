"""Which target does a unit pick, and when does it change its mind?

Pathfinding and targeting were the two places the two engines could still
differ structurally. Pathfinding turned out to hold a real bug; this is the
other one, and it has never been compared.

The reference commits hard: `_should_switch_target` refuses to retarget while
the unit is in attack range of what it already has, and otherwise lets a troop
outrank a building. Ours keeps a target while it is alive and inside sight,
and re-sweeps on a timer. Those are not the same rule, and the difference shows
up in the most common situation in the game - something hitting a tower when a
defender arrives.

Measured 2026-08-31, five of twelve scenarios disagree:

    attacker   defender   ours              reference
    musketeer  knight     -> knight         -> tower
    musketeer  skeletons  -> tower          -> Skeletons
    knight     knight     -> knight         -> tower
    hog_rider  anything   -> tower          -> tower        (agree)

Building-targeters agree everywhere, which is the easy half. The rest is a
genuine behavioural split in the most frequent situation in the game, and
**which one is right is not settled here**. Ours retargets onto the defender;
theirs holds the tower. Clash Royale does retarget on a short timer, which
argues for ours, but the reference's rule is deliberate and the owner reports
that engine as behaving correctly - so this is evidence about a difference, not
a verdict on it.

What would settle it: a recording of a Musketeer hitting a tower when a Knight
walks into her range, and whether her next shot lands on the tower or the
Knight. One clip, one answer. Until then nothing here is changed - the
targeting rule is not something to swap on a coin flip.
"""
from __future__ import annotations
import contextlib, io, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "_references" / "clash-royale-simulator" / "src" / "clasher_new"
sys.path.insert(0, str(ROOT))
TICK = 0.05


def ours_switches(attacker: str, defender: str, arrive_s: float,
                  seconds: float = 45.0):
    """Is the attacker still on the tower when a defender walks up to it?

    Returns (what it was hitting before, what it is hitting after).
    """
    from sim import arena
    from sim.arena import MT, Point
    from sim.engine import Battle
    from sim.entities import make_tower, make_unit
    from sim.gamedata import load_gamedata
    C = load_gamedata(level=11); P = (3052, 109, 800, 7500)
    b = Battle()
    tower = b.add(make_tower(0, -1, arena.ENEMY_PRINCESS["right"], *P))
    b.add(make_tower(0, -1, arena.ENEMY_PRINCESS["left"], *P))
    k = b.add(make_tower(0, -1, arena.ENEMY_KING, *P, king=True))
    k.target_only_buildings = True
    # Deploy where it is legal and let it walk, rather than teleporting it
    # into contact: the reference refuses a deploy in enemy ground, and an
    # attacker that started already in range was never comparable to one that
    # arrived. The defender lands once the attacker is actually hitting the
    # tower, which is the moment the question is about.
    a = b.add(make_unit(0, C[attacker].unit, 1, Point(14 * MT, 20 * MT)))
    before = after = None
    dropped = False
    engaged_at = None
    last_y = -99_000
    from sim.engine import distance
    for tick in range(int(seconds / TICK)):
        b.step()
        if not a.alive:
            break
        if engaged_at is None:
            if (a.target_uid == tower.uid and tower.alive
                    and abs(a.pos.y - last_y) < 10 and tick > 20):
                engaged_at = tick * TICK
            last_y = a.pos.y
        elif not dropped and tick * TICK >= engaged_at + arrive_s:
            dropped = True
            before = _name(b, a.target_uid, tower.uid)
            b.add(make_unit(0, C[defender].unit, -1,
                            Point(a.pos.x, a.pos.y - MT)))
        elif dropped and tick * TICK >= engaged_at + arrive_s + 2.0:
            after = _name(b, a.target_uid, tower.uid)
            break
    return before, after


def _name(b, uid, tower_uid):
    if not uid:
        return "nothing"
    e = b.entities.get(uid)
    if e is None:
        return "gone"
    return "tower" if e.is_tower else e.name


def ref_switches(attacker: str, defender: str, arrive_s: float,
                 seconds: float = 45.0):
    if not REF.exists():
        return None, None
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
            # Player 0 owns the top and may only deploy at y < 15.
            if not st.deploy_card(0, names[attacker], Position(14.5, 11.0)):
                return None, None
            a = next((e for e in st.entities.values()
                      if type(e).__name__ != "Building"), None)
            tower = next((e for e in st.entities.values()
                          if getattr(e, "player", None) == 1
                          and getattr(e, "name", "") != "KingTower"
                          and e.position.x > 9), None)
            if a is None or tower is None:
                return None, None
            before = after = None
            dropped = False
            engaged_at = None
            last_y = -99.0
            for tick in range(int(seconds / TICK)):
                st.step(TICK)
                if not getattr(a, "is_alive", True):
                    break
                if engaged_at is None:
                    # "Stopped, with the tower targeted" rather than a distance
                    # threshold: a Knight halts 2.7 tiles out and a threshold
                    # built from range plus collision radius came to 2.6, so it
                    # never fired and every run read as "never engaged".
                    if (getattr(a, "target_id", None) == tower.id
                            and abs(a.position.y - last_y) < 0.01
                            and tick > 20):
                        engaged_at = tick * TICK
                    last_y = a.position.y
                elif not dropped and tick * TICK >= engaged_at + arrive_s:
                    dropped = True
                    before = _ref_name(st, a)
                    st.deploy_card(1, names[defender],
                                   Position(14.5, max(18.0, a.position.y - 1.0)))
                elif dropped and tick * TICK >= engaged_at + arrive_s + 2.0:
                    after = _ref_name(st, a)
                    break
        return before, after
    finally:
        os.chdir(prev)


def _ref_name(st, entity):
    tid = getattr(entity, "target_id", None)
    if tid is None:
        return "nothing"
    t = st.entities.get(tid)
    if t is None:
        return "gone"
    return "tower" if type(t).__name__ == "Building" else getattr(t, "name", "?")


if __name__ == "__main__":
    print("A unit is attacking a tower; a defender arrives. Does it switch?\n")
    print(f"{'attacker':<11}{'defender':<11}{'after':>7}"
          f"{'ours: before -> after':>28}{'   reference'}")
    print("-" * 78)
    differs = 0
    for attacker in ("musketeer", "knight", "hog_rider"):
        for defender in ("knight", "skeletons"):
            for arrive in (0.5, 3.0):
                ab, aa = ours_switches(attacker, defender, arrive)
                bb, ba = ref_switches(attacker, defender, arrive)
                ours = f"{ab} -> {aa}"
                ref = f"{bb} -> {ba}"
                same = (ab == bb) and (aa == ba)
                if not same:
                    differs += 1
                print(f"{attacker:<11}{defender:<11}{arrive:>8.1f}"
                      f"{ours:>28}   {ref}{'' if same else '   <-- differs'}")
    print(f"\n{differs} of 12 scenarios disagree.")
