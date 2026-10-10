"""One battle frame from game memory, read level by level in batches.

Each level of the pointer graph is one `adb exec-out` call that runs many
`dd` reads on the device and returns them concatenated, so a frame costs about
four round trips instead of one per field.

    py -3.12 scripts/gamestate/frame.py            # one frame, printed
    py -3.12 scripts/gamestate/frame.py --watch 10 # ten frames, timings

What a frame holds (public to a player watching the screen, plus our own
hand and elixir, which we see on our own screen):
    tick, my elixir, my hand / next (deck indexes), every unit's side, x, y,
    card id, level, hp, max hp.
The OPPONENT's hand and elixir are deliberately not read. They are hidden
in the real game; the bot estimates elixir from the plays it sees.

Field offsets for build 160402012 (see memread.py for where they come from):
    battle +0x60 tick | +0xA8 player state | +0x1BC replay tick
    player state +0xE0: two player pointers
    player +0x2F8 elixir (raw, 1e-4 elixir) | +0x210/+0x218/+0x21C hand
        vector/capacity/size | +0x220/+0x228/+0x22C cycle vector/capacity/size
    player state +0x08 registry -> +0x40 collection -> +0x08 data, +0x14 count
    entity +0x08 category | +0x30 kind | +0x78 side | +0x7C x | +0x80 y
        | +0xAC card id | +0x120 level | +0x18 components -> +0x10 hp object
        -> +0x10 (hp, max hp)

**Unvalidated until a battle is read**: on the home screen the battle pointer
is null (checked 2026-10-09), so nothing below the battle has been seen live
yet. `validate()` lists what to check in the first friendly match.
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gamestate.memread import BATTLE_GLOBAL, BATTLE_HOPS, GameMemory, _sh  # noqa: E402

MAX_ENTITIES = 256


class FrameReader:
    def __init__(self, mem: Optional[GameMemory] = None):
        self.my_side = 1                      # see --side; UNVERIFIED that ours is always 1
        self.mem = mem or GameMemory()

    def batch(self, reads: List[Tuple[int, int]]) -> List[bytes]:
        """Many (addr, size) reads in one adb round trip."""
        if not reads:
            return []
        pid = self.mem.pid
        parts = []
        for addr, size in reads:
            if addr <= 0:
                parts.append(f"head -c {size} /dev/zero")
            else:
                parts.append(f"dd if=/proc/{pid}/mem bs=4096 iflag=skip_bytes,count_bytes "
                             f"skip={addr} count={size} 2>/dev/null || head -c {size} /dev/zero")
        blob = _sh("; ".join(parts), self.mem.serial, binary=True, timeout=30)
        out, at = [], 0
        for _, size in reads:
            out.append(blob[at:at + size])
            at += size
        return out

    @staticmethod
    def u64(b: bytes, off: int = 0) -> int:
        return struct.unpack_from("<Q", b, off)[0] if len(b) >= off + 8 else 0

    @staticmethod
    def i32(b: bytes, off: int = 0) -> int:
        return struct.unpack_from("<i", b, off)[0] if len(b) >= off + 4 else -1

    def battle(self) -> int:
        p = self.mem.u64(self.mem.libg + BATTLE_GLOBAL)
        for hop in BATTLE_HOPS:
            if not p:
                return 0
            p = self.mem.u64(p + hop)
        return p

    def frame(self, my_side: Optional[int] = None) -> Optional[Dict]:
        t0 = time.perf_counter()
        battle = self.battle()
        if not battle:
            return None
        head, = self.batch([(battle, 0x200)])
        tick = self.i32(head, 0x60)
        pstate = self.u64(head, 0xA8)
        if not pstate:
            return None
        ps, = self.batch([(pstate, 0x100)])
        players = [self.u64(ps, 0xE0), self.u64(ps, 0xE8)]
        registry = self.u64(ps, 0x08)
        reads = [(players[0], 0x300), (players[1], 0x300), (registry, 0x48)]
        p0, p1, reg = self.batch(reads)
        collection = self.u64(reg, 0x40)
        coll, = self.batch([(collection, 0x18)])
        data, count = self.u64(coll, 0x08), self.i32(coll, 0x14)
        if not (0 < count <= MAX_ENTITIES):
            return {"tick": tick, "units": [], "note": f"entity count {count}"}
        ptrs_raw, = self.batch([(data, 8 * count)])
        ptrs = [self.u64(ptrs_raw, 8 * i) for i in range(count)]
        raws = self.batch([(p, 0x124) for p in ptrs])
        comps = [self.u64(r, 0x18) for r in raws]
        hp_objs_raw = self.batch([(c + 0x10, 8) if c else (0, 8) for c in comps])
        hp_objs = [self.u64(b) for b in hp_objs_raw]
        pairs = self.batch([(h + 0x10, 8) if h else (0, 8) for h in hp_objs])
        units = []
        for p, r, pair in zip(ptrs, raws, pairs):
            hp, mx = self.i32(pair, 0), self.i32(pair, 4)
            units.append({"addr": p, "category": self.i32(r, 0x08), "kind": self.i32(r, 0x30),
                          "side": self.i32(r, 0x78), "x": self.i32(r, 0x7C),
                          "y": self.i32(r, 0x80), "card_id": self.i32(r, 0xAC),
                          "level": self.i32(r, 0x120), "hp": hp, "max_hp": mx})
        me = {}
        mine = None
        for raw in (p0, p1):
            side = self.i32(raw, 0x78) if len(raw) > 0x7C else -1
            if side == (self.my_side if my_side is None else my_side):
                mine = raw
                break
        if mine is not None:
            hand_vec = self.u64(mine, 0x210)
            hand_size = self.i32(mine, 0x21C)
            cyc_vec, cyc_size = self.u64(mine, 0x220), self.i32(mine, 0x22C)
            reads = [(hand_vec, 4 * max(0, min(hand_size, 8))),
                     (cyc_vec, 4 * max(0, min(cyc_size, 8)))]
            hv, cv = self.batch(reads)
            me = {"elixir_raw": self.i32(mine, 0x2F8),
                  "hand": [self.i32(hv, 4 * i) for i in range(max(0, min(hand_size, 8)))],
                  "cycle": [self.i32(cv, 4 * i) for i in range(max(0, min(cyc_size, 8)))]}
        return {"tick": tick, "units": units, "me": me,
                "read_ms": (time.perf_counter() - t0) * 1000}


def validate(frames: List[Dict]) -> List[str]:
    """What must hold in a real battle before the bot is allowed to use frames."""
    issues = []
    ticks = [f["tick"] for f in frames if f]
    if ticks != sorted(ticks):
        issues.append("tick not monotonic")
    for f in frames:
        if not f:
            continue
        towers = [u for u in f["units"] if 0 < u["hp"] and u["max_hp"] >= 2000]
        if len(towers) < 6 and f["tick"] < 600:
            issues.append(f"tick {f['tick']}: {len(towers)} tower-sized bodies, expected 6")
        for u in f["units"]:
            if not (0 <= u["x"] <= 18_000 and 0 <= u["y"] <= 32_000):
                issues.append(f"tick {f['tick']}: unit off board {u['x']},{u['y']}")
                break
        hand = f.get("me", {}).get("hand", [])
        if len(hand) != 4 or len(set(hand)) != 4 or not all(0 <= h < 8 for h in hand):
            issues.append(f"tick {f['tick']}: hand {hand} is not 4 distinct deck indexes")
        e = f.get("me", {}).get("elixir_raw", -1)
        if not (0 <= e <= 100_000):
            issues.append(f"tick {f['tick']}: elixir_raw {e} out of range")
    return issues


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=1)
    ap.add_argument("--side", type=int, default=1,
                    help="our player record's side; 1 in the 2026-10-10 friendly, "
                         "UNVERIFIED that it is always 1")
    args = ap.parse_args()
    reader = FrameReader()
    reader.my_side = args.side
    frames = []
    for _ in range(args.watch):
        f = reader.frame()
        frames.append(f)
        if f is None:
            print("no battle (battle pointer is null)")
        else:
            print(f"tick {f['tick']}  units {len(f['units'])}  me {f.get('me')}  "
                  f"read {f.get('read_ms', 0):.0f} ms")
    issues = validate([f for f in frames if f])
    print("validation:", "ok" if not issues else issues[:10])
    return 0


if __name__ == "__main__":
    sys.exit(main())
