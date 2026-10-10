"""Read-only access to the Clash Royale process on MuMu instance 3.

Nothing here writes to the game, injects code, or sends input. It reads
`/proc/<pid>/mem` through `adb shell` (MuMu's "ADB Root" setting must be on),
which is what the kernel offers any root process.

Guards, all checked before the first read:
* the device is the Clash Royale instance (`scripts/emulator.verify`);
* the installed client is exactly the build the offsets belong to: versionCode
  and the SHA-256 of `libg.so` (`IDENTITY`). Any other build is refused - the
  offsets would point at something else and every value read would be wrong.

This is the slow path: one `adb` round trip per read, tens of milliseconds.
It exists to validate offsets and structure. The live reader batches a whole
frame in one call (`snapshot`) and a native helper is the next step.

Offsets: the pointer chain and field layout for build 160402012 come from
vegetableleaf's public ClashAI repo - the summary in
scratchpad/gauntlet/L70/reader/FINDINGS.md and the field tables in his reader
sources (L70/reader/live_sampler2.c, L72/.../live_sampler3.c: player elixir,
hand and cycle vectors, unit data-name chain). His repo has no licence; no
code was copied, our readers are written here, but the offsets are his
findings. Every one is re-verified against the live process before use
(`probe`); which player record is ours was found here (2026-10-10).
"""

from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from emulator import DEFAULT_ADB, DEFAULT_SERIAL, verify  # noqa: E402

PACKAGE = "com.supercell.clashroyale"
IDENTITY = {
    "version_code": "160402012",
    "libg_sha256": "79c4e485a4e730db4482a08d625d92ddfccaceee2033ee6c55046b57088918c2",
}
# libg-relative global, then the hops to the battle object (FINDINGS.md).
BATTLE_GLOBAL = 0x1AEEF98
BATTLE_HOPS = (0x18, 0x90)


class Refused(SystemExit):
    pass


def _sh(cmd: str, serial: str = DEFAULT_SERIAL, binary: bool = False, timeout: int = 20):
    out = subprocess.run([str(DEFAULT_ADB), "-s", serial, "exec-out", cmd],
                         capture_output=True, timeout=timeout)
    return out.stdout if binary else out.stdout.decode("utf-8", "replace")


class GameMemory:
    def __init__(self, serial: str = DEFAULT_SERIAL):
        verify(serial)
        self.serial = serial
        if _sh("id -u", serial).strip() != "0":
            raise Refused("adb is not root: enable MuMu's 'ADB Root' on instance 3 only")
        version = _sh(f"dumpsys package {PACKAGE} | grep -m1 versionCode", serial)
        if IDENTITY["version_code"] not in version:
            raise Refused(f"client build changed ({version.strip()}); offsets are for "
                          f"{IDENTITY['version_code']} only")
        pid = _sh(f"pidof {PACKAGE}", serial).strip()
        if not pid:
            raise Refused("Clash Royale is not running")
        self.pid = int(pid.split()[0])
        maps = _sh(f"grep libg.so /proc/{self.pid}/maps", serial).splitlines()
        if not maps:
            raise Refused("libg.so is not mapped")
        path = maps[0].split()[-1]
        sha = _sh(f"sha256sum {path}", serial).split()[0]
        if sha != IDENTITY["libg_sha256"]:
            raise Refused(f"libg.so hash {sha} is not the pinned build")
        self.libg = min(int(line.split("-")[0], 16) for line in maps)

    def read(self, addr: int, size: int) -> bytes:
        if addr <= 0:
            return b""
        cmd = (f"dd if=/proc/{self.pid}/mem bs=4096 iflag=skip_bytes,count_bytes "
               f"skip={addr} count={size} 2>/dev/null")
        return _sh(cmd, self.serial, binary=True)

    def u64(self, addr: int) -> int:
        b = self.read(addr, 8)
        return struct.unpack("<Q", b)[0] if len(b) == 8 else 0

    def battle(self) -> int:
        """Address of the battle object, or 0 outside a battle."""
        p = self.u64(self.libg + BATTLE_GLOBAL)
        for hop in BATTLE_HOPS:
            if not p:
                return 0
            p = self.u64(p + hop)
        return p


def probe() -> Dict[str, object]:
    g = GameMemory()
    first = g.u64(g.libg + BATTLE_GLOBAL)
    hop1 = g.u64(first + BATTLE_HOPS[0]) if first else 0
    return {"pid": g.pid, "libg": hex(g.libg), "global": hex(first),
            "hop1": hex(hop1), "battle": hex(g.battle())}


if __name__ == "__main__":
    for k, v in probe().items():
        print(f"{k:8} {v}")
