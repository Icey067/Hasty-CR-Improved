"""Download the public IL_Replay dataset (FirstLight, on Hugging Face) and verify it.

252,238 anonymised ladder matches: both decks with levels and forms, every card
play with its tick, side and native board coordinates, and the end result
(crowns, final tower hitpoints). The file list with sizes and SHA-256 is taken
from the dataset's own manifest, fetched fresh, so nothing here is pinned to a
copy of someone else's tooling.

Resumable: a file whose size and hash already match is skipped, a partial file
is resumed with an HTTP Range request, and a hash mismatch is deleted and
fetched again once.

    python tools/il_replay_download.py              # everything, ~825 MB
    python tools/il_replay_download.py --parts 2    # first two parts only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "VanguardX101/IL_Replay"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/"
CHUNK = 1 << 20


def _get_json(url: str):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def listing() -> list[dict]:
    """Every parquet file in the dataset with its size and LFS SHA-256."""
    out = []
    for folder in ("replays", "actions"):
        for item in _get_json(API + folder):
            if item.get("type") != "file" or not item["path"].endswith(".parquet"):
                continue
            lfs = item.get("lfs") or {}
            out.append({"path": item["path"],
                        "bytes": int(lfs.get("size", item.get("size", 0))),
                        "sha256": lfs.get("oid", "")})
    return sorted(out, key=lambda f: f["path"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(entry: dict, dest: Path) -> str:
    """Download one file to `dest`; returns 'skip', 'ok' or raises."""
    target = dest / entry["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == entry["bytes"]:
        if not entry["sha256"] or sha256(target) == entry["sha256"]:
            return "skip"
        target.unlink()
    for attempt in range(2):
        have = target.stat().st_size if target.exists() else 0
        request = urllib.request.Request(BASE + entry["path"])
        if have:
            request.add_header("Range", f"bytes={have}-")
        with urllib.request.urlopen(request, timeout=120) as response, \
                target.open("ab" if have else "wb") as handle:
            if have and response.status != 206:      # server ignored the range
                handle.seek(0)
                handle.truncate()
            for block in iter(lambda: response.read(CHUNK), b""):
                handle.write(block)
        if target.stat().st_size == entry["bytes"] and (
                not entry["sha256"] or sha256(target) == entry["sha256"]):
            return "ok"
        target.unlink()
    raise RuntimeError(f"{entry['path']}: size or hash still wrong after a refetch")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=ROOT / "data" / "il_replay")
    ap.add_argument("--parts", type=int, default=0,
                    help="only the first N parts of each folder (0 = all)")
    args = ap.parse_args()

    files = listing()
    if args.parts:
        keep = {f"part-{i:06d}.parquet" for i in range(args.parts)}
        files = [f for f in files if Path(f["path"]).name in keep]
    total = sum(f["bytes"] for f in files)
    print(f"{len(files)} files, {total / 1e6:.0f} MB -> {args.dest}", flush=True)
    args.dest.mkdir(parents=True, exist_ok=True)
    (args.dest / "listing.json").write_text(json.dumps(files, indent=1))

    started, done = time.time(), 0
    failed = []
    for entry in files:
        try:
            state = fetch(entry, args.dest)
        except Exception as exc:                     # keep going; report at the end
            failed.append(f"{entry['path']}: {exc}")
            state = "FAIL"
        done += entry["bytes"]
        rate = done / max(1e-6, time.time() - started) / 1e6
        print(f"  {state:4} {entry['path']}  ({done / total:.0%}, {rate:.1f} MB/s)",
              flush=True)
    if failed:
        print("\nFAILED:\n  " + "\n  ".join(failed))
        return 1
    print(f"\nall {len(files)} files verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
