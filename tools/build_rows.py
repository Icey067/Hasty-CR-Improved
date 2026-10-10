"""Build training shards from IL_Replay: every Hog 2.6 match plus a sample of others.

    py -3.12 tools/build_rows.py --others 20000 --workers 8

Writes data/il_rows/shard-XXXX.npz (one per worker batch) and manifest.json.
Matches are split into train / val / test by a hash of the match tag, so a
match never straddles splits and reruns agree on the split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "il_rows"
HOG26 = frozenset({"cannon", "fireball", "hog_rider", "ice_golemite",
                   "ice_spirits", "musketeer", "skeletons", "log"})
MODES = {"Ranked", "Ladder", "1v1 Battle", "Grand Challenge", "Classic Challenge"}


def split_of(tag: str) -> str:
    h = int(hashlib.sha1(tag.encode()).hexdigest()[:8], 16) % 100
    return "test" if h < 5 else "val" if h < 10 else "train"


def _start_elixir():
    from sim import match
    match.START_ELIXIR = 6000
    match.PlayerState.__init__.__defaults__ = tuple(
        6000 if d == 5000 else d for d in match.PlayerState.__init__.__defaults__)


def build(batch_id: int, recs) -> dict:
    _start_elixir()
    from sim.replay_rows import rows_for, stack
    rows, tags, info = [], [], []
    for rec in recs:
        try:
            r, res = rows_for(rec)
        except Exception as exc:            # one bad match must not end a shard
            info.append({"tag": rec.tag, "error": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        hog = [s for s in rec.decks if rec.base_deck(s) == HOG26]
        for row in r:
            row["hog"] = (("team" if row["side"] == 1 else "opponent") in hog)
        rows += r
        tags += [rec.tag] * len(r)
        info.append({"tag": rec.tag, "rows": len(r), "split": split_of(rec.tag),
                     "hog_sides": hog, "ok": res.ok})
    if not rows:
        return {"batch": batch_id, "rows": 0, "info": info}
    data = stack(rows, tags)
    data["hog"] = np.asarray([row["hog"] for row in rows])
    data["split"] = np.asarray([split_of(t) for t in tags])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"shard-{batch_id:04d}.npz", **data)
    return {"batch": batch_id, "rows": len(rows), "info": info}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--others", type=int, default=20000,
                    help="non-Hog matches to include alongside every Hog 2.6 match")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--parts", type=int, default=0)
    args = ap.parse_args()
    from sim.replay_corpus import iter_corpus

    t0 = time.time()
    hog, others = [], []
    for rec in iter_corpus(limit_parts=args.parts):
        if not rec.loadable() or not rec.plays or rec.game_mode not in MODES:
            continue
        if HOG26 in (rec.base_deck("team"), rec.base_deck("opponent")):
            hog.append(rec)
        elif len(others) < args.others:
            others.append(rec)
    picked = hog + others
    print(f"selected {len(hog):,} Hog 2.6 matches + {len(others):,} others "
          f"({time.time() - t0:.0f}s)", flush=True)

    batches = [picked[i:i + args.batch] for i in range(0, len(picked), args.batch)]
    manifest, total = [], 0
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(build, i, b) for i, b in enumerate(batches)]
        for k, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            total += res["rows"]
            manifest += res["info"]
            print(f"  shard {res['batch']:4d}: {res['rows']:6,} rows   "
                  f"[{k}/{len(batches)}, {total:,} rows, {time.time() - t0:.0f}s]", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(
        {"hog_matches": len(hog), "other_matches": len(others), "rows": total,
         "start_elixir_milli": 6000, "matches": manifest}), encoding="utf-8")
    print(f"done: {total:,} rows in {time.time() - t0:.0f}s -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
