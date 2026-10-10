"""Re-drive a sample of IL_Replay through our engine; write the fidelity report.

    py -3.12 tools/redrive_bench.py --matches 2000 --workers 8 [--start-elixir 6000]

Fixed sample: the first N loadable matches of the first parts, so two runs
with different engine settings are paired match for match.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_START = None


def _init(start_elixir):
    global _START
    _START = start_elixir
    if start_elixir is not None:
        from sim import match
        match.START_ELIXIR = start_elixir
        match.PlayerState.__dataclass_fields__["elixir"].default = start_elixir
        match.PlayerState.__init__.__defaults__ = tuple(
            start_elixir if d == 5000 else d
            for d in match.PlayerState.__init__.__defaults__)


def _one(rec):
    from sim.redrive import redrive
    r = redrive(rec)
    decks = {s: sorted({c.base_engine for c in rec.decks[s] if c.base_engine})
             for s in rec.decks}
    return {
        "tag": r.tag, "ok": r.ok, "error": r.error, "level": r.level,
        "spread": r.level_spread, "plays": r.plays, "forced": dict(r.forced),
        "short": r.elixir_short_milli, "elixir_events": r.elixir_events,
        "first_forced": r.first_forced_tick, "first_div": r.first_divergence_tick,
        "end": r.end_tick, "winner_agree": r.winner_agrees,
        "crowns_agree": r.crowns_agree, "engine_crowns": r.engine_crowns,
        "recorded_crowns": r.recorded_crowns, "decks": decks,
        "abilities": r.abilities, "abilities_refused": r.abilities_refused,
        "missing_spawns": r.missing_spawns,
        "tower_cards": rec.tower_cards,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--start-elixir", type=int, default=None,
                    help="milli-elixir; default = the engine's own value")
    ap.add_argument("--name", default="")
    args = ap.parse_args()
    from sim.replay_corpus import iter_corpus

    sample = []
    for rec in iter_corpus():
        if rec.loadable() and rec.plays and rec.game_mode in ("Ranked", "Ladder", "1v1 Battle"):
            sample.append(rec)
            if len(sample) >= args.matches:
                break
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(args.start_elixir,)) as pool:
        rows = list(pool.map(_one, sample, chunksize=8))
    name = args.name or f"se{args.start_elixir or 'default'}"
    out = ROOT / "reports" / "redrive" / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    ok = [r for r in rows if r["ok"]]
    n = len(ok)
    forced = Counter()
    for r in ok:
        forced.update(r["forced"])
    plays = sum(r["plays"] for r in ok)
    early = [e for r in ok for e in r["elixir_events"] if e[0] < 600]
    print(f"{name}: {len(rows)} matches, {n} ran, {time.time() - t0:.0f}s")
    print(f"  winner agree {sum(r['winner_agree'] for r in ok) / n:.1%}, "
          f"crowns agree {sum(r['crowns_agree'] for r in ok) / n:.1%}")
    print(f"  plays {plays:,}; forced per 100 plays: " +
          ", ".join(f"{k} {100 * v / plays:.2f}" for k, v in forced.most_common()))
    print(f"  elixir refusals in first 30 s: {len(early)} "
          f"(mean shortfall {sum(e[3] for e in early) / max(1, len(early)):.0f} milli)")
    errors = Counter(r["error"].split(":")[0] for r in rows if not r["ok"])
    if errors:
        print("  errors:", dict(errors))
    return 0


if __name__ == "__main__":
    sys.exit(main())
