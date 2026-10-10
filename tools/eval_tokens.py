"""Play a token-policy checkpoint against the engine's opponents.

    py -3.12 tools/eval_tokens.py --ckpt runs/tokens/awr/best.pt --matches 60 \
        --opponent brain   # or scripted (meta archetypes), simple

The agent plays Hog 2.6 as the bottom side, against the rule engine (`brain`,
also Hog 2.6), the scripted meta archetypes, or the simple opponent. Seeds are
fixed, so two checkpoints are compared on the same matches. Reports wins with
a Wilson 95% interval, crowns, and plays per match.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def one(args):
    ckpt, seed, opponent, start_elixir, search = args
    import torch
    torch.set_num_threads(1)
    from sim import match as M
    if start_elixir:
        M.PlayerState.__init__.__defaults__ = tuple(
            start_elixir if d == 5000 else d for d in M.PlayerState.__init__.__defaults__)
    from sim.gamedata import load_gamedata
    from sim.runner import (DECIDE_EVERY_MS, DECK_26, BrainPolicy, SimpleOpponent,
                            resolve_deck)
    from sim.spells import load_spells
    from sim.token_agent import TokenAgent
    base = load_gamedata(level=11)
    cards = resolve_deck(base, DECK_26)
    top_deck = list(DECK_26)
    if opponent == "scripted":
        from sim.meta_decks import deck_pool
        from sim.opponents import ScriptedOpponent
        pool = deck_pool(base)
        name, style, top_deck = pool[seed % len(pool)]
        for c in top_deck:
            cards.setdefault(c, base[c])
    match = M.Match(cards=cards, decks=(DECK_26, top_deck), seed=seed,
                    spells=dict(load_spells(level=11)))
    if search:
        from sim.search_agent import SearchAgent
        bottom = SearchAgent(ckpt, side=1, seed=seed)
    else:
        bottom = TokenAgent(ckpt, side=1, seed=seed)
    if opponent == "brain":
        top = BrainPolicy(cards, side=-1)
    elif opponent == "scripted":
        top = ScriptedOpponent(cards, side=-1, deck=top_deck, style=style, seed=seed + 1)
    else:
        top = SimpleOpponent(cards, side=-1, seed=seed + 1)
    bottom.reset()
    top.reset()
    nxt, turn = 0, 0
    while not match.finished and match.elapsed_ms < 310_000:
        match.step()
        if match.elapsed_ms >= nxt:
            nxt = match.elapsed_ms + DECIDE_EVERY_MS
            turn += 1
            for pol in ((bottom, top) if turn % 2 else (top, bottom)):
                pol.act(match)
    return {"seed": seed, "result": match.result,
            "crowns": [match.crowns_for(1), match.crowns_for(-1)],
            "plays": dict(bottom.plays), "secs": match.elapsed_ms / 1000}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--matches", type=int, default=60)
    ap.add_argument("--opponent", default="brain", choices=("brain", "scripted", "simple"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--start-elixir", type=int, default=0)
    ap.add_argument("--search", action="store_true", help="decision-time search (sim.search_agent)")
    args = ap.parse_args()
    t0 = time.time()
    jobs = [(args.ckpt, args.seed + i, args.opponent, args.start_elixir, args.search)
            for i in range(args.matches)]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(one, jobs))
    wins = sum(r["result"] == "bottom" for r in rows)
    draws = sum(r["result"] == "draw" for r in rows)
    lo, hi = wilson(wins, len(rows))
    plays = Counter()
    for r in rows:
        plays.update(r["plays"])
    cd = sum(r["crowns"][0] - r["crowns"][1] for r in rows) / len(rows)
    print(f"{Path(args.ckpt).parent.name}{' +search' if args.search else ''} vs {args.opponent}: {wins}/{len(rows)} wins "
          f"({wins / len(rows):.1%}, 95% [{lo:.1%}, {hi:.1%}]), {draws} draws, "
          f"crown diff {cd:+.2f}, plays/match {sum(plays.values()) / len(rows):.1f} "
          f"({time.time() - t0:.0f}s)")
    print("  card mix:", ", ".join(f"{k} {v}" for k, v in plays.most_common()))
    out = ROOT / "reports" / "tokens_eval.jsonl"
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ckpt": args.ckpt, "opponent": args.opponent, "search": args.search,
                            "wins": wins, "n": len(rows), "ci": [lo, hi],
                            "crown_diff": cd, "plays": dict(plays)}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
