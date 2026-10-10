"""How much of IL_Replay our engine can play, and what blocks the rest.

Writes reports/il_replay_coverage.md. A match counts as loadable when all 16
cards (both decks, base form) have an engine row; evolution and hero rows are
counted separately because the engine can still play the base card.

    py -3.12 tools/il_replay_coverage.py [--parts N]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.replay_corpus import OPPONENT, TEAM, iter_corpus  # noqa: E402

HOG26 = frozenset({"cannon", "fireball", "hog_rider", "ice_golemite",
                   "ice_spirits", "musketeer", "skeletons", "log"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "il_replay_coverage.md")
    args = ap.parse_args()

    n = loadable = hog_sides = hog_loadable = 0
    no_plays = 0
    blockers = Counter()
    missing_forms = Counter()
    modes = Counter()
    orient_check = Counter()
    for rec in iter_corpus(limit_parts=args.parts):
        n += 1
        modes[rec.game_mode] += 1
        ok = rec.loadable()
        loadable += ok
        if not rec.plays:
            no_plays += 1
        for key in rec.missing():
            blockers[key] += 1
        for side in (TEAM, OPPONENT):
            for c in rec.decks[side]:
                if c.base_engine and c.form != "base" and not c.engine:
                    missing_forms[c.key] += 1
            if rec.base_deck(side) == HOG26:
                hog_sides += 1
                hog_loadable += ok
        # Orientation sanity: after the rotation the team's troops should sit
        # on the bottom half (y >= 16 tiles) and the opponent's on the top.
        for p in rec.plays:
            if p.kind != "play_card" or p.engine is None:
                continue
            from sim.replay_corpus import _is_troop_like
            if not _is_troop_like(p.engine):
                continue
            own_half = (p.y >= 16_000) == (p.side == TEAM)
            orient_check["own half" if own_half else "other half"] += 1

    pct = lambda a, b: f"{100 * a / max(1, b):.1f}%"
    lines = [
        "# IL_Replay coverage in our engine",
        "",
        f"Matches read: **{n:,}** ({'all parts' if not args.parts else f'{args.parts} parts'})",
        "",
        "| measure | count | share |",
        "|---|---|---|",
        f"| all 16 cards load (base form) | {loadable:,} | {pct(loadable, n)} |",
        f"| exact Hog 2.6 sides | {hog_sides:,} | {pct(hog_sides, 2 * n)} of sides |",
        f"| Hog 2.6 sides in a loadable match | {hog_loadable:,} | {pct(hog_loadable, hog_sides)} |",
        f"| matches with no plays recorded | {no_plays:,} | {pct(no_plays, n)} |",
        f"| troop plays on the player's own half after rotation | {orient_check['own half']:,} | "
        f"{pct(orient_check['own half'], sum(orient_check.values()))} |",
        "",
        "Own-half share is the frame check: a player deploys troops on its own half far more "
        "often than past the river, so a share near 50% would mean the rotation is wrong.",
        "",
        "## Cards with no engine row, by matches they block",
        "",
        "| card | matches |",
        "|---|---|",
        *[f"| {k} | {v:,} |" for k, v in blockers.most_common(40)],
        "",
        "## Evolution / hero forms the engine lacks (base card still plays)",
        "",
        "| form | deck slots |",
        "|---|---|",
        *[f"| {k} | {v:,} |" for k, v in missing_forms.most_common(30)],
        "",
        "## Game modes",
        "",
        *[f"- {k or '(blank)'}: {v:,}" for k, v in modes.most_common()],
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:14]))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
