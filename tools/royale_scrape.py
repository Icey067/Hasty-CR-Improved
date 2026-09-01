"""Pull tick-accurate replay timelines for our archetype off RoyaleAPI.

Wraps `_references/clash-replay-scraper` and drives its pipeline directly
rather than its TUI, for two reasons: the TUI imports `readline`, which does
not exist on Windows, and an interactive picker cannot run unattended.

**One step needs a person, once.** RoyaleAPI now serves an interactive
Cloudflare Turnstile - a "Verify you are human" checkbox - and it does not
self-solve the way the upstream README describes. Measured against both
bundled Chromium and real Chrome: the checkbox sits there indefinitely.
Solving it programmatically is the one thing it exists to prevent, so this
does not try. A browser window opens and waits; everything after is automatic.

Both gates are handled in that one window. `/data/replay` carries the
play-by-play and is login-gated, so a session is needed too - and rather than
prising it out of an installed browser's cookie store (Chrome's app-bound
encryption wants admin, and a running Chrome holds the file open), this reads
it straight out of the window you just used. The profile is persistent, so the
login survives and later runs need only the checkbox.

Why this is worth the trouble: `tools/vod/plays.py` infers card plays from YOLO
detections at a few frames a second, documents its own over-count, and cost
45GB of video to produce a few thousand noisy events. This returns the plays
themselves, already tick-accurate, at roughly 10KB a match.

    python tools/royale_scrape.py --players 20 --pages 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRAPER = ROOT / "_references" / "clash-replay-scraper"
if str(SCRAPER) not in sys.path:
    sys.path.insert(0, str(SCRAPER))

# Our deck. Evolution cards carry `-ev1`, champions `-hero`; only true
# variations of these eight are kept, so a substituted card is not a match.
SEED = "cannon-ev1,fireball,hog-rider,ice-golem,ice-spirit,musketeer,skeletons,the-log"

# How long to leave the browser open for the checkbox before giving up.
TURNSTILE_WAIT_S = 600


def existing_tags(outdir: Path) -> set:
    """Replay tags already downloaded, so a later sitting does not refetch."""
    import csv
    battles = outdir / "battles.csv"
    if not battles.exists():
        return set()
    with battles.open(newline="", encoding="utf-8") as handle:
        return {row["replay_tag"] for row in csv.DictReader(handle)
                if row.get("replay_tag")}


def append_csv(outdir: Path, got: list) -> tuple:
    """`pipeline.write_csv`, but adding to what is already there.

    Upstream opens both files in "w" because it assumes one crawl. A full pass
    over this archetype takes several sittings - the Cloudflare pass expires
    and only a person can renew it - so the first sitting's replays have to
    survive the second.
    """
    import csv

    from royale.pipeline import BATTLE_FIELDS, PLAY_FIELDS
    outdir.mkdir(parents=True, exist_ok=True)
    battles_csv, plays_csv = outdir / "battles.csv", outdir / "plays.csv"
    fresh = not battles_csv.exists()
    n_plays = 0
    with battles_csv.open("a", newline="", encoding="utf-8") as bf, \
            plays_csv.open("a", newline="", encoding="utf-8") as pf:
        bw = csv.DictWriter(bf, BATTLE_FIELDS, extrasaction="ignore")
        pw = csv.DictWriter(pf, PLAY_FIELDS, extrasaction="ignore")
        if fresh:
            bw.writeheader()
            pw.writeheader()
        for b, (stats, plays) in got:
            bw.writerow({**b, **stats, "plays": len(plays)})
            pw.writerows(plays)
            n_plays += len(plays)
    return battles_csv, plays_csv, n_plays


def patch_transport_encoding() -> None:
    """Make the vendored `curl` shell-out decode as UTF-8.

    `royale.transport.Curl.get` runs curl with `text=True` and no encoding, so
    Python decodes with the locale codec. That is cp1252 on this machine and
    RoyaleAPI serves UTF-8, which kills the very first fetch on a player name
    with a non-Latin character:

        UnicodeDecodeError: 'charmap' codec can't decode byte 0x81

    Patched here rather than in `_references/` so that a re-clone of the
    upstream repo does not quietly take the fix away again.
    """
    import subprocess

    from royale import transport

    real_run = subprocess.run

    def run_utf8(*a, **kw):
        if kw.get("text") and "encoding" not in kw:
            kw["encoding"] = "utf-8"
            kw["errors"] = "replace"
        return real_run(*a, **kw)

    transport.subprocess.run = run_utf8


def _report(failures: dict, attempted: int) -> None:
    """What `_fan` swallowed, grouped, so a silent crawl explains itself."""
    if not failures:
        return
    total = sum(len(v) for v in failures.values())
    print()
    print(f"  {total} of {attempted} fetches failed:")
    for reason, items in sorted(failures.items(), key=lambda kv: -len(kv[1])):
        print(f"    {len(items):>5}x  {reason}")
        print(f"           e.g. {', '.join(items[:3])}")


def wait_for_clearance(pages, timeout_s: int = TURNSTILE_WAIT_S) -> str:
    """Block until a person clears Cloudflare and is signed in, in that window.

    Returns the RoyaleAPI session cookie. Both cookies are read from the live
    context rather than from an installed browser's store: that store is
    DPAPI-sealed on current Chrome and locked while Chrome is running, and
    neither obstacle is worth fighting when the window is right here.
    """
    from royale.cookies import SESSION_COOKIE, Session
    from royale.limiter import Limiter
    from royale.transport import Curl
    print("\n" + "=" * 70)
    print("  A browser window has opened on royaleapi.com. Two things:")
    print("    1. tick the 'Verify you are human' box")
    print("    2. SIGN IN, in that window. Signing in elsewhere does not")
    print("       count: this is a separate browser profile, and /data/replay")
    print("       is login-gated, so an anonymous run returns no timelines.")
    print(f"  Then leave it alone. Waiting up to {timeout_s // 60} minutes.")
    print("=" * 70 + "\n", flush=True)
    started = time.time()
    told: set = set()
    jar: dict = {}
    while time.time() - started < timeout_s:
        jar = {c["name"]: c["value"] for c in pages._ctx.cookies()}
        if "cf_clearance" in jar and "cf_clearance" not in told:
            told.add("cf_clearance")
            print(f"  cloudflare cleared ({time.time() - started:.0f}s)",
                  flush=True)
        if "cf_clearance" in jar and SESSION_COOKIE in jar:
            pages.clearance = jar["cf_clearance"]
            # The session cookie is NOT proof of a login: RoyaleAPI hands one
            # to anonymous visitors too. A first run read its presence as
            # "signed in", sailed past here, and every replay came back
            # `success: false` - which the pipeline reports as "login
            # expired?" - because the session was simply anonymous the whole
            # time. `/me` answers 200 only for a real login, so ask that.
            probe = Curl(pages, Session("probe", jar[SESSION_COOKIE]),
                         Limiter(ceiling=2.0))
            # `logged_in` catches AuthError, but an anonymous /me answers 403,
            # which is raised as ClearanceExpired and escapes it - so the probe
            # crashed rather than reporting "not signed in".
            try:
                signed_in = probe.logged_in()
            except Exception:
                signed_in = False
            if signed_in:
                print(f"  signed in ({time.time() - started:.0f}s)", flush=True)
                return jar[SESSION_COOKIE]
            if "anon" not in told:
                told.add("anon")
                print("  cloudflare cleared, but this session is anonymous - "
                      "sign in in that window", flush=True)
        time.sleep(3.0)
    missing = "the checkbox" if "cf_clearance" not in jar else "a sign-in"
    raise SystemExit(f"gave up waiting for {missing}; nothing was fetched.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--players", type=int, default=20,
                    help="how many of the top-rated players on this archetype")
    ap.add_argument("--pages", type=int, default=3,
                    help="battle-history pages per player, 10 battles each; "
                         "0 walks every page a player has")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ceiling", type=float, default=4.0,
                    help="requests per second the limiter may never exceed. "
                         "The upstream default probes upward until RoyaleAPI "
                         "answers 429; this keeps it well under that instead, "
                         "because the account being throttled is not mine")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "replays")
    ap.add_argument("--profile", type=Path,
                    default=ROOT / "tmp" / "royale_profile",
                    help="persistent browser profile, so the login and the "
                         "Cloudflare pass survive between runs")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    patch_transport_encoding()

    from royale import pipeline
    from royale.cookies import Session
    from royale.limiter import Limiter
    from royale.transport import Curl, Pages

    # Our own persistent profile rather than an installed browser's. The login
    # and the Cloudflare pass both live in it, so a second run needs at most
    # the checkbox, and often not even that.
    args.profile.mkdir(parents=True, exist_ok=True)
    pages = Pages.__new__(Pages)      # skip __init__: it calls renew(), which
    pages._pw = sync_playwright().start()           # blocks on the Turnstile
    ctx = pages._pw.chromium.launch_persistent_context(
        user_data_dir=str(args.profile), channel="chrome", headless=False,
        args=["--disable-blink-features=AutomationControlled"])
    pages._browser = ctx.browser
    pages._ctx = ctx
    pages.page = ctx.pages[0] if ctx.pages else ctx.new_page()
    pages.ua = pages.page.evaluate("navigator.userAgent")
    pages.clearance = ""
    pages.page.goto("https://royaleapi.com/", wait_until="domcontentloaded",
                    timeout=60_000)

    session = Session("the window you just used", wait_for_clearance(pages))
    curl = Curl(pages, session, Limiter(ceiling=args.ceiling))

    started = time.time()
    decks = pipeline.similar_decks(curl, SEED)
    print(f"{len(decks)} variations of the seed deck")

    # `pipeline._fan` drops an item that raises and carries on, which is right
    # for a long crawl and invisible without a hook: a first run reported "19
    # battles" from 150 players and gave no hint that ~148 fetches had failed.
    failures: dict = {}

    def note(item, exc):
        key = f"{type(exc).__name__}: {str(exc)[:90]}"
        failures.setdefault(key, []).append(str(item)[:20])

    players, found_on = pipeline.rated_players(curl, decks, workers=args.workers,
                                               on_error=note)
    order = sorted(players, key=lambda t: -int(players[t]["rating"] or 0))
    keep = {t: players[t] for t in order[:args.players]}
    print(f"{len(players)} rated players; taking the top {len(keep)} "
          f"(rating {players[order[0]]['rating']} down to "
          f"{players[order[min(len(keep), len(order)) - 1]]['rating']})")

    # The Cloudflare pass expires on its own clock and only a person can renew
    # it, so an expiry mid-crawl must keep what already landed rather than
    # throw the run away. `pipeline._fan` stops the wave and returns partial
    # results for exactly this reason; these guards cover the single fetches
    # around it, which do not go through it.
    from royale.transport import ClearanceExpired, RateLimited

    try:
        rows, dropped = pipeline.battles(curl, keep, found_on, SEED,
                                         max_pages=args.pages,
                                         workers=args.workers, on_error=note)
    except (ClearanceExpired, RateLimited) as exc:
        raise SystemExit(f"stopped while listing battles: {exc}\n"
                         "Rerun and tick the checkbox again.")
    # The second return is what the archetype filter *removed*, not what was
    # seen. Reporting it as "seen" hid how little the crawl had fetched.
    print(f"{len(rows) + dropped} battles fetched, {len(rows)} on this "
          f"archetype, {dropped} dropped as a different deck")
    _report(failures, len(keep))
    if not rows:
        raise SystemExit("nothing to fetch")

    # Sessions accumulate. The Cloudflare pass expires on its own clock and
    # only a person can renew it, so a full crawl of this archetype is several
    # ordinary sittings rather than one long one - and upstream's `write_csv`
    # opens both files in "w", so a second sitting would have erased the first.
    # Anything already on disk is skipped and new rows are appended.
    already = existing_tags(args.out)
    if already:
        before = len(rows)
        rows = [r for r in rows if r["replay_tag"] not in already]
        print(f"{len(already):,} replays already on disk; skipping "
              f"{before - len(rows):,} of them, {len(rows):,} new to fetch")
    if not rows:
        raise SystemExit("nothing new - every battle found is already downloaded")

    try:
        got = pipeline.replays(curl, rows, workers=args.workers,
                               on_error=note)
    except (ClearanceExpired, RateLimited) as exc:
        print(f"\nstopped early: {exc}")
        got = []
    if not got:
        _report(failures, len(rows))
        raise SystemExit("no replays came back; nothing written.")
    args.out.mkdir(parents=True, exist_ok=True)
    battles_csv, plays_csv, n_plays = append_csv(args.out, got)

    elapsed = time.time() - started
    size = sum(p.stat().st_size for p in (battles_csv, plays_csv))
    print(f"\n{len(got)} replays, {n_plays:,} card plays in {elapsed / 60:.1f} min")
    print(f"  {battles_csv}")
    print(f"  {plays_csv}")
    print(f"  {size / 1024:.0f} KB total, "
          f"{size / max(1, len(got)) / 1024:.1f} KB per replay, "
          f"{elapsed / max(1, len(got)):.2f}s per replay")
    print(f"  limiter: {curl.limiter}")
    pages.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
