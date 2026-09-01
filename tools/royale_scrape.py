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

**`/data/replay` is blocked as of 2026-09-01, and no amount of header-fiddling
gets past it.** Measured against a live, signed-in session whose clearance was
crawling every other page fine:

    curl, as the scraper sends it      403
    curl + Referer                     403
    curl + Referer + X-Requested-With  403
    the browser's own fetch()          403, a Cloudflare challenge page
    a real page navigation             403, a Cloudflare challenge page

RoyaleAPI protects that one endpoint far more strictly than the rest of the
site, and the challenge does not self-solve on navigation the way the site's
front page does. Defeating it is the thing this tool refuses to do, so replay
timelines are simply unavailable through here.

What still works is everything up to that point, and it is worth running on its
own: the crawl lists battles for every rated player on the archetype, and each
row carries both decks, both crown counts, the result, and the player's rating.
One sitting produced 1,467 top-ladder 2.6 games and 573 distinct opponent
decks - the current meta, measured, with our win rate into each. See
`data/ladder/`.

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

# Consecutive 403s on the replay endpoint before concluding the Cloudflare pass
# is gone rather than that one request was unlucky.
CLEARANCE_STREAK = 25


def _norm(card: str) -> str:
    """Card name or deck slug -> a comparable key.

    The deck is slugs (`hog-rider`, `cannon-ev1`); the plays are RoyaleAPI's
    display names (`HogRider`). Evolution and champion suffixes are dropped so
    an Evo Cannon still matches a Cannon.
    """
    import re
    card = re.sub(r"-(?:ev\d+|hero)$", "", card.strip().lower())
    return re.sub(r"[^a-z0-9]", "", card)


def learn_side(battle: dict, plays: list) -> str:
    """Which side of this replay is the player we are trying to learn from.

    Half of every replay is the opponent, so this is the difference between
    learning 2.6 and learning whatever was played against it. The scraper's own
    convention is that the "team" side is the player whose battle log was
    walked, which is the one filtered to our archetype - but that is a comment
    in someone else's parser, and reading it wrong would silently train on the
    wrong half of every match.

    So it is derived instead: whichever side's cards actually match the deck we
    filtered on. Returns "" when the two sides cannot be told apart, which is
    the honest answer for a mirror.
    """
    deck = {_norm(c) for c in battle.get("team_deck", "").split(",") if c}
    if not deck:
        return ""
    hits: dict = {}
    totals: dict = {}
    for play in plays:
        side = str(play.get("side", ""))
        totals[side] = totals.get(side, 0) + 1
        if _norm(play.get("card", "")) in deck:
            hits[side] = hits.get(side, 0) + 1
    if not totals:
        return ""
    # The share of a side's plays that are cards from the filtered deck.
    scored = sorted(((hits.get(s, 0) / n, s) for s, n in totals.items()),
                    reverse=True)
    if len(scored) < 2:
        return scored[0][1]
    (best, side), (second, _) = scored[0], scored[1]
    # A mirror gives both sides the same deck and no way to choose; so does a
    # replay whose cards did not parse. Say so rather than guess.
    if best < 0.8 or best - second < 0.2:
        return ""
    return side


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
    # Which side to learn from, worked out per battle rather than assumed.
    fields = list(BATTLE_FIELDS) + ["learn_side"]
    outdir.mkdir(parents=True, exist_ok=True)
    battles_csv, plays_csv = outdir / "battles.csv", outdir / "plays.csv"
    fresh = not battles_csv.exists()
    n_plays = 0
    with battles_csv.open("a", newline="", encoding="utf-8") as bf, \
            plays_csv.open("a", newline="", encoding="utf-8") as pf:
        bw = csv.DictWriter(bf, fields, extrasaction="ignore")
        pw = csv.DictWriter(pf, PLAY_FIELDS, extrasaction="ignore")
        if fresh:
            bw.writeheader()
            pw.writeheader()
        for b, (stats, plays) in got:
            bw.writerow({**b, **stats, "plays": len(plays),
                         "learn_side": learn_side(b, plays)})
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


def patch_crawl_isolation() -> None:
    """Stop one bad page from ending the whole crawl, and tidy the pager URL.

    First sitting: 436 rated players found, top 150 taken, and then *zero*
    battles, with a single failure reported out of 150. The one failure was

        ClearanceExpired: 403 on /player/.../battles/history?before=...&&

    and it was fatal to everything else. `pipeline._fan` treats
    `ClearanceExpired` as "the Cloudflare pass is dead, stop the wave" and
    breaks, so the remaining 149 players were cancelled rather than fetched.
    That is the right instinct for an expired pass and the wrong one for a
    single page that happens to 403: the pass was one second old and the
    ratings boards had just been crawled with it.

    So a 403 is contained to the player it happened on, which keeps whatever
    pages already arrived and lets the crawl carry on. If the pass really has
    expired, every player fails and the run ends with nothing fetched, which
    reads clearly in the failure report either way.

    The `&&` is RoyaleAPI's own markup - the pager emits a trailing empty
    query separator - and is stripped because a URL ending in `&&` is the most
    likely reason that request was refused when its neighbours were not.
    """
    from royale import parse, pipeline
    from royale.transport import ClearanceExpired, RateLimited

    original_next = parse.next_history_page

    def next_page(html):
        href = original_next(html)
        return href.rstrip("&?") if href else href

    parse.next_history_page = next_page
    pipeline.parse.next_history_page = next_page

    original_battles = pipeline.player_battles

    def player_battles(curl, tag, max_pages=0, out=None):
        rows = [] if out is None else out
        try:
            return original_battles(curl, tag, max_pages, rows)
        except (ClearanceExpired, RateLimited):
            return rows          # keep this player's pages, spare the rest

    pipeline.player_battles = player_battles

    # The replay fetch is the *second* place a 403 ends the wave, and patching
    # only the first left the same bug one step later: a sitting listed 1,925
    # battles, found 1,414 on the archetype, then cancelled all of them on one
    # `403 on /data/replay` and wrote nothing.
    #
    # Contained the same way, but with a budget. A scattered 403 is transient
    # and worth stepping over; a genuinely expired pass fails every request
    # after it, and burning through 1,400 of those at four a second serves
    # nobody. Consecutive failures are what separates the two, so a run of them
    # re-raises and stops the wave, while an isolated one does not.
    budget = {"streak": 0}
    original_fetch = pipeline.fetch_replay

    def fetch_replay(curl, battle):
        try:
            got = original_fetch(curl, battle)
        except ClearanceExpired:
            budget["streak"] += 1
            if budget["streak"] >= CLEARANCE_STREAK:
                raise            # the pass really is gone; stop the wave
            raise RuntimeError(f"403 on one replay "
                               f"({budget['streak']} in a row)") from None
        budget["streak"] = 0
        return got

    pipeline.fetch_replay = fetch_replay
    pipeline.replays.__globals__["fetch_replay"] = fetch_replay


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
    patch_crawl_isolation()

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
    # The listing phase takes about ten minutes and produced nothing usable the
    # first time the replay phase died, so it is cached. A rerun reuses it and
    # goes straight to fetching.
    pending = args.out / "pending_battles.json"
    if pending.exists():
        import json
        cached = json.loads(pending.read_text(encoding="utf-8"))
        print(f"reusing {len(cached):,} listed battles from a previous sitting")
        rows = cached + [r for r in rows
                         if r["replay_tag"] not in
                         {c["replay_tag"] for c in cached}]
    args.out.mkdir(parents=True, exist_ok=True)
    import json as _json
    pending.write_text(_json.dumps(rows), encoding="utf-8")

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
