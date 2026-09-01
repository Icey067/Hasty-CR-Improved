"""Pull tick-accurate replay timelines for our archetype off RoyaleAPI.

Wraps `_references/clash-replay-scraper` and drives its pipeline directly
rather than its TUI, for two reasons: the TUI imports `readline`, which does
not exist on Windows, and an interactive picker cannot run unattended.

**A stale Cloudflare pass breaks `/data/replay`, and nothing else does.**

That one line took three wrong answers to reach, so the wrong ones are kept
here too. Every replay came back 403 while every other page kept working, and
it was written up first as "RoyaleAPI has blocked the endpoint - measured five
ways". All five ways ran through the same browser profile, so they agreed with
each other, and that agreement was read as confirmation. Then the cause was
guessed a second time, as cookie pollution from a persistent profile. Also
wrong.

What actually matters is the age of `cf_clearance`. A pass that is hours old
still gets served the ordinary pages and is refused by `/data/replay`
specifically, which is exactly the shape of a blocked endpoint. Calling
`Pages.renew()` before crawling fixes it: measured, 3 of 3 replays from the
same persistent profile that had just failed 25 in a row.

So the profile is persistent - the login survives between runs, and a long
series of sittings needs a person exactly once - and the pass is refreshed at
the start of every run.

Two things upstream does not survive on this machine, patched from here so a
re-clone cannot quietly drop them: `Curl.get` decodes curl's output with the
locale codec (cp1252 here, against UTF-8 from the server), and a single 403
inside `pipeline._fan` cancels the whole wave rather than the one request.

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


def renew_or_ask(pages, timeout_s: int = 600) -> bool:
    """Refresh the Cloudflare pass, waiting for a person if it is challenged.

    `Pages.renew()` assumes the interstitial self-solves. It usually does - but
    ask for a fresh pass often enough and Cloudflare escalates to the
    interactive checkbox, which is exactly what this tool must not solve on its
    own. Renewing before every chunk provoked that within seconds and killed a
    run outright, so renewals are now rare; when one is challenged anyway, the
    window is left open for a person rather than the run thrown away.
    """
    import time as _t
    try:
        pages.renew()
        return True
    except Exception:
        pass
    print()
    print("=" * 68)
    print("  Cloudflare wants the checkbox again. Tick 'Verify you are human'")
    print("  in the browser window; the crawl carries on by itself after that.")
    print(f"  Waiting up to {timeout_s // 60} minutes.")
    print("=" * 68)
    print(flush=True)
    deadline = _t.time() + timeout_s
    while _t.time() < deadline:
        # Wait for the challenge PAGE to be gone, not merely for a cookie to
        # exist. An earlier version took any cf_clearance it could see, and at
        # startup `pages.clearance` is empty - so the stale cookie already in
        # the profile satisfied it instantly, the run reported "cleared", and
        # then every request 403'd with the same dead pass it began with.
        try:
            title = pages.page.title()
        except Exception:
            title = "Just a moment"
        if not title.startswith("Just a moment"):
            for cookie in pages._ctx.cookies():
                if cookie["name"] == "cf_clearance" and cookie["value"]:
                    pages.clearance = cookie["value"]
                    print("  cleared, carrying on", flush=True)
                    return True
        _t.sleep(3.0)
    print("  nobody ticked it; stopping", flush=True)
    return False


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

    def _reset():
        budget["streak"] = 0

    globals()["budget_reset"] = _reset
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--players", type=int, default=20,
                    help="how many of the top-rated players on this archetype")
    ap.add_argument("--pages", type=int, default=3,
                    help="battle-history pages per player, 10 battles each; "
                         "0 walks every page a player has")
    ap.add_argument("--workers", type=int, default=16,
                    help="parallel REPLAY fetches. That endpoint is slow "
                         "rather than rate limited - measured at 7.7s a "
                         "request, so four workers gave 0.52/s against a 4/s "
                         "ceiling - so more workers is a near-linear win up "
                         "to that ceiling")
    ap.add_argument("--list-workers", type=int, default=4,
                    help="parallel players during the battle-listing phase, "
                         "which is a different problem and wants far fewer. "
                         "Listing paginates sequentially per player, so 16 "
                         "workers there simply rate-limits: raising this to "
                         "match --workers cut a crawl from about sixteen "
                         "battles a player to one, and the 429s were "
                         "swallowed as empty histories rather than reported")
    ap.add_argument("--max-renewals", type=int, default=6,
                    help="how many times to refresh the Cloudflare pass in one "
                         "run. Each refresh risks an interactive challenge, so "
                         "this is bounded rather than unlimited")
    ap.add_argument("--chunk", type=int, default=200,
                    help="replays fetched between refreshes of the Cloudflare "
                         "pass, and the unit of work written to disk")
    ap.add_argument("--ceiling", type=float, default=4.0,
                    help="requests per second the limiter may never exceed. "
                         "The upstream default probes upward until RoyaleAPI "
                         "answers 429; this keeps it well under that instead, "
                         "because the account being throttled is not mine")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "replays")
    ap.add_argument("--profile", type=Path,
                    default=ROOT / "tmp" / "royale_loop_profile",
                    help="persistent browser profile, so the login and the "
                         "Cloudflare pass survive between runs")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    patch_transport_encoding()
    patch_crawl_isolation()

    from royale import pipeline
    from royale.cookies import SESSION_COOKIE, Session
    from royale.limiter import Limiter
    from royale.transport import Curl, Pages

    # A persistent profile whose Cloudflare pass is refreshed every run.
    #
    # Two wrong explanations were written here before this one, so the finding
    # is stated plainly: what breaks /data/replay is a STALE clearance, not a
    # persistent profile and not cookie pollution. A profile carrying an
    # hours-old pass 403s on that endpoint while every other page still works,
    # which is what made it look like the endpoint had been blocked. Calling
    # `renew()` first fixes it, and then the profile is worth having - the
    # login survives between runs, so a long series of sittings needs a person
    # exactly once.
    from playwright.sync_api import sync_playwright
    args.profile.mkdir(parents=True, exist_ok=True)
    pages = Pages.__new__(Pages)
    pages._pw = sync_playwright().start()
    ctx = pages._pw.chromium.launch_persistent_context(
        user_data_dir=str(args.profile), channel="chrome", headless=False,
        args=["--disable-blink-features=AutomationControlled"])
    pages._browser, pages._ctx = ctx.browser, ctx
    pages.page = ctx.pages[0] if ctx.pages else ctx.new_page()
    # No resource filtering: Cloudflare's check needs the whole page, which is
    # the bug upstream fixed by moving its own filter to after clearance.
    pages._route = lambda route: route.continue_()
    pages.ua = pages.page.evaluate("navigator.userAgent")
    pages.clearance = ""
    if not renew_or_ask(pages):
        raise SystemExit("no Cloudflare pass; nothing fetched.")
    print("cloudflare pass ready", flush=True)

    jar = {c["name"]: c["value"] for c in ctx.cookies()}
    session = Session("profile", jar.get(SESSION_COOKIE, ""))
    curl = Curl(pages, session, Limiter(ceiling=args.ceiling))

    def signed_in(c) -> bool:
        # `logged_in` catches AuthError, but an unusable pass answers 403,
        # which the transport raises as ClearanceExpired and which escaped it
        # and killed the run.
        try:
            return c.logged_in()
        except Exception:
            return False

    if not signed_in(curl):
        print()
        print("  Not signed in - sign in to RoyaleAPI in the browser window.")
        print("  This is needed once; the profile keeps it afterwards.")
        print(flush=True)
        session = Session("profile", pages.login())
        curl = Curl(pages, session, Limiter(ceiling=args.ceiling))
        if not signed_in(curl):
            raise SystemExit("still not signed in; stopping rather than "
                             "crawling for nothing")
    print("signed in", flush=True)

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

    players, found_on = pipeline.rated_players(curl, decks,
                                               workers=args.list_workers,
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
                                         workers=args.list_workers,
                                         on_error=note)
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

    # Fetched in chunks, with the Cloudflare pass refreshed before each one
    # and every chunk written to disk as it lands.
    #
    # A pass expires on its own clock. The first long run fetched for 56
    # minutes, went stale partway, hit the consecutive-403 stop, and lost
    # everything - because upstream writes only once, at the end. Both halves
    # of that are fixed here: `renew()` between chunks keeps the pass young,
    # and an expiry now costs one chunk rather than the whole sitting.
    total_plays = 0
    written = 0
    battles_csv = plays_csv = None
    renewals = 0
    for start_at in range(0, len(rows), args.chunk):
        chunk = rows[start_at:start_at + args.chunk]
        budget_reset()
        try:
            got = pipeline.replays(curl, chunk, workers=args.workers,
                                   on_error=note)
        except (ClearanceExpired, RateLimited):
            got = []

        # A chunk that returns nothing means the pass has gone stale, which is
        # the one thing that breaks this endpoint. Renew THEN, not before every
        # chunk: asking Cloudflare for a fresh pass repeatedly makes it escalate
        # to an interactive challenge, and a run that renewed each time aborted
        # on its very first chunk with nothing written.
        if not got and renewals < args.max_renewals:
            renewals += 1
            print(f"  chunk empty - refreshing the pass ({renewals})", flush=True)
            if not renew_or_ask(pages):
                break
            budget_reset()
            try:
                got = pipeline.replays(curl, chunk, workers=args.workers,
                                       on_error=note)
            except (ClearanceExpired, RateLimited):
                got = []

        if got:
            battles_csv, plays_csv, n = append_csv(args.out, got)
            total_plays += n
            written += len(got)
            print(f"  {written:,}/{len(rows):,} replays, "
                  f"{total_plays:,} card plays "
                  f"({time.time() - started:.0f}s)", flush=True)
        else:
            print("  chunk still empty after a refresh; stopping", flush=True)
            break

    if not written:
        _report(failures, len(rows))
        raise SystemExit("no replays came back; nothing written.")
    elapsed = time.time() - started
    size = sum(p.stat().st_size for p in (battles_csv, plays_csv))
    print(f"\n{written:,} replays, {total_plays:,} card plays "
          f"in {elapsed / 60:.1f} min")
    print(f"  {battles_csv}")
    print(f"  {plays_csv}")
    print(f"  {size / 1024:.0f} KB total, "
          f"{size / max(1, written) / 1024:.1f} KB per replay, "
          f"{elapsed / max(1, written):.2f}s per replay")
    print(f"  limiter: {curl.limiter}")
    pages.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
