#!/usr/bin/env bash
# Keep pulling replays until told to stop.
#
# Each pass re-lists the archetype and fetches whatever is not already on disk,
# so the corpus grows as these players keep playing. The Cloudflare pass is
# refreshed at the start of every pass and the login lives in the profile, so
# no sitting after the first needs a person.
#
# Stop it by creating the stop file:  touch tmp/STOP_SCRAPE
cd "C:/Users/aksha/Downloads/HastyCR" || exit 1
PY=".venvs/scraper/Scripts/python.exe"
PASS=0
while [ ! -f tmp/STOP_SCRAPE ]; do
  PASS=$((PASS + 1))
  echo "=== pass $PASS  $(date '+%H:%M:%S') ==="
  "$PY" -u tools/royale_scrape.py --players 400 --pages 0 --out data/replays 2>&1 \
    | grep -vE "^\s+at |node:|coreBundle|EPIPE|errno:|syscall|throw er|Emitted|Node\.js"
  if [ -f data/replays/battles.csv ]; then
    echo "    corpus now: $(($(wc -l < data/replays/battles.csv) - 1)) replays"
  fi
  # A pass that finds nothing new means these players have not played since;
  # wait longer before asking again rather than spinning on the same battles.
  [ -f tmp/STOP_SCRAPE ] && break
  sleep 900
done
echo "stopped: tmp/STOP_SCRAPE present"
