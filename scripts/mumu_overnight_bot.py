from __future__ import annotations

import argparse
import io
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "ClashRoyaleBuildABot"))

from clashroyalebuildabot.detectors.detector import Detector  # noqa: E402
from clashroyalebuildabot.namespaces.cards import Cards  # noqa: E402


DECK = [
    Cards.CANNON, Cards.FIREBALL, Cards.HOG_RIDER, Cards.ICE_GOLEM,
    Cards.ICE_SPIRIT, Cards.MUSKETEER, Cards.SKELETONS, Cards.THE_LOG,
]
CARD_CENTRES = ((334, 1711), (538, 1711), (742, 1711), (946, 1711))
HOME_BATTLE = (540, 1490)


def adb(path: Path, serial: str, *args: str, binary: bool = False):
    out = subprocess.run(
        [str(path), "-s", serial, *args], capture_output=True, check=True
    ).stdout
    return out if binary else out.decode(errors="replace")


_fast_cap = None

# Everything downstream - tower_hp, popup_guard, the card classifier, every
# tap constant - was calibrated against 1080x1920. Rather than recalibrate all
# of it, the device resolution is normalised here at the boundary: frames are
# upscaled to 1080x1920 on the way in and taps are scaled back to the device's
# real resolution on the way out.
#
# Why bother: a raw screencap is uncompressed, so its cost is entirely the
# transfer. Measured on this machine, a 1080x1920 frame is 8.3MB and takes
# 406ms, while 540x960 is 2.07MB and takes 117ms - and capture was 90% of the
# whole decision loop. Running the emulator at 540x960 therefore roughly
# triples the bot's reaction rate, and the upscale costs a few milliseconds.
NATIVE_W, NATIVE_H = 1080, 1920
_device_size = (NATIVE_W, NATIVE_H)


def _normalise(full: Image.Image, record_size: bool = True) -> Image.Image:
    global _device_size
    if record_size:
        _device_size = full.size
    if full.size != (NATIVE_W, NATIVE_H):
        full = full.resize((NATIVE_W, NATIVE_H), Image.Resampling.BILINEAR)
    return full


# Frames from an on-device H.264 stream instead of one screencap per frame.
#
# The reasoning above still holds for screencap - transfer is the whole cost,
# so a smaller frame is a faster one - but it is an argument about the wrong
# mechanism. Measured on this machine against MuMu instance 3:
#
#     screencap (raw)          218ms   1080x1920
#     FastScreenCap            130ms    540x960, upscaled back to 1080x1920
#     DeviceStream.grab()     0.00ms   1080x1920, 52 frames/sec
#
# The stream decodes continuously on a worker thread, so `grab` is a memory
# read rather than a transfer, and it delivers the calibrated resolution
# instead of a bilinear upscale of half of it. That second part matters beyond
# speed: the detector and the card reader had been working from an upscaled
# 540x960 image, and this engine has just had a run of units it could not see.
#
# Opt-in for now. It changes the frame every part of perception reads, and it
# has not yet been through a live match - set HASTYCR_STREAM_CAPTURE=1, or pass
# --stream-capture to cr_bot.py, to try it. Any failure falls back to the
# existing path rather than stopping the bot.
_stream = None
_stream_failed = False


_device_res_cache: tuple | None = None


def _device_resolution(path: Path, serial: str) -> tuple:
    """The device's real pixel size, asked once and remembered."""
    global _device_res_cache
    if _device_res_cache is None:
        try:
            out = adb(path, serial, "shell", "wm", "size")
            text = out if isinstance(out, str) else out.decode("utf-8", "replace")
            size = text.strip().rsplit(":", 1)[-1].strip()
            width, height = (int(n) for n in size.lower().split("x"))
            _device_res_cache = (width, height)
        except Exception:
            _device_res_cache = (NATIVE_W, NATIVE_H)
    return _device_res_cache


def _stream_frame(path: Path, serial: str):
    """A frame from the H.264 stream, or None to fall back."""
    global _stream, _stream_failed
    import os
    if _stream_failed or os.environ.get("HASTYCR_STREAM_CAPTURE") != "1":
        return None
    if _stream is None:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from studio.devicecap import DeviceStream
            _stream = DeviceStream(str(path), serial, size=f"{NATIVE_W}x{NATIVE_H}")
            if not _stream.ready or not _stream.start():
                raise RuntimeError(_stream.last_error or "stream would not start")
        except Exception as exc:
            _stream_failed = True
            print(f"stream capture unavailable, using screencap: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            return None
    array = _stream.grab()
    if array is None:
        return None                      # still warming up; screencap this one
    # `_normalise` records the frame size as the device size, and `tap` scales
    # taps back down by it. Asking the stream for 1080x1920 on a 540x960 device
    # therefore switched tap scaling off and every tap would have landed at
    # double its coordinate. The device resolution is asked for once, here,
    # rather than inferred from a frame we chose the size of.
    global _device_size
    _device_size = _device_resolution(path, serial)
    # BGRA from the decoder; PIL wants RGB.
    frame = Image.fromarray(array[:, :, 2::-1])
    if frame.size != (NATIVE_W, NATIVE_H):
        frame = frame.resize((NATIVE_W, NATIVE_H), Image.Resampling.BILINEAR)
    return frame


def capture(path: Path, serial: str):
    frame = _stream_frame(path, serial)
    if frame is not None:
        full = _normalise(frame, record_size=False)
        return full, full.resize((368, 652))
    global _fast_cap
    if _fast_cap is None:
        try:
            from screencap_fast import FastScreenCap
            _fast_cap = FastScreenCap(path, serial)
        except Exception:
            _fast_cap = None
    if _fast_cap is not None:
        full = _fast_cap.capture_frame()
        if full is not None:
            full = _normalise(full)
            return full, full.resize((368, 652))
    raw = adb(path, serial, "exec-out", "screencap", "-p", binary=True)
    full = _normalise(Image.open(io.BytesIO(raw)).convert("RGB"))
    return full, full.resize((368, 652))


def tap(path: Path, serial: str, xy) -> None:
    """Tap, in 1080x1920 coordinates whatever the device is actually running."""
    width, height = _device_size
    x = round(xy[0] * width / NATIVE_W)
    y = round(xy[1] * height / NATIVE_H)
    adb(path, serial, "shell", "input", "tap", str(x), str(y))


def home_guard(full: Image.Image, state) -> bool:
    if state.screen.name != "lobby":
        return False
    pixels = np.asarray(full)
    roi = pixels[1380:1600, 350:730]
    r, g, b = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
    orange = (r > 180) & (g > 80) & (g < 210) & (b < 100) & (r > g * 1.25)
    return float(orange.mean()) > 0.35


def battle_guard(state) -> bool:
    if state.screen.name != "in_game":
        return False
    hp = (
        state.numbers.left_ally_princess_hp.number,
        state.numbers.right_ally_princess_hp.number,
        state.numbers.left_enemy_princess_hp.number,
        state.numbers.right_enemy_princess_hp.number,
    )
    valid_hand = sum(card.name != "blank" for card in state.cards[1:])
    elixir = getattr(getattr(state.numbers, "elixir", None), "number", 0) or 0

    # Tower bars are the wrong signal for "am I in a battle", and it took two
    # abandoned matches to see why. The rule was two of four princess bars above
    # 10%. Captured mid-match with the frame saved: both our towers destroyed
    # (rubble on screen, king at 5348) and theirs on 282 and 493 hitpoints - 8%
    # and 15% of a 3346 tower. Every bar at or under the threshold, all of them
    # correct, and the guard concluded there was no battle. The bot stopped
    # playing with a king tower to defend and two nearly-dead towers to shoot at.
    #
    # A hand of real cards on an `in_game` screen is the honest test. The bars
    # stay only as a weak sanity check - any tower still showing, or any elixir
    # in the bar - which a Collection preview does not have.
    return (valid_hand >= 3
            and (any(value > 0.0 for value in hp) or elixir > 0))


def detect(detector: Detector, path: Path, serial: str):
    full, small = capture(path, serial)
    return full, detector.run(small)


def validate_battle(detector, path, serial, slot=None, card_name=None, count=1):
    latest = None
    for _ in range(count):
        full, state = detect(detector, path, serial)
        if state is None or not battle_guard(state):
            return None
        if slot is not None and card_name is not None:
            if state.cards[slot + 1].name != card_name:
                # Dynamically locate the card if it shifted slots in hand
                found_slot = None
                for s in state.ready:
                    if 0 <= s < 4 and state.cards[s + 1].name == card_name:
                        found_slot = s
                        break
                if found_slot is not None:
                    slot = found_slot
                else:
                    return None
        latest = (full, state, slot if slot is not None else 0)
    return latest


def choose(state):
    available = []
    for slot in state.ready:
        card = state.cards[slot + 1]
        if card.name != "blank" and state.numbers.elixir.number >= card.cost:
            available.append((slot, card))
    if not available:
        return None
    by_name = {card.name: (slot, card) for slot, card in available}
    enemies = list(state.enemies)
    elixir = state.numbers.elixir.number
    lane_x = 300
    if enemies:
        lane_x = 300 if sum(e.position.tile_x for e in enemies) / len(enemies) < 9 else 780

    if enemies:
        deep = [enemy for enemy in enemies if 31 - enemy.position.tile_y >= 20]
        order = ["cannon", "musketeer", "ice_golem", "skeletons", "ice_spirit"]
        if len(enemies) >= 2 or deep:
            order = ["cannon", "musketeer", "fireball", "the_log", "ice_golem", "skeletons", "ice_spirit"]
        name = next((item for item in order if item in by_name), None)
        if name is None:
            return None
        targets = {
            "cannon": (540, 1130), "the_log": (lane_x, 1090),
            "fireball": (lane_x, 930),
        }
        target = targets.get(name, (lane_x, 1230))
    elif "hog_rider" in by_name and elixir >= 5:
        name, target = "hog_rider", (300 if lane_x < 540 else 780, 880)
    elif elixir >= 9:
        order = ["skeletons", "ice_spirit", "ice_golem", "musketeer", "cannon"]
        name = next((item for item in order if item in by_name), None)
        if name is None:
            return None
        targets = {
            "cannon": (540, 1130),
        }
        target = targets.get(name, (lane_x, 1430))
    else:
        return None
    slot, card = by_name[name]
    return slot, card.name, target, elixir, len(enemies)


def main() -> int:
    parser = argparse.ArgumentParser(description="Guarded MuMu ladder heuristic")
    parser.add_argument("--adb", required=True, type=Path)
    parser.add_argument("--serial", default="127.0.0.1:16480")
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--log", type=Path, default=ROOT / "tmp" / "live" / "overnight-bot.log")
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    detector = Detector(DECK)
    deadline = time.monotonic() + args.hours * 3600
    last_action = 0.0
    last_nav = 0.0
    actions = matches = 0

    def log(message):
        line = f"{datetime.now():%H:%M:%S} {message}"
        print(line, flush=True)
        with args.log.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    log("START guarded ladder runner; unknown screens are no-click")
    while time.monotonic() < deadline:
        try:
            full, state = detect(detector, args.adb, args.serial)
            if state is None:
                time.sleep(1)
                continue
            now = time.monotonic()
            screen = state.screen.name

            if home_guard(full, state) and now - last_nav > 8:
                full2, state2 = detect(detector, args.adb, args.serial)
                if state2 is not None and home_guard(full2, state2):
                    tap(args.adb, args.serial, HOME_BATTLE)
                    last_nav = now
                    matches += 1
                    log(f"QUEUE match={matches}")
                time.sleep(1)
                continue

            if screen in {"end_of_game", "bypass_end_of_game"} and now - last_nav > 5:
                _, state2 = detect(detector, args.adb, args.serial)
                if state2 is not None and state2.screen.name == screen:
                    xy720 = state.screen.click_xy
                    tap(args.adb, args.serial, (round(xy720[0] * 1.5), round(xy720[1] * 1.5)))
                    last_nav = now
                    log(f"RESULT exit screen={screen}")
                time.sleep(1)
                continue

            if not battle_guard(state) or now - last_action < 3.5:
                time.sleep(0.8)
                continue
            decision = choose(state)
            if decision is None:
                time.sleep(0.8)
                continue
            slot, card_name, target, elixir, enemies = decision

            checked = validate_battle(detector, args.adb, args.serial, slot, card_name, count=2)
            if checked is None:
                log("BLOCK pre-card battle validation failed")
                time.sleep(1)
                continue
            tap(args.adb, args.serial, CARD_CENTRES[slot])
            time.sleep(0.15)
            # Selecting a card adds a highlight that can change its image hash. Re-check
            # the battle/tower/hand guard, but do not require the highlighted slot hash.
            checked = validate_battle(detector, args.adb, args.serial, count=1)
            if checked is None:
                log("BLOCK pre-target battle validation failed")
                time.sleep(1)
                continue
            tap(args.adb, args.serial, target)
            actions += 1
            last_action = time.monotonic()
            log(f"PLAY #{actions} {card_name} target={target} elixir={elixir} enemies={enemies}")
            time.sleep(0.8)
        except Exception as exc:
            log(f"ERROR {type(exc).__name__}: {exc}")
            time.sleep(2)
    log(f"STOP deadline matches={matches} actions={actions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
