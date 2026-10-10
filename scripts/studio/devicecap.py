"""Capture the emulator at the device's own resolution, not the screen's.

The window mirror is bounded by the monitor. MuMu renders Android at 1080x1920
but the desktop only has 1080 rows to show it in, so the visible surface is
560x996 and the studio then upscales that into a 1080x1920 canvas. Everything
downstream inherits the blur - it is a 1.3x upscale of a source that was
already half the target's height - and no encoder setting recovers detail that
was never captured.

`screenrecord` encodes on the device, before any of that. It hands back true
1080x1920 H.264, which PyAV decodes here into the same BGRA frames the window
grabber produces. Same interface, four times the pixels.

The costs, stated plainly:

* Latency. The device buffers before emitting, so the mirror runs roughly a
  second behind the game. Irrelevant for recording, wrong for anything making
  decisions - the bot keeps using its own capture path regardless.
* `screenrecord` stops itself at 180 seconds. The reader restarts it, which
  drops a frame or two at the seam.
* It shares the ADB channel with the bot.

So this is opt-in, for when the run is going into a video.
"""

from __future__ import annotations

import subprocess
import re
import threading
import time
from typing import Optional

import numpy as np

# screenrecord's own hard limit. Restart before it, not after, so the gap
# lands where we choose rather than mid-frame.
SEGMENT_SECONDS = 170


class _LivePipe:
    """Read-only PyAV input that never advertises seeking on a live pipe.

    Windows FileIO.seekable() can return True for subprocess stdout even
    though seek() raises EINVAL. Exposing only read prevents PyAV from
    registering a seek callback in FFmpeg. The subprocess owns the pipe.
    """

    def __init__(self, pipe):
        self._pipe = pipe

    def read(self, size=-1):
        return self._pipe.read(size)

    def readable(self):
        return True

    def seekable(self):
        return False


def bitrate_bps(value: str) -> int:
    """screenrecord takes integer bits/s; accept human-readable CLI rates."""
    match = re.fullmatch(r"(\d+)([kKmM]?)", str(value))
    if match is None:
        raise ValueError("Bitrate must be a positive integer, optionally ending in K or M")
    rate = int(match[1]) * {"": 1, "k": 1000, "m": 1_000_000}[match[2].lower()]
    if rate <= 0:
        raise ValueError("Bitrate must be positive")
    return rate


class DeviceStream:
    """Native-resolution frames, decoded from an on-device H.264 stream."""

    def __init__(self, adb: str, serial: str, bitrate: str = "24M",
                 size: Optional[str] = None):
        self.adb = str(adb)
        self.serial = serial
        self.bitrate = str(bitrate_bps(bitrate))
        self.size = size
        self.error = ""
        self.frames = 0
        # Frames decoded but never converted, because the studio had not yet
        # collected the previous one.  Only the newest frame is ever drawn, so
        # these cost nothing to lose - see `_run`.
        self.dropped = 0
        self.width = 0
        self.height = 0
        self._received_at = 0.0
        self._taken = True
        self._latest: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        try:
            import av                                # noqa: F401
            self._ok = True
        except Exception as exc:                     # pragma: no cover
            self._ok = False
            self.error = f"PyAV unavailable: {type(exc).__name__}: {exc}"

    @property
    def ready(self) -> bool:
        return self._ok

    @property
    def last_error(self) -> str:
        return self.error

    @property
    def stale_seconds(self) -> float:
        return time.monotonic() - self._received_at if self._received_at else float("inf")

    def start(self) -> bool:
        if not self._ok or self._thread is not None:
            return self._ok
        self._thread = threading.Thread(target=self._run, name="device-stream",
                                        daemon=True)
        self._thread.start()
        return True

    def grab(self) -> Optional[np.ndarray]:
        """The newest decoded frame as BGRA, or None before the first arrives.

        Taking a frame is also what tells the reader to convert the next one:
        while nobody has collected the last frame, further decodes are dropped
        rather than converted into an array that would be overwritten unseen.
        """
        if not self._ok:
            return None
        if self._thread is None:
            self.start()
        with self._lock:
            self._taken = True
            return self._latest

    def close(self) -> None:
        self._stop.set()
        self._kill()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ---------------------------------------------------------------- inside

    def _command(self) -> list[str]:
        cmd = [self.adb, "-s", self.serial, "exec-out", "screenrecord",
               "--output-format=h264", f"--bit-rate={self.bitrate}",
               f"--time-limit={SEGMENT_SECONDS}"]
        if self.size:
            cmd.append(f"--size={self.size}")
        cmd.append("-")
        return cmd

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.kill()
            proc.wait(timeout=2)
        except Exception:
            pass

    def _run(self) -> None:
        import av

        while not self._stop.is_set():
            try:
                self._proc = subprocess.Popen(
                    self._command(), stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, bufsize=0)
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                time.sleep(1.0)
                continue

            container = None
            try:
                # PyAV reads the pipe directly; the stream is raw Annex-B, so
                # the demuxer needs to be told rather than guess from an
                # extension it does not have.
                container = av.open(_LivePipe(self._proc.stdout), format="h264", mode="r")
                for frame in container.decode(video=0):
                    if self._stop.is_set():
                        break
                    # Decoding is cheap - 690fps on this machine - but the
                    # YUV->BGRA conversion is 3.2ms of an eight megabyte
                    # allocation, and at 60fps that is 190ms of every second
                    # spent on a thread that is competing with the compositor
                    # and the detector for the GIL.  Under load the studio
                    # collects fewer frames than arrive, so most of those
                    # conversions produced an array that was overwritten
                    # before anything drew it.  Convert for a reader that is
                    # ready, and drop the rest: only the newest frame is ever
                    # shown, so a dropped one is not a frame lost.
                    with self._lock:
                        waiting = self._latest is not None and not self._taken
                    if waiting:
                        self.dropped += 1
                        continue
                    array = frame.to_ndarray(format="bgra")
                    self.height, self.width = array.shape[:2]
                    with self._lock:
                        self._latest = array
                        self._taken = False
                        self._received_at = time.monotonic()
                    self.frames += 1
            except Exception as exc:
                # A restart at the segment boundary lands here as a decode or
                # EOF error. That is expected once every SEGMENT_SECONDS and is
                # not worth reporting as a fault; anything else is.
                self.error = f"{type(exc).__name__}: {exc}"
            finally:
                try:
                    if container is not None:
                        container.close()
                finally:
                    self._kill()

            if not self._stop.is_set():
                time.sleep(0.2)          # brief gap, then a fresh segment
