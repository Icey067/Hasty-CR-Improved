import threading
import time

import numpy as np

from scripts.studio.capture import ThreadedSurfaceGrabber


def test_slow_capture_does_not_block_reader_or_mutate_published_frame():
    blocked = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    class Capture:
        frames = 0
        stale_seconds = 0

        def __init__(self, surface):
            self.buffer = np.zeros((2, 2, 4), dtype=np.uint8)

        def grab(self):
            self.frames += 1
            if self.frames == 2:
                blocked.set()
                release.wait(3)
            self.buffer.fill(self.frames)
            return self.buffer

        def close(self):
            closed.set()

    worker = ThreadedSurfaceGrabber(None, factory=Capture)
    try:
        assert blocked.wait(3)
        started = time.monotonic()
        first = worker.grab()
        assert time.monotonic() - started < .1
        assert np.all(first == 1)
        release.set()
        deadline = time.monotonic() + 3
        while worker.frames < 2 and time.monotonic() < deadline:
            time.sleep(.005)
        assert worker.frames >= 2
        assert np.all(first == 1)
    finally:
        release.set()
        worker.close()
    assert closed.is_set()
