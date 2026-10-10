"""Decode actual H.264 through a pipe that falsely advertises seeking."""
import io

import numpy as np
import pytest

from scripts.studio.devicecap import _LivePipe


def test_live_pipe_decodes_without_seeking(capfd):
    av = pytest.importorskip('av')
    encoded = io.BytesIO()
    with av.open(encoded, mode='w', format='h264') as output:
        stream = output.add_stream('libx264', rate=30)
        stream.width = stream.height = 32
        stream.pix_fmt = 'yuv420p'
        for value in range(5):
            frame = av.VideoFrame.from_ndarray(
                np.full((32, 32, 3), value * 40, dtype=np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)

    class MisreportedPipe(io.BytesIO):
        seek_calls = 0

        def seek(self, *args):
            self.seek_calls += 1
            raise OSError(22, 'Invalid argument')

    pipe = MisreportedPipe(encoded.getvalue())
    assert pipe.seekable()  # Reproduce Windows' misleading pipe capability.
    with av.open(_LivePipe(pipe), format='h264', mode='r') as source:
        frames = list(source.decode(video=0))
    assert len(frames) == 5
    assert all(frame.width == frame.height == 32 for frame in frames)
    assert pipe.seek_calls == 0
    assert 'PyAV library exception' not in capfd.readouterr().err
