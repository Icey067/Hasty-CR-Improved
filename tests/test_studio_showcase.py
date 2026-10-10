"""Portrait composition, startup and simulator timing regression coverage."""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
pytest.importorskip("PyQt6")
from PyQt6.QtCore import QRect
from PyQt6.QtWidgets import QApplication
from scripts.studio.render import CANVAS, CAPTION_ZONE_H, Layout, Telemetry
from scripts.studio.simfeed import SimFeed


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_showcase_keeps_simulator_and_its_decoration_off_gameplay():
    layout = Layout.build("showcase")
    assert layout.showcase
    assert layout.header.isEmpty() and layout.rail.isEmpty() and layout.feed.isEmpty()
    canvas = QRect(0, 0, *CANVAS)
    assert canvas.contains(layout.game)
    surround = layout.sim.adjusted(-4, -36, 4, 4)
    assert canvas.contains(surround)
    assert not layout.game.intersects(surround)
    assert not layout.game.intersects(surround.translated(3, 5))
    assert layout.game.width() > 780
    # Small, but a schematic survives it; below this the tower HP stops reading.
    assert layout.sim.width() >= 240
    # The band above and below the mirror is caption room, and it is a trade:
    # too little and there is nowhere to put the text, too much and the game
    # reads as a stamp in a black frame. Keep it between 8% and 16% a side.
    assert 0.08 <= layout.game.top()/CANVAS[1] <= 0.16
    assert CANVAS[1]-layout.game.bottom()-1 == pytest.approx(layout.game.top(), abs=1)
    # The simulator is centred on the mirror, not hung from its top.
    assert abs(layout.sim.center().y()-layout.game.center().y()) <= 1
    for rect in (layout.game, layout.sim):
        assert abs(rect.height()/rect.width()-16/9) < .01


def test_default_sim_runs_at_three_times_wall_clock(app, monkeypatch):
    feed = SimFeed()
    assert feed.ready, feed.error
    try:
        feed.watcher.timer.stop()
        assert feed.speed == feed.watcher.args.speed == 3.0
        feed.watcher.last = 100.0
        feed.watcher.carry = 0
        before = feed.watcher.match.elapsed_ms
        monkeypatch.setattr("sim.watch.time.perf_counter", lambda: 100.1)
        feed.watcher.tick()
        assert 250 <= feed.watcher.match.elapsed_ms-before <= 300
    finally:
        feed.stop()


def test_telemetry_has_source_dimensions_before_first_frame():
    assert Telemetry().source_w == Telemetry().source_h == 0


def test_launcher_defaults_and_startup_include_sim(monkeypatch):
    from scripts.studio import __main__, app as studio_app
    captured = {}
    monkeypatch.setattr(studio_app, "run", lambda args: captured.update(vars(args)) or 0)
    assert __main__.main([]) == 0
    assert captured["layout"] == "showcase"
    assert captured["sim_speed"] == 3
    assert not captured["hq"] and not captured["adb_fallback"]
    assert captured["record_fps"] == 60


def test_stream_frames_age_even_when_decoder_stalls(monkeypatch):
    from scripts.studio.devicecap import DeviceStream
    stream = DeviceStream("unused", "unused")
    assert stream.stale_seconds == float("inf")
    stream._received_at = 100
    monkeypatch.setattr("scripts.studio.devicecap.time.monotonic", lambda: 103)
    assert stream.stale_seconds == 3


def test_native_stream_uses_numeric_android_bitrate():
    from scripts.studio.devicecap import DeviceStream, bitrate_bps
    assert "--bit-rate=24000000" in DeviceStream("adb", "serial")._command()
    assert bitrate_bps("8000K") == 8_000_000
    with pytest.raises(ValueError):
        bitrate_bps("0")


def test_portrait_mode_releases_toolbar_minimum_width(app):
    from PyQt6.QtWidgets import QWidget, QVBoxLayout
    from scripts.studio.app import Studio

    class Window(QWidget):
        _toggle_presentation = Studio._toggle_presentation

    window = Window()
    window._presenting = False
    layout = QVBoxLayout(window)
    window.controls_bar = QWidget()
    window.controls_bar.setMinimumWidth(900)
    window.console = QWidget()
    layout.addWidget(window.console)
    layout.addWidget(window.controls_bar)
    window.show()
    app.processEvents()
    try:
        window._toggle_presentation()
        app.processEvents()
        assert abs(window.width()/window.height()-9/16) < .005
        assert not window.controls_bar.isVisible()
        window._toggle_presentation()
        app.processEvents()
        assert window.controls_bar.isVisible()
    finally:
        window.close()
