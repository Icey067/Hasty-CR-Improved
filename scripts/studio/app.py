"""The studio window: mirror, brain rail, log feed, recorder, and bot controls.

Three loops run at three different rates, and keeping them independent is the
whole design:

    mirror     60fps   PrintWindow on the emulator surface (~4ms)
    detector   12fps   YOLO on a background thread, boxes published async
    brain      ~0.3fps whatever the bot logs, tailed read-only

The UI thread only ever does work bounded by the frame budget: grab, compose,
blit.  Anything that could stall - model inference, H.264 encoding, enumerating
the process table - lives on its own thread behind a latest-value slot.

The window is two parts: the 1080x1920 canvas, and a control bar underneath it.
Only the canvas is recorded, so buttons, spin boxes and the bot's pid stay out
of the video while still being one click away.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QRect, Qt, QTimer
from PyQt6.QtGui import (QFont, QGuiApplication, QImage, QKeySequence,
                         QPainter, QShortcut)
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLayout,
                             QLabel, QPlainTextEdit, QPushButton, QSpinBox,
                             QSizePolicy, QVBoxLayout, QWidget)

from . import coach, render
from .botctl import BotController
from .capture import AdbGrabber, Surface, ThreadedSurfaceGrabber, find_surface
from .detect import DetectorWorker
from .feed import LogFeed
from .recorder import Recorder

ROOT = Path(__file__).resolve().parents[2]
CLIPS = ROOT / "tmp" / "live" / "studio"
# Where training writes its checkpoints; the brain selector lists these.
RL_CHECKPOINTS = ROOT / "tmp" / "rl"
# Policies frozen after a held-out evaluation, each beside a manifest saying
# what it scored. Scratch checkpoints under tmp/rl change while a run is going.
VETTED_CHECKPOINTS = ROOT / "checkpoints"

HELP = """  R  start / stop recording      1  layout: balanced
  S  save a PNG still            2  layout: bigger mirror
  L  detection labels on / off   3  layout: bigger log
  D  detector overlay on / off   4  layout: live + simulator
  F  fullscreen                 5  layout: portrait showcase
  P  portrait preview / controls
  Q  quit                        controls are in the bar below the canvas"""

BAR_STYLE = """
QWidget#bar { background:#0c1220; }
QLabel { color:#94a3b8; font-family:Consolas,'Cascadia Mono'; font-size:12px; }
QLabel#status { color:#e2e8f0; font-size:13px; font-weight:600; }
QCheckBox { color:#cbd5e1; spacing:6px; font-size:12px; }
QCheckBox::indicator { width:14px; height:14px; border:1px solid #475569; border-radius:3px; background:#101828; }
QCheckBox::indicator:checked { background:#60a5fa; border-color:#60a5fa; }
QPushButton {
    background:#16203a; color:#e2e8f0; border:1px solid #24314f;
    border-radius:6px; padding:5px 11px;
    font-family:Consolas,'Cascadia Mono'; font-size:12px;
}
QPushButton:hover { background:#1d294a; }
QPushButton:disabled { color:#475569; border-color:#182238; background:#101828; }
QPushButton#go { border-color:#2f6b45; color:#86efac; }
QPushButton#halt { border-color:#7f3038; color:#fca5a5; }
QPushButton#rec { border-color:#7f3038; color:#fca5a5; }
QSpinBox, QDoubleSpinBox, QComboBox {
    background:#101828; color:#e2e8f0; border:1px solid #24314f;
    border-radius:6px; padding:3px 6px;
    font-family:Consolas,'Cascadia Mono'; font-size:12px;
}
"""


class CanvasView(QWidget):
    """Shows the composed canvas, letterboxed and never stretched."""

    def __init__(self, canvas: QImage):
        super().__init__()
        self.canvas = canvas
        self.setMinimumSize(270, 480)
        # Qt coalesces update() calls, so a paint is not guaranteed per tick.
        # Counting them here is the only place the on-screen rate is knowable.
        self.paints = 0

    def paintEvent(self, event) -> None:
        self.paints += 1
        painter = QPainter(self)
        painter.fillRect(self.rect(), render.BG)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        width, height = self.canvas.width(), self.canvas.height()
        scale = min(self.width() / width, self.height() / height)
        target = QRect(0, 0, int(width * scale), int(height * scale))
        target.moveCenter(self.rect().center())
        painter.drawImage(target, self.canvas)
        painter.end()


# Wide enough for a timestamp, a kind and a card with its placement -
# the rest of a log line is detail that can run off the edge.
CONSOLE_W = 366


class Console(QPlainTextEdit):
    """The brain log at a size a person can actually read.

    The canvas is 9:16 and letterboxed into a wide window, which leaves most of
    the screen empty while the one thing worth watching - what the bot is
    deciding and why - is six pixels tall inside the mirror. This fills that
    space with the log at full size.

    It is a window panel, not part of the canvas, so the studio's own recorder
    (which captures the 1080x1920 canvas) will not see it. Capture the window
    instead if it needs to be in the video; the canvas keeps its own compact
    BRAIN LOG for that case.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setMaximumBlockCount(4000)
        # Fixed, not stretchy. With a stretch factor it swallowed every pixel
        # of a maximised window, which is the opposite of the point: the studio
        # has to stay a compact portrait block that crops straight into a reel.
        self.setFixedWidth(CONSOLE_W)
        self.setFrameStyle(0)
        font = QFont()
        font.setFamilies(["Cascadia Mono", "Consolas", "DejaVu Sans Mono"])
        font.setPixelSize(12)
        self.setFont(font)
        self.setStyleSheet(
            "QPlainTextEdit{background:#09090b;color:#a1a1aa;border:none;"
            "selection-background-color:#27272a;padding:14px 16px;}"
            "QScrollBar:vertical{background:#09090b;width:10px;margin:0;}"
            "QScrollBar::handle:vertical{background:#27272a;border-radius:5px;}"
            "QScrollBar::add-line,QScrollBar::sub-line{height:0;}")
        self._last = None

    def sync(self, events) -> None:
        """Append whatever is new since the last call."""
        rows = list(events)
        if not rows:
            return
        start = 0
        if self._last is not None:
            for index in range(len(rows) - 1, -1, -1):
                row = rows[index]
                if (row.at, row.kind, row.text) == self._last:
                    start = index + 1
                    break
        at_end = (self.verticalScrollBar().value()
                  >= self.verticalScrollBar().maximum() - 4)
        for row in rows[start:]:
            # Minutes and seconds only, and no padding after the kind. The hour
            # is the same all session and the column has to earn every
            # character: at this width the tag is the first thing to fall off
            # the edge, and the tag is the interesting part.
            self.appendPlainText(f"{row.at[-5:]} {row.kind:<6s} {row.text}")
        newest = rows[-1]
        self._last = (newest.at, newest.kind, newest.text)
        if at_end:
            bar = self.verticalScrollBar()
            bar.setValue(bar.maximum())


class Studio(QWidget):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.layout_name = args.layout
        self.canvas_size = (render.PRESENTATION_CANVAS if args.layout == "recording"
                            else render.CANVAS)
        self.layout_spec = render.Layout.build(args.layout, self.canvas_size)
        # Built on demand, the first time a layout asks for it. Loading the
        # card and spell tables costs about a second, and most sessions never
        # switch to the side-by-side view.
        self.sim = None
        # Every card the log has shown, for the 1v1 deck check.
        self._seen_cards: set[str] = set()
        self.canvas = QImage(*self.canvas_size, QImage.Format.Format_RGB32)
        self.canvas.fill(render.BG)

        # `--frame` swaps the live window for a still image.  That makes layout
        # and detector work testable with the emulator on the home screen, or
        # closed entirely, which is most of the time during development.
        self.still: Optional[np.ndarray] = None
        self.surface: Optional[Surface] = None
        self.grabber: Optional[ThreadedSurfaceGrabber] = None
        if args.frame:
            from PIL import Image
            rgb = np.asarray(Image.open(args.frame).convert("RGB"))
            self.still = np.dstack([rgb[:, :, ::-1],
                                    np.full(rgb.shape[:2], 255, np.uint8)]).copy()
        else:
            self.surface = find_surface(args.hwnd, getattr(args, 'instance', None))
            self.grabber = ThreadedSurfaceGrabber(self.surface, fps=args.fps)
        # MuMu's renderer can stall while the emulator keeps playing perfectly,
        # leaving the window presenting one frame for ever - measured at 180
        # back-to-back grabs returning a single distinct frame with the window in
        # the foreground. ADB reads the framebuffer instead, so it cannot freeze.
        # It is slow and shares the channel the bot needs, so it is only used
        # once the window has demonstrably gone stale.
        # Native-resolution capture, for when the run is going into a video.
        # The window mirror cannot exceed the monitor: MuMu renders 1080x1920
        # but a 1080-row desktop shows it at 560x996, and the canvas then
        # upscales that 1.3x. screenrecord encodes on the device instead, so
        # the frames arrive at full size. About a second behind, which is fine
        # for recording and wrong for deciding - the bot never reads this.
        self.device_stream = None
        if getattr(args, "hq", False) and not args.frame:
            from .devicecap import DeviceStream
            stream = DeviceStream(args.adb, args.serial, args.hq_bitrate)
            if stream.ready and stream.start():
                self.device_stream = stream
                print(f"mirror: device stream at native resolution "
                      f"({args.hq_bitrate}); window capture is the fallback",
                      flush=True)
            else:
                print(f"hq capture unavailable: {stream.error}", flush=True)

        self.adb_grabber: Optional[AdbGrabber] = None
        if getattr(args, "adb_fallback", False) and not args.frame:
            self.adb_grabber = AdbGrabber(args.adb, args.serial, args.adb_fps)
            if not self.adb_grabber.ready:
                print(f"adb fallback unavailable: {self.adb_grabber.error}")
                self.adb_grabber = None
        self.using_adb = False

        self.feed = LogFeed(args.log)
        self.bot = BotController()
        self.detector = DetectorWorker(
            weights=args.weights, rate=args.detect_fps, confidence=args.conf)
        self.detector_on = args.detect and self.detector.start()
        self.show_labels = True

        self.recorder: Optional[Recorder] = None
        self.telemetry = render.Telemetry(
            surface=str(self.surface) if self.surface else f"still {args.frame}")
        self._frame_times: Deque[float] = deque(maxlen=90)
        # (wall clock, counter) pairs for the two rates the eye cares about:
        # frames delivered by the capture source, and frames painted.
        self._source_marks: Deque[Tuple[float, int]] = deque(maxlen=90)
        self._paint_marks: Deque[Tuple[float, int]] = deque(maxlen=90)
        self._last_detect = 0.0
        self._marker: Optional[Tuple[float, float]] = None
        self._marker_at = -99.0
        self._seen_plays = (self.feed.state.match_index, self.feed.state.plays)
        self._render_ms = 0.0
        self._probed = 0
        self._warned_stale = False

        self._build_ui()
        self._build_shortcuts()
        self.set_layout(self.layout_name)
        if getattr(args, "presentation", False):
            self._toggle_presentation()

        print(f"surface   {self.telemetry.surface}")
        print(f"layout    {args.layout}  game {self.layout_spec.game.width()}x"
              f"{self.layout_spec.game.height()}  feed {self.layout_spec.feed.height()}px")
        print(f"detector  {self.detector.weights or 'disabled'}")
        print(f"log       {args.log}")
        print(HELP)

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.tick)
        self.timer.start(max(1, round(1000 / args.fps)))

        if args.record:
            self.toggle_recording()

    # -------------------------------------------------------------------- ui

    def _build_ui(self) -> None:
        self.setWindowTitle("HastyCR Studio")
        self.setStyleSheet("background:#080b14;")

        self.view = CanvasView(self.canvas)
        self.view.setToolTip("R records the portrait canvas. Gameplay and simulator have separate columns.")
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding,
                                QSizePolicy.Policy.Expanding)

        bar = QWidget()
        bar.setObjectName("bar")
        bar.setStyleSheet(BAR_STYLE)
        self.controls_bar = bar
        rows = QVBoxLayout(bar)
        rows.setContentsMargins(12, 8, 12, 8)
        row = QHBoxLayout()
        rows.addLayout(row)
        row.setContentsMargins(8, 5, 8, 5)
        row.setSpacing(5)

        self.status = QLabel("bot: …")
        self.status.setObjectName("status")
        # The bar sets the floor on how narrow the window can be, and the
        # window has to stay narrow enough to crop into a portrait reel. Every
        # control here is abbreviated for that reason; the tooltips carry the
        # full meaning.
        self.status.setMinimumWidth(96)
        row.addWidget(self.status)

        self.play_forever = QPushButton("Loop")
        self.play_forever.setObjectName("go")
        self.play_forever.setToolTip("Supervisor: blocks of 5 matches, reviewed "
                                     "and tuned between blocks, until stopped")
        self.play_forever.clicked.connect(self._start_forever)
        row.addWidget(self.play_forever)

        self.play_n = QPushButton("Start")
        self.play_n.setObjectName("go")
        self.play_n.setToolTip("Play a set number of matches, with an hours cap")
        self.play_n.clicked.connect(self._start_matches)
        row.addWidget(self.play_n)

        # Which policy decides the plays. One combo rather than a checkbox
        # plus a path box: the bar sets the floor on how narrow the window can
        # be, and a checkpoint is only ever chosen from what is on disk.
        self.brain = QComboBox()
        # The closed box stays narrow so the control bar keeps its shape; the
        # popup does not have to. At 150px every label was elided to
        # "Hog vs H...1 model)", which is worse than no label at all.
        self.brain.setFixedWidth(178)
        self.brain.view().setMinimumWidth(430)
        self.brain.setToolTip(
            "Qwen: the hand-written rules, with the local LLM biasing which "
            "card and lane.\n"
            "Sim-trained: a policy trained in the simulator, deciding every "
            "play itself (the advisor is not consulted).\n"
            "Perception is the same either way - only judgement changes.")
        self._load_brain_choices()
        row.addWidget(self.brain)

        # Play, but never press Battle - the bot joins whatever match it finds
        # itself in. Without it, it queues ladder in the gap between two
        # friendlies and you end up watching it play a stranger.
        self.friendly = QCheckBox("1v1")
        self.friendly.setToolTip(
            "Friendly battle: you start the match, the bot only plays.\n"
            "Off, it presses Battle itself and queues for ladder.")
        row.addWidget(self.friendly)

        self.matches = QSpinBox()
        self.matches.setFixedWidth(72)
        self.matches.setRange(1, 500)
        self.matches.setValue(5)
        self.matches.setSuffix("g")
        row.addWidget(self.matches)

        self.hours = QDoubleSpinBox()
        self.hours.setFixedWidth(72)
        self.hours.setRange(0.1, 24.0)
        self.hours.setSingleStep(0.5)
        self.hours.setValue(2.0)
        self.hours.setDecimals(1)
        self.hours.setSuffix("h")
        row.addWidget(self.hours)

        self.halt = QPushButton("Stop")
        self.halt.setObjectName("halt")
        self.halt.clicked.connect(self.bot.stop)
        row.addWidget(self.halt)

        row.addStretch(1)

        row = QHBoxLayout()
        row.setSpacing(8)
        rows.addLayout(row)

        self.record_button = QPushButton("● Rec")
        self.record_button.setObjectName("rec")
        self.record_button.clicked.connect(self.toggle_recording)
        row.addWidget(self.record_button)

        still = QPushButton("Still")
        still.clicked.connect(self.save_still)
        row.addWidget(still)

        self.layout_box = QComboBox()
        self.layout_box.addItems(["showcase", "balanced", "game", "feed", "sim", "recording"])
        self.layout_box.setFixedWidth(94)
        self.layout_box.setCurrentText(self.layout_name)
        self.layout_box.currentTextChanged.connect(self.set_layout)
        row.addWidget(self.layout_box)

        clips = QPushButton("Clips")
        clips.setToolTip(str(CLIPS))
        clips.clicked.connect(self._open_clips)
        row.addWidget(clips)

        logs = QPushButton("Logs")
        logs.clicked.connect(lambda: self.console.setVisible(not self.console.isVisible()))
        row.addWidget(logs)
        present = QPushButton("Portrait preview  ·  P")
        present.clicked.connect(self._toggle_presentation)
        row.addWidget(present)
        row.addStretch(1)
        self.quality_status = QLabel("1080 × 1920 export")
        row.addWidget(self.quality_status)

        # Controls must never swallow the keyboard shortcuts.
        for widget in bar.findChildren(QWidget):
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # The canvas is 9:16 inside a wide window, so most of the screen was
        # empty. The console takes that space and the mirror keeps its own.
        self.console = Console()
        self.console.hide()
        stage = QHBoxLayout()
        stage.setContentsMargins(0, 0, 0, 0)
        stage.setSpacing(0)
        # Both panels keep their width and the pair is centred, so making the
        # window bigger shows more desktop rather than more console. The block
        # stays the same shape, which is what makes it croppable.
        stage.addWidget(self.console, 0)
        stage.addWidget(self.view, 1)

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addLayout(stage, 1)
        column.addWidget(bar, 0)

        self.bar_height = 44
        scale = self.args.scale
        # The mirror keeps the width `--scale` asks for; the console takes
        # whatever is left, which on a wide screen is most of it. Opening at
        # roughly twice the canvas width means the console starts useful rather
        # than as a sliver the user has to drag open.
        canvas_w = int(self.canvas_size[0] * scale)
        # Snug: console plus mirror and nothing else. Taller than it is wide,
        # so a portrait crop of the desktop takes the whole studio and a margin
        # of whatever is behind it.
        available = QGuiApplication.primaryScreen().availableGeometry()
        self.resize(min(960, available.width()),
                    min(int(self.canvas_size[1] * scale)+100, available.height()-48))
        self._presenting = False

    def _build_shortcuts(self) -> None:
        # QShortcut rather than keyPressEvent: a spin box that has taken focus
        # would otherwise eat every key.
        for keys, handler in (
            ("R", self.toggle_recording), ("S", self.save_still),
            ("L", self._toggle_labels), ("D", self._toggle_detector),
            ("F", self._toggle_fullscreen), ("Q", self.close),
            ("P", self._toggle_presentation),
            ("5", lambda: self.layout_box.setCurrentText("showcase")),
            ("Esc", self.close),
            ("1", lambda: self.layout_box.setCurrentText("balanced")),
            ("2", lambda: self.layout_box.setCurrentText("game")),
            ("3", lambda: self.layout_box.setCurrentText("feed")),
            ("4", lambda: self.layout_box.setCurrentText("sim")),
        ):
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(handler)

    @staticmethod
    def _manifest(path: Path) -> dict:
        """The manifest sitting beside a checkpoint, or an empty dict.

        Named for the checkpoint or simply `manifest.json`; a directory that
        holds one shippable policy uses the latter.
        """
        for candidate in (path.with_name(path.stem + ".json"),
                          path.with_name("manifest.json")):
            if candidate.exists():
                try:
                    loaded = json.loads(candidate.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if isinstance(loaded, dict):
                    return loaded
        return {}

    @staticmethod
    def _checkpoint_record(path: Path) -> str:
        """"68% hog 14% n60" for a checkpoint, from the eval saved beside it.

        Worth the lookup rather than listing bare filenames. Fifteen
        checkpoints accumulate over a few days of training and several of them
        are *bad* - two runs here collapsed to never playing the win condition
        and won 30% - so a name alone makes picking one a coin flip.

        `hog` is on the label because win rate alone does not separate them:
        the very first run scored 81% while playing its win condition 0% of
        the time, by chipping and taking the overtime tiebreak. A high number
        next to `hog 0%` is that policy, not a good one.

        The sample size is shown for the same reason. That 81% was sixteen
        matches; v6's 68% was sixty. They are not the same claim, and the
        evals also ran against different opponents across runs, so treat these
        as a sanity check rather than a ranking.

        Three shapes are read, and which one a number came from is part of the
        number. A training run drops a flat `{wins, losses, ...}` beside each
        `_best.pt`. A frozen checkpoint carries a manifest, either named for
        the checkpoint or simply `manifest.json`; its `held_out` block was
        measured on games the run never trained on, and its `eval` block is
        the run's own selection score.

        Those last two differ by more than noise. `live_candidate` reports 82%
        under `eval` and scored 24% held out - the same policy, the same day.
        A held-out number is shown bare; a self-eval is prefixed `self` so it
        can never be mistaken for one.
        """
        for report in (path.with_name(path.stem + ".json"),
                       path.with_name("manifest.json")):
            if not report.exists():
                continue
            try:
                data = json.loads(report.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            held_out = data.get("held_out") or {}
            # Largest sample first: brain_300 over brain_100 over whatever
            # else is there. A 300-game number and a 40-game number are not
            # the same claim and the label prints n for that reason.
            scored = max(
                (block for block in held_out.values()
                 if isinstance(block, dict) and "wins" in block),
                key=lambda block: block["wins"] + block["losses"],
                default=None)
            prefix = "  "
            if scored is None:
                # A run's own `_best.json` is a self-eval too, but it is the
                # only number those checkpoints ever have and the `scratch:`
                # prefix on the row already says what they are.
                # `trainer_best` is what scripts/rl_supervisor.py writes: the
                # numbers the training run selected on, which are a 40-episode
                # smoke test against meta decks - a self-eval like the others.
                scored = data.get("trainer_best") or data.get("eval")
                prefix = "  self "
                if not isinstance(scored, dict):
                    scored, prefix = (data if "wins" in data else None), "  "
            # Every branch above reads a *foreign* file whose shape this code
            # does not own - manifests, run outputs, and whatever a later
            # script decides to write beside a checkpoint. One of them used
            # `eval` for a description string rather than a results block,
            # and indexing that string raised straight out of the dropdown
            # builder, leaving the studio with no brains to pick from at all.
            # A label is a convenience; it must never be able to do that.
            if not isinstance(scored, dict) or "wins" not in scored:
                continue
            played = scored["wins"] + scored["losses"] + scored.get("draws", 0)
            if not played:
                continue
            hog = scored.get("hog_share", data.get("hog_share"))
            label = f"{prefix}{scored['wins'] / played:.0%}"
            if hog is not None:
                label += f"  hog {hog:.0%}"
            return f"{label}  n{played}"
        return ""

    # How many of the scratch checkpoints to offer, newest first.
    SCRATCH_SHOWN = 3
    # And how many vetted-but-unlabelled ones. Both are small on purpose: the
    # list is for choosing what to play, and eleven near-identical rows named
    # after their files is not a choice, it is a puzzle.
    OTHERS_SHOWN = 2

    def _add_mode(self, path: Path, record: dict) -> None:
        """One named mode: short label, numbers on the hover."""
        self.brain.addItem(record["mode"], ("rl", str(path)))
        detail = record.get("measured", "")
        blurb = record.get("blurb", "")
        tip = "\n".join(part for part in (record["mode"], blurb, detail) if part)
        self.brain.setItemData(self.brain.count() - 1, tip,
                               Qt.ItemDataRole.ToolTipRole)

    def _load_brain_choices(self) -> None:
        """Qwen, then the vetted checkpoints, then the scratch ones.

        Listed rather than typed: a mistyped path would be caught by run.ps1,
        but only after a launch that then refuses to start.

        The two directories are not interchangeable and the ordering says so.
        `checkpoints/` holds policies that were frozen after a held-out
        evaluation, each with a manifest recording what it scored and against
        whom. `tmp/rl/` is the scratch space a training run writes into: it
        contains collapsed policies, mid-run snapshots, and files that change
        under you while a run is going. Sorting the whole lot by modification
        time - which is what this did - puts whatever is training *right now*
        at the top of the list, which is the one thing you never want to hand
        a live ladder match.
        """
        self.brain.clear()
        self.brain.addItem("Qwen (rules)", ("rules", ""))

        # Checkpoints whose manifest names a mode go first, described by what
        # they were trained against rather than by their filename. "Hog vs
        # Hog" and "Hog vs All" are the distinction that decides which one to
        # play; `mirror_best` and `ladder_best` are not.
        vetted = sorted(VETTED_CHECKPOINTS.glob("**/*.pt"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
        modes, plain = [], []
        for path in vetted:
            record = self._manifest(path)
            mode = record.get("mode")
            (modes if mode else plain).append((record.get("order", 99), path,
                                               record))
        # The two worth testing go at the top, on their own, with everything
        # else behind a separator. The measured numbers move to the hover:
        # they are what you read once when choosing, not every time you open
        # the list, and inline they crowded out the name.
        ranked = sorted(modes, key=lambda row: row[0])
        primary = [row for row in ranked if row[0] < 5]
        rest = [row for row in ranked if row[0] >= 5]
        if primary:
            self.brain.insertSeparator(self.brain.count())
        for _order, path, record in primary:
            self._add_mode(path, record)
        if rest or plain:
            self.brain.insertSeparator(self.brain.count())
        for _order, path, record in rest:
            self._add_mode(path, record)
        for _order, path, _record in plain[:self.OTHERS_SHOWN]:
            self.brain.addItem(f"Sim: {path.stem}{self._checkpoint_record(path)}",
                               ("rl", str(path)))
        # Only the newest handful of scratch files. Every training run this
        # project has done has left a checkpoint here, several of them
        # collapsed policies - one reads "35%  hog 0%", a run that never
        # played its win condition - and listing thirty of them buries the
        # three that are worth playing. The full set is still on disk for
        # anyone who wants a specific one.
        scratch = sorted(RL_CHECKPOINTS.glob("*.pt"),
                         key=lambda p: p.stat().st_mtime,
                         reverse=True)[:self.SCRATCH_SHOWN]
        for path in scratch:
            # `_last` is a mid-run snapshot and `_best` is the one selected on
            # evaluation score, so `_best` is what you want to play. Both are
            # offered; the ordering above puts whichever is newer first.
            self.brain.addItem(
                f"scratch: {path.stem}{self._checkpoint_record(path)}",
                ("rl", str(path)))
        if not vetted and not scratch:
            self.brain.setToolTip(self.brain.toolTip()
                                  + "\n\nNo checkpoints on disk yet.")

    def _brain_choice(self) -> tuple[str, str]:
        data = self.brain.currentData()
        return data if data else ("rules", "")

    def _start_forever(self) -> None:
        brain, checkpoint = self._brain_choice()
        self.bot.start_supervisor(brain, checkpoint)

    def _draw_coach(self, state) -> None:
        """The 1v1 checklist, over the canvas, only while 1v1 is ticked.

        It clears itself once the bot is in a match with nothing outstanding:
        at that point the checklist has served its purpose and the arena is
        what you want the window to be showing.
        """
        if not self.friendly.isChecked():
            return
        brain, checkpoint = self._brain_choice()
        # Every card the log has shown us this session - the hand right now
        # plus everything played - so a wrong deck is caught from the first
        # frame rather than only when an odd card happens to be played.
        self._seen_cards.update(state.hand or ())
        self._seen_cards.update(state.play_counts)
        advice = coach.advise(
            running=self.bot.state.running,
            friendly=True,
            mode=(Path(checkpoint).parent.name if checkpoint else "default"),
            screen=state.screen,
            observed=sorted(self._seen_cards),
            matches_done=state.matches_done,
            brain=brain,
        )
        if advice.next_step is None and not advice.warning:
            return
        painter = QPainter(self.canvas)
        try:
            width = min(880, render.CANVAS[0] - 80)
            height = 210 + 28 * len(advice.steps) + (44 if advice.warning else 0)
            if advice.next_step is not None:
                height += 22 * len(coach.WATCH_FOR)
            box = QRect(0, 0, width, height)
            box.moveCenter(QRect(0, 0, *render.CANVAS).center())
            render.draw_coach(painter, box, advice,
                              coach.WATCH_FOR if advice.next_step else ())
        finally:
            painter.end()

    def _start_matches(self) -> None:
        brain, checkpoint = self._brain_choice()
        self.bot.start_matches(self.matches.value(), self.hours.value(),
                               brain, checkpoint,
                               friendly=self.friendly.isChecked())

    def _open_clips(self) -> None:
        CLIPS.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(CLIPS))                       # noqa: S606 - Windows only
        except Exception as exc:
            print(f"could not open {CLIPS}: {type(exc).__name__}: {exc}")

    def _refresh_controls(self) -> None:
        state = self.bot.state
        if state.running:
            minutes = int(state.uptime // 60)
            brain = "sim" if state.brain == "rl" else "qwen"
            self.status.setText(
                f"bot: {state.mode} [{brain}]  pid {state.pid}  {minutes}m")
            self.status.setStyleSheet("color:#4ade80;")
        else:
            action = self.bot.last_action
            # The bar sets the floor on how narrow the window can go, and a
            # launcher refusal names a full path - so the status shows the
            # front of it and the tooltip carries the rest.
            short = action if len(action) <= 58 else action[:57] + "…"
            self.status.setText("bot: stopped" + (f"  ({short})" if short else ""))
            self.status.setToolTip(action)
            failed = action.startswith(("start failed", "checkpoint gone",
                                        "launch failed", "scan failed"))
            self.status.setStyleSheet("color:#f87171;" if failed else "color:#94a3b8;")
        self.play_forever.setEnabled(not state.running)
        self.play_n.setEnabled(not state.running)
        # Deliberately NOT disabled while the bot runs. The brain only takes
        # effect at the next launch, but greying the control out reads as
        # broken - there is no other way to see what the options even are, and
        # the bot is running most of the time you would want to look.
        if self.brain.isEnabled() is False:
            self.brain.setEnabled(True)
        self.matches.setEnabled(not state.running)
        self.hours.setEnabled(not state.running)
        self.halt.setEnabled(state.running)

        if self.recorder is None:
            self.record_button.setText("● Record")
        else:
            seconds = int(self.recorder.elapsed)
            self.record_button.setText(f"■ Stop  {seconds // 60}:{seconds % 60:02d}")

    # ------------------------------------------------------------------ loop

    def tick(self) -> None:
        started = time.perf_counter()
        self._frame_times.append(started)

        raw = self.still if self.grabber is None else self.grabber.grab()
        native_active = False
        stale = self.grabber.stale_seconds if self.grabber is not None else 0.0
        if self.device_stream is not None:
            native = self.device_stream.grab()
            if native is not None and self.device_stream.stale_seconds < self.args.stale_seconds:
                # Full-resolution frames win outright; the window grab above
                # still runs so `stale` keeps meaning what it meant, and so a
                # dead stream falls back without a branch here.
                raw = native
                stale = 0.0
                native_active = True
        if self.adb_grabber is not None and stale >= self.args.stale_seconds:
            fallback = self.adb_grabber.grab()
            if fallback is not None:
                raw = fallback
                if not self.using_adb:
                    self.using_adb = True
                    print(f"mirror: window stale {stale:.0f}s, falling back to ADB "
                          f"at {self.args.adb_fps:g}fps", flush=True)
        elif self.using_adb and stale < 1.0:
            self.using_adb = False
            print("mirror: window is rendering again, back to the fast path", flush=True)
        frame: Optional[QImage] = None
        if raw is not None:
            height, width = raw.shape[:2]
            # tobytes() hands Qt an owned copy, which is the safe way to view a
            # buffer the next grab is about to overwrite.
            frame = QImage(raw.tobytes(), width, height, width * 4,
                           QImage.Format.Format_RGB32)
            if self.detector_on and started - self._last_detect >= 1.0 / self.args.detect_fps:
                self._last_detect = started
                self.detector.submit(raw.copy())

        self.feed.poll()
        state = self.feed.state
        if (state.match_index, state.plays) != self._seen_plays:
            self._seen_plays = (state.match_index, state.plays)
            self._marker = render.marker_for(state.last_grid)
            self._marker_at = started

        bot = self.bot.state
        self.telemetry.mirror_fps = self._fps()
        self.telemetry.source_fps = self._rate(self._source_marks, started,
                                               self._source_frames(native_active))
        self.telemetry.paint_fps = self._rate(self._paint_marks, started,
                                              self.view.paints)
        self.telemetry.mirror_stale = 0.0 if self.using_adb else stale
        self.telemetry.mirror_source = "device" if native_active else ("adb" if self.using_adb else "window")
        if raw is not None:
            self.telemetry.source_h, self.telemetry.source_w = raw.shape[:2]
        self.telemetry.render_ms = self._render_ms
        self.telemetry.detector = self.detector.status if self.detector_on else "off"
        self.telemetry.detector_ms = self.detector.inference_ms
        self.telemetry.bot = (f"{bot.mode} {int(bot.uptime // 60)}m"
                              if bot.running else "stopped")
        boxes = self.detector.boxes if self.detector_on else ()
        self.telemetry.detections = len(boxes)
        self.telemetry.recording = self.recorder is not None
        if self.recorder is not None:
            self.telemetry.recorded_seconds = self.recorder.elapsed
            self.telemetry.record_behind = self.recorder.behind_seconds
            if self.recorder.error:
                print(f"REC ERROR {self.recorder.error}", flush=True)
                self.toggle_recording()

        render.compose(
            self.canvas, self.layout_spec, frame, state, self.feed.events, boxes,
            self.telemetry, self._marker, started - self._marker_at,
            self.show_labels, self.sim,
        )
        if not self.layout_spec.showcase:
            self._draw_coach(state)
        quality = f"{self.telemetry.source_w} × {self.telemetry.source_h} source"
        if self.telemetry.source_w and self.telemetry.source_w < self.layout_spec.game.width():
            quality += " · LOW RES"
        self.quality_status.setText(quality)
        if self.recorder is not None and self.recorder.behind_seconds > 1:
            self.quality_status.setText("Encoder behind · use 30 fps")

        if self.recorder is not None:
            if self.telemetry.mirror_stale >= 3.0 and not self._warned_stale:
                self._warned_stale = True
                print('WARNING mirror frozen: the emulator window is fully '
                      'occluded, so DWM is returning its last presented frame. '
                      'Bring the MuMu window to the front.', flush=True)
            elif self.telemetry.mirror_stale < 1.0:
                self._warned_stale = False
            self.recorder.submit(self._canvas_array().copy())

        self.console.sync(self.feed.events)
        self._refresh_controls()
        self._render_ms = (time.perf_counter() - started) * 1000.0
        self.view.update()

        # --probe: composite a fixed number of frames, prove the numbers, exit.
        # Cheaper than eyeballing a live window when checking a layout change.
        if self.args.probe:
            self._probed += 1
            if self._probed >= self.args.probe:
                self.save_still()
                print(f"probe    {self.telemetry.source_fps:.1f} fps source, "
                      f"{self.telemetry.paint_fps:.1f} fps painted, "
                      f"{self.telemetry.mirror_fps:.1f} fps compose loop, "
                      f"{self._render_ms:.1f} ms compose, "
                      f"detector={self.telemetry.detector} "
                      f"boxes={self.telemetry.detections} "
                      f"events={len(self.feed.events)} "
                      f"bot={self.telemetry.bot}", flush=True)
                self.close()
                QGuiApplication.quit()

    def _fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    def _source_frames(self, native_active: bool) -> int:
        """How many distinct frames the live source has produced so far.

        Whichever path is feeding the mirror owns the count; a still image
        never changes, so it has none and the rate stays at zero rather than
        reporting the timer back to itself.
        """
        if native_active and self.device_stream is not None:
            return self.device_stream.frames
        if self.using_adb and self.adb_grabber is not None:
            return self.adb_grabber.frames
        return self.grabber.frames if self.grabber is not None else 0

    @staticmethod
    def _rate(marks: Deque[Tuple[float, int]], now: float, count: int) -> float:
        """Frames per second from a counter sampled over the recent window."""
        marks.append((now, count))
        if len(marks) < 2:
            return 0.0
        span = marks[-1][0] - marks[0][0]
        return (marks[-1][1] - marks[0][1]) / span if span > 0 else 0.0

    def _canvas_array(self) -> np.ndarray:
        pointer = self.canvas.bits()
        pointer.setsize(self.canvas.sizeInBytes())
        stride = self.canvas.bytesPerLine() // 4
        array = np.frombuffer(pointer, dtype=np.uint8).reshape(
            self.canvas.height(), stride, 4)
        return array[:, : self.canvas.width(), :]

    # --------------------------------------------------------------- actions

    def toggle_recording(self) -> None:
        if self.recorder is not None:
            path, frames = self.recorder.path, self.recorder.frames
            behind = self.recorder.behind_seconds
            self.recorder.close()
            self.recorder = None
            size_mb = path.stat().st_size / 1e6 if path.exists() else 0.0
            print(f"REC stopped  {path}  {frames} frames  {size_mb:.1f}MB"
                  + (f"  encoder ran {behind:.1f}s behind" if behind > 1 else ""),
                  flush=True)
            return
        out = self.args.out or (CLIPS / f"clip_{datetime.now():%Y%m%d_%H%M%S}.mp4")
        try:
            self.recorder = Recorder(out, self.canvas_size, fps=self.args.record_fps,
                                     crf=self.args.crf, preset=self.args.preset)
            print(f"REC started  {out}  {self.canvas_size[0]}x{self.canvas_size[1]} "
                  f"@{self.args.record_fps}fps crf{self.args.crf}", flush=True)
        except Exception as exc:
            print(f"REC failed: {type(exc).__name__}: {exc}", flush=True)
            self.recorder = None

    def save_still(self) -> None:
        CLIPS.mkdir(parents=True, exist_ok=True)
        out = CLIPS / f"still_{datetime.now():%Y%m%d_%H%M%S}.png"
        self.canvas.save(str(out))
        print(f"still  {out}", flush=True)

    def set_layout(self, name: str) -> None:
        if name == "recording" and self.canvas_size != render.PRESENTATION_CANVAS:
            print("recording layout is available at launch: use --layout recording", flush=True)
            return
        self.layout_name = name
        self.layout_spec = render.Layout.build(name, self.canvas_size)
        if self.layout_spec.sim is not None and self.sim is None:
            from .simfeed import SimFeed
            self.sim = SimFeed(speed=self.args.sim_speed)
            if not self.sim.ready:
                print(f"simulator panel unavailable: {self.sim.error}", flush=True)
        print(f"layout {name}  feed {self.layout_spec.feed.height()}px", flush=True)

    def _toggle_presentation(self) -> None:
        self._presenting = not self._presenting
        if self._presenting:
            self._studio_geometry = self.saveGeometry()
            self._console_visible = self.console.isVisible()
            self.controls_bar.hide()
            self.console.hide()
            # Qt retains the toolbar's previous minimum width until its next
            # layout pass. Remove that constraint before requesting portrait.
            self.layout().setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
            self.setMinimumSize(270, 480)
            self.layout().activate()
            screen = QGuiApplication.primaryScreen().availableGeometry()
            height = max(480, screen.height()-80)
            self.resize(round(height*9/16), height)
        else:
            self.controls_bar.show()
            self.console.setVisible(self._console_visible)
            self.layout().setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
            self.restoreGeometry(self._studio_geometry)

    def _toggle_labels(self) -> None:
        self.show_labels = not self.show_labels

    def _toggle_detector(self) -> None:
        if self.detector_on:
            self.detector_on = False
        else:
            self.detector_on = (self.detector.status in ("live", "loading")
                                or self.detector.start())

    def _toggle_fullscreen(self) -> None:
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def closeEvent(self, event) -> None:
        self.timer.stop()
        if self.sim is not None:
            self.sim.stop()
        if self.recorder is not None:
            self.toggle_recording()
        self.detector.stop()
        if self.device_stream is not None:
            self.device_stream.close()
        self.bot.close()
        if self.grabber is not None:
            self.grabber.close()
        if self.adb_grabber is not None:
            self.adb_grabber.close()
        super().closeEvent(event)


def run(args) -> int:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QFontDatabase
    app = QGuiApplication.instance() or QApplication([])
    # Qt's offscreen Windows platform exposes no system fonts. Register the
    # installed faces so probes and exports use the same typography as live UI.
    if not QFontDatabase.families():
        for name in ("segoeui.ttf", "segoeuib.ttf", "consola.ttf"):
            path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / name
            if path.exists():
                QFontDatabase.addApplicationFont(str(path))
    window = Studio(args)
    # Shown even under --probe: paintEvent is only ever called on a visible
    # window, and a probe that skipped it once let a crash in exactly that
    # method reach the user.
    window.show()
    return app.exec()
