"""Portrait gameplay and simulator in separate columns."""
from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor
from . import render as r


def draw(p, layout, frame, state, events, boxes, telemetry,
         marker, marker_age, labels, sim):
    r.draw_game(p, layout.game, frame, boxes, marker, marker_age, labels)
    side = layout.sim
    surround = side.adjusted(-4, -36, 4, 4)
    r.panel(p, surround.translated(3, 5), QColor(0, 0, 0, 100), radius=8, edge=None)
    r.panel(p, surround, QColor("#11151b"), radius=6, edge=QColor("#66717e"))
    speed = getattr(sim, "speed", 3.0)
    r.label(p, QRect(side.x()+8, side.y()-32, side.width()-16, 28),
            f"SIM · {speed:g}×", r.font(20, r.SANS, True), r.TEXT)
    if sim is not None and sim.ready:
        sim.draw(p, side)
    else:
        r.label(p, side, "Simulator unavailable", r.font(16), r.DIM,
                Qt.AlignmentFlag.AlignCenter)
