"""Start, stop and watch the bot from the studio.

Two decisions worth recording.

**Liveness comes from the process table, not from a handle we kept.**  The bot is
often started from `bot.cmd`, from a terminal, or by the watchdog, and a studio
that only knew about children it spawned itself would show "stopped" while a run
was plainly in progress.  Scanning for `cr_bot.py` / `supervisor.py` in process
command lines reports what is actually true, whoever started it.

**Launching goes through `run.ps1` rather than invoking Python directly.**  That
script does the pre-flight the bot depends on - reconnect ADB, re-apply
`wm size 540x960` for capture speed, check Ollama before promising an advisor -
and duplicating that here would mean two versions of it to keep in step.

The scan is polled on a background thread: enumerating every process costs tens
of milliseconds, which is most of a frame budget at 60fps.
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import psutil

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "run.ps1"
LAUNCH_LOG = ROOT / "tmp" / "live" / "studio" / "launcher.log"

# Script -> label, most specific first.  Matched with the `scripts/` prefix
# attached, and confirmed against the process's working directory, because a
# bare name is not unique on this machine: an unrelated Clash of Clans bot runs
# its own `supervisor.py`, and matching that would have the studio report
# someone else's process as ours and refuse to start.
SIGNATURES = (("supervisor.py", "supervisor"), ("cr_bot.py", "match run"))

POLL_SECONDS = 2.0

# How long a launch is assumed to be in flight but invisible to the scan.
# `run.ps1` does its ADB and Ollama pre-flight before starting Python, so the
# process the scan looks for appears two to three seconds in; 8s is that with
# room for a cold disk, and it only ever delays a *second* start.
LAUNCH_SETTLE_SECONDS = 8.0


def _label_for(line: str) -> Optional[str]:
    for script, label in SIGNATURES:
        for prefix in ("scripts/", "scripts\\"):
            if prefix + script in line:
                return label
    return None


def _in_this_repo(process: psutil.Process, line: str) -> bool:
    """Best effort: the command line names this checkout, or the cwd is it."""
    if str(ROOT) in line:
        return True
    try:
        return Path(process.cwd()) == ROOT
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        # Unreadable cwd: the scripts/ prefix already ruled out the obvious
        # collision, so accept rather than report a live run as stopped.
        return True


@dataclass(frozen=True)
class BotState:
    running: bool = False
    mode: str = "stopped"
    pid: Optional[int] = None
    uptime: float = 0.0
    brain: str = ""
    detail: str = ""


def _scan() -> BotState:
    for process in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            info = process.info
            name = (info.get("name") or "").lower()
            if not name.startswith("python"):
                continue
            line = " ".join(info.get("cmdline") or ())
            label = _label_for(line)
            if label and _in_this_repo(process, line):
                # Read the brain off the running command line rather than
                # remembering what we launched: the bot is often started from
                # bot.cmd, a terminal or the watchdog, and "which brain is
                # actually playing" is exactly the thing worth not guessing.
                brain = "rl" if "--rl" in line.split() else "rules"
                return BotState(True, label, info["pid"],
                                time.time() - info["create_time"], brain)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return BotState()


class BotController:
    """Poll the bot's liveness, and start or stop it on request."""

    def __init__(self):
        self.state = _scan()
        self.last_action = ""
        self._launched_at = 0.0
        self._pending: Optional[subprocess.Popen] = None
        self._pending_log: Optional[object] = None
        # Where launcher.log ended when the current launch started, so a
        # failure can be quoted back without replaying the whole file.
        self._log_mark = 0
        self._expect_bot = False
        self._saw_bot = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="botctl", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.state = _scan()
                if self.state.running:
                    self._saw_bot = True
                self._check_launch()
            except Exception as exc:                 # never take the UI down
                self.last_action = f"scan failed: {type(exc).__name__}: {exc}"
            self._stop.wait(POLL_SECONDS)

    def _check_launch(self) -> None:
        """Say why a start produced no bot.

        `run.ps1` does the pre-flight and can refuse - no emulator at the
        serial, a checkpoint that has since been deleted out of `tmp/rl` - and
        it says so on its stdout, which goes to `launcher.log`.  Nothing read
        that back, so a refused start looked exactly like a click that did
        nothing at all: the button depressed, the status stayed "bot: stopped",
        and the only way to find the reason was to open a file the studio never
        mentions.  Now the launcher's exit is watched, and if it is gone while
        no bot ever appeared, its last line becomes the status.
        """
        pending = self._pending
        if pending is None or pending.poll() is None:
            return
        self._pending = None
        if self._pending_log is not None:
            try:
                self._pending_log.close()
            finally:
                self._pending_log = None
        if not self._expect_bot:
            return                                   # a -Stop, which is done
        if self._saw_bot:
            self.last_action = "run finished"
            return
        # Never a bot, and the launcher has exited: it refused.  Let a refusal
        # be startable again immediately rather than sitting out the settle.
        self._launched_at = 0.0
        self.last_action = f"start failed: {self._launch_tail()}"

    # Lines the launcher always prints, which say nothing about a refusal.
    LAUNCH_NOISE = ("+", "At ", "CategoryInfo", "FullyQualifiedErrorId",
                    "display:", "playing ", "starting supervisor",
                    "watch it with", "stop it with")

    def _launch_tail(self) -> str:
        """Why the launcher gave up, as one line.

        The *first* interesting line, not the last: run.ps1 refuses by saying
        what is wrong and then how to fix it ("no checkpoint at X" / "train
        one, or pass -Checkpoint"), and the fix on its own does not identify
        the problem.  A Python traceback is the other way round, so that one
        case reads from the bottom.
        """
        try:
            with LAUNCH_LOG.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(self._log_mark)
                lines = [line.strip() for line in handle if line.strip()]
        except OSError:
            lines = []
        lines = [line for line in lines if not line.startswith(self.LAUNCH_NOISE)]
        if not lines:
            return "see tmp/live/studio/launcher.log"
        if lines[0].startswith("Traceback"):
            return lines[-1][:120]
        return lines[0][:120]

    # ---------------------------------------------------------------- actions

    def _launch(self, arguments: list[str], description: str) -> None:
        LAUNCH_LOG.parent.mkdir(parents=True, exist_ok=True)
        # Claim the slot before spawning.  See _guard_is_clear.
        self._launched_at = time.time()
        self._saw_bot = False
        try:
            self._log_mark = LAUNCH_LOG.stat().st_size
        except OSError:
            self._log_mark = 0
        command = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                   "-File", str(LAUNCHER), "-FullRes", *arguments]
        try:
            # Detached with its own console-less process group: the bot must
            # outlive the studio, so closing the window mid-run does not end it.
            self._pending_log = LAUNCH_LOG.open("a", encoding="utf-8")
            self._pending = subprocess.Popen(
                command, cwd=str(ROOT),
                stdout=self._pending_log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
            self.last_action = description
        except Exception as exc:
            self.last_action = f"launch failed: {type(exc).__name__}: {exc}"

    def _guard_is_clear(self) -> bool:
        """Is it safe to start a bot right now?

        `self.state` is refreshed by a poller every POLL_SECONDS, and `run.ps1`
        takes two to three seconds to get as far as a `python.exe` the scan can
        see.  So the cached flag reads "stopped" for several seconds *after* a
        launch, and the two start buttons were both guarded on it alone.

        That is not theoretical.  On 2026-08-22 the studio launched `-Matches 5
        -Hours 2` at 17:03:00 and `-Forever` at 17:03:02; both cleared the guard,
        and two bots then drove the same emulator through the same serial for
        thirteen minutes.  They interleaved into one `cr_bot.log`, each played
        cards out of the other's hand, and the match files written in that window
        are unusable - hand_flips ran to 99 in a match against a clean-run
        baseline of 33, because from either bot's point of view the hand kept
        changing on its own.

        Two gates, because they fail in different ways: a launch we know about
        but the scan cannot see yet, and a re-scan so a bot started from a
        terminal or the watchdog in the last two seconds is also caught.
        """
        if time.time() - self._launched_at < LAUNCH_SETTLE_SECONDS:
            self.last_action = "a launch is still settling; try again"
            return False
        self.state = _scan()
        if self.state.running:
            self.last_action = "already running"
            return False
        return True

    # ------------------------------------------------------------------ brain
    #
    # Which policy decides the plays. `run.ps1` validates the checkpoint and
    # refuses to start rather than falling back to the rules, so a mistyped
    # path is visible instead of quietly playing the wrong brain.

    @staticmethod
    def _brain_arguments(brain: str, checkpoint: str = "") -> list[str]:
        if brain != "rl":
            return []
        arguments = ["-Brain", "rl"]
        if checkpoint:
            arguments += ["-Checkpoint", checkpoint]
        return arguments

    @staticmethod
    def _brain_label(brain: str) -> str:
        return "simulator-trained" if brain == "rl" else "rules + advisor"

    def _brain_is_on_disk(self, brain: str, checkpoint: str) -> bool:
        """A checkpoint listed at startup can be gone by the time it is picked.

        `tmp/rl` is scratch: a training run rotates its files while the studio
        is open, so the combo can be offering a path that no longer exists.
        `run.ps1` refuses in that case, which is right, but checking here names
        the file in the status bar instead of burying it in launcher.log.
        """
        if brain != "rl" or not checkpoint:
            return True
        if Path(checkpoint).exists():
            return True
        self.last_action = f"checkpoint gone: {Path(checkpoint).name}"
        return False

    def start_supervisor(self, brain: str = "rules", checkpoint: str = "") -> None:
        """Blocks of 5 matches, reviewed and tuned between blocks, indefinitely."""
        if not self._brain_is_on_disk(brain, checkpoint):
            return
        if not self._guard_is_clear():
            return
        self._expect_bot = True
        self._launch(["-Forever"] + self._brain_arguments(brain, checkpoint),
                     f"started supervisor ({self._brain_label(brain)})")

    def start_matches(self, matches: int, hours: float,
                      brain: str = "rules", checkpoint: str = "",
                      friendly: bool = False) -> None:
        """`friendly` plays without ever pressing Battle.

        For a 1v1 against a person: they start the match, the bot plays
        whatever it finds itself in. Without it the bot taps Battle in the
        lobby and drags itself into ladder between your friendlies.
        """
        if not self._brain_is_on_disk(brain, checkpoint):
            return
        if not self._guard_is_clear():
            return
        self._expect_bot = True
        arguments = ["-Matches", str(int(matches)), "-Hours", f"{hours:g}"]
        arguments += self._brain_arguments(brain, checkpoint)
        if friendly:
            arguments.append("-NoQueue")
        self._launch(
            arguments,
            f"started {int(matches)} matches (cap {hours:g}h, "
            f"{self._brain_label(brain)}"
            f"{', friendly - you start the match' if friendly else ''})")

    def stop(self) -> None:
        # run.ps1 -Stop matches on the command line rather than on a pid, so it
        # also stops a bot this studio did not start.
        self._expect_bot = False
        self._launch(["-Stop"], "stopping")
