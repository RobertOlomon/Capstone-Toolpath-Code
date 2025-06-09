from __future__ import annotations

"""
toolpath_gui.py
===============

PyQt 6 desktop application that lets you

1. **Generate** a new tool-path via `Main.main(...)` (after the user selects
   an STL file).
2. **Preview** the resulting Plotly figure(s)—static or animated—inside a
   `QWebEngineView` (no external browser tab).
3. **Load** a previously saved NumPy command array (`*.npy`) and preview it.
4. **Run** the command array on the robot over a serial link.

The file is self-contained apart from several project-specific modules that
already exist in your repo (`Main`, `robot`, `transmitter`, etc.).
"""

###############################################################################
# Imports
###############################################################################

import os
import sys
import threading
import queue
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PyQt6 import QtWidgets, QtWebEngineWidgets
from PyQt6.QtCore import Qt, QUrl, QEvent
import plotly.io as pio

# ---------------------------------------------------------------------------
# QtWebEngine compatibility shim (PyQt 6 moved QWebEngineSettings)
# ---------------------------------------------------------------------------
from PyQt6.QtWebEngineCore import QWebEngineSettings as _CoreQWebEngineSettings

# Alias so legacy references keep working
QtWebEngineWidgets.QWebEngineSettings = _CoreQWebEngineSettings  # type: ignore[attr-defined]

# Map legacy attribute names → Enum values so `settings.setAttribute(...)` works
for _name in (
    "LocalContentCanAccessRemoteUrls",
    "LocalContentCanAccessFileUrls",
    "AllowRunningInsecureContent",
    "JavascriptEnabled",
    "JavascriptCanOpenWindows",
    "LocalStorageEnabled",
    "AutoLoadImages",
):
    if not hasattr(QtWebEngineWidgets.QWebEngineSettings, _name):  # type: ignore[attr-defined]
        try:
            setattr(
                QtWebEngineWidgets.QWebEngineSettings,  # type: ignore[attr-defined]
                _name,
                getattr(_CoreQWebEngineSettings.WebAttribute, _name),
            )
        except AttributeError:
            # Some attributes vanished or were renamed—ignore quietly.
            pass

# ---------------------------------------------------------------------------
# Project-specific imports (deferred until after Qt initialises)
# ---------------------------------------------------------------------------
import transmitter            # noqa: E402 – your repo
from command_sender import reader_task, wait_for_ack  # noqa: E402 – your repo
from robot import Robot, Laser                         # noqa: E402 – your repo
import Main                                            # noqa: E402 – tool-path generator

###############################################################################
# Configurable constants
###############################################################################

PORT = "COM9"                        # Serial port for the robot
BAUD = 921_600                       # Baud rate
CHUCK_DRAWBACK_DISTANCE_MM = 30.0    # How far to pull back on MUST_PULL_BACK

###############################################################################
# Plotly helpers
###############################################################################

@contextmanager
def _suppress_plotly_show():
    """Silence any accidental `fig.show()` calls in third-party code."""
    prev = pio.renderers.default
    try:
        pio.renderers.default = "json"   # no-op renderer
        yield
    finally:
        pio.renderers.default = prev


def _ensure_fig_list(
    figs: "pio.Figure | Sequence[pio.Figure] | None"  # type: ignore[name-defined]
) -> List["pio.Figure"]:                              # type: ignore[name-defined]
    """Turn None / single figure / iterable → list of figures."""
    if figs is None:
        return []
    return list(figs) if isinstance(figs, Sequence) else [figs]  # type: ignore[arg-type]


def _compose_html(figs: Sequence["pio.Figure"]) -> str:  # type: ignore[name-defined]
    """Return a minimal HTML page embedding all Plotly figs."""
    if not figs:
        return (
            "<html><body style='margin:0;background:#222;color:#eee;"
            "font-family:monospace'>No preview available.</body></html>"
        )

    parts: List[str] = []
    for idx, fig in enumerate(figs):
        parts.append(
            pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs="cdn" if idx == 0 else False,
                auto_play=False,
            )
        )
    body = (
        "<hr style='border:none;height:1px;background:#444;margin:16px 0'>"
    ).join(parts)
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'></head>"
        f"<body style='margin:0'>{body}</body></html>"
    )

###############################################################################
# Machine control helpers
###############################################################################

def convert_npy_to_gcode(
    npy_path: os.PathLike | str,
    gcode_path: os.PathLike | str,
    drawback_dist: float = CHUCK_DRAWBACK_DISTANCE_MM,
) -> None:
    """Handy debug utility—turn binary command array into human G-code."""
    data = np.load(npy_path)
    with open(gcode_path, "w", encoding="utf-8") as f:
        f.write("G28 A\0\n")  # Home rotary axis first
        for i, row in enumerate(data):
            angle_rad = np.deg2rad(row[7])
            pull_back = bool(row[8])

            # Look-ahead: merge consecutive pull-backs at identical angle
            next_same_angle_pull = (
                pull_back
                and i < len(data) - 1
                and np.isclose(angle_rad, np.deg2rad(data[i + 1][7]))
                and bool(data[i + 1][8])
            )

            if pull_back:
                f.write(f"G01 A{angle_rad:.4f} C-30.0\0\n")
                f.write(f"G01 A{angle_rad:.4f} Y{drawback_dist} C-30.0\0\n")
                if not next_same_angle_pull:
                    f.write(f"G01 A{angle_rad:.4f} Y-{drawback_dist} C0\0\n")
            else:
                f.write(f"G01 A{angle_rad:.4f}\0\n")

        f.write("G01 A0 Y0\0\nG01 A0 Y0 C0\0\n")


def run_toolpath(
    npy_path: os.PathLike | str,
    drawback_dist: float = CHUCK_DRAWBACK_DISTANCE_MM,
) -> None:
    """Stream a command array to the robot over serial."""
    steps = np.load(npy_path)
    robot = Robot()

    tx = transmitter.Transmitter(PORT, BAUD, write_timeout=None, timeout=None)
    log_q: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=reader_task, args=(tx.serial, log_q), daemon=True).start()

    def _send(cmd: str) -> None:
        if not cmd.endswith("\0"):
            cmd += "\0"
        tx.send_msg(transmitter.CommandMessage(cmd))
        wait_for_ack(log_q)

    # Home rotary axis once at start
    _send("G28 A")

    for i, step in enumerate(steps):
        angle_rad = np.deg2rad(step[7])
        pull_back = bool(step[8])

        next_same_angle_pull = (
            pull_back
            and i < len(steps) - 1
            and np.isclose(angle_rad, np.deg2rad(steps[i + 1][7]))
            and bool(steps[i + 1][8])
        )

        if pull_back:
            _send(f"G01 A{angle_rad:.4f} C-30.0")
            _send(f"G01 A{angle_rad:.4f} Y{drawback_dist} C-30.0")
        else:
            _send(f"G01 A{angle_rad:.4f}")

        # Update your digital twin / live visualiser, if present
        quat_xyzw = [step[4], step[5], step[6], step[3]]
        robot_pos = np.array([step[0], step[1], step[2], *quat_xyzw])
        robot.step(robot_pos)
        Laser.ablate()

        if pull_back and not next_same_angle_pull:
            _send(f"G01 A{angle_rad:.4f} Y-{drawback_dist} C0")

    _send("G01 A0 Y0")
    _send("G01 A0 Y0 C0")

###############################################################################
# Qt main window
###############################################################################

class ToolpathGUI(QtWidgets.QWidget):
    """Main window for planning, previewing and executing tool-paths."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Toolpath Runner")
        self.resize(1280, 860)

        # ---------------- Layout ----------------
        vbox = QtWidgets.QVBoxLayout(self)

        # Top button row
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("Load Existing .npy")
        self.btn_gen = QtWidgets.QPushButton("Generate New Toolpath")
        btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_gen)
        vbox.addLayout(btn_row)

        # Web view for Plotly
        self.web = QtWebEngineWidgets.QWebEngineView()
        settings = self.web.settings()
        settings.setAttribute(
            QtWebEngineWidgets.QWebEngineSettings.JavascriptEnabled, True
        )
        settings.setAttribute(
            QtWebEngineWidgets.QWebEngineSettings.LocalContentCanAccessRemoteUrls,  # type: ignore[attr-defined]
            True,
        )
        vbox.addWidget(self.web, 1)

        # Run button
        self.btn_run = QtWidgets.QPushButton("Run Toolpath")
        self.btn_run.setEnabled(False)
        vbox.addWidget(self.btn_run)

        # ---------------- Internal state ----------------
        self.npy_path: Optional[Path] = None
        self.temp_html_path: Optional[str] = None  # To store the path to the temp file

        # ---------------- Signals ----------------
        self.btn_load.clicked.connect(self.load_existing)
        self.btn_gen.clicked.connect(self.generate_toolpath)
        self.btn_run.clicked.connect(self.execute_toolpath)

    # ------------------------------------------------------------------
    # Helper: display figures in the embedded browser
    # ------------------------------------------------------------------
    def _show_figures(
        self, figs: "pio.Figure | Sequence[pio.Figure] | None" = None  # type: ignore[name-defined]
    ) -> None:
        # Clean up the previous temporary file if it exists
        if self.temp_html_path and os.path.exists(self.temp_html_path):
            try:
                os.remove(self.temp_html_path)
            except OSError as e:
                print(f"Error removing temporary file: {e}")
            finally:
                self.temp_html_path = None

        figs_list = _ensure_fig_list(figs)
        html = _compose_html(figs_list)

        if not figs_list:
            # For simple text, setHtml is fine
            self.web.setHtml(html)
            return

        # For complex JS, save to a temporary file and load it
        with tempfile.NamedTemporaryFile(
            "w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            self.temp_html_path = f.name
            f.write(html)
        
        self.web.load(QUrl.fromLocalFile(os.path.abspath(self.temp_html_path)))

    # ------------------------------------------------------------------
    # Helper: normalise return value from Main.main
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_main_return(
        ret: "str | os.PathLike | Tuple[object, ...]"
    ) -> Tuple[Path, List["pio.Figure"]]:  # type: ignore[name-defined]
        if isinstance(ret, (str, Path, os.PathLike)):
            return Path(ret), []
        if isinstance(ret, tuple):
            if not ret:
                raise ValueError("Empty tuple returned from Main.main")
            path = Path(ret[0])
            figs = _ensure_fig_list(ret[1:])
            return path, figs
        raise TypeError(f"Unexpected return type from Main.main: {type(ret)}")

    # ------------------------------------------------------------------
    # Slot: load existing .npy
    # ------------------------------------------------------------------
    def load_existing(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select command array (.npy)", "", "NumPy files (*.npy)"
        )
        if not file_path:
            return

        self.npy_path = Path(file_path)
        self.btn_run.setEnabled(True)

        # Show a placeholder message
        self._show_figures(None)

    # ------------------------------------------------------------------
    # Slot: generate a new tool-path
    # ------------------------------------------------------------------
    def generate_toolpath(self) -> None:
        stl_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select STL file", "", "STL files (*.stl)"
        )
        if not stl_path:
            return

        # Run the planner (could be lengthy—consider threading)
        with _suppress_plotly_show():
            # Pass display_animation=False so the backend returns the figures
            # instead of trying to show them itself.
            ret = Main.main(stl_path, display_animation=False)

        try:
            npy_path, figs = self._normalize_main_return(ret)
        except Exception as exc:  # broad, but we want to GUI-notify
            QtWidgets.QMessageBox.critical(self, "Error", str(exc))
            return

        self.npy_path = npy_path
        self.btn_run.setEnabled(True)
        self._show_figures(figs)

    # ------------------------------------------------------------------
    # Slot: execute the loaded/generated tool-path
    # ------------------------------------------------------------------
    def execute_toolpath(self) -> None:
        if self.npy_path is None:
            QtWidgets.QMessageBox.warning(self, "No path", "Load or generate a tool-path first.")
            return

        self.btn_run.setEnabled(False)

        def _worker():
            try:
                run_toolpath(self.npy_path)
            finally:
                # Re-enable button when done by posting a custom event to the main thread
                QtWidgets.QApplication.instance().postEvent(
                    self,
                    QEvent(QEvent.Type(QEvent.registerEventType())),
                )

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Override: custom event for re-enabling the Run button
    # ------------------------------------------------------------------
    def event(self, ev: QEvent) -> bool:  # type: ignore[override]
        if ev.type() >= QEvent.Type.User:
            self.btn_run.setEnabled(True)
            return True
        return super().event(ev)

    # ------------------------------------------------------------------
    # Override: clean up temp file on exit
    # ------------------------------------------------------------------
    def closeEvent(self, event: QEvent) -> None: # type: ignore[override]
        if self.temp_html_path and os.path.exists(self.temp_html_path):
            try:
                os.remove(self.temp_html_path)
            except OSError as e:
                print(f"Error removing temporary file on close: {e}")
        super().closeEvent(event)


###############################################################################
# Launcher
###############################################################################

def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    gui = ToolpathGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()