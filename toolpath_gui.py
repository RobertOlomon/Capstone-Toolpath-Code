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
from typing import List, Optional, Sequence, Tuple, Callable

import numpy as np
from PyQt6 import QtWidgets, QtWebEngineWidgets
from PyQt6.QtCore import Qt, QUrl, QEvent, QTimer, pyqtSignal, QObject, QThread
import plotly.io as pio
import serial.tools.list_ports

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

BAUD = 921_600                       # Baud rate
CHUCK_DRAWBACK_DISTANCE_MM = 30.0    # How far to pull back on MUST_PULL_BACK

def detect_serial_port() -> Optional[str]:
    """Return the first serial port that looks like an Arduino/ESP32."""
    for port in serial.tools.list_ports.comports():
        desc = (port.description or "").lower()
        if any(key in desc for key in ("arduino", "ch340", "cp210", "usb")):
            return port.device
        # fallback to ttyACM/ttyUSB naming
        if port.device.lower().startswith(("/dev/ttyacm", "/dev/ttyusb")):
            return port.device
    return None

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
    port: Optional[str] = None,
) -> None:
    """Stream a command array to the robot over serial."""
    if port is None:
        port = detect_serial_port()
    if port is None:
        raise RuntimeError("No Arduino/ESP32 serial device found")

    steps = np.load(npy_path)
    robot = Robot()

    tx = transmitter.Transmitter(port, BAUD, write_timeout=None, timeout=None)
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


class PlanWorker(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object, list)
    error = pyqtSignal(str)

    def __init__(self, stl_path: str) -> None:
        super().__init__()
        self.stl_path = stl_path

    def run(self) -> None:
        try:
            with _suppress_plotly_show():
                ret = Main.main(
                    self.stl_path,
                    display_animation=False,
                    progress_callback=self._on_progress,
                )
            path, figs = ToolpathGUI._normalize_main_return(ret)
            self.finished.emit(path, figs)
        except Exception as exc:  # pragma: no cover - gui
            self.error.emit(str(exc))

    def _on_progress(self, current: int, total: int) -> None:
        self.progress.emit(current, total)

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

        # Serial connection status
        self.status_label = QtWidgets.QLabel()
        vbox.addWidget(self.status_label)

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

        self.loading_label = QtWidgets.QLabel("Loading")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.setVisible(False)
        vbox.addWidget(self.loading_label)
        self.loading_timer = QTimer(self)
        self.loading_timer.timeout.connect(self._animate_loading)
        self._loading_dots = 0
        self.web.loadFinished.connect(self._stop_loading_animation)

        # Progress bar for planning
        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        vbox.addWidget(self.progress)

        # Run button
        self.btn_run = QtWidgets.QPushButton("Run Toolpath")
        self.btn_run.setEnabled(False)
        vbox.addWidget(self.btn_run)

        # ---------------- Internal state ----------------
        self.npy_path: Optional[Path] = None
        self.temp_html_path: Optional[str] = None  # To store the path to the temp file
        self.serial_port: Optional[str] = detect_serial_port()

        self._update_serial_status()
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_serial_status)
        self.status_timer.start(3000)

        # ---------------- Signals ----------------
        self.btn_load.clicked.connect(self.load_existing)
        self.btn_gen.clicked.connect(self.generate_toolpath)
        self.btn_run.clicked.connect(self.execute_toolpath)

    def _update_serial_status(self) -> None:
        self.serial_port = detect_serial_port()
        if self.serial_port:
            self.status_label.setText(f"Arduino detected on {self.serial_port}")
        else:
            self.status_label.setText("No Arduino/ESP32 detected")

    def _on_plan_progress(self, current: int, total: int) -> None:
        self.progress.setMaximum(total)
        self.progress.setValue(current)

    def _on_plan_finished(self, npy_path: Path, figs: list) -> None:
        self.npy_path = npy_path
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self._show_figures(figs)

    def _on_plan_error(self, msg: str) -> None:
        self.progress.setVisible(False)
        QtWidgets.QMessageBox.critical(self, "Error", msg)

    def _animate_loading(self) -> None:
        self._loading_dots = (self._loading_dots + 1) % 4
        self.loading_label.setText("Loading" + "." * self._loading_dots)

    def _start_loading_animation(self) -> None:
        self._loading_dots = 0
        self.loading_label.setText("Loading")
        self.loading_label.setVisible(True)
        self.loading_timer.start(300)

    def _stop_loading_animation(self, _ok: bool) -> None:
        self.loading_timer.stop()
        self.loading_label.setVisible(False)

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

        self._start_loading_animation()
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
        self.progress.setVisible(True)
        self.progress.setValue(0)

        self.worker = PlanWorker(stl_path)
        self.worker.progress.connect(self._on_plan_progress)
        self.worker.finished.connect(self._on_plan_finished)
        self.worker.error.connect(self._on_plan_error)
        self.worker.start()

    # ------------------------------------------------------------------
    # Slot: execute the loaded/generated tool-path
    # ------------------------------------------------------------------
    def execute_toolpath(self) -> None:
        if self.npy_path is None:
            QtWidgets.QMessageBox.warning(self, "No path", "Load or generate a tool-path first.")
            return

        if self.serial_port is None:
            QtWidgets.QMessageBox.warning(self, "No device", "No Arduino/ESP32 detected.")
            return

        self.btn_run.setEnabled(False)

        def _worker():
            try:
                run_toolpath(self.npy_path, port=self.serial_port)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Error", str(exc))
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