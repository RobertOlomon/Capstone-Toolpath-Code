from __future__ import annotations

"""
toolpath_gui.py
===============

PyQt 6 desktop application that lets you

1. **Generate** a new tool-path via `toolpath.main(...)` (after the user selects
   an STL file).
2. **Preview** the resulting Plotly figure(s)—static or animated—inside a
   `QWebEngineView` (no external browser tab).
3. **Load** a previously saved NumPy command array (`*.npy`) and preview it.
4. **Run** the command array on the robot over a serial link.

The file is self-contained apart from several project-specific modules that
already exist in your repo (`toolpath`, `robot`, `transmitter`, etc.).
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
import socket
import plotly.io as pio
import serial.tools.list_ports

import transmitter
from command_sender import reader_task, wait_for_ack_timeout
from robot import Robot, Laser  
import toolpath              
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

###############################################################################
# Configurable constants
###############################################################################

BAUD = 921600                       # Baud rate
CHUCK_DRAWBACK_DISTANCE_MM = 300    # How far to pull back on MUST_PULL_BACK

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

def is_robot_connected(host: str = "192.168.100.50", port: int = 80, timeout: float = 0.5) -> bool:
    """Return True if the ABB robot responds on the configured host/port."""
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except OSError:
        return False

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
        #f.write("G28 A0\0\n")  # Home rotary axis first
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
                f.write(f"G01 A{angle_rad:.4f} C30.0\0\n")
                f.write(f"G01 A{angle_rad:.4f} Y{drawback_dist} C30.0\0\n")
                if not next_same_angle_pull:
                    f.write(f"G01 A{angle_rad:.4f} Y-{drawback_dist} C0\0\n")
            else:
                f.write(f"G01 A{angle_rad:.4f}\0\n")

        f.write("G01 A0 Y0\0\nG01 A0 Y0 C0\0\n")


def run_toolpath(
    npy_path: os.PathLike | str,
    drawback_dist: float = CHUCK_DRAWBACK_DISTANCE_MM,
    port: Optional[str] = None,
    *,
    run_machine: bool = True,
    run_arm: bool = True,
    laser_on: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    ack_timeout: float = 15.0,
) -> None:
    """Stream a command array to the robot and/or machine."""

    steps = np.load(npy_path)

    total_steps = len(steps)
    if progress_callback:
        progress_callback(0, total_steps)

    if run_machine:
        if port is None:
            port = detect_serial_port()
        if port is None:
            raise RuntimeError("No Arduino/ESP32 serial device found")
        tx = transmitter.Transmitter(port, BAUD, write_timeout=None, timeout=None)
        log_q: "queue.Queue[str]" = queue.Queue()
        threading.Thread(target=reader_task, args=(tx.serial, log_q), daemon=True).start()

        def _send(cmd: str) -> None:
            if not cmd.endswith("\0"):
                cmd += "\0"
            tx.send_msg(transmitter.CommandMessage(cmd))
            try:
                wait_for_ack_timeout(log_q, ack_timeout)
            except TimeoutError as exc:
                raise TimeoutError(f"No ACK for '{cmd.strip()}'") from exc
    else:
        def _send(_cmd: str) -> None:
            pass

    robot = Robot() if run_arm else None

    if run_machine:
        #_send("G28 A \0")

        for i, step in enumerate(steps, 1):
            angle_rad = np.deg2rad(step[7])
            pull_back = bool(step[8])

            next_same_angle_pull = (
                pull_back
                and i < len(steps) - 1
                and np.isclose(angle_rad, np.deg2rad(steps[i + 1][7]))
                and bool(steps[i + 1][8])
            )

            if run_machine:
                if pull_back:
                    _send(f"G01 A-{angle_rad:.4f} C5 B0\0")
                    _send(f"G01 A-{angle_rad:.4f} Y{drawback_dist} C5 B0 \0")
                else:
                    _send(f"G01 A-{angle_rad:.4f} B1 \0")

            if run_arm:
                quat_xyzw = [step[4], step[5], step[6], step[3]]
                robot_pos = np.array([step[0], step[1], step[2], *quat_xyzw])
                robot.step(robot_pos)
                if laser_on:
                    Laser.ablate()

            if run_machine and pull_back and not next_same_angle_pull:
                _send(f"G01 A{angle_rad:.4f} Y-{drawback_dist} C0")

            if progress_callback:
                progress_callback(i, total_steps)

        if run_machine:
            _send("G01 A0 Y0 B1\0")
            _send("G01 A0 Y0 C0 B0\0")

        if progress_callback:
            progress_callback(total_steps, total_steps)



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
                ret = toolpath.main(
                    self.stl_path,
                    display_animation=False,
                    progress_callback=self._on_progress,
                )
            path, figs = ClientGUI._normalize_main_return(ret)
            self.finished.emit(path, figs)
        except Exception as exc:  # pragma: no cover - gui
            self.error.emit(str(exc))

    def _on_progress(self, current: int, total: int) -> None:
        self.progress.emit(current, total)


class RunWorker(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        npy_path: Path,
        port: Optional[str],
        *,
        run_machine: bool,
        run_arm: bool,
        laser_on: bool,
    ) -> None:
        super().__init__()
        self.npy_path = npy_path
        self.port = port
        self.run_machine = run_machine
        self.run_arm = run_arm
        self.laser_on = laser_on

    def run(self) -> None:
        try:
            run_toolpath(
                self.npy_path,
                port=self.port,
                run_machine=self.run_machine,
                run_arm=self.run_arm,
                laser_on=self.laser_on,
                progress_callback=self._on_progress,
            )
            self.finished.emit()
        except Exception as exc:  # pragma: no cover - gui
            self.error.emit(str(exc))

    def _on_progress(self, current: int, total: int) -> None:
        self.progress.emit(current, total)

###############################################################################
# Qt main window
###############################################################################

class ClientGUI(QtWidgets.QWidget):
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

        # Connection status labels
        self.status_serial = QtWidgets.QLabel()
        self.status_robot = QtWidgets.QLabel()
        vbox.addWidget(self.status_serial)
        vbox.addWidget(self.status_robot)


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

        # Run buttons
        self.btn_run_machine = QtWidgets.QPushButton("Run with machine only")
        self.btn_run_arm = QtWidgets.QPushButton("Run with arm only (laser off)")
        self.btn_run_sync_off = QtWidgets.QPushButton("Run synchronously (laser off)")
        self.btn_run_sync_on = QtWidgets.QPushButton("Run synchronously (laser on)")
        for b in (
            self.btn_run_machine,
            self.btn_run_arm,
            self.btn_run_sync_off,
            self.btn_run_sync_on,
        ):
            b.setEnabled(False)
            vbox.addWidget(b)


        # ---------------- Internal state ----------------
        self.npy_path: Optional[Path] = None
        self.temp_html_path: Optional[str] = None  # To store the path to the temp file
        self.serial_port: Optional[str] = detect_serial_port()
        self.robot_connected: bool = is_robot_connected()

        self._update_connection_status()
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_connection_status)

        self.status_timer.start(3000)

        # ---------------- Signals ----------------
        self.btn_load.clicked.connect(self.load_existing)
        self.btn_gen.clicked.connect(self.generate_toolpath)
        self.btn_run_machine.clicked.connect(self.run_machine_only)
        self.btn_run_arm.clicked.connect(self.run_arm_only)
        self.btn_run_sync_off.clicked.connect(lambda: self.run_synchronous(False))
        self.btn_run_sync_on.clicked.connect(lambda: self.run_synchronous(True))

    def _update_connection_status(self) -> None:
        self.serial_port = detect_serial_port()
        self.robot_connected = is_robot_connected()
        if self.serial_port:
            self.status_serial.setText(f"Arduino detected on {self.serial_port}")
        else:
            self.status_serial.setText("No Arduino/ESP32 detected")
        self.status_robot.setText(
            "Robot connected" if self.robot_connected else "Robot not reachable"
        )
        self._update_run_buttons()

    def _update_run_buttons(self) -> None:
        has_path = self.npy_path is not None
        machine_ready = has_path and self.serial_port is not None
        arm_ready = has_path and self.robot_connected
        both_ready = machine_ready and arm_ready
        self.btn_run_machine.setEnabled(machine_ready)
        self.btn_run_arm.setEnabled(arm_ready)
        self.btn_run_sync_off.setEnabled(both_ready)
        self.btn_run_sync_on.setEnabled(both_ready)

    def _on_plan_progress(self, current: int, total: int) -> None:
        self.progress.setMaximum(total)
        self.progress.setValue(current)

    def _on_plan_finished(self, npy_path: Path, figs: list) -> None:
        self.npy_path = npy_path
        self.progress.setVisible(False)
        self._show_figures(figs)
        self._update_run_buttons()

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
        # Show a placeholder message
        self._show_figures(None)
        self._update_run_buttons()

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


    def _execute_run(self, *, run_machine: bool, run_arm: bool, laser_on: bool) -> None:
        if self.npy_path is None:
            QtWidgets.QMessageBox.warning(self, "No path", "Load or generate a tool-path first.")
            return
        if run_machine and self.serial_port is None:
            QtWidgets.QMessageBox.warning(self, "No device", "No Arduino/ESP32 detected.")
            return
        if run_arm and not self.robot_connected:
            QtWidgets.QMessageBox.warning(self, "No robot", "Robot not reachable.")
            return
        for b in (
            self.btn_run_machine,
            self.btn_run_arm,
            self.btn_run_sync_off,
            self.btn_run_sync_on,
        ):
            b.setEnabled(False)


        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.worker = RunWorker(
            self.npy_path,
            self.serial_port,
            run_machine=run_machine,
            run_arm=run_arm,
            laser_on=laser_on,
        )
        self.worker.progress.connect(self._on_run_progress)
        self.worker.finished.connect(self._on_run_finished)
        self.worker.error.connect(self._on_run_error)
        self.worker.start()

    def _on_run_progress(self, current: int, total: int) -> None:
        self.progress.setMaximum(total)
        self.progress.setValue(current)

    def _on_run_finished(self) -> None:
        self.progress.setVisible(False)
        QtWidgets.QApplication.instance().postEvent(
            self, QEvent(QEvent.Type(QEvent.registerEventType()))
        )

    def _on_run_error(self, msg: str) -> None:
        self.progress.setVisible(False)
        QtWidgets.QMessageBox.critical(self, "Run Error", msg)
        QtWidgets.QApplication.instance().postEvent(
            self, QEvent(QEvent.Type(QEvent.registerEventType()))
        )

    def run_machine_only(self) -> None:
        self._execute_run(run_machine=True, run_arm=False, laser_on=False)

    def run_arm_only(self) -> None:
        self._execute_run(run_machine=False, run_arm=True, laser_on=False)

    def run_synchronous(self, laser_on: bool) -> None:
        self._execute_run(run_machine=True, run_arm=True, laser_on=laser_on)

    # ------------------------------------------------------------------
    # Override: custom event for re-enabling the Run button
    # ------------------------------------------------------------------
    def event(self, ev: QEvent) -> bool:  # type: ignore[override]
        if ev.type() >= QEvent.Type.User:
            self._update_run_buttons()
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
    gui = ClientGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
