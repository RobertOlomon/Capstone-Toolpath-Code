from __future__ import annotations
import os
import sys
import threading
import queue
import numpy as np
from PyQt5 import QtWidgets, QtWebEngineWidgets
from PyQt5.QtCore import QUrl
import plotly.io as pio

import transmitter
from command_sender import reader_task, wait_for_ack
from robot import Robot, Laser
import Main

PORT = "COM9"
BAUD = 921600
CHUCK_DRAWBACK_DISTANCE_MM = 30.0

def convert_npy_to_gcode(
    npy_path: str, gcode_path: str,
    drawback_dist: float = CHUCK_DRAWBACK_DISTANCE_MM
) -> None:
    data = np.load(npy_path)
    with open(gcode_path, "w", encoding="utf-8") as f:
        f.write("G28 A\0\n")
        for i, row in enumerate(data):
            angle_rad = np.deg2rad(row[7])
            pull_back = bool(row[8])
            next_same_angle_and_pullback = False
            if pull_back and i < len(data) - 1:
                next_row = data[i + 1]
                next_angle_rad = np.deg2rad(next_row[7])
                next_pull_back = bool(next_row[8])
                if np.isclose(next_angle_rad, angle_rad) and next_pull_back:
                    next_same_angle_and_pullback = True

            if pull_back:
                f.write(f"G01 A{angle_rad:.4f} C-30.0\0\n")
                f.write(
                    f"G01 A{angle_rad:.4f} Y{drawback_dist} C-30.0\0\n"
                )
                if not next_same_angle_and_pullback:
                    f.write(
                        f"G01 A{angle_rad:.4f} Y-{drawback_dist} C0\0\n"
                    )
            else:
                f.write(f"G01 A{angle_rad:.4f}\0\n")

        f.write("G01 A0 Y0\0\n")
        f.write("G01 A0 Y0 C0\0\n")


def run_toolpath(npy_path: str, drawback_dist: float = CHUCK_DRAWBACK_DISTANCE_MM) -> None:
    commands = np.load(npy_path)
    robot = Robot()

    tx = transmitter.Transmitter(PORT, BAUD, write_timeout=None, timeout=None)
    log_q: queue.Queue[str] = queue.Queue()
    threading.Thread(target=reader_task, args=(tx.serial, log_q), daemon=True).start()

    def send(cmd: str):
        if not cmd.endswith("\0"):
            cmd += "\0"
        tx.send_msg(transmitter.CommandMessage(cmd))
        wait_for_ack(log_q)

    send("G28 A")

    for i, step in enumerate(commands):
        angle_rad = np.deg2rad(step[7])
        pull_back = bool(step[8])

        next_same_angle_and_pullback = False
        if pull_back and i < len(commands) - 1:
            next_step = commands[i + 1]
            next_angle_rad = np.deg2rad(next_step[7])
            next_pull_back = bool(next_step[8])
            if np.isclose(next_angle_rad, angle_rad) and next_pull_back:
                next_same_angle_and_pullback = True

        if pull_back:
            send(f"G01 A{angle_rad:.4f} C-30.0")
            send(f"G01 A{angle_rad:.4f} Y{drawback_dist} C-30.0")
        else:
            send(f"G01 A{angle_rad:.4f}")

        quat_xyzw = [step[4], step[5], step[6], step[3]]
        robot_pos = np.array([step[0], step[1], step[2], *quat_xyzw])
        robot.step(robot_pos)
        Laser.ablate()

        if pull_back and not next_same_angle_and_pullback:
            send(f"G01 A{angle_rad:.4f} Y-{drawback_dist} C0")

    send("G01 A0 Y0")
    send("G01 A0 Y0 C0")


class ToolpathGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Toolpath Runner")
        self.resize(1000, 800)
        layout = QtWidgets.QVBoxLayout(self)

        btn_layout = QtWidgets.QHBoxLayout()
        self.load_btn = QtWidgets.QPushButton("Use Existing Commands")
        self.gen_btn = QtWidgets.QPushButton("Generate New Toolpath")
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.gen_btn)
        layout.addLayout(btn_layout)

        self.web_view = QtWebEngineWidgets.QWebEngineView()
        layout.addWidget(self.web_view, 1)

        self.run_btn = QtWidgets.QPushButton("Run Toolpath")
        self.run_btn.setEnabled(False)
        layout.addWidget(self.run_btn)

        self.npy_path: str | None = None
        self.html_temp: str | None = None

        self.load_btn.clicked.connect(self.load_existing)
        self.gen_btn.clicked.connect(self.generate_toolpath)
        self.run_btn.clicked.connect(self.execute_toolpath)

    def display_figures(self, figs):
        if not figs:
            return
        html = "".join(pio.to_html(fig, include_plotlyjs='cdn', full_html=False) for fig in figs)
        tmp = os.path.abspath("_temp_plot.html")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(html)
        self.web_view.load(QUrl.fromLocalFile(tmp))
        self.html_temp = tmp

    def load_existing(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select ee_robot_commands.npy", os.getcwd(), "NumPy Files (*.npy)")
        if path:
            self.npy_path = path
            QtWidgets.QMessageBox.information(self, "Loaded", f"Loaded {os.path.basename(path)}. No preview available.")
            self.run_btn.setEnabled(True)

    def generate_toolpath(self):
        stl_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select STL file", os.getcwd(), "STL Files (*.stl)")
        if not stl_path:
            return
        figs = Main.main(stl_path)
        self.npy_path = "ee_robot_commands.npy"
        self.display_figures(figs)
        self.run_btn.setEnabled(True)

    def execute_toolpath(self):
        if not self.npy_path:
            return
        run_toolpath(self.npy_path)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = ToolpathGUI()
    gui.show()
    sys.exit(app.exec_())
