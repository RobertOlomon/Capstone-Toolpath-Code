<<<<<<< HEAD
import argparse
import threading
import time
import queue
import sys
import transmitter
import serial.tools.list_ports
from typing import Optional

BAUD = 921600
PRINT_HZ = 30
ACK_MSG = "At Pos"

def detect_serial_port() -> Optional[str]:
    for port in serial.tools.list_ports.comports():
        desc = (port.description or "").lower()
        if any(key in desc for key in ("arduino", "ch340", "cp210", "usb")):
            return port.device
        if port.device.lower().startswith(("/dev/ttyacm", "/dev/ttyusb")):
            return port.device
    return None

def reader_task(ser, log_q: queue.Queue):
    """Background thread to continuously read from the serial port."""
    buf = bytearray()
    next_emit = time.time()
    while True:
        chunk = ser.read(4096)
        if chunk:
            buf.extend(chunk)
        while True:
            nl = buf.find(b"\n")
=======
import threading, time, queue, sys
import transmitter           # your existing module
import math

PORT      = "COM9"
BAUD      = 921600
PRINT_HZ  = 30               # visible lines per second (raise/lower as you like)

# ─────────────────────────── background reader ────────────────────────────────
def reader_task(ser, log_q: queue.Queue):
    """Drain the serial port fast; drop decoded lines into log_q."""
    buf = bytearray()
    next_emit = time.time()
    while True:
        chunk = ser.read(4096)         # as fast as driver gives data
        if chunk:
            buf.extend(chunk)

        # pull out complete lines
        while True:
            nl = buf.find(b'\n')
>>>>>>> a1c92f2b1a320fddc7718ceea9f2a41a2e0c8cc8
            if nl == -1:
                break
            line = buf[:nl]
            del buf[:nl+1]
<<<<<<< HEAD
            now = time.time()
            if now >= next_emit:
                log_q.put(line.decode(errors="replace").rstrip())
                next_emit = now + 1.0 / PRINT_HZ
        if not chunk:
            time.sleep(0.0005)

def wait_for_ack(log_q: queue.Queue):
    """Block until the controller reports command completion."""
    while True:
        line = log_q.get()
        print(line)
        if line.strip() == ACK_MSG:
            return

def send_from_file(path: str, tx: transmitter.Transmitter, log_q: queue.Queue) -> None:
    """Send all commands from the given gcode file."""
    with open(path, "r", encoding="utf-8") as infile:
        for raw in infile:
            cmd = raw.strip()
            if not cmd:
                continue
            if not cmd.endswith("\0"):
                cmd += "\0"
            print(f"[sending] {cmd}")
            tx.send_msg(transmitter.CommandMessage(cmd))
            wait_for_ack(log_q)
            print("[done]")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Send gcode commands to the cleaner")
    parser.add_argument("gcode_file", nargs="?", help="path to gcode text file")
    args = parser.parse_args(argv)

    port = detect_serial_port()
    if port is None:
        print("No Arduino/ESP32 serial device found")
        return
    tx = transmitter.Transmitter(port, BAUD, write_timeout=None, timeout=None)
=======

            now = time.time()
            # rate-limit what we show so console survives
            if now >= next_emit:
                log_q.put(line.decode(errors="replace").rstrip())
                next_emit = now + 1.0 / PRINT_HZ

        if not chunk:
            time.sleep(0.0005)         # tiny sleep keeps CPU ~0 %

cmd2 = [
    "G0 A1 C-.1\0",  # Move to position A1 with C-.1
    "G0 A2 C-.1\0",  # Move to position A2 with C-.1
    "G0 A3 C-.1\0",  # Move to position A3 with C-.1
    "G0 A4 C-.1\0",  # Move to position A4 with C-.1
    "G0 A5 C-.1\0",  # Move to position A5 with C-.1
    "G0 A6 C-.1\0",  # Move to position A6 with C-.1
    "G0 A7 C-.1\0",  # Move to position A7 with C-.1
    "G0 A8 C-.1\0",  # Move to position A8 with C-.1
    "G0 A0 C-.1\0",  # Move to position A8 with C-.1
]

cmd3 = [
    "G0 A0 Y0 C-.1 B1\0",
    "G0 A3 Y0 C-.1 B1\0",
    "G0 A3 Y0 C-.1 B1\0",
    "G0 A3 Y0 C-.1 B1\0",
    "G0 A3 Y0 C-.1 B1\0",
    "G0 A3 Y0 C-.1 B1\0",
    "G0 A3 Y0 C-.1 B1\0",
    "G0 A3 Y0 C-.1 B1\0",
    "G0 A3 Y0 C-.1 B1\0",
    f"G0 A{math.pi * 2} Y0 C-.1 B1\0",
    f"G0 A{math.pi * 2} Y0 C-.1 B1\0",
    f"G0 A{math.pi * 2} Y0 C-.1 B1\0",
    f"G0 A{math.pi * 2} Y0 C-.1 B1\0",
    f"G0 A{math.pi * 2} Y0 C-.1 B1\0",
    f"G0 A{math.pi * 2} Y0 C-.1 B1\0",
    f"G0 A{math.pi * 2} Y0 C-.1 B1\0",
    f"G0 A{math.pi * 2} Y0 C-.1 B1\0",
    f"G0 A{math.pi * 2} Y0 C-.1 B1\0",
    f"G0 A{math.pi * 2} Y0 C-.1 B1\0",
    f"G0 A{math.pi * 2} Y0 C-.1 B1\0",
    f"G0 A{math.pi * 2} Y0 C-.1 B1\0",
    f"G0 A{math.pi * 2} Y0 C-.1 B1\0",
    f"G0 A{math.pi * 2} Y0 C-.1 B1\0",
    f"G0 A{math.pi * 2} Y0 C-.1 B1\0",
    f"G0 A{math.pi * 2} Y0 C-.1 B0\0",
    "G0 A6.2831 Y0 C5 B0\0",
    "G0 A6.2831 Y0 C5 B0\0",
    "G0 A6.2831 Y200 C5 B0\0",
    "G0 A6.2831 Y200 C5 B0\0",
    "G0 A6.2831 Y200 C5 B0\0",
    "G0 A6.2831 Y200 C5 B0\0",
    "G0 A6.2831 Y200 C5 B0\0",
    "G0 A6.2831 Y200 C5 B0\0",
    "G0 A6.2831 Y200 C5 B0\0",
    "G0 A6.2831 Y280 C5 B0\0",
    "G0 A6.2831 Y280 C5 B0\0",
    "G0 A6.2831 Y280 C5 B0\0",
    "G0 A6.2831 Y280 C5 B0\0",
    "G0 A6.2831 Y280 C5 B0\0",
    "G0 A6.2831 Y280 C5 B0\0",
    "G0 A6.2831 Y280 C5 B0\0",
    "G0 A6.2831 Y280 C5 B0\0",
    "G0 C5 B0\0",
    "G0 C5 B0\0",
    "G0 C5 B0\0",
    "G0 C5 B0\0",
    "G0 C5 B0\0",
    "G0 C2.5 B0\0",
    "G0 C2.5 B0\0",
    "G0 C2.5 B0\0",
    "G0 C2.5 B0\0",
    "G0 C2.5 B0\0",
    "G0 C2.5 B0\0",
    "G0 C2.5 B0\0",
    "G0 C2.5 B0\0",
    "G0 C2.5 B0\0",
    "G0 C2.5 B0\0",
    "G0 C2.5 B0\0",
    "G0 C2.5 B0\0",
    "G0 C2.5 B0\0",
    "G0 C2.5 B0\0",
    "G0 C2.5 B0\0",
    "G0 C2.5 B0\0",
    "G0 C2.5 B0\0",
    "G0 C2.5 B0\0",
    "G0 B0\0",
    "G0 B0\0",
    "G0 B0\0",
    "G0 B0\0",
    "G0 B0\0"
    "G0 B0\0"
    "G0 B0\0"
]
# ─────────────────────────────── main program ────────────────────────────────
def main():
    tx = transmitter.Transmitter(PORT, BAUD, write_timeout=None, timeout=None)

>>>>>>> a1c92f2b1a320fddc7718ceea9f2a41a2e0c8cc8
    log_q: queue.Queue[str] = queue.Queue()
    threading.Thread(target=reader_task, args=(tx.serial, log_q), daemon=True).start()

    try:
<<<<<<< HEAD
        if args.gcode_file:
            send_from_file(args.gcode_file, tx, log_q)
        else:
            while True:
                while not log_q.empty():
                    print(log_q.get())
                cmd = input("Enter command (exit to quit): ").strip()
                if cmd.lower() == "exit":
                    break
                if not cmd:
                    continue
                if not cmd.endswith("\0"):
                    cmd += "\0"
                print(f"[sending] {cmd}")
                tx.send_msg(transmitter.CommandMessage(cmd))
                wait_for_ack(log_q)
                print("[done]")
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print("Error has occurred:", exc)
=======
        while True:
            # flush any waiting log lines *before* we prompt
            while not log_q.empty():
                print(log_q.get())

            cmd = input("Enter command (exit to quit): ").strip()
            if cmd.lower() == "exit":
                break
            if not cmd:
                continue
            if not cmd.endswith("\0"):
                cmd += "\0"
            cmd2 = cmd3
            while(1):
                for i in range(len(cmd2)):
                    print(f"[sending] {cmd2[i]}")
                    tx.send_msg(transmitter.CommandMessage(cmd2[i]))
                    time.sleep(1)
            
            tx.send_msg(transmitter.CommandMessage(cmd))
            print("[sent]")
    except:
        print("Error has occurred:", sys.exc_info()[0])
>>>>>>> a1c92f2b1a320fddc7718ceea9f2a41a2e0c8cc8
    finally:
        tx.serial.close()
        print("Serial connection closed.")

if __name__ == "__main__":
    main()