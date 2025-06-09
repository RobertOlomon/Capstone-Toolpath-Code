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
            if nl == -1:
                break
            line = buf[:nl]
            del buf[:nl+1]
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
    log_q: queue.Queue[str] = queue.Queue()
    threading.Thread(target=reader_task, args=(tx.serial, log_q), daemon=True).start()

    try:
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
    finally:
        tx.serial.close()
        print("Serial connection closed.")

if __name__ == "__main__":
    main()