#!/usr/bin/env python3
"""
Thermal Data Collector for Motor Health ML Pipeline
====================================================
Reads JSON thermal frames from PyGamer (AMG8833) over USB serial.
Lets you label data in real-time for supervised learning.

Usage:
    python serial_collector.py --port COM5 --label healthy
    python serial_collector.py --port /dev/ttyACM0 --label overheating
    python serial_collector.py --port COM5 --label healthy --duration 120

Controls:
    Ctrl+C  → Stop collection and save
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import serial


def find_thermal_line(line: str) -> dict | None:
    """Parse a serial line looking for THERMAL: JSON data."""
    line = line.strip()
    if line.startswith("THERMAL:"):
        try:
            return json.loads(line[8:])  # Skip "THERMAL:" prefix
        except json.JSONDecodeError:
            return None
    return None


def collect_data(port: str, baud: int, label: str, output_dir: str,
                 duration: int | None = None, interval: float = 0.5):
    """
    Collect thermal frames from PyGamer serial and save to a labeled JSON-lines file.

    Args:
        port: Serial port (e.g., COM5 or /dev/ttyACM0)
        baud: Baud rate (default 115200)
        label: Label for this session (e.g., 'healthy', 'overheating', 'misaligned')
        output_dir: Directory to save data files
        duration: Optional max seconds to collect
        interval: Minimum seconds between saved frames
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"thermal_{label}_{timestamp}.jsonl"
    filepath = os.path.join(output_dir, filename)

    print(f"╔══════════════════════════════════════════════╗")
    print(f"║  Thermal Data Collector                      ║")
    print(f"╠══════════════════════════════════════════════╣")
    print(f"║  Port:     {port:<33s} ║")
    print(f"║  Label:    {label:<33s} ║")
    print(f"║  Saving:   {filename:<33s} ║")
    if duration:
        print(f"║  Duration: {duration:<33d} ║")
    print(f"╠══════════════════════════════════════════════╣")
    print(f"║  Press Ctrl+C to stop collection             ║")
    print(f"╚══════════════════════════════════════════════╝")
    print()

    frame_count = 0
    start_time = time.time()
    last_save_time = 0

    try:
        ser = serial.Serial(port, baud, timeout=1)
        print(f"[✓] Connected to {port} at {baud} baud")
        print(f"[…] Waiting for thermal data...\n")

        # Give the board time to reset after serial connection
        time.sleep(2)
        ser.reset_input_buffer()

        with open(filepath, "w") as f:
            while True:
                # Check duration limit
                elapsed = time.time() - start_time
                if duration and elapsed >= duration:
                    print(f"\n[✓] Duration limit ({duration}s) reached.")
                    break

                # Read serial line
                try:
                    raw = ser.readline().decode("utf-8", errors="replace")
                except serial.SerialException as e:
                    print(f"\n[✗] Serial error: {e}")
                    break

                if not raw:
                    continue

                # Parse thermal data
                frame = find_thermal_line(raw)
                if frame is None:
                    continue

                # Rate-limit saving
                now = time.time()
                if now - last_save_time < interval:
                    continue
                last_save_time = now

                # Validate the frame has 8x8 grid
                grid = frame.get("grid", [])
                if len(grid) != 8 or any(len(row) != 8 for row in grid):
                    print(f"  [!] Skipped malformed frame (grid shape != 8x8)")
                    continue

                # Add metadata
                record = {
                    "timestamp": datetime.now().isoformat(),
                    "elapsed_s": round(elapsed, 2),
                    "label": label,
                    "max_c": frame.get("max_c"),
                    "min_c": frame.get("min_c"),
                    "avg_c": frame.get("avg_c"),
                    "grid": grid,
                }

                f.write(json.dumps(record) + "\n")
                f.flush()
                frame_count += 1

                # Live status
                max_t = record["max_c"]
                min_t = record["min_c"]
                avg_t = record["avg_c"]
                print(
                    f"\r  Frame {frame_count:4d} | "
                    f"Max: {max_t:5.1f}°C  Min: {min_t:5.1f}°C  Avg: {avg_t:5.1f}°C | "
                    f"Elapsed: {elapsed:6.1f}s",
                    end="",
                )

    except serial.SerialException as e:
        print(f"\n[✗] Could not open serial port: {e}")
        print("    Tip: Check the port name with 'Device Manager' (Windows)")
        print("         or 'ls /dev/ttyACM*' (Linux/Mac)")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n\n[✓] Collection stopped by user.")
    finally:
        if "ser" in dir() and ser.is_open:
            ser.close()

    print(f"\n{'='*50}")
    print(f"  Collection Summary")
    print(f"{'='*50}")
    print(f"  Frames saved:  {frame_count}")
    print(f"  Label:         {label}")
    print(f"  File:          {filepath}")
    print(f"  Duration:      {time.time() - start_time:.1f}s")
    print(f"{'='*50}")

    if frame_count == 0:
        print("\n  [!] No frames were captured.")
        print("      Make sure the PyGamer is running the modified code.py")
        print("      and DATA_STREAM = True is set.")

    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Collect thermal imaging data from PyGamer for ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python serial_collector.py --port COM5 --label healthy
  python serial_collector.py --port COM5 --label overheating --duration 120
  python serial_collector.py --port /dev/ttyACM0 --label misaligned --interval 1.0
        """,
    )
    parser.add_argument("--port", required=True, help="Serial port (e.g., COM5, /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument(
        "--label",
        required=True,
        help="Label for this session (e.g., healthy, overheating, misaligned, worn_bearing)",
    )
    parser.add_argument("--output", default="thermal_data", help="Output directory (default: thermal_data)")
    parser.add_argument("--duration", type=int, default=None, help="Max seconds to collect (default: unlimited)")
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Min seconds between saved frames (default: 0.5)",
    )

    args = parser.parse_args()
    collect_data(args.port, args.baud, args.label, args.output, args.duration, args.interval)


if __name__ == "__main__":
    main()
