#!/usr/bin/env python3
"""
mpu_reader.py — MPU6050 Background Reader (Raspberry Pi only)
Uses the mpu6050-raspberrypi library for reliable I2C reads.

Install:
    pip install mpu6050-raspberrypi

Accel units : m/s²  (divide by 9.81 to get g)
Gyro units  : °/s
"""

import math
import threading
import time
from collections import deque

# Attempt to import mpu6050 library — only available on RPi
try:
    from mpu6050 import mpu6050 as MPU6050Lib
    MPU_LIB_AVAILABLE = True
except ImportError:
    MPU_LIB_AVAILABLE = False

G = 9.81  # m/s² per g

# Vibration severity thresholds (RMS in g)
THRESH_NORMAL  = 0.3
THRESH_WARNING = 0.8


class MPUReader:
    """
    Background thread that reads MPU6050 at ~20Hz using mpu6050-raspberrypi.
    Thread-safe access to latest readings via get_snapshot().
    """

    def __init__(self, address: int = 0x68, window_sec: float = 0.5,
                 sample_rate_hz: int = 20):
        self.address     = address
        self._interval   = 1.0 / sample_rate_hz
        self._window_sz  = int(window_sec * sample_rate_hz)

        self.available   = False
        self.error       = None

        self._lock       = threading.RLock()   # re-entrant so properties can be called inside lock
        self._running    = False
        self._thread     = None
        self._sensor     = None

        self._mag_window = deque(maxlen=self._window_sz)

        # Latest values (SI units stored, g converted on read)
        self._ax_ms2 = 0.0; self._ay_ms2 = 0.0; self._az_ms2 = 0.0
        self._gx     = 0.0; self._gy     = 0.0; self._gz     = 0.0
        self._temp_c = 0.0
        self._rms_g  = 0.0
        self._peak_g = 0.0

        # Try to open sensor
        if MPU_LIB_AVAILABLE:
            try:
                self._sensor = MPU6050Lib(address)
                self.available = True
            except Exception as e:
                self.error = str(e)
        else:
            self.error = "mpu6050-raspberrypi not installed (pip install mpu6050-raspberrypi)"

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def ax(self) -> float:
        """Acceleration X in g"""
        with self._lock: return self._ax_ms2 / G

    @property
    def ay(self) -> float:
        with self._lock: return self._ay_ms2 / G

    @property
    def az(self) -> float:
        with self._lock: return self._az_ms2 / G

    @property
    def gx(self) -> float:
        with self._lock: return self._gx

    @property
    def gy(self) -> float:
        with self._lock: return self._gy

    @property
    def gz(self) -> float:
        with self._lock: return self._gz

    @property
    def temp_c(self) -> float:
        with self._lock: return self._temp_c

    @property
    def rms_g(self) -> float:
        with self._lock: return self._rms_g

    @property
    def peak_g(self) -> float:
        with self._lock: return self._peak_g

    @property
    def status(self) -> str:
        rms = self.rms_g
        if rms < THRESH_NORMAL:
            return "NORMAL"
        elif rms < THRESH_WARNING:
            return "WARNING"
        return "HIGH"

    def get_snapshot(self) -> dict:
        """Return all values as a dict (thread-safe)."""
        with self._lock:
            ax = self._ax_ms2 / G
            ay = self._ay_ms2 / G
            az = self._az_ms2 / G
            rms  = self._rms_g
            peak = self._peak_g
            # Compute status inline — avoids re-acquiring lock via property
            if rms < THRESH_NORMAL:
                status = "NORMAL"
            elif rms < THRESH_WARNING:
                status = "WARNING"
            else:
                status = "HIGH"
            return {
                "ax":        round(ax,           4),
                "ay":        round(ay,           4),
                "az":        round(az,           4),
                "gx":        round(self._gx,     2),
                "gy":        round(self._gy,     2),
                "gz":        round(self._gz,     2),
                "temp_c":    round(self._temp_c, 1),
                "rms_g":     round(rms,          4),
                "peak_g":    round(peak,         4),
                "status":    status,
                "available": self.available,
            }

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        if not self.available:
            print(f"[MPU] Not available: {self.error}")
            return
        self._running = True
        self._thread  = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print("[MPU] Reader started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    # ── Internal read loop ────────────────────────────────────────────────────

    def _read_loop(self):
        peak_reset_interval = 2.0
        last_peak_reset     = time.time()
        consecutive_errors  = 0

        while self._running:
            t0 = time.time()
            try:
                accel = self._sensor.get_accel_data()   # m/s²
                gyro  = self._sensor.get_gyro_data()    # °/s
                temp  = self._sensor.get_temp()         # °C

                ax_ms2 = accel['x']
                ay_ms2 = accel['y']
                az_ms2 = accel['z']
                gx     = gyro['x']
                gy     = gyro['y']
                gz     = gyro['z']

                # Vibration magnitude — remove gravity (≈9.81 on Z)
                vib_ax = ax_ms2 / G
                vib_ay = ay_ms2 / G
                vib_az = az_ms2 / G - 1.0   # subtract 1g upright Z gravity
                mag = math.sqrt(vib_ax**2 + vib_ay**2 + vib_az**2)

                self._mag_window.append(mag ** 2)
                rms = math.sqrt(sum(self._mag_window) / len(self._mag_window))

                consecutive_errors = 0
                now = time.time()

                with self._lock:
                    self._ax_ms2 = ax_ms2
                    self._ay_ms2 = ay_ms2
                    self._az_ms2 = az_ms2
                    self._gx     = gx
                    self._gy     = gy
                    self._gz     = gz
                    self._temp_c = temp
                    self._rms_g  = rms
                    if now - last_peak_reset > peak_reset_interval:
                        self._peak_g   = 0.0
                        last_peak_reset = now
                    if mag > self._peak_g:
                        self._peak_g = mag

            except Exception as e:
                consecutive_errors += 1
                with self._lock:
                    self.error = str(e)
                if consecutive_errors <= 3:
                    print(f"[MPU] Read error ({consecutive_errors}): {e}")
                time.sleep(0.1)

            elapsed = time.time() - t0
            time.sleep(max(0.0, self._interval - elapsed))


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing MPU6050 reader (Ctrl+C to stop)...")
    reader = MPUReader()
    if not reader.available:
        print(f"ERROR: {reader.error}")
        exit(1)
    reader.start()
    try:
        while True:
            s = reader.get_snapshot()
            print(f"Ax:{s['ax']:+6.3f}g  Ay:{s['ay']:+6.3f}g  Az:{s['az']:+6.3f}g  "
                  f"RMS:{s['rms_g']:5.3f}g  Peak:{s['peak_g']:5.3f}g  "
                  f"Temp:{s['temp_c']:.1f}°C  [{s['status']}]")
            time.sleep(0.2)
    except KeyboardInterrupt:
        reader.stop()
        print("Done.")
