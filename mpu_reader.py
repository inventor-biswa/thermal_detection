#!/usr/bin/env python3
"""
mpu_reader.py — MPU6050 Background Reader (Raspberry Pi only)
Reads accelerometer and gyroscope data via I2C (smbus2).
Falls back gracefully if sensor is not found.
"""

import math
import threading
import time
from collections import deque

# Attempt to import smbus2 — only available on RPi
try:
    import smbus2
    SMBUS_AVAILABLE = True
except ImportError:
    SMBUS_AVAILABLE = False

# ── MPU6050 Register Map ──────────────────────────────────────────────────────
MPU_ADDR        = 0x68
PWR_MGMT_1      = 0x6B
ACCEL_XOUT_H    = 0x3B
GYRO_XOUT_H     = 0x43
ACCEL_CONFIG    = 0x1C
GYRO_CONFIG     = 0x1B

# Sensitivity scales (±2g accel, ±250°/s gyro by default)
ACCEL_SCALE = 16384.0  # LSB/g  for ±2g
GYRO_SCALE  = 131.0    # LSB/°s for ±250°/s

# Vibration severity thresholds (RMS g)
THRESH_NORMAL  = 0.3
THRESH_WARNING = 0.8


class MPUReader:
    """
    Background thread that reads MPU6050 at ~50Hz.
    Thread-safe access to latest readings via properties.
    """

    def __init__(self, i2c_bus: int = 1, address: int = MPU_ADDR,
                 window_sec: float = 0.5, sample_rate_hz: int = 50):
        self.address      = address
        self.window_sec   = window_sec
        self.sample_rate  = sample_rate_hz
        self._interval    = 1.0 / sample_rate_hz

        self.available    = False   # True if sensor found
        self.error        = None

        self._lock        = threading.Lock()
        self._running     = False
        self._thread      = None
        self._bus         = None

        # Rolling window for RMS calculation
        self._window_size = int(window_sec * sample_rate_hz)
        self._mag_window  = deque(maxlen=self._window_size)

        # Latest values
        self._ax = 0.0; self._ay = 0.0; self._az = 0.0
        self._gx = 0.0; self._gy = 0.0; self._gz = 0.0
        self._rms_g  = 0.0
        self._peak_g = 0.0

        # Try to open bus
        if SMBUS_AVAILABLE:
            try:
                self._bus = smbus2.SMBus(i2c_bus)
                # Wake the MPU6050 (clear sleep bit)
                self._bus.write_byte_data(address, PWR_MGMT_1, 0x00)
                time.sleep(0.1)
                self.available = True
            except Exception as e:
                self.error = str(e)
                self.available = False
        else:
            self.error = "smbus2 not installed"

    # ── Public read-only properties ───────────────────────────────────────────

    @property
    def ax(self) -> float:
        with self._lock: return self._ax

    @property
    def ay(self) -> float:
        with self._lock: return self._ay

    @property
    def az(self) -> float:
        with self._lock: return self._az

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
    def rms_g(self) -> float:
        with self._lock: return self._rms_g

    @property
    def peak_g(self) -> float:
        with self._lock: return self._peak_g

    @property
    def status(self) -> str:
        """Returns 'NORMAL', 'WARNING', or 'HIGH' based on RMS."""
        rms = self.rms_g
        if rms < THRESH_NORMAL:
            return "NORMAL"
        elif rms < THRESH_WARNING:
            return "WARNING"
        return "HIGH"

    def get_snapshot(self) -> dict:
        """Return all values as a dict (thread-safe)."""
        with self._lock:
            return {
                "ax": round(self._ax, 4),
                "ay": round(self._ay, 4),
                "az": round(self._az, 4),
                "gx": round(self._gx, 4),
                "gy": round(self._gy, 4),
                "gz": round(self._gz, 4),
                "rms_g":  round(self._rms_g, 4),
                "peak_g": round(self._peak_g, 4),
                "status": self.status,
                "available": self.available,
            }

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        if not self.available:
            print(f"[MPU] Sensor not available: {self.error}")
            return
        self._running = True
        self._thread  = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print("[MPU] Reader started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._bus:
            self._bus.close()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _read_word_2c(self, reg: int) -> float:
        """Read a signed 16-bit word from the MPU6050."""
        high = self._bus.read_byte_data(self.address, reg)
        low  = self._bus.read_byte_data(self.address, reg + 1)
        val  = (high << 8) + low
        return val - 65536 if val >= 0x8000 else val

    def _read_loop(self):
        peak_reset_interval = 2.0  # Reset peak every 2 seconds
        last_peak_reset = time.time()

        while self._running:
            t0 = time.time()
            try:
                ax = self._read_word_2c(ACCEL_XOUT_H)     / ACCEL_SCALE
                ay = self._read_word_2c(ACCEL_XOUT_H + 2) / ACCEL_SCALE
                az = self._read_word_2c(ACCEL_XOUT_H + 4) / ACCEL_SCALE
                gx = self._read_word_2c(GYRO_XOUT_H)      / GYRO_SCALE
                gy = self._read_word_2c(GYRO_XOUT_H + 2)  / GYRO_SCALE
                gz = self._read_word_2c(GYRO_XOUT_H + 4)  / GYRO_SCALE

                # Vibration magnitude (remove gravity component from az)
                vib_az = az - 1.0  # subtract 1g (gravity on Z)
                mag = math.sqrt(ax**2 + ay**2 + vib_az**2)

                self._mag_window.append(mag ** 2)
                rms = math.sqrt(sum(self._mag_window) / len(self._mag_window)) if self._mag_window else 0.0

                now = time.time()
                with self._lock:
                    self._ax, self._ay, self._az = ax, ay, az
                    self._gx, self._gy, self._gz = gx, gy, gz
                    self._rms_g = rms

                    # Reset peak periodically
                    if now - last_peak_reset > peak_reset_interval:
                        self._peak_g = 0.0
                        last_peak_reset = now
                    if mag > self._peak_g:
                        self._peak_g = mag

            except Exception as e:
                self.error = str(e)

            # Sleep for the remainder of the interval
            elapsed = time.time() - t0
            sleep_t = max(0, self._interval - elapsed)
            time.sleep(sleep_t)


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing MPU6050 reader (Ctrl+C to stop)...")
    reader = MPUReader()
    reader.start()
    try:
        while True:
            s = reader.get_snapshot()
            print(f"Ax:{s['ax']:6.3f}  Ay:{s['ay']:6.3f}  Az:{s['az']:6.3f}  "
                  f"RMS:{s['rms_g']:5.3f}g  Peak:{s['peak_g']:5.3f}g  [{s['status']}]")
            time.sleep(0.2)
    except KeyboardInterrupt:
        reader.stop()
        print("Done.")
