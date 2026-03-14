#!/usr/bin/env python3
"""
app.py — Motor Health Web Dashboard Server
Flask + Flask-SocketIO server for Raspberry Pi.

Usage:
    source venv/bin/activate
    python app.py --port /dev/ttyACM0
    python app.py --port /dev/ttyACM0 --model models/motor_health_rf_*.pkl

Then open: http://raspberrypi.local:5000  (or http://<Pi-IP>:5000)
"""

# ── Fix: pre-register stdlib 'code' before Flask/werkzeug imports it ──────────
# werkzeug.debug.console does `import code` which finds our PyGamer code.py.
# Solution: load the real stdlib code.py directly and register it first.
import importlib.util as _ilu, sysconfig as _sc, os as _os, sys as _sys
_stdlib_dir = _sc.get_path('stdlib')
_code_file  = _os.path.join(_stdlib_dir, 'code.py')
if _os.path.exists(_code_file) and 'code' not in _sys.modules:
    _spec = _ilu.spec_from_file_location('code', _code_file)
    _mod  = _ilu.module_from_spec(_spec)
    _sys.modules['code'] = _mod
    _spec.loader.exec_module(_mod)
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import glob
import json
import os
import pickle
import subprocess
import sys
import threading
import time
from datetime import datetime

import serial
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

# Local modules
from mpu_reader import MPUReader
try:
    from train_model import extract_features
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

import numpy as np

# ── Flask App Setup ───────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "thermal_motor_health_2024"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Global State ──────────────────────────────────────────────────────────────
serial_port   = None
baud_rate     = 115200
ml_model      = None
mpu           = MPUReader()
collecting    = False
collect_label = ""
collect_file  = None

last_thermal_frame = None
last_prediction    = {"label": "N/A", "confidence": 0.0, "color": "gray"}
last_vib_snapshot  = {}

# ── Label Normalisation ───────────────────────────────────────────────────────
LABEL_MAP = {
    "moderate_healthy": "healthy",
    "moderate healthy": "healthy",
    "moderately_healthy": "healthy",
}

def normalise_label(raw: str) -> str:
    key = raw.strip().lower()
    if key in LABEL_MAP:
        return LABEL_MAP[key]
    if "healthy" in key and "un" not in key:
        return "healthy"
    return key


# ── Serial Thermal Reader ─────────────────────────────────────────────────────
class ThermalSerialReader(threading.Thread):
    def __init__(self, port, baud):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.connected = False
        self.error = None
        self._running = True

    def run(self):
        global last_thermal_frame, last_prediction, collecting, collect_file
        try:
            ser = serial.Serial(self.port, self.baud, timeout=1)
            self.connected = True
            time.sleep(1)
            ser.reset_input_buffer()
            print(f"[Serial] Connected to {self.port}")

            while self._running:
                try:
                    raw = ser.readline().decode("utf-8", errors="replace").strip()
                except serial.SerialException:
                    self.connected = False
                    self.error = "Serial disconnected"
                    break

                if not raw.startswith("THERMAL:"):
                    continue

                try:
                    frame = json.loads(raw[8:])
                except json.JSONDecodeError:
                    continue

                grid = frame.get("grid", [])
                if len(grid) != 8 or any(len(r) != 8 for r in grid):
                    continue

                last_thermal_frame = frame

                # Save to file if collecting
                if collecting and collect_file:
                    record = {
                        "timestamp": datetime.now().isoformat(),
                        "label": collect_label,
                        "max_c": frame.get("max_c"),
                        "min_c": frame.get("min_c"),
                        "avg_c": frame.get("avg_c"),
                        "grid": grid,
                    }
                    collect_file.write(json.dumps(record) + "\n")
                    collect_file.flush()

                # Run ML inference
                if ml_model and ML_AVAILABLE:
                    try:
                        grid_np = np.array(grid, dtype=np.float32)
                        features = extract_features(grid_np).reshape(1, -1)
                        features_scaled = ml_model["scaler"].transform(features)
                        raw_pred = ml_model["label_encoder"].inverse_transform(
                            ml_model["model"].predict(features_scaled)
                        )[0]
                        pred = normalise_label(raw_pred)
                        if hasattr(ml_model["model"], "predict_proba"):
                            proba = ml_model["model"].predict_proba(features_scaled)[0]
                            conf = float(max(proba))
                        else:
                            conf = 0.0
                        color = "green" if pred == "healthy" else "orange"
                        last_prediction = {"label": pred, "confidence": conf, "color": color}
                    except Exception as e:
                        last_prediction = {"label": "error", "confidence": 0.0, "color": "gray"}

                # Broadcast via WebSocket
                socketio.emit("thermal_frame", {
                    "grid":   grid,
                    "max_c":  frame.get("max_c"),
                    "min_c":  frame.get("min_c"),
                    "avg_c":  frame.get("avg_c"),
                    "prediction": last_prediction,
                })

        except serial.SerialException as e:
            self.error = str(e)
            self.connected = False
            print(f"[Serial] Error: {e}")

    def stop(self):
        self._running = False


# ── Vibration Broadcast Loop ──────────────────────────────────────────────────
def vibration_loop():
    global last_vib_snapshot
    while True:
        snap = mpu.get_snapshot()
        last_vib_snapshot = snap
        socketio.emit("vibration", snap)
        time.sleep(0.1)  # 10Hz to browser


# ── Flask Routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def status():
    return jsonify({
        "serial_port": serial_port,
        "serial_connected": reader.connected if 'reader' in globals() else False,
        "mpu_available": mpu.available,
        "model_loaded": ml_model is not None,
        "collecting": collecting,
        "collect_label": collect_label,
    })


@app.route("/collect/start", methods=["POST"])
def collect_start():
    global collecting, collect_label, collect_file
    if collecting:
        return jsonify({"error": "Already collecting"}), 400
    data = request.get_json()
    label = data.get("label", "unknown").strip().replace(" ", "_")
    os.makedirs("thermal_data", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"thermal_data/thermal_{label}_{ts}.jsonl"
    collect_file = open(path, "w")
    collect_label = label
    collecting = True
    return jsonify({"status": "started", "label": label, "file": path})


@app.route("/collect/stop", methods=["POST"])
def collect_stop():
    global collecting, collect_file
    if not collecting:
        return jsonify({"error": "Not collecting"}), 400
    collecting = False
    if collect_file:
        collect_file.close()
        collect_file = None
    return jsonify({"status": "stopped"})


@app.route("/train", methods=["POST"])
def train():
    """Trigger model training in background."""
    def run_training():
        try:
            result = subprocess.run(
                [sys.executable, "train_model.py", "--data", "thermal_data/", "--model", "rf"],
                capture_output=True, text=True, timeout=300
            )
            socketio.emit("train_complete", {
                "success": result.returncode == 0,
                "output": result.stdout[-2000:] if result.stdout else result.stderr[-2000:]
            })
            # Reload the model
            load_latest_model()
        except subprocess.TimeoutExpired:
            socketio.emit("train_complete", {"success": False, "output": "Timed out after 5 min"})

    threading.Thread(target=run_training, daemon=True).start()
    return jsonify({"status": "training started"})


@app.route("/models")
def list_models():
    models = sorted(glob.glob("models/*.pkl"), reverse=True)
    return jsonify({"models": [os.path.basename(m) for m in models]})


@app.route("/models/load", methods=["POST"])
def load_model_route():
    data = request.get_json()
    name = data.get("name")
    path = os.path.join("models", name)
    if not os.path.exists(path):
        return jsonify({"error": "Model not found"}), 404
    load_model(path)
    return jsonify({"status": "loaded", "model": name})


# ── Model Helpers ─────────────────────────────────────────────────────────────
def load_model(path: str):
    global ml_model
    try:
        with open(path, "rb") as f:
            ml_model = pickle.load(f)
        classes = list(ml_model["label_encoder"].classes_)
        print(f"[Model] Loaded: {classes} from {path}")
    except Exception as e:
        print(f"[Model] Failed to load: {e}")


def load_latest_model():
    models = sorted(glob.glob("models/*.pkl"), reverse=True)
    if models:
        load_model(models[0])


# ── SocketIO Events ───────────────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    print(f"[WS] Client connected")
    # Send current state immediately
    emit("status", {
        "mpu_available": mpu.available,
        "model_loaded": ml_model is not None,
        "serial_port": serial_port,
    })


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motor Health Web Dashboard")
    parser.add_argument("--port", required=True, help="Serial port for PyGamer (e.g. /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--model", default=None, help="Path to trained model .pkl")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--web-port", type=int, default=5000, help="Web server port (default: 5000)")
    args = parser.parse_args()

    serial_port = args.port
    baud_rate   = args.baud

    # Load model
    if args.model:
        load_model(args.model)
    else:
        load_latest_model()

    # Start MPU6050
    mpu.start()

    # Start serial reader
    reader = ThermalSerialReader(serial_port, baud_rate)
    reader.start()

    # Start vibration broadcast loop
    vib_thread = threading.Thread(target=vibration_loop, daemon=True)
    vib_thread.start()

    print(f"\n{'='*50}")
    print(f"  Motor Health Dashboard")
    print(f"{'='*50}")
    print(f"  URL:    http://0.0.0.0:{args.web_port}")
    print(f"  Serial: {serial_port}")
    print(f"  MPU:    {'Connected' if mpu.available else 'Not found'}")
    print(f"  Model:  {'Loaded' if ml_model else 'None'}")
    print(f"  Open:   http://raspberrypi.local:{args.web_port}")
    print(f"{'='*50}\n")

    socketio.run(app, host=args.host, port=args.web_port, debug=False)
