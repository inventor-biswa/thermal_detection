#!/usr/bin/env python3
"""
Real-Time Motor Health Inference
==================================
Reads live thermal data from PyGamer serial and predicts motor health
using a trained model.

Usage:
    python realtime_predict.py --port COM5 --model models/motor_health_rf_20260213.pkl
"""

import argparse
import json
import sys
import pickle
import time

import numpy as np
import serial

from train_model import extract_features


def run_inference(port: str, baud: int, model_path: str, interval: float = 1.0):
    """Run real-time motor health prediction from live thermal data."""

    # Load model bundle
    print(f"[…] Loading model from: {model_path}")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    scaler = bundle["scaler"]
    le = bundle["label_encoder"]
    class_names = list(le.classes_)

    print(f"[✓] Model loaded. Classes: {class_names}")
    print(f"[…] Connecting to {port}...\n")

    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        ser.reset_input_buffer()
        print(f"[✓] Connected. Running inference every {interval}s...")
        print(f"    Press Ctrl+C to stop.\n")
        print(f"{'Time':>10s}  {'Prediction':<15s}  {'Confidence':>10s}  {'Max °C':>7s}  {'Avg °C':>7s}")
        print(f"{'-'*60}")

        last_predict_time = 0

        while True:
            try:
                raw = ser.readline().decode("utf-8", errors="replace").strip()
            except serial.SerialException:
                break

            if not raw.startswith("THERMAL:"):
                continue

            now = time.time()
            if now - last_predict_time < interval:
                continue
            last_predict_time = now

            try:
                frame = json.loads(raw[8:])
            except json.JSONDecodeError:
                continue

            grid = frame.get("grid", [])
            if len(grid) != 8 or any(len(r) != 8 for r in grid):
                continue

            grid_np = np.array(grid, dtype=np.float32)
            features = extract_features(grid_np).reshape(1, -1)
            features_scaled = scaler.transform(features)

            prediction = le.inverse_transform(model.predict(features_scaled))[0]

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features_scaled)[0]
                confidence = max(proba)
            else:
                confidence = 0.0

            max_c = frame.get("max_c", 0)
            avg_c = frame.get("avg_c", 0)

            # Color-code output
            if prediction == "healthy":
                symbol = "✅"
            else:
                symbol = "⚠️"

            timestamp = time.strftime("%H:%M:%S")
            print(f"{timestamp:>10s}  {symbol} {prediction:<13s}  {confidence:>9.1%}  {max_c:>6.1f}  {avg_c:>6.1f}")

    except serial.SerialException as e:
        print(f"\n[✗] Serial error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n\n[✓] Inference stopped.")
    finally:
        if "ser" in dir() and ser.is_open:
            ser.close()


def main():
    parser = argparse.ArgumentParser(description="Real-time motor health prediction from thermal data")
    parser.add_argument("--port", required=True, help="Serial port (e.g., COM5)")
    parser.add_argument("--model", required=True, help="Path to trained model .pkl file")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between predictions (default: 1.0)")
    args = parser.parse_args()

    run_inference(args.port, args.baud, args.model, args.interval)


if __name__ == "__main__":
    main()
