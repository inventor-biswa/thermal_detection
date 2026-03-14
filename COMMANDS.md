# Thermal Camera Motor Health Monitor — Full Command Reference

A complete step-by-step guide to set up, collect data, train, and run the motor health ML pipeline using the Adafruit PyGamer + AMG8833 thermal sensor.

---

## Prerequisites

### Hardware Required
- Adafruit PyGamer (with AMG8833 thermal sensor connected via I2C)
- USB cable (PyGamer → Laptop)

### Software Required
- Python 3.10+
- Windows PowerShell or Terminal

---

## Step 0 — Install Dependencies

Run once at the start:

```bash
pip install pyserial numpy scikit-learn matplotlib pygame flask flask-socketio eventlet smbus2
```

### Raspberry Pi One-Step Setup
Downloads the fixed script and installs all dependencies including Flask, smbus2, and ARM-optimised packages:
```bash
curl -O https://raw.githubusercontent.com/inventor-biswa/thermal_detection/main/setup_pi.sh
chmod +x setup_pi.sh
./setup_pi.sh
```

---

## 🌐 Web Dashboard (Raspberry Pi — Recommended)

```bash
source venv/bin/activate
python app.py --port /dev/ttyACM0

# OR use the launcher (option 1):
bash ~/run_thermal.sh
```

Then open from **any device on the same network**:
```
http://raspberrypi.local:5000
```

**Includes:** Live thermal heatmap • MPU6050 vibration chart • ML prediction • Collect/Train from browser

---

## Step 1 — Upload Code to PyGamer

1. Plug in the PyGamer via USB — it appears as a drive called **`CIRCUITPY`**
2. Copy **only** `code.py` from your project folder to the `CIRCUITPY` drive:

```
Source:      d:\Thynx\College Projects\Thermal_imaging_diy\code.py
Destination: CIRCUITPY:\code.py   (replace existing file)
```

> The board auto-restarts instantly. No compile step needed.

---

## Step 2 — Find the Serial Port

**Windows:** Open Device Manager → Ports (COM & LPT) → look for "USB Serial Device (COMx)"

Verify it's printing data (optional sanity check):
```bash
python -m serial.tools.miniterm COM10 115200
```
You should see lines like:
```
THERMAL:{"ts": 12.3, "max_c": 34.5, "min_c": 22.1, "avg_c": 28.3, "grid": [[...]]}
```
Press `Ctrl+]` to exit miniterm.

---

## Step 3 — Collect Training Data

Run once for each motor condition. Replace `COM10` with your actual port.

```bash
# Healthy motor (~2 minutes)
python serial_collector.py --port COM10 --label healthy --duration 120

# Unhealthy motor (~2 minutes)
python serial_collector.py --port COM10 --label unhealthy --duration 120
```

**Optional flags:**
```bash
--duration 120     # seconds to collect (default: unlimited, stop with Ctrl+C)
--interval 0.5     # seconds between saved frames (default: 0.5)
--output my_data/  # custom output folder (default: thermal_data/)
```

Data is saved to `thermal_data/thermal_<label>_<timestamp>.jsonl`

> Aim for **200+ frames per label** for reliable accuracy.

---

## Step 4 — Train the ML Model

```bash
python train_model.py --data thermal_data/ --model rf --visualize
```

**Model options (`--model`):**
| Flag | Model |
|---|---|
| `rf` | Random Forest *(default, recommended)* |
| `svm` | Support Vector Machine |
| `gb` | Gradient Boosting |

**Output:**
- Cross-validation accuracy + confusion matrix printed to terminal
- Trained model saved to `models/motor_health_rf_<timestamp>.pkl`
- Plots saved to `plots/` (if `--visualize` used)

---

## Step 5 — Live Thermal Camera UI

```bash
python thermal_ui.py --port COM10
```

With ML predictions overlaid (after Step 4):
```bash
python thermal_ui.py --port COM10 --model models/motor_health_rf_<timestamp>.pkl
```

### UI Keyboard Controls

| Key | Action |
|---|---|
| `H` | Toggle Hold (freeze frame) |
| `F` | Toggle Focus (auto-range to current min/max) |
| `P` | Cycle palette: Iron → Grayscale → Inferno |
| `S` | Save screenshot to `screenshots/` |
| `+` / `-` | Adjust alarm threshold |
| `Q` / `Esc` | Quit |

---

## Step 6 — Real-Time Inference (Terminal)

For a lightweight text-only prediction feed (no UI):

```bash
python realtime_predict.py --port COM10 --model models/motor_health_rf_<timestamp>.pkl
```

**Output:**
```
      Time  Prediction       Confidence   Max °C   Avg °C
------------------------------------------------------------
  14:30:01  ✅ healthy           94.2%    45.3    38.7
  14:30:03  ⚠️ unhealthy         87.3%    62.1    55.2
```

**Optional flags:**
```bash
--interval 1.0     # seconds between predictions (default: 1.0)
--baud 115200      # baud rate (default: 115200)
```

Press `Ctrl+C` to stop.

---

## File Reference

| File | Location | Purpose |
|---|---|---|
| `code.py` | PyGamer CIRCUITPY drive | Streams thermal JSON over USB serial |
| `serial_collector.py` | Laptop / RPi | Captures & labels training data |
| `train_model.py` | Laptop / RPi | Trains ML classifier |
| `thermal_ui.py` | Laptop / RPi | pygame live heatmap display |
| `realtime_predict.py` | Laptop / RPi | Text-based live predictions |
| `app.py` | RPi | **Flask web dashboard server** |
| `mpu_reader.py` | RPi | MPU6050 vibration reader thread |
| `templates/index.html` | RPi | Web dashboard HTML |
| `static/app.js` | RPi | WebSocket client + charts |
| `static/style.css` | RPi | Dashboard dark theme |
| `thermalcamera_config.py` | PyGamer | Alarm threshold & range config |
| `setup_pi.sh` | RPi | One-step Pi setup script |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `CIRCUITPY` drive not showing | Double-tap the reset button on PyGamer |
| `No module named 'serial'` | `pip install pyserial` |
| `No module named 'sklearn'` | `pip install scikit-learn` |
| `ModuleNotFoundError: pygame` | `pip install pygame` |
| `ModuleNotFoundError: flask` | `pip install flask flask-socketio eventlet` |
| MPU6050 not found in dashboard | Run `i2cdetect -y 1` — should show `0x68` |
| No `THERMAL:` lines in serial | Check `DATA_STREAM = True` in `code.py` |
| Port not found | Run `ls /dev/ttyACM*` or `dmesg \| tail` |
| Low accuracy after training | Collect more frames (200+ per label) |
