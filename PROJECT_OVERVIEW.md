# Motor Health Monitoring Using Thermal Imaging & Machine Learning

**Project Type:** Embedded Systems + Machine Learning  
**Domain:** Predictive Maintenance / Industrial IoT  
**Hardware:** Adafruit PyGamer, AMG8833 Thermal Sensor  
**Software:** Python, CircuitPython, scikit-learn  

---

## Overview

This project focuses on using **thermal imaging** to monitor the health of electric motors in real time. A low-cost infrared thermal sensor (AMG8833) mounted on an Adafruit PyGamer captures the heat signature of a running motor. The thermal data is streamed to a laptop or Raspberry Pi over USB serial, where a **machine learning classifier** analyzes it to determine whether the motor is **healthy or unhealthy**.

The goal is to demonstrate how affordable embedded hardware and ML can together enable **predictive maintenance** — detecting a motor fault before it leads to failure, reducing downtime and repair costs.

---

## Problem Statement

Industrial motors often fail due to overheating, bearing wear, or overloading. Traditional monitoring systems are expensive and reactive — they raise an alarm only after failure. This project proposes a **low-cost, proactive system** that continuously monitors motor temperature patterns and flags anomalies using a trained ML model.

---

## System Architecture

```
[AMG8833 Thermal Sensor]
         │  I2C (8×8 thermal grid, ~10fps)
         ▼
[Adafruit PyGamer]  ──── USB Serial (JSON) ────►  [Laptop / Raspberry Pi]
  CircuitPython                                       │
  code.py                                             ├── serial_collector.py  (Data Collection)
                                                      ├── train_model.py       (ML Training)
                                                      ├── thermal_ui.py        (Live Heatmap UI)
                                                      └── realtime_predict.py  (Live Prediction)
```

---

## Key Features

- **Live Thermal Heatmap UI** — Replicates the PyGamer display on your laptop with iron pseudocolor palette, interpolated from 8×8 to 400×400 pixels
- **Labeled Data Collection** — Capture and label thermal frames for healthy and unhealthy motors
- **ML Classification** — Train a Random Forest / SVM / Gradient Boosting classifier on 22 extracted features per frame
- **Real-Time Prediction** — Feed live sensor data to the trained model and display health status with confidence score

---

## Technologies Used

| Area | Tools / Libraries |
|---|---|
| Embedded firmware | CircuitPython 8.x, `adafruit_amg88xx`, `ulab` (numpy) |
| Serial communication | `pyserial` |
| Data processing | `numpy` |
| Machine learning | `scikit-learn` (Random Forest, SVM, Gradient Boosting) |
| Visualization | `matplotlib`, `pygame` |
| Data format | JSON Lines (`.jsonl`) |

---

## ML Pipeline

### 1. Data Collection
- Mount the thermal sensor facing the motor
- Record 200+ frames per condition (healthy / unhealthy)
- Each frame = 8×8 grid of temperatures (°C)

### 2. Feature Extraction (22 features per frame)
From each 8×8 thermal frame, the following features are extracted:

| Category | Features |
|---|---|
| Statistical | max, min, mean, std deviation, range |
| Spatial | hotspot location (row, col), coldspot location, center temp, edge temp |
| Gradient | mean gradient, max gradient, horizontal & vertical gradients |
| Pattern | left-right asymmetry, top-bottom asymmetry, quadrant variance, diagonal difference |
| Distribution | skewness, kurtosis, % pixels above mean |

### 3. Training
- 80/20 train-test split with 5-fold cross-validation
- Outputs: accuracy, confusion matrix, feature importance ranking

### 4. Deployment
- Trained model loaded for real-time inference
- Predictions displayed in the live UI or terminal

---

## Expected Outcomes

- A working real-time thermal monitoring system
- A trained binary classifier (healthy / unhealthy) with measurable accuracy
- Understanding of the end-to-end embedded ML pipeline: **Sense → Stream → Classify → Alert**

---

## Learning Outcomes

By completing this project, students will:

1. Interface an I2C thermal sensor with a CircuitPython microcontroller
2. Stream structured sensor data over USB serial using JSON
3. Understand feature engineering for time-series sensor data
4. Train and evaluate a multi-feature ML classifier
5. Build a real-time desktop application with live sensor visualization
6. Apply the concept of predictive maintenance in an industrial context

---

## Project Files

| File | Description |
|---|---|
| `code.py` | CircuitPython firmware for PyGamer — reads sensor, streams JSON |
| `serial_collector.py` | Captures and labels thermal frames to `.jsonl` files |
| `train_model.py` | Extracts features, trains classifier, saves model |
| `thermal_ui.py` | Live heatmap display with ML overlay |
| `realtime_predict.py` | Terminal-based real-time motor health prediction |
| `thermalcamera_config.py` | Alarm threshold and range configuration |
| `COMMANDS.md` | Step-by-step command reference |

---

## Dataset Requirements

| Condition | Description | Minimum Frames |
|---|---|---|
| Healthy | Motor running normally under rated load | 200 |
| Unhealthy | Motor overheating, overloaded, or faulty | 200 |

> **Important:** Keep sensor distance (~30 cm) and angle consistent across all captures.

---

## References

- Adafruit AMG8833 Thermal Camera Library: https://github.com/adafruit/Adafruit_CircuitPython_AMG88xx
- Adafruit PyGamer Learn Guide: https://learn.adafruit.com/improved-amg8833-pygamer-thermal-camera
- scikit-learn Documentation: https://scikit-learn.org/stable/
- CircuitPython Documentation: https://docs.circuitpython.org/
