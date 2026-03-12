# Thermal Detection for Motor Health Monitoring

A real-time thermal imaging and Machine Learning project for predictive maintenance of industrial motors. This system uses an **AMG8833 Thermal Sensor** and an **Adafruit PyGamer** to detect and classify motor conditions (Healthy vs. Unhealthy).

## 🚀 Overview

This project provides a complete pipeline to:
1.  **Capture** 8x8 thermal data from an infrared sensor.
2.  **Stream** raw data via USB Serial to a host computer.
3.  **Collect** and label thermal signatures for different motor states.
4.  **Train** a Machine Learning model (Random Forest, SVM, or Gradient Boosting) to predict motor health.
5.  **Visualize** the thermal signatures in a live, high-resolution interpolated UI.

## 🛠️ Hardware Requirements

- **Adafruit PyGamer** (or PyBadge)
- **AMG8833 Thermal Sensor** (connected via I2C)
- USB Cable (PyGamer to Laptop)

## 📦 Software Setup

Ensure you have Python 3.10+ installed, then run:

```bash
pip install pyserial numpy scikit-learn matplotlib pygame
```

### 🍓 Raspberry Pi One-Step Setup

If you are using a Raspberry Pi as your host:
```bash
chmod +x setup_pi.sh
./setup_pi.sh
```
This script handles system updates, handles the `EXTERNALLY-MANAGED` pip restrictions by creating a virtual environment, and optionally enables I2C for direct sensor connection.

## 📋 Project Structure

- `code.py`: CircuitPython firmware for the PyGamer sensor node.
- `serial_collector.py`: Script to collect and label thermal data for ML training.
- `train_model.py`: Extracts 22 features and trains the ML classifier.
- `thermal_ui.py`: A real-time, high-res UI for monitoring and live inference.
- `realtime_predict.py`: Lightweight CLI-based inference tool.
- `PROJECT_OVERVIEW.md`: Detailed documentation of the project's architecture and goals.
- `COMMANDS.md`: A quick reference for all terminal commands.

## 🚦 How to Use

1.  **Deploy Firmware**: Copy `code.py` to your PyGamer's `CIRCUITPY` drive.
2.  **Collect Data**: 
    ```bash
    python serial_collector.py --port COMx --label healthy --duration 120
    python serial_collector.py --port COMx --label unhealthy --duration 120
    ```
3.  **Train Model**:
    ```bash
    python train_model.py --data thermal_data/ --visualize
    ```
4.  **Run Live UI**:
    ```bash
    python thermal_ui.py --port COMx --model models/your_trained_model.pkl
    ```

## 🔒 Note on Data & Models

The following directories are **ignored by Git** (via `.gitignore`) because they contain temporary, large, or machine-specific data:
- `thermal_data/`: Raw collected thermal frames.
- `models/`: Trained ML model binaries (`.pkl`).
- `plots/`: Generated training visualizations.
- `screenshots/`: Captured thermal camera images.

You will need to run the collection and training steps yourself to generate these for your specific hardware environment.

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
