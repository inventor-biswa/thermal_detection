#!/usr/bin/env python3
"""
Motor Health ML Training Pipeline
===================================
Trains a classifier to predict motor health from AMG8833 thermal data.

Usage:
    python train_model.py --data thermal_data/
    python train_model.py --data thermal_data/ --model svm --visualize

Expects JSONL files in the data directory created by serial_collector.py.
Each line: {"label": "healthy", "grid": [[...8 values...], ...8 rows...], ...}
"""

import argparse
import glob
import json
import os
import sys
import pickle
from datetime import datetime

import numpy as np


def load_dataset(data_dir: str) -> tuple[list[np.ndarray], list[str]]:
    """Load all .jsonl files from the data directory."""
    grids = []
    labels = []

    jsonl_files = glob.glob(os.path.join(data_dir, "*.jsonl"))
    if not jsonl_files:
        print(f"[✗] No .jsonl files found in '{data_dir}'")
        print("    Run serial_collector.py first to collect data.")
        sys.exit(1)

    print(f"[…] Loading data from {len(jsonl_files)} file(s)...")

    for filepath in sorted(jsonl_files):
        fname = os.path.basename(filepath)
        count = 0
        with open(filepath, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"  [!] Skipping malformed line {line_num} in {fname}")
                    continue

                grid = record.get("grid")
                label = record.get("label")

                if grid is None or label is None:
                    continue
                if len(grid) != 8 or any(len(row) != 8 for row in grid):
                    continue

                grids.append(np.array(grid, dtype=np.float32))
                labels.append(label)
                count += 1

        print(f"  ✓ {fname}: {count} frames (label: {labels[-1] if count > 0 else 'N/A'})")

    print(f"\n[✓] Loaded {len(grids)} total frames")

    # Summary by label
    unique_labels = sorted(set(labels))
    print(f"    Labels: {unique_labels}")
    for lbl in unique_labels:
        n = labels.count(lbl)
        print(f"    - {lbl}: {n} frames ({n/len(labels)*100:.1f}%)")

    return grids, labels


def extract_features(grid: np.ndarray) -> np.ndarray:
    """
    Extract meaningful features from an 8x8 thermal grid.

    Features (22 total):
        Statistical (5):  max, min, mean, std, range
        Spatial (6):      hotspot_row, hotspot_col, coldspot_row, coldspot_col,
                          center_temp (avg of center 4 pixels), edge_temp (avg of border)
        Gradient (4):     mean gradient magnitude, max gradient, horizontal_grad, vertical_grad
        Pattern (4):      left_right_diff, top_bottom_diff, quadrant_std, diagonal_diff
        Distribution (3): skewness, kurtosis, percent_above_mean
    """
    flat = grid.flatten()

    # === Statistical features ===
    f_max = np.max(flat)
    f_min = np.min(flat)
    f_mean = np.mean(flat)
    f_std = np.std(flat)
    f_range = f_max - f_min

    # === Spatial features ===
    hotspot = np.unravel_index(np.argmax(grid), grid.shape)
    coldspot = np.unravel_index(np.argmin(grid), grid.shape)
    f_hot_row, f_hot_col = hotspot[0], hotspot[1]
    f_cold_row, f_cold_col = coldspot[0], coldspot[1]

    # Center 4 pixels (3:5, 3:5)
    f_center = np.mean(grid[3:5, 3:5])
    # Edge pixels (border ring)
    edge = np.concatenate([grid[0, :], grid[7, :], grid[1:7, 0], grid[1:7, 7]])
    f_edge = np.mean(edge)

    # === Gradient features ===
    grad_y = np.diff(grid, axis=0)  # vertical gradient (7x8)
    grad_x = np.diff(grid, axis=1)  # horizontal gradient (8x7)
    grad_mag = np.sqrt(
        np.mean(grad_y**2) + np.mean(grad_x**2)
    )
    f_grad_mean = grad_mag
    f_grad_max = max(np.max(np.abs(grad_y)), np.max(np.abs(grad_x)))
    f_grad_h = np.mean(grad_x)  # average horizontal gradient
    f_grad_v = np.mean(grad_y)  # average vertical gradient

    # === Pattern features (symmetry / distribution) ===
    left = grid[:, :4]
    right = grid[:, 4:]
    f_lr_diff = np.mean(left) - np.mean(right)

    top = grid[:4, :]
    bottom = grid[4:, :]
    f_tb_diff = np.mean(top) - np.mean(bottom)

    # Quadrant standard deviation
    q1 = np.mean(grid[:4, :4])
    q2 = np.mean(grid[:4, 4:])
    q3 = np.mean(grid[4:, :4])
    q4 = np.mean(grid[4:, 4:])
    f_quad_std = np.std([q1, q2, q3, q4])

    # Diagonal difference
    diag1 = np.mean([grid[i, i] for i in range(8)])
    diag2 = np.mean([grid[i, 7 - i] for i in range(8)])
    f_diag_diff = abs(diag1 - diag2)

    # === Distribution features ===
    n = len(flat)
    if f_std > 0:
        f_skewness = np.mean(((flat - f_mean) / f_std) ** 3)
        f_kurtosis = np.mean(((flat - f_mean) / f_std) ** 4) - 3
    else:
        f_skewness = 0.0
        f_kurtosis = 0.0
    f_pct_above = np.sum(flat > f_mean) / n

    return np.array([
        f_max, f_min, f_mean, f_std, f_range,
        f_hot_row, f_hot_col, f_cold_row, f_cold_col, f_center, f_edge,
        f_grad_mean, f_grad_max, f_grad_h, f_grad_v,
        f_lr_diff, f_tb_diff, f_quad_std, f_diag_diff,
        f_skewness, f_kurtosis, f_pct_above,
    ], dtype=np.float32)


FEATURE_NAMES = [
    "max", "min", "mean", "std", "range",
    "hotspot_row", "hotspot_col", "coldspot_row", "coldspot_col", "center_temp", "edge_temp",
    "grad_mean", "grad_max", "grad_h", "grad_v",
    "lr_diff", "tb_diff", "quad_std", "diag_diff",
    "skewness", "kurtosis", "pct_above_mean",
]


def train_and_evaluate(X: np.ndarray, y: np.ndarray, model_type: str = "rf",
                       output_dir: str = "models"):
    """Train a classifier and evaluate with cross-validation."""
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report, confusion_matrix

    os.makedirs(output_dir, exist_ok=True)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_

    print(f"\n{'='*55}")
    print(f"  Training: {model_type.upper()} Classifier")
    print(f"  Classes:  {list(class_names)}")
    print(f"  Samples:  {len(y_encoded)}")
    print(f"  Features: {X.shape[1]}")
    print(f"{'='*55}\n")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Choose model
    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == "svm":
        model = SVC(kernel="rbf", probability=True, random_state=42)
    elif model_type == "gb":
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    else:
        print(f"[✗] Unknown model type: {model_type}")
        sys.exit(1)

    # Cross-validation
    print("[…] Running 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=5, scoring="accuracy")
    print(f"    CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"    Per fold:    {[f'{s:.3f}' for s in cv_scores]}")

    # Train final model
    print("\n[…] Training final model on 80% split...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(f"\n--- Test Set Results ({len(y_test)} samples) ---\n")
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"{'':>15s}", end="")
    for name in class_names:
        print(f"{name:>12s}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>15s}", end="")
        for val in row:
            print(f"{val:>12d}", end="")
        print()

    # Feature importance (for tree-based models)
    if hasattr(model, "feature_importances_"):
        print(f"\n--- Top 10 Most Important Features ---")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for rank, idx in enumerate(indices[:10], 1):
            print(f"  {rank:2d}. {FEATURE_NAMES[idx]:<20s}  {importances[idx]:.4f}")

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_dir, f"motor_health_{model_type}_{timestamp}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "label_encoder": le,
                      "feature_names": FEATURE_NAMES}, f)
    print(f"\n[✓] Model saved to: {model_path}")

    return model, scaler, le


def visualize_data(X: np.ndarray, y: np.ndarray, grids: list[np.ndarray],
                   labels: list[str]):
    """Generate visual analysis of the dataset."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] matplotlib not installed. Skipping visualization.")
        print("    Install with: pip install matplotlib")
        return

    os.makedirs("plots", exist_ok=True)
    unique_labels = sorted(set(labels))

    # 1. Average thermal image per class
    fig, axes = plt.subplots(1, len(unique_labels), figsize=(5 * len(unique_labels), 4))
    if len(unique_labels) == 1:
        axes = [axes]
    for ax, lbl in zip(axes, unique_labels):
        mask = [i for i, l in enumerate(labels) if l == lbl]
        avg_grid = np.mean([grids[i] for i in mask], axis=0)
        im = ax.imshow(avg_grid, cmap="inferno", interpolation="bilinear")
        ax.set_title(f"{lbl}\n(n={len(mask)})")
        plt.colorbar(im, ax=ax, label="°C")
    fig.suptitle("Average Thermal Image by Motor Condition", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/average_thermal_by_class.png", dpi=150)
    print("[✓] Saved: plots/average_thermal_by_class.png")

    # 2. Feature distribution boxplots for key features
    key_features = ["max", "mean", "std", "range", "grad_mean", "center_temp"]
    key_indices = [FEATURE_NAMES.index(f) for f in key_features]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feat_name, feat_idx in zip(axes.flatten(), key_features, key_indices):
        data_by_label = []
        for lbl in unique_labels:
            mask = [i for i, l in enumerate(labels) if l == lbl]
            data_by_label.append(X[mask, feat_idx])
        ax.boxplot(data_by_label, tick_labels=unique_labels)
        ax.set_title(feat_name, fontweight="bold")
        ax.set_ylabel("Value")
    fig.suptitle("Key Feature Distributions by Motor Condition", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/feature_distributions.png", dpi=150)
    print("[✓] Saved: plots/feature_distributions.png")

    plt.close("all")


def predict_single(model_path: str, grid: np.ndarray):
    """Make a prediction for a single 8x8 thermal grid."""
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    scaler = bundle["scaler"]
    le = bundle["label_encoder"]

    features = extract_features(grid).reshape(1, -1)
    features_scaled = scaler.transform(features)

    prediction = le.inverse_transform(model.predict(features_scaled))[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features_scaled)[0]
        confidence = {le.inverse_transform([i])[0]: round(float(p), 3)
                      for i, p in enumerate(proba)}
    else:
        confidence = {}

    return prediction, confidence


def main():
    parser = argparse.ArgumentParser(
        description="Train a motor health classifier from thermal imaging data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_model.py --data thermal_data/
  python train_model.py --data thermal_data/ --model svm
  python train_model.py --data thermal_data/ --model rf --visualize
        """,
    )
    parser.add_argument("--data", required=True, help="Path to thermal_data directory")
    parser.add_argument("--model", default="rf", choices=["rf", "svm", "gb"],
                        help="Model type: rf (Random Forest), svm (SVM), gb (Gradient Boosting)")
    parser.add_argument("--output", default="models", help="Directory to save trained models")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization plots")

    args = parser.parse_args()

    # Load data
    grids, labels = load_dataset(args.data)

    if len(set(labels)) < 2:
        print(f"\n[✗] Need at least 2 different labels to train a classifier.")
        print(f"    Found only: {set(labels)}")
        print(f"    Collect data with different --label values using serial_collector.py")
        sys.exit(1)

    # Extract features
    print(f"\n[…] Extracting {len(FEATURE_NAMES)} features from {len(grids)} frames...")
    X = np.array([extract_features(g) for g in grids])
    y = np.array(labels)
    print(f"[✓] Feature matrix: {X.shape}")

    # Visualize
    if args.visualize:
        print("\n[…] Generating visualizations...")
        visualize_data(X, y, grids, labels)

    # Train
    train_and_evaluate(X, y, model_type=args.model, output_dir=args.output)


if __name__ == "__main__":
    main()
