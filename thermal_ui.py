#!/usr/bin/env python3
"""
Thermal Camera Live UI
========================
Replicates the PyGamer thermal camera display on your laptop.
Reads real-time 8×8 AMG8833 data from serial and renders an
interpolated heatmap with iron pseudocolor palette.

Usage:
    python thermal_ui.py --port COM10
    python thermal_ui.py --port COM10 --model models/motor_health_rf_*.pkl

Controls:
    H        → Toggle HOLD (freeze frame)
    F        → Toggle FOCUS (auto-range to current min/max)
    S        → Screenshot (saves to screenshots/ folder)
    P        → Toggle palette (Iron / Grayscale / Inferno)
    M        → Toggle ML prediction overlay (if model loaded)
    +/-      → Adjust alarm threshold
    ESC / Q  → Quit
"""

import argparse
import json
import math
import os
import pickle
import sys
import threading
import time
from datetime import datetime

import numpy as np

try:
    import pygame
    import pygame.freetype
except ImportError:
    print("[✗] pygame not installed. Run: pip install pygame")
    sys.exit(1)

try:
    import serial
except ImportError:
    print("[✗] pyserial not installed. Run: pip install pyserial")
    sys.exit(1)

# Try to import ML components
try:
    from train_model import extract_features
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


# ── Iron Palette (matches PyGamer's iron.py) ──────────────────────────────────

def map_range(x, in_min, in_max, out_min, out_max):
    in_range = in_max - in_min
    in_delta = x - in_min
    if in_range != 0:
        mapped = in_delta / in_range
    elif in_delta != 0:
        mapped = in_delta
    else:
        mapped = 0.5
    mapped *= out_max - out_min
    mapped += out_min
    if out_min <= out_max:
        return max(min(mapped, out_max), out_min)
    return min(max(mapped, out_max), out_min)


def iron_color(index, gamma=0.5):
    """Iron thermographic pseudocolor: index 0.0–1.0 → (R, G, B)."""
    index = max(0.0, min(1.0, index))
    band = index * 600

    if band < 70:
        red = 0.1
        grn = 0.1
        blu = (0.2 + (0.8 * map_range(band, 0, 70, 0.0, 1.0))) ** gamma
    elif band < 200:
        red = map_range(band, 70, 200, 0.0, 0.6) ** gamma
        grn = 0.0
        blu = 1.0 ** gamma
    elif band < 300:
        red = map_range(band, 200, 300, 0.6, 1.0) ** gamma
        grn = 0.0
        blu = map_range(band, 200, 300, 1.0, 0.0) ** gamma
    elif band < 400:
        red = 1.0 ** gamma
        grn = map_range(band, 300, 400, 0.0, 0.5) ** gamma
        blu = 0.0
    elif band < 500:
        red = 1.0 ** gamma
        grn = map_range(band, 400, 500, 0.5, 1.0) ** gamma
        blu = 0.0
    else:
        red = 1.0 ** gamma
        grn = 1.0 ** gamma
        blu = map_range(band, 500, 580, 0.0, 1.0) ** gamma

    return (int(red * 255), int(grn * 255), int(blu * 255))


def grayscale_color(index, gamma=1.0):
    """Grayscale: index 0.0–1.0 → (R, G, B)."""
    index = max(0.0, min(1.0, index))
    v = int((index ** gamma) * 255)
    return (v, v, v)


def inferno_color(index, gamma=1.0):
    """Simplified inferno-style palette."""
    index = max(0.0, min(1.0, index))
    # Approximate inferno colormap
    r = int(max(0, min(255, (1.46 * index - 0.27) * 255)) ** gamma)
    g = int(max(0, min(255, (1.5 * index ** 2) * 255)) ** gamma)
    b = int(max(0, min(255, (0.7 * math.sin(index * 3.14) + 0.1) * 255)) ** gamma)
    return (r, g, b)


PALETTES = {
    "Iron": iron_color,
    "Grayscale": grayscale_color,
    "Inferno": inferno_color,
}


# ── Precompute palette LUTs ───────────────────────────────────────────────────

def build_palette_lut(color_func, steps=256):
    """Pre-build a lookup table for fast color mapping."""
    return [color_func(i / (steps - 1)) for i in range(steps)]


# ── Bilinear Interpolation ────────────────────────────────────────────────────

def interpolate_grid(grid_8x8, scale=16):
    """Bilinear interpolation of 8×8 grid to (8*scale)×(8*scale)."""
    h, w = 8, 8
    out_h, out_w = h * scale, w * scale
    result = np.zeros((out_h, out_w), dtype=np.float32)

    for y in range(out_h):
        for x in range(out_w):
            gx = x * (w - 1) / (out_w - 1)
            gy = y * (h - 1) / (out_h - 1)

            x0 = int(gx)
            y0 = int(gy)
            x1 = min(x0 + 1, w - 1)
            y1 = min(y0 + 1, h - 1)

            fx = gx - x0
            fy = gy - y0

            val = (grid_8x8[y0][x0] * (1 - fx) * (1 - fy) +
                   grid_8x8[y0][x1] * fx * (1 - fy) +
                   grid_8x8[y1][x0] * (1 - fx) * fy +
                   grid_8x8[y1][x1] * fx * fy)
            result[y][x] = val

    return result


def interpolate_fast(grid_8x8, scale=16):
    """Vectorized bilinear interpolation — much faster."""
    h, w = 8, 8
    out_h, out_w = h * scale, w * scale
    grid = np.array(grid_8x8, dtype=np.float32)

    # Create coordinate grids
    y_out = np.linspace(0, h - 1, out_h)
    x_out = np.linspace(0, w - 1, out_w)
    xv, yv = np.meshgrid(x_out, y_out)

    x0 = np.floor(xv).astype(int)
    y0 = np.floor(yv).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)

    fx = xv - x0
    fy = yv - y0

    result = (grid[y0, x0] * (1 - fx) * (1 - fy) +
              grid[y0, x1] * fx * (1 - fy) +
              grid[y1, x0] * (1 - fx) * fy +
              grid[y1, x1] * fx * fy)

    return result


# ── Serial Reader Thread ──────────────────────────────────────────────────────

class ThermalReader:
    """Background thread to read thermal data from serial port."""

    def __init__(self, port, baud=115200):
        self.port = port
        self.baud = baud
        self.latest_frame = None
        self.frame_count = 0
        self.fps = 0.0
        self.connected = False
        self.error = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._fps_times = []

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def get_frame(self):
        with self._lock:
            return self.latest_frame

    def _read_loop(self):
        try:
            ser = serial.Serial(self.port, self.baud, timeout=1)
            self.connected = True
            time.sleep(1)
            ser.reset_input_buffer()

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

                with self._lock:
                    self.latest_frame = frame
                    self.frame_count += 1

                    # FPS calculation
                    now = time.time()
                    self._fps_times.append(now)
                    self._fps_times = [t for t in self._fps_times if now - t < 2.0]
                    if len(self._fps_times) > 1:
                        self.fps = len(self._fps_times) / (self._fps_times[-1] - self._fps_times[0])

        except serial.SerialException as e:
            self.error = str(e)
            self.connected = False


# ── Main UI Application ──────────────────────────────────────────────────────

def run_ui(port, baud, model_path=None):
    pygame.init()

    # ── Window Setup ──
    HEATMAP_SIZE = 400        # Heatmap area (square)
    SIDEBAR_W = 200           # Right sidebar width
    COLORBAR_W = 30           # Color bar width
    BOTTOM_H = 60             # Bottom status bar
    WIN_W = HEATMAP_SIZE + COLORBAR_W + SIDEBAR_W
    WIN_H = HEATMAP_SIZE + BOTTOM_H

    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("🌡️ Thermal Camera — Motor Health Monitor")
    clock = pygame.time.Clock()

    # ── Fonts ──
    try:
        font_large = pygame.font.SysFont("Consolas", 28, bold=True)
        font_med = pygame.font.SysFont("Consolas", 18)
        font_small = pygame.font.SysFont("Consolas", 14)
        font_title = pygame.font.SysFont("Segoe UI", 20, bold=True)
    except Exception:
        font_large = pygame.font.Font(None, 32)
        font_med = pygame.font.Font(None, 22)
        font_small = pygame.font.Font(None, 18)
        font_title = pygame.font.Font(None, 24)

    # ── Colors ──
    BG = (18, 18, 24)
    SIDEBAR_BG = (25, 25, 35)
    BOTTOM_BG = (20, 20, 30)
    WHITE = (255, 255, 255)
    RED = (255, 60, 60)
    YELLOW = (255, 220, 50)
    CYAN = (80, 220, 255)
    GREEN = (60, 220, 100)
    GRAY = (120, 120, 140)
    ORANGE = (255, 160, 40)
    DARK_GRAY = (40, 40, 55)

    # ── State ──
    palette_name = "Iron"
    palette_lut = build_palette_lut(PALETTES[palette_name])
    alarm_f = 120
    min_range_f = 60
    max_range_f = 120
    hold = False
    focus = False
    show_ml = True
    interp_scale = 50  # 8 * 50 = 400px = HEATMAP_SIZE
    held_frame = None

    # ── Label display mapping (normalise labels for display) ──
    # Any label containing 'healthy' but not 'un' is shown as 'healthy'
    LABEL_MAP = {
        "moderate_healthy": "healthy",
        "moderate healthy": "healthy",
        "moderately_healthy": "healthy",
    }

    def normalise_label(raw: str) -> str:
        """Map raw model label to a clean display label."""
        key = raw.strip().lower()
        if key in LABEL_MAP:
            return LABEL_MAP[key]
        # Fallback: anything containing 'healthy' but NOT 'un' → healthy
        if "healthy" in key and "un" not in key:
            return "healthy"
        return raw.lower()

    # ── ML Model ──
    ml_model = None
    ml_prediction = ""
    ml_confidence = 0.0
    if model_path and ML_AVAILABLE:
        try:
            with open(model_path, "rb") as f:
                ml_bundle = pickle.load(f)
            ml_model = ml_bundle
            print(f"[✓] ML model loaded: {list(ml_bundle['label_encoder'].classes_)}")
        except Exception as e:
            print(f"[!] Could not load model: {e}")

    # ── Start serial reader ──
    reader = ThermalReader(port, baud)
    reader.start()
    print(f"[…] Connecting to {port}...")

    # ── Heatmap surface ──
    heatmap_surface = pygame.Surface((HEATMAP_SIZE, HEATMAP_SIZE))

    # ── Helper: render text with shadow ──
    def draw_text(surface, text, font, pos, color, shadow=True):
        if shadow:
            shadow_surf = font.render(text, True, (0, 0, 0))
            surface.blit(shadow_surf, (pos[0] + 1, pos[1] + 1))
        text_surf = font.render(text, True, color)
        surface.blit(text_surf, pos)

    # ── Helper: draw color bar ──
    def draw_colorbar(surface, x, y, w, h, lut, min_t, max_t):
        for i in range(h):
            idx = int((1.0 - i / h) * (len(lut) - 1))
            color = lut[idx]
            pygame.draw.line(surface, color, (x, y + i), (x + w - 1, y + i))

        # Border
        pygame.draw.rect(surface, GRAY, (x - 1, y - 1, w + 2, h + 2), 1)

        # Labels
        draw_text(surface, f"{max_t:.0f}°F", font_small, (x + w + 4, y - 2), RED, shadow=False)
        draw_text(surface, f"{min_t:.0f}°F", font_small, (x + w + 4, y + h - 14), CYAN, shadow=False)
        mid_t = (max_t + min_t) / 2
        draw_text(surface, f"{mid_t:.0f}°F", font_small, (x + w + 4, y + h // 2 - 7), GRAY, shadow=False)

    running = True
    last_ml_time = 0

    while running:
        # ── Events ──
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_h:
                    hold = not hold
                    if hold:
                        held_frame = reader.get_frame()
                elif event.key == pygame.K_f:
                    focus = not focus
                elif event.key == pygame.K_p:
                    names = list(PALETTES.keys())
                    idx = (names.index(palette_name) + 1) % len(names)
                    palette_name = names[idx]
                    palette_lut = build_palette_lut(PALETTES[palette_name])
                elif event.key == pygame.K_s:
                    os.makedirs("screenshots", exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = f"screenshots/thermal_{ts}.png"
                    pygame.image.save(screen, path)
                elif event.key == pygame.K_m:
                    show_ml = not show_ml
                elif event.key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
                    alarm_f = min(200, alarm_f + 5)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    alarm_f = max(50, alarm_f - 5)

        # ── Get frame ──
        if hold:
            frame = held_frame
        else:
            frame = reader.get_frame()

        # ── Clear ──
        screen.fill(BG)

        if frame is None:
            # Waiting for data
            msg = "Waiting for thermal data..."
            if reader.error:
                msg = f"Error: {reader.error}"
            draw_text(screen, msg, font_med, (20, HEATMAP_SIZE // 2), YELLOW)
            draw_text(screen, f"Port: {port}", font_small, (20, HEATMAP_SIZE // 2 + 30), GRAY)
            pygame.display.flip()
            clock.tick(30)
            continue

        grid = np.array(frame["grid"], dtype=np.float32)
        max_c = frame.get("max_c", float(np.max(grid)))
        min_c = frame.get("min_c", float(np.min(grid)))
        avg_c = frame.get("avg_c", float(np.mean(grid)))

        # Convert to Fahrenheit for display
        max_f = max_c * 9 / 5 + 32
        min_f = min_c * 9 / 5 + 32
        avg_f = avg_c * 9 / 5 + 32

        # Focus mode: auto-range
        if focus:
            disp_min = min_c
            disp_max = max_c
        else:
            disp_min = (min_range_f - 32) * 5 / 9
            disp_max = (max_range_f - 32) * 5 / 9

        # Normalize grid to 0–1
        temp_range = disp_max - disp_min
        if temp_range <= 0:
            temp_range = 1
        normalized = np.clip((grid - disp_min) / temp_range, 0.0, 1.0)

        # Interpolate
        interp = interpolate_fast(normalized, interp_scale)

        # ── Render heatmap ──
        pixel_array = pygame.surfarray.pixels3d(heatmap_surface)
        lut_len = len(palette_lut) - 1
        indices = np.clip((interp * lut_len).astype(int), 0, lut_len)

        # Transpose because pygame uses (x, y) not (row, col)
        for y in range(HEATMAP_SIZE):
            for x in range(HEATMAP_SIZE):
                pixel_array[x, y] = palette_lut[indices[y, x]]
        del pixel_array

        screen.blit(heatmap_surface, (0, 0))

        # ── Crosshair at hotspot ──
        hot_idx = np.unravel_index(np.argmax(grid), grid.shape)
        hot_x = int(hot_idx[1] * HEATMAP_SIZE / 8 + HEATMAP_SIZE / 16)
        hot_y = int(hot_idx[0] * HEATMAP_SIZE / 8 + HEATMAP_SIZE / 16)
        pygame.draw.circle(screen, WHITE, (hot_x, hot_y), 12, 1)
        pygame.draw.line(screen, WHITE, (hot_x - 16, hot_y), (hot_x - 8, hot_y), 1)
        pygame.draw.line(screen, WHITE, (hot_x + 8, hot_y), (hot_x + 16, hot_y), 1)
        pygame.draw.line(screen, WHITE, (hot_x, hot_y - 16), (hot_x, hot_y - 8), 1)
        pygame.draw.line(screen, WHITE, (hot_x, hot_y + 8), (hot_x, hot_y + 16), 1)

        # ── Color bar ──
        cb_x = HEATMAP_SIZE + 8
        cb_y = 10
        cb_h = HEATMAP_SIZE - 20
        draw_colorbar(screen, cb_x, cb_y, 18, cb_h, palette_lut,
                       min_range_f if not focus else min_f,
                       max_range_f if not focus else max_f)

        # ── Sidebar ──
        sb_x = HEATMAP_SIZE + COLORBAR_W
        pygame.draw.rect(screen, SIDEBAR_BG, (sb_x, 0, SIDEBAR_W, HEATMAP_SIZE))

        # Title
        draw_text(screen, "THERMAL CAM", font_title, (sb_x + 12, 10), WHITE, shadow=False)
        pygame.draw.line(screen, DARK_GRAY, (sb_x + 10, 38), (sb_x + SIDEBAR_W - 10, 38))

        # Stats
        sy = 50
        draw_text(screen, "ALM", font_small, (sb_x + 12, sy), GRAY, shadow=False)
        alarm_c = (alarm_f - 32) * 5 / 9
        alarm_color = RED if max_c >= alarm_c else WHITE
        draw_text(screen, f"{alarm_f}°F", font_large, (sb_x + 12, sy + 15), alarm_color)

        sy += 60
        draw_text(screen, "MAX", font_small, (sb_x + 12, sy), GRAY, shadow=False)
        draw_text(screen, f"{max_f:.1f}°F", font_large, (sb_x + 12, sy + 15), RED)
        draw_text(screen, f"{max_c:.1f}°C", font_small, (sb_x + 130, sy + 20), GRAY, shadow=False)

        sy += 60
        draw_text(screen, "AVG", font_small, (sb_x + 12, sy), GRAY, shadow=False)
        draw_text(screen, f"{avg_f:.1f}°F", font_large, (sb_x + 12, sy + 15), YELLOW)
        draw_text(screen, f"{avg_c:.1f}°C", font_small, (sb_x + 130, sy + 20), GRAY, shadow=False)

        sy += 60
        draw_text(screen, "MIN", font_small, (sb_x + 12, sy), GRAY, shadow=False)
        draw_text(screen, f"{min_f:.1f}°F", font_large, (sb_x + 12, sy + 15), CYAN)
        draw_text(screen, f"{min_c:.1f}°C", font_small, (sb_x + 130, sy + 20), GRAY, shadow=False)

        # ── ML Prediction ──
        if ml_model and show_ml and not hold:
            now = time.time()
            if now - last_ml_time > 1.0:
                last_ml_time = now
                try:
                    features = extract_features(grid).reshape(1, -1)
                    features_scaled = ml_model["scaler"].transform(features)
                    pred = ml_model["label_encoder"].inverse_transform(
                        ml_model["model"].predict(features_scaled)
                    )[0]
                    if hasattr(ml_model["model"], "predict_proba"):
                        proba = ml_model["model"].predict_proba(features_scaled)[0]
                        conf = max(proba)
                    else:
                        conf = 0.0
                    ml_prediction = normalise_label(pred)  # ← clean label
                    ml_confidence = conf
                except Exception:
                    ml_prediction = "error"
                    ml_confidence = 0.0

            sy += 70
            pygame.draw.line(screen, DARK_GRAY, (sb_x + 10, sy), (sb_x + SIDEBAR_W - 10, sy))
            sy += 8
            draw_text(screen, "ML PREDICT", font_small, (sb_x + 12, sy), GRAY, shadow=False)
            pred_color = GREEN if ml_prediction == "healthy" else ORANGE  # healthy=green, unhealthy=orange
            draw_text(screen, ml_prediction.upper(), font_large, (sb_x + 12, sy + 18), pred_color)
            draw_text(screen, f"{ml_confidence:.0%}", font_med, (sb_x + 12, sy + 48), GRAY, shadow=False)

        # ── Alarm flash ──
        if max_c >= alarm_c:
            # Red border flash
            t = time.time()
            if int(t * 4) % 2 == 0:
                pygame.draw.rect(screen, RED, (0, 0, HEATMAP_SIZE, HEATMAP_SIZE), 4)
                draw_text(screen, "⚠ ALARM", font_large, (HEATMAP_SIZE // 2 - 60, 10), RED)

        # ── Bottom status bar ──
        pygame.draw.rect(screen, BOTTOM_BG, (0, HEATMAP_SIZE, WIN_W, BOTTOM_H))
        pygame.draw.line(screen, DARK_GRAY, (0, HEATMAP_SIZE), (WIN_W, HEATMAP_SIZE))

        bx = 12
        by = HEATMAP_SIZE + 8

        # Status indicators
        status_items = []
        if hold:
            status_items.append(("HOLD", RED))
        if focus:
            status_items.append(("FOCUS", CYAN))
        status_items.append((f"Palette: {palette_name}", GRAY))
        status_items.append((f"FPS: {reader.fps:.1f}", GRAY))
        status_items.append((f"Frames: {reader.frame_count}", GRAY))

        for text, color in status_items:
            draw_text(screen, text, font_small, (bx, by), color, shadow=False)
            bx += font_small.size(text)[0] + 20

        # Controls hint
        hint = "H:Hold  F:Focus  P:Palette  S:Screenshot  +/-:Alarm  Q:Quit"
        draw_text(screen, hint, font_small, (12, by + 20), (70, 70, 90), shadow=False)

        # ── Connected indicator ──
        status_color = GREEN if reader.connected else RED
        pygame.draw.circle(screen, status_color, (WIN_W - 20, HEATMAP_SIZE + 15), 6)
        draw_text(screen, "Serial", font_small, (WIN_W - 70, HEATMAP_SIZE + 8), GRAY, shadow=False)

        pygame.display.flip()
        clock.tick(30)

    reader.stop()
    pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description="Live thermal camera UI — replicate PyGamer display on laptop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  H        Toggle HOLD (freeze frame)
  F        Toggle FOCUS (auto-range)
  P        Cycle palette (Iron/Grayscale/Inferno)
  S        Save screenshot
  +/-      Adjust alarm threshold
  M        Toggle ML prediction overlay
  Q/ESC    Quit
        """,
    )
    parser.add_argument("--port", required=True, help="Serial port (e.g., COM10)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--model", default=None, help="Path to trained ML model .pkl")

    args = parser.parse_args()
    run_ui(args.port, args.baud, args.model)


if __name__ == "__main__":
    main()
