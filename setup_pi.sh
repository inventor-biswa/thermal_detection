#!/bin/bash
# =============================================================================
#   setup_pi.sh — Full Raspberry Pi 4 Setup
#   Thermal Imaging Motor Health Monitoring
# =============================================================================
# Replicates the complete laptop workflow on a Raspberry Pi 4.
# PyGamer connects via USB serial → RPi collects data, trains model, shows UI.
#
# Usage:
#   chmod +x setup_pi.sh
#   ./setup_pi.sh
# =============================================================================

set -e  # Stop on any error

# ── Colors for pretty output ──────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC}   $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERR]${NC}  $1"; exit 1; }

REPO_URL="https://github.com/inventor-biswa/thermal_detection.git"
PROJECT_DIR="$HOME/thermal_detection"
VENV_DIR="$PROJECT_DIR/venv"
LAUNCHER="$HOME/run_thermal.sh"

echo -e "${BOLD}"
echo "============================================================="
echo "   Thermal Camera Motor Health — Raspberry Pi 4 Setup"
echo "============================================================="
echo -e "${NC}"

# ── Step 1: System Update ─────────────────────────────────────────────────────
info "Step 1/6: Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y
success "System updated."

# ── Step 2: Install System-Level Dependencies ─────────────────────────────────
# These are installed via apt because they have native C/Fortran extensions
# that are pre-compiled for ARM — much faster than pip compiling from source.
info "Step 2/6: Installing system dependencies..."
sudo apt-get install -y \
    python3 python3-pip python3-venv git \
    python3-numpy python3-scipy \
    python3-matplotlib \
    python3-pygame \
    python3-serial \
    i2c-tools libopenblas-dev
success "System dependencies installed."

# ── Step 3: Clone the Project Repository ─────────────────────────────────────
info "Step 3/6: Setting up project directory..."
if [ -d "$PROJECT_DIR/.git" ]; then
    warn "Repository already exists. Pulling latest changes..."
    cd "$PROJECT_DIR"
    git pull origin main
else
    info "Cloning repository..."
    git clone "$REPO_URL" "$PROJECT_DIR"
fi
cd "$PROJECT_DIR"
success "Project at: $PROJECT_DIR"

# ── Step 4: Create Virtual Environment ───────────────────────────────────────
# RPi OS Bookworm+ uses "externally managed" Python.
# --system-site-packages lets us use the apt-installed numpy/pygame/scipy
# which are pre-compiled for ARM — no waiting 20+ mins for pip to build them.
info "Step 4/6: Creating Python virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR" --system-site-packages
    success "Virtual environment created."
else
    warn "Virtual environment already exists. Skipping creation."
fi

source "$VENV_DIR/bin/activate"

# Install pip-only packages
info "Installing ML + Web Dashboard packages via pip..."
pip install --upgrade pip
pip install scikit-learn flask flask-socketio eventlet smbus2

success "All Python packages installed."

# ── Step 5: Create Data & Output Directories ──────────────────────────────────
info "Step 5/6: Creating project folders..."
mkdir -p "$PROJECT_DIR/thermal_data"
mkdir -p "$PROJECT_DIR/models"
mkdir -p "$PROJECT_DIR/plots"
mkdir -p "$PROJECT_DIR/screenshots"
success "Folders created."

# ── Step 6: Create Convenience Launcher Scripts ───────────────────────────────
info "Step 6/6: Creating launcher scripts..."

# --- Main launcher (activates venv) ---
cat > "$LAUNCHER" << EOF
#!/bin/bash
# Launcher: activates venv and runs the motor health monitor
cd $PROJECT_DIR
source venv/bin/activate

export DISPLAY=:0

echo "Select a mode:"
echo "  1) Web Dashboard (recommended)"
echo "  2) Live Thermal UI (pygame)"
echo "  3) Collect Data"
echo "  4) Train Model"
echo "  5) Real-time Predict (CLI)"
read -p "Enter choice [1-5]: " choice

case "\$choice" in
    1)
        read -p "Serial port (default: /dev/ttyACM0): " port
        port=\${port:-/dev/ttyACM0}
        model=\$(ls models/*.pkl 2>/dev/null | sort -r | head -1 || echo "")
        echo "Starting web dashboard..."
        echo "Open: http://\$(hostname -I | awk '{print \$1}'):5000"
        if [ -n "\$model" ]; then
            python app.py --port "\$port" --model "\$model"
        else
            python app.py --port "\$port"
        fi
        ;;
    2)
        read -p "Serial port (default: /dev/ttyACM0): " port
        port=\${port:-/dev/ttyACM0}
        model=\$(ls models/*.pkl 2>/dev/null | sort -r | head -1 || echo "")
        if [ -n "\$model" ]; then
            python thermal_ui.py --port "\$port" --model "\$model"
        else
            python thermal_ui.py --port "\$port"
        fi
        ;;
    3)
        read -p "Serial port (default: /dev/ttyACM0): " port
        port=\${port:-/dev/ttyACM0}
        read -p "Label (healthy/unhealthy): " label
        read -p "Duration in seconds (default: 120): " dur
        dur=\${dur:-120}
        python serial_collector.py --port "\$port" --label "\$label" --duration "\$dur"
        ;;
    4)
        python train_model.py --data thermal_data/ --visualize
        ;;
    5)
        read -p "Serial port (default: /dev/ttyACM0): " port
        port=\${port:-/dev/ttyACM0}
        model=\$(ls models/*.pkl 2>/dev/null | sort -r | head -1 || echo "")
        if [ -z "\$model" ]; then
            echo "No trained model found. Run option 4 first."
            exit 1
        fi
        python realtime_predict.py --port "\$port" --model "\$model"
        ;;
    *)
        echo "Invalid choice."
        ;;
esac
EOF
chmod +x "$LAUNCHER"

# ── Optional: Give permission to user for serial port ─────────────────────────
if ! groups "$USER" | grep -q "dialout"; then
    warn "Adding user to 'dialout' group (needed to access USB serial)..."
    sudo usermod -aG dialout "$USER"
    warn "You will need to REBOOT for this to take effect."
fi

echo ""
echo -e "${GREEN}${BOLD}============================================================="
echo "   Setup Complete!"
echo "=============================================================${NC}"
echo ""
echo -e "  ${BOLD}Project:${NC}      $PROJECT_DIR"
echo -e "  ${BOLD}Launcher:${NC}     $LAUNCHER"
echo ""
echo -e "  ${BOLD}To run:${NC}"
echo -e "  ${CYAN}  bash ~/run_thermal.sh${NC}"
echo ""
echo -e "  ${BOLD}Or manually:${NC}"
echo -e "  ${CYAN}  cd $PROJECT_DIR"
echo -e "  source venv/bin/activate"
echo -e "  python thermal_ui.py --port /dev/ttyACM0${NC}"
echo ""
echo -e "${YELLOW}  NOTE: Plug the PyGamer USB into the Raspberry Pi."
echo -e "  The serial port is usually /dev/ttyACM0 on Linux.${NC}"
echo -e "  ${YELLOW}Run 'ls /dev/ttyACM*' or 'dmesg | tail' after plugging in to confirm.${NC}"
echo ""

# Ask about reboot if dialout group was just added
if ! groups "$USER" | grep -q "dialout"; then
    read -p "Reboot now to apply serial port permissions? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo reboot
    fi
fi
