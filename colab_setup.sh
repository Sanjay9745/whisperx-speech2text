#!/usr/bin/env bash
# =============================================================================
# colab_setup.sh — Bootstrap Speech-to-Text API on Google Colab (GPU runtime)
#
# Usage (in a Colab cell):
#   !git clone https://github.com/Sanjay9745/whisperx-speech2text.git /content/speech2text
#   %cd /content/speech2text
#   !bash colab_setup.sh
#
# The script assumes:
#   • A GPU runtime is selected (Runtime → Change runtime type → T4 GPU)
#   • CUDA 12.x is already present in the Colab image (CUDA 12.1 as of 2024)
# =============================================================================

set -euo pipefail

echo "============================================================"
echo " Speech-to-Text API — Google Colab Setup"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. System dependencies
# ---------------------------------------------------------------------------
echo "[1/7] Installing system packages …"
apt-get update -qq
apt-get install -y -qq ffmpeg redis-server

# ---------------------------------------------------------------------------
# 2. Start Redis (background)
# ---------------------------------------------------------------------------
echo "[2/7] Starting Redis …"
redis-server --daemonize yes --loglevel warning
sleep 1
redis-cli ping && echo "Redis is up" || { echo "ERROR: Redis failed to start"; exit 1; }

# ---------------------------------------------------------------------------
# 3. Upgrade pip / setuptools
# ---------------------------------------------------------------------------
echo "[3/7] Upgrading pip / build tools …"
pip install --quiet --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# 4. Install PyTorch with the CUDA 12.1 wheel
#    Colab ships CUDA 12.x; cu121 wheels work on both 12.1 and 12.4.
#    We pin torch==2.5.1 to match requirements.txt.
# ---------------------------------------------------------------------------
echo "[4/7] Installing PyTorch 2.5.1 + CUDA 12.1 wheels …"
pip install --quiet \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Verify GPU access
python - <<'PYEOF'
import torch
assert torch.cuda.is_available(), "CUDA not available — did you select a GPU runtime?"
print(f"  torch {torch.__version__}  |  CUDA {torch.version.cuda}  |  GPU: {torch.cuda.get_device_name(0)}")
PYEOF

# ---------------------------------------------------------------------------
# 5. Install application requirements (whisperx excluded — handled below)
# ---------------------------------------------------------------------------
echo "[5/7] Installing Python requirements …"
# Install numpy first to guarantee 1.x is used (some deps pull 2.x).
pip install --quiet "numpy==1.26.4"

# ctranslate2 must be installed before faster-whisper.
pip install --quiet "ctranslate2==4.4.0"

# Install the rest of requirements.txt (torch/torchaudio already satisfied).
pip install --quiet -r requirements.txt

# ---------------------------------------------------------------------------
# 6. Install WhisperX AFTER torch (no-deps avoids torch version conflicts)
# ---------------------------------------------------------------------------
echo "[6/7] Installing WhisperX (--no-deps, pinned commit) …"
pip install --quiet --no-deps \
    git+https://github.com/m-bain/whisperX.git@v3.3.1

# ---------------------------------------------------------------------------
# 7. Create working directories
# ---------------------------------------------------------------------------
echo "[7/7] Creating working directories …"
mkdir -p uploads outputs temp models

echo ""
echo "============================================================"
echo " Setup complete!  Next steps:"
echo ""
echo "   • Set environment variables (see .env.example):"
echo "       export WHISPER_HF_TOKEN=hf_..."
echo "       export WHISPER_API_KEYS=my-secret-key"
echo ""
echo "   • Start the API in one cell:"
echo "       !uvicorn app.main:app --host 0.0.0.0 --port 8000 &"
echo ""
echo "   • Start the worker in another cell:"
echo "       !rq worker transcription --url redis://localhost:6379/0 &"
echo "============================================================"
