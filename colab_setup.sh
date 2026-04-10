#!/usr/bin/env bash
# =============================================================================
# colab_setup.sh — Bootstrap Speech-to-Text API on Google Colab (GPU runtime)
#
# Usage (in a Colab cell):
#   !git clone https://github.com/Sanjay9745/whisperx-speech2text.git /content/speech2text
#   %cd /content/speech2text
#   !bash colab_setup.sh
#
# Strategy:
#   • Do NOT reinstall or downgrade torch / numpy — Colab already ships
#     torch 2.10+cu128 and numpy 2.x, which satisfy all our dependencies.
#   • Install only the packages Colab is missing.
#   • Install whisperx v3.8.5 with --no-deps to avoid pip touching torch.
#
# Prerequisites:
#   • GPU runtime selected (Runtime → Change runtime type → T4 GPU)
# =============================================================================

set -uo pipefail   # note: no -e so pip warnings don't abort the script

echo "============================================================"
echo " Speech-to-Text API — Google Colab Setup"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. System dependencies
# ---------------------------------------------------------------------------
echo "[1/5] Installing system packages …"
apt-get update -qq
apt-get install -y -qq ffmpeg libsndfile1 redis-server

# ---------------------------------------------------------------------------
# 2. Start Redis (background)
# ---------------------------------------------------------------------------
echo "[2/5] Starting Redis …"
redis-server --daemonize yes --loglevel warning
sleep 1
if redis-cli ping | grep -q PONG; then
    echo "      Redis is up ✅"
else
    echo "      ERROR: Redis failed to start ❌"
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. Upgrade pip / build tools
# ---------------------------------------------------------------------------
echo "[3/5] Upgrading pip / build tools …"
pip install --quiet --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# 4. Install Python dependencies
#    torch, torchaudio, numpy are already installed by Colab — pip skips them.
#    We override only the packages whose Colab versions are too old for us.
# ---------------------------------------------------------------------------
echo "[4/5] Installing Python dependencies …"

# WhisperX v3.8.5 runtime deps first (without whisperx itself)
pip install --quiet \
    "ctranslate2>=4.5.0" \
    "faster-whisper>=1.2.0" \
    "pyannote.audio>=4.0.0" \
    "pyannote.core>=5.0.0" \
    "pyannote.pipeline>=3.0.1" \
    "omegaconf>=2.3.0"

# App dependencies (ranges let Colab's other packages coexist)
pip install --quiet \
    "fastapi>=0.124.1,<2.0.0" \
    "uvicorn[standard]>=0.34.0,<2.0.0" \
    "starlette>=0.49.1,<2.0.0" \
    "python-multipart>=0.0.18" \
    "redis>=5.2.0,<6.0.0" \
    "rq>=2.0.0,<3.0.0" \
    "pydub>=0.25.1" \
    "soundfile>=0.12.1" \
    "librosa>=0.10.2" \
    "silero-vad>=5.1.2" \
    "pandas>=2.2.3" \
    "nltk>=3.9.1" \
    "huggingface-hub>=0.33.5,<1.0.0" \
    "transformers>=4.48.0" \
    "httpx>=0.28.1" \
    "aiofiles>=24.1.0" \
    "pyyaml>=6.0.2" \
    "loguru>=0.7.3"

# ---------------------------------------------------------------------------
# 5. Install WhisperX v3.8.5 with --no-deps (keeps Colab torch intact)
# ---------------------------------------------------------------------------
echo "[5/5] Installing WhisperX v3.8.5 (--no-deps) …"
pip install --quiet --no-deps \
    "git+https://github.com/m-bain/whisperX.git@v3.8.5"

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
python - <<'PYEOF'
import sys
ok = True
checks = {
    "torch"         : lambda: __import__("torch").__version__,
    "numpy"         : lambda: __import__("numpy").__version__,
    "faster_whisper": lambda: __import__("faster_whisper").__version__,
    "whisperx"      : lambda: __import__("whisperx").__version__,
    "fastapi"       : lambda: __import__("fastapi").__version__,
    "pyannote.audio": lambda: __import__("pyannote.audio").__version__,
}
for pkg, fn in checks.items():
    try:
        ver = fn()
        print(f"  ✅  {pkg:<20} {ver}")
    except Exception as e:
        print(f"  ❌  {pkg:<20} FAILED: {e}")
        ok = False

import torch
cuda = torch.cuda.is_available()
print(f"\n  CUDA available : {'✅ Yes — ' + torch.cuda.get_device_name(0) if cuda else '❌ No — CPU only'}")
if not ok:
    sys.exit(1)
PYEOF

# ---------------------------------------------------------------------------
# Create working directories
# ---------------------------------------------------------------------------
mkdir -p /content/uploads /content/outputs /content/temp /content/models

echo ""
echo "============================================================"
echo " Setup complete!  Next steps:"
echo ""
echo "   1. Fill in env vars (Section 5 of the Colab notebook):"
echo "        export WHISPER_HF_TOKEN=hf_..."
echo "        export WHISPER_API_KEYS=my-secret-key"
echo ""
echo "   2. Start the API:"
echo "        uvicorn app.main:app --host 0.0.0.0 --port 8000 &"
echo ""
echo "   3. Start the worker:"
echo "        rq worker transcription --url redis://localhost:6379/0 &"
echo "============================================================"
