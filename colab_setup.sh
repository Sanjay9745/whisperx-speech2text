#!/bin/bash
echo "Setting up environment for Google Colab..."

# Install system dependencies
apt-get update
apt-get install -y ffmpeg redis-server

# Start Redis for the queue (in background)
redis-server --daemonize yes
echo "Redis started."

# Install Python dependencies
pip install --upgrade pip setuptools wheel
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install core packages explicitly to avoid missing modules if requirements.txt fails midway
pip install fastapi uvicorn python-multipart loguru rq redis
pip install "gradio>=4.44.1" "huggingface_hub<0.26.0" "pydantic<3.0.0"

# Install everything else
pip install -r requirements.txt

# Finish with compiling whisperx bypassing pip dependency checks
pip install --no-deps git+https://github.com/m-bain/whisperX.git@v3.7.9

# Create necessary directories
mkdir -p uploads outputs temp models
echo "Setup complete!"
