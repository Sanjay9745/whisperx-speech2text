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
pip install -r requirements.txt

# Install Gradio for the testing UI
pip install gradio==5.0.0

# Create necessary directories
mkdir -p uploads outputs temp models
echo "Setup complete!"
