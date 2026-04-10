# ============================================================================
# GPU-enabled Dockerfile for Speech-to-Text
# Base: NVIDIA CUDA 12.4 + cuDNN + Ubuntu 22.04
# ============================================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        ffmpeg git curl build-essential && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# ---------- Python dependencies ----------
COPY requirements.txt .

# Install PyTorch with CUDA 12.4 support first
RUN pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install remaining requirements (skip torch/torchaudio, installed above with CUDA wheels)
RUN grep -Ev "^(torch|torchaudio)([<>=].*)?$" requirements.txt > /tmp/reqs_no_torch.txt && \
    pip install -r /tmp/reqs_no_torch.txt && \
    rm /tmp/reqs_no_torch.txt

# ---------- Application code ----------
COPY config.yaml .
COPY app/ ./app/

# Create directories
RUN mkdir -p uploads outputs temp models

# ---------- Defaults ----------
EXPOSE 8000

# Default: run the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
