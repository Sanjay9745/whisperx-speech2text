#!/bin/bash

echo "Pulling latest code from GitHub..."
git pull

echo "Restarting services..."

# Kill existing processes
pkill -f "uvicorn app.main:app"
pkill -f "rq worker"
pkill -f "python colab_ui.py"

# Ensure Redis is running
if ! pgrep -x "redis-server" > /dev/null
then
    redis-server --daemonize yes
    echo "Started Redis."
fi

# Start RQ Worker in the background
echo "Starting RQ worker..."
rq worker &

# Start FastAPI server in the background (optional if only using UI directly, but keeping it for full setup)
echo "Starting FastAPI server..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Start the UI
echo "Starting UI..."
python colab_ui.py
