# Speech-to-Text API

Production-grade, GPU-optimized transcription system built on **faster-whisper**, **WhisperX**, **Silero-VAD**, and **NVIDIA NeMo** (speaker diarization).

## Quick Start

### 1. Configure

Edit `config.yaml`:

- Set `security.api_keys` to your own secret keys.
- Adjust `model.size`, `performance.batch_size`, etc. as needed.
- Speaker diarization uses NVIDIA NeMo — no HuggingFace token required.

### 2. Launch (Docker)

```bash
docker compose up --build -d
```

Scale workers:

```bash
docker compose up --build -d --scale worker=4
```

### 3. Transcribe

**File upload:**

```bash
curl -X POST http://localhost:8000/transcribe \
  -H "x-api-key: your-secret-key-1" \
  -F "file=@audio.wav" \
  -F 'metadata={"project":"demo"}' \
  -F "webhook_url=https://example.com/hook"
```

**Audio URL:**

```bash
curl -X POST http://localhost:8000/transcribe \
  -H "x-api-key: your-secret-key-1" \
  -F "audio_url=https://example.com/audio.mp3" \
  -F 'metadata={"source":"url"}'
```

### 4. Check Status

```bash
curl http://localhost:8000/status/{job_id} \
  -H "x-api-key: your-secret-key-1"
```

### 5. Get Result

```bash
curl http://localhost:8000/result/{job_id} \
  -H "x-api-key: your-secret-key-1"
```

## API Reference

| Endpoint               | Method | Description                  |
|------------------------|--------|------------------------------|
| `/transcribe`          | POST   | Submit audio for transcription |
| `/status/{job_id}`     | GET    | Get job status & progress    |
| `/result/{job_id}`     | GET    | Get transcription result     |
| `/health`              | GET    | Health check                 |

## Architecture

```
Client → FastAPI (API) → Redis (RQ Queue)
                              ↓
                        RQ Worker(s)
                     ┌─────────────────┐
                     │ 1. VAD / Chunk  │
                     │ 2. Transcribe   │
                     │ 3. Align        │
                     │ 4. Diarize      │
                     │ 5. Webhook      │
                     └─────────────────┘
```

## Environment Variables

| Variable                      | Default              | Description                     |
|-------------------------------|----------------------|---------------------------------|
| `REDIS_URL`                   | `redis://localhost:6379/0` | Redis connection string   |
| `WHISPER_MODEL_SIZE`          | from config.yaml     | Override model size             |
| `WHISPER_DEVICE`              | from config.yaml     | Override device (cuda/cpu)      |
| `WHISPER_COMPUTE_TYPE`        | from config.yaml     | Override compute type           |
| `WHISPER_BATCH_SIZE`          | from config.yaml     | Override batch size             |
| `WHISPER_MAX_WORKERS`         | from config.yaml     | Override max workers            |
| `WHISPER_HF_TOKEN`            | (optional)           | HuggingFace token (not needed for NeMo) |
| `WHISPER_DIARIZATION_ENABLED` | from config.yaml     | Enable/disable diarization      |

## License

MIT
