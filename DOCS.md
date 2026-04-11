# Speech-to-Text API — Full Workflow Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Technology Stack](#2-technology-stack)
3. [Repository Structure](#3-repository-structure)
4. [System Architecture](#4-system-architecture)
5. [Configuration](#5-configuration)
6. [API Reference](#6-api-reference)
7. [Request Lifecycle — Step by Step](#7-request-lifecycle--step-by-step)
   - [Step 1 · Security & CORS middleware](#step-1--security--cors-middleware)
   - [Step 2 · POST /transcribe — audio intake](#step-2--post-transcribe--audio-intake)
   - [Step 3 · Redis / RQ — job queuing](#step-3--redis--rq--job-queuing)
   - [Step 4 · Worker warm-up](#step-4--worker-warm-up)
   - [Step 5 · VAD chunking](#step-5--vad-chunking)
   - [Step 6 · Parallel transcription](#step-6--parallel-transcription)
   - [Step 7 · Word alignment](#step-7--word-alignment)
   - [Step 8 · Speaker diarization](#step-8--speaker-diarization)
   - [Step 9 · Re-segmentation by speaker](#step-9--re-segmentation-by-speaker)
   - [Step 10 · Result assembly & storage](#step-10--result-assembly--storage)
   - [Step 11 · Webhook delivery](#step-11--webhook-delivery)
8. [Output Format](#8-output-format)
9. [Speaker Diarization — Deep Dive](#9-speaker-diarization--deep-dive)
10. [Deployment](#10-deployment)
11. [Environment Variables](#11-environment-variables)
12. [Running in Google Colab](#12-running-in-google-colab)
13. [Common Issues & Fixes](#13-common-issues--fixes)

---

## 1. Project Overview

This is a **production-grade, GPU-accelerated speech-to-text API** that accepts audio files or URLs and returns a fully timestamped, speaker-labeled transcript.

**What it does:**

- Accepts any audio/video format (`.wav`, `.mp3`, `.mp4`, `.mkv`, `.webm`, …)
- Detects speech regions with **Silero-VAD** and skips silence
- Transcribes with **faster-whisper** (supports 100+ languages, `transcribe` and `translate` modes)
- Refines word-level timestamps with **WhisperX**
- Assigns speaker labels using **pyannote.audio** diarization
- Splits any mixed-speaker segment so each output segment has exactly one speaker
- Delivers results via **webhook** or polling
- Scales horizontally by adding more RQ workers

---

## 2. Technology Stack

| Layer | Library | Purpose |
|-------|---------|---------|
| API | FastAPI + uvicorn | HTTP server, request validation |
| Queue | Redis + RQ | Async job queue, state storage |
| VAD | Silero-VAD | Voice activity detection |
| Transcription | faster-whisper | CTranslate2-optimized Whisper |
| Alignment | WhisperX | Precise word-level timestamps |
| Diarization | pyannote.audio 4.x | Speaker identification |
| Audio I/O | librosa, soundfile, pydub | Load / resample audio |
| HTTP client | httpx | URL download, webhook delivery |
| Logging | loguru | Structured logging |
| Containers | Docker + NVIDIA CUDA 12.4 | GPU-enabled deployment |

---

## 3. Repository Structure

```
speech2text/
├── app/
│   ├── main.py              # FastAPI app, all HTTP endpoints
│   ├── config.py            # Typed config loader (config.yaml + env vars)
│   ├── queue.py             # Redis connection, RQ job helpers
│   ├── worker.py            # RQ job handler — the full pipeline
│   ├── downloader.py        # Stream-download audio from URL
│   ├── security.py          # API-key middleware
│   ├── webhook.py           # Async webhook delivery with retries
│   ├── result_formatter.py  # Build [HH:MM:SS] Speaker: text lines
│   └── transcriber/
│       ├── vad.py           # Silero-VAD wrapper
│       ├── chunker.py       # Load audio, VAD-guided chunking
│       ├── whisper.py       # faster-whisper model + transcribe_chunk()
│       ├── align.py         # WhisperX word alignment
│       └── diarization.py   # pyannote pipeline, speaker assignment,
│                            #   re-segmentation
├── config.yaml              # All tuneable knobs (model, perf, security…)
├── requirements.txt         # Python dependencies
├── Dockerfile               # CUDA 12.4 / Python 3.11 image
├── docker-compose.yml       # api + worker + redis services
├── speech2text_colab.ipynb  # Google Colab notebook
└── colab_setup.sh           # Colab install script
```

---

## 4. System Architecture

```
┌─────────────┐     HTTP      ┌───────────────────────────────────────┐
│   Client    │ ──────────── ▶│            FastAPI (api)               │
│  (curl /    │               │  ┌────────────┐  ┌───────────────────┐ │
│  JS / app)  │               │  │CORS middle-│  │ APIKey middleware │ │
└─────────────┘               │  │    ware    │  │  x-api-key header │ │
                              │  └────────────┘  └───────────────────┘ │
                              │         ↓                               │
                              │  POST /transcribe                       │
                              │  ┌────────────────────────────────────┐ │
                              │  │ 1. Save upload  OR  download URL   │ │
                              │  │ 2. Validate metadata / params      │ │
                              │  │ 3. create_job() → Redis + RQ       │ │
                              │  │ 4. Return 202 { job_id }           │ │
                              │  └────────────────────────────────────┘ │
                              └───────────────────────────────────────┘
                                            │ RQ enqueue
                                            ▼
                              ┌───────────────────────────────────────┐
                              │           Redis (queue)                │
                              │   job state  •  progress  •  result   │
                              └───────────────────────────────────────┘
                                            │ dequeue
                                            ▼
                              ┌───────────────────────────────────────┐
                              │          RQ Worker(s)                  │
                              │                                        │
                              │  ① VAD → chunks                       │
                              │  ② faster-whisper transcribe          │
                              │  ③ WhisperX word align                │
                              │  ④ pyannote diarize                   │
                              │  ⑤ resegment by speaker               │
                              │  ⑥ save JSON → outputs/               │
                              │  ⑦ webhook POST (optional)            │
                              └───────────────────────────────────────┘
                                            │ poll
                                            ▼
                              GET /status/{job_id}
                              GET /result/{job_id}
```

Multiple workers can run in parallel — each independently dequeues a job. Workers **share model weights via the filesystem** (`./models/`) so the large Whisper binary is only downloaded once.

---

## 5. Configuration

All configuration lives in **`config.yaml`** and can be overridden by environment variables (highest priority).

### Priority chain
```
ENV VAR  >  config.yaml value  >  code default
```

### config.yaml sections

#### `model`
```yaml
model:
  size: large-v2          # tiny | base | small | medium | large-v2 | large-v3
  device: cuda            # cuda | cpu
  compute_type: float16   # float16 | int8 | float32
  download_root: ./models # local cache for weights
```

#### `performance`
```yaml
performance:
  batch_size: 16           # chunks sent to GPU at once
  max_workers: 1           # parallel transcription threads per worker process
  chunk_duration_sec: 30   # max audio chunk length before forced split
  use_vad: true            # VAD-guided chunking (strongly recommended)
  queue_soft_limit: 10000  # log warning when queue exceeds this
  queue_hard_limit: null   # reject requests when queue hits this (null = off)
```

#### `accuracy`
```yaml
accuracy:
  beam_size: 5
  temperature: 0.0
  best_of: 5
```

#### `diarization`
```yaml
diarization:
  enabled: true
  hf_token: ""         # Hugging Face token (required for pyannote)
  min_speakers:        # optional hint — null or e.g. 2
  max_speakers:        # optional hint — null or e.g. 5
```

#### `security`
```yaml
security:
  api_key_enabled: true
  api_keys:
    - "your-key-1"
    - "your-key-2"
```

#### `webhook`
```yaml
webhook:
  enabled: true
  timeout_sec: 15
  retry_count: 3
```

#### `paths`
```yaml
paths:
  upload_dir: ./uploads   # saved file uploads
  output_dir: ./outputs   # saved result JSONs
  temp_dir: ./temp        # downloaded URL files
```

---

## 6. API Reference

All endpoints (except `/health`) require the `x-api-key` header when `security.api_key_enabled: true`.

### `POST /transcribe`

Submit audio for transcription. Returns a `job_id` immediately (HTTP 202).

| Form field | Type | Required | Description |
|---|---|---|---|
| `file` | file upload | one of these | Audio file in any ffmpeg-supported format |
| `audio_url` | string | one of these | Public URL to download audio from |
| `metadata` | JSON string | no | Arbitrary key/value object; stored with result |
| `webhook_url` | string | no | URL to POST the result to when done |
| `language` | string | no | ISO-639-1 code, e.g. `ml`, `en`, `hi` |
| `task` | string | no | `transcribe` (default) or `translate` (→ English) |
| `initial_prompt` | string | no | Whisper context hint for first chunk |
| `language_hints` | string | no | Comma-separated language codes, e.g. `ml,en` |
| `min_speakers` | integer | no | Minimum expected speakers (hint for pyannote) |
| `max_speakers` | integer | no | Maximum expected speakers (hint for pyannote) |

**Response (202):**
```json
{ "job_id": "550e8400-e29b-41d4-a716-446655440000", "status": "queued" }
```

---

### `GET /status/{job_id}`

Poll job progress.

**Response:**
```json
{
  "job_id": "...",
  "status": "processing",   // queued | processing | completed | failed
  "progress": 65,           // 0-100
  "created_at": "2026-04-11T10:00:00",
  "updated_at": "2026-04-11T10:00:05",
  "error": ""
}
```

**Progress milestones:**

| Value | Stage |
|---|---|
| 0 | Job accepted |
| 5 | Audio chunked |
| 10–80 | Transcription (scales with chunks) |
| 82 | Alignment |
| 88 | Diarization |
| 95 | Result assembly |
| 100 | Saved / webhook sent |

---

### `GET /result/{job_id}`

Fetch the completed result. Returns HTTP 202 if still processing, HTTP 500 if failed.

**Response (200):**
```json
{
  "job_id": "...",
  "status": "completed",
  "result": { ... }   // see Output Format section
}
```

---

### `GET /health`

Liveness probe — no auth required.

```json
{ "status": "ok" }
```

---

## 7. Request Lifecycle — Step by Step

### Step 1 · Security & CORS middleware

Every request passes through two layers before reaching route handlers:

1. **CORSMiddleware** (outermost) — allows all origins; handles preflight `OPTIONS` requests before auth runs.
2. **APIKeyMiddleware** (inner) — reads `x-api-key` header. Paths `/health`, `/docs`, `/redoc`, `/openapi.json` are public. All others must carry a valid key from `config.yaml → security.api_keys`.

```
Request → CORS → APIKeyMiddleware → Route handler
```

---

### Step 2 · POST /transcribe — audio intake

`app/main.py` handles audio ingestion:

1. **Parse metadata** — the `metadata` form field is a JSON string; it's decoded into a dict. Top-level params (`language`, `task`, `initial_prompt`, `language_hints`, `min_speakers`, `max_speakers`) are merged into this dict.
2. **Audio source resolution:**
   - If `file` is provided → save to `./uploads/<uuid><ext>`
   - Else if `audio_url` is provided → validate URL (`http`/`https` only) then stream-download to `./temp/<uuid><ext>` via `app/downloader.py`
   - Otherwise → HTTP 400
3. **Enqueue** — call `create_job(file_path, metadata, webhook_url)` which creates a Redis hash `job:<job_id>` and pushes `process_job(job_id)` onto the `transcription` RQ queue.
4. **Return 202** with the `job_id`.

---

### Step 3 · Redis / RQ — job queuing

`app/queue.py` manages all Redis state:

- Each job is stored as a Redis hash: `job:<job_id>` with keys `status`, `progress`, `file_path`, `metadata`, `webhook_url`, `created_at`, `updated_at`, `result`, `error`.
- The RQ `transcription` queue holds references to `process_job(job_id)` callables.
- Workers call `update_job()` as they progress, updating the `progress` field (0–100) so clients can poll `/status`.
- On completion: `set_job_result()` writes the full result JSON into `job:<job_id>:result`.
- On failure: `set_job_failed()` writes the error string.

---

### Step 4 · Worker warm-up

`app/worker.py → _warm_up()` runs once per worker process (guarded by `_warmed_up` flag):

- Calls `load_model()` to load the faster-whisper model into GPU/CPU memory.
- Logs the diarization status (token present / missing).
- Subsequent jobs on the same worker reuse the loaded model, avoiding repeated load overhead.

---

### Step 5 · VAD chunking

`app/transcriber/chunker.py → prepare_chunks()`

Long audio is split into chunks that Whisper can handle efficiently:

1. **Load audio** — `load_audio()` uses `librosa` to load any format, resampling to 16 kHz mono float32.
2. **Voice Activity Detection** — `app/transcriber/vad.py` runs **Silero-VAD** to find speech regions, returning `[{start, end}, ...]` in seconds.
3. **Padding & merge** — speech segments are padded by 0.2 s each side and neighboring segments are merged, so short pauses don't create tiny chunks.
4. **Chunking** — speech regions are grouped into windows ≤ `chunk_duration_sec` (default 30 s).
5. **Fallback** — if VAD is disabled or finds nothing, fixed 30-second windows are used instead.

Each chunk is `(audio_array, chunk_start_sec, chunk_end_sec)`.

---

### Step 6 · Parallel transcription

`app/transcriber/whisper.py → transcribe_chunk()`

Chunks are processed in parallel up to `max_workers` threads:

1. **Model** — `WhisperModel` (faster-whisper / CTranslate2) is loaded once and cached globally per process.
2. **Device detection** — CUDA if available; falls back to CPU + int8 if CUDA is absent or the GPU's compute capability < 7.0.
3. **Language detection** — If `language` is not supplied, Whisper detects it automatically on the first chunk. `language_hints` biases detection toward the specified languages.
4. **Transcription** — `model.transcribe()` with `word_timestamps=True`, `beam_size`, `temperature`, `best_of` from config.
5. **Timestamp offset** — each segment's `start`/`end` are offset by `chunk_start_sec` so all segments have absolute timestamps relative to the original file.

Failed chunks are retried once; if the retry also fails the chunk is skipped and a warning is added to the result.

---

### Step 7 · Word alignment

`app/transcriber/align.py → align_segments()`

After transcription, **WhisperX** is used to refine word timestamps:

- Loads a language-specific forced-alignment model (wav2vec2 based).
- Aligns each word to the audio using CTC forced alignment, producing precise `start`/`end` per word.
- Safety check: if alignment returns fewer words or empty text, the original segments are preserved.
- **Skipped** when `task == "translate"` (English output doesn't match source-language audio) or when WhisperX is not installed.

---

### Step 8 · Speaker diarization

`app/transcriber/diarization.py`

Speaker diarization runs the **pyannote.audio** speaker-diarization-3.1 pipeline:

#### 8a. Audio pre-loading

Instead of letting pyannote read the file directly (which can cause duration-mismatch errors on `.mp4`/`.mkv` container formats), the audio is pre-loaded with librosa into an in-memory dict:

```python
{"waveform": Tensor(1, T), "sample_rate": 16000}
```

This bypasses pyannote's internal ffmpeg/torchaudio file I/O entirely.

#### 8b. Speaker count hints

The pipeline accepts optional hints that significantly improve accuracy:

```python
pipeline(audio_input, min_speakers=2, max_speakers=5)
```

These are resolved in priority order: **API param** → **metadata JSON** → **config.yaml** → **none** (pyannote decides automatically).

#### 8c. Output normalisation

Raw pyannote speaker labels (`SPEAKER_00`, `SPEAKER_01`, …) are mapped to human-readable `Speaker 1`, `Speaker 2`, … in chronological order of first appearance. The raw ID is preserved as `speaker_id`.

#### 8d. Speaker assignment

`assign_speakers()` maps each **word** (not just each segment) to a speaker turn:

1. For each word, find the diarization turn with the greatest time overlap.
2. Handle collapsed timestamps (translate mode): distribute words evenly across the segment's time span before looking up the speaker.
3. Set the **segment-level** speaker to whichever speaker owns the **majority** of words in that segment (Counter vote).

---

### Step 9 · Re-segmentation by speaker

`app/transcriber/diarization.py → resegment_by_speakers()`

This is the critical step that solves the "all Speaker 1" problem.

**The problem:** Whisper creates segments based on *content boundaries* (sentences, natural pauses), NOT speaker changes. In a conversation, Speaker A's question and Speaker B's answer may land in a single Whisper segment. The majority-vote then always assigns Speaker 1 because they dominate.

**The fix:** After every word has a speaker label, walk through each segment's word list. Whenever the speaker changes between consecutive words, split the segment at that boundary:

```
Before:
  Segment 1: [word1/Spk1 word2/Spk1 word3/Spk2 word4/Spk2]  → Speaker 1 (majority)

After:
  Segment 1: [word1/Spk1 word2/Spk1]  → Speaker 1
  Segment 2: [word3/Spk2 word4/Spk2]  → Speaker 2
```

Each output segment gets:
- `text` — space-joined words in that group
- `start` / `end` — min/max of word timestamps
- `speaker` — the single speaker for this group
- `words` — the word objects (with individual timestamps)

Segment IDs are renumbered sequentially after splitting.

---

### Step 10 · Result assembly & storage

`app/worker.py` assembles the final result dict:

```
final_text         — all segment texts joined
formatted_text     — [HH:MM:SS → HH:MM:SS] Speaker N: text  (one line per segment)
segments           — list of segment objects (with words, speaker, timestamps)
words              — flat list of all word objects
speakers           — raw diarization turn list [{start, end, speaker}]
language           — detected ISO language code
task               — "transcribe" | "translate"
metadata           — original metadata dict from client
warnings           — list of human-readable warning strings
stats              — { segment_count, speaker_count, speakers_in_segments,
                       word_count, chunks_total, chunks_with_content,
                       chunks_failed, speaker_turn_count }
```

The result is:
1. Written to Redis via `set_job_result()`.
2. Saved to `./outputs/<job_id>.json` on disk.

---

### Step 11 · Webhook delivery

`app/webhook.py → deliver_webhook_sync()`

If a `webhook_url` was provided, a POST is made with:

```json
{
  "job_id": "...",
  "status": "completed",
  "result": { ... },
  "metadata": { ... }
}
```

On failure the same structure is posted with `"status": "failed"` and an `"error"` field.

Delivery uses exponential back-off retries (up to `webhook.retry_count`, default 3). Timeouts default to `webhook.timeout_sec` (15 s).

---

## 8. Output Format

### Full result structure

```jsonc
{
  "text": "Hello how are you. I am fine thank you.",
  "formatted_text": "[00:00:01.250 -> 00:00:03.100] Speaker 1: Hello how are you.\n[00:00:03.400 -> 00:00:05.200] Speaker 2: I am fine thank you.",
  "segments": [
    {
      "id": 1,
      "start": 1.25,
      "end": 3.1,
      "text": "Hello how are you.",
      "speaker": "Speaker 1",
      "speaker_id": "SPEAKER_00",
      "words": [
        { "word": "Hello",  "start": 1.25, "end": 1.60, "score": 0.98, "speaker": "Speaker 1" },
        { "word": "how",    "start": 1.62, "end": 1.80, "score": 0.99, "speaker": "Speaker 1" },
        { "word": "are",    "start": 1.82, "end": 1.95, "score": 0.97, "speaker": "Speaker 1" },
        { "word": "you.",   "start": 1.97, "end": 2.30, "score": 0.99, "speaker": "Speaker 1" }
      ]
    },
    {
      "id": 2,
      "start": 3.4,
      "end": 5.2,
      "text": "I am fine thank you.",
      "speaker": "Speaker 2",
      "speaker_id": "SPEAKER_01",
      "words": [ ... ]
    }
  ],
  "words": [ /* flat list of all word objects */ ],
  "speakers": [
    { "start": 0.0,  "end": 10.5, "speaker": "Speaker 1", "speaker_id": "SPEAKER_00" },
    { "start": 3.4,  "end":  5.2, "speaker": "Speaker 2", "speaker_id": "SPEAKER_01" }
  ],
  "language": "en",
  "task": "transcribe",
  "metadata": { "project": "demo" },
  "warnings": [],
  "stats": {
    "chunks_total": 2,
    "chunks_with_content": 2,
    "chunks_failed": 0,
    "segment_count": 12,
    "speaker_turn_count": 23,
    "speaker_count": 2,
    "speakers_in_segments": 2,
    "word_count": 147
  }
}
```

### `formatted_text` format

```
[HH:MM:SS.mmm -> HH:MM:SS.mmm] Speaker N: transcript text
```

Example:
```
[00:00:01.250 -> 00:00:03.100] Speaker 1: Hello how are you.
[00:00:03.400 -> 00:00:05.200] Speaker 2: I am fine thank you.
```

---

## 9. Speaker Diarization — Deep Dive

### Requirements

1. A **Hugging Face account** with accepted terms for both models:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
2. A **HF token** with `read` scope — set as `WHISPER_HF_TOKEN` or in `config.yaml`.

### Controlling speaker count

| Method | How |
|---|---|
| Per-request (API) | `min_speakers=2` form field |
| Per-request (metadata) | `metadata={"min_speakers":2}` JSON |
| All requests (config.yaml) | `diarization: min_speakers: 2` |
| All requests (env var) | `WHISPER_DIARIZATION_MIN_SPEAKERS=2` |

Setting `min_speakers=2` is recommended for **conversational audio** (interviews, phone calls, meetings) where one speaker dominates but you know there are at least two participants.

### Why `speakers_in_segments` might be < `speaker_count`

`speaker_count` counts distinct speakers found by the diarization pipeline.
`speakers_in_segments` counts distinct speakers actually present in the output segments.

If they differ, it means pyannote detected a speaker but their turns were too short relative to Whisper's segments and were absorbed. Passing `min_speakers=N` forces pyannote to find more distinct turns.

### torch.load compatibility patch

pyannote checkpoints use OmegaConf objects that are not in PyTorch's safe-globals allowlist introduced in torch 2.6+. The code automatically patches `torch.load` to force `weights_only=False` before importing pyannote. The patch is guarded by a sentinel attribute to prevent double-wrapping.

---

## 10. Deployment

### Docker (recommended)

```bash
# 1. Copy and edit config
cp config.yaml config.yaml   # edit: api_keys, hf_token (or use env vars)

# 2. Build and start
docker compose up --build -d

# 3. Scale workers (one GPU per worker is ideal)
docker compose up --build -d --scale worker=4

# 4. View logs
docker compose logs -f worker
docker compose logs -f api
```

Services:
- `api` — FastAPI on port `8000`
- `worker` — RQ workers (scalable)
- `redis` — Redis 7 on port `6379`, with 4 GB memory cap and LRU eviction

Volumes mounted from host:
```
./models   → /app/models    # shared model weight cache
./uploads  → /app/uploads   # uploaded audio files
./outputs  → /app/outputs   # saved result JSONs
./temp     → /app/temp      # temp downloaded files
./config.yaml               # live config (editable without rebuild)
```

### Environment variables for secrets

Never put real tokens in `config.yaml`. Use Docker env instead:

```bash
# docker-compose.yml → worker / api environment section
environment:
  - WHISPER_HF_TOKEN=hf_xxxxxxxxxxxx
  - WHISPER_API_KEYS=my-secret-key
```

Or export them on the host before `docker compose up`:
```bash
export WHISPER_HF_TOKEN=hf_xxxxxxxxxxxx
export WHISPER_API_KEYS=my-secret-key
```

---

## 11. Environment Variables

| Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `WHISPER_MODEL_SIZE` | `large-v2` | Model size: `tiny` / `base` / `small` / `medium` / `large-v2` / `large-v3` |
| `WHISPER_DEVICE` | `cuda` | `cuda` or `cpu` |
| `WHISPER_COMPUTE_TYPE` | `float16` | `float16` / `int8` / `float32` |
| `WHISPER_DOWNLOAD_ROOT` | `./models` | Local model weight cache path |
| `WHISPER_BATCH_SIZE` | `16` | Chunks per GPU batch |
| `WHISPER_MAX_WORKERS` | `1` | Parallel transcription threads per worker |
| `WHISPER_CHUNK_DURATION` | `30` | Max chunk length in seconds |
| `WHISPER_USE_VAD` | `true` | Enable Silero-VAD chunking |
| `WHISPER_BEAM_SIZE` | `5` | Beam search width |
| `WHISPER_TEMPERATURE` | `0.0` | Sampling temperature |
| `WHISPER_BEST_OF` | `5` | Best-of sampling count |
| `WHISPER_HF_TOKEN` | — | **Required** for diarization — HuggingFace read token |
| `WHISPER_DIARIZATION_ENABLED` | `true` | `1`/`true`/`yes` to enable |
| `WHISPER_DIARIZATION_MIN_SPEAKERS` | — | Minimum speaker hint for pyannote |
| `WHISPER_DIARIZATION_MAX_SPEAKERS` | — | Maximum speaker hint for pyannote |
| `WHISPER_WEBHOOK_ENABLED` | `true` | Enable webhook delivery |
| `WHISPER_WEBHOOK_TIMEOUT` | `15` | Webhook HTTP timeout in seconds |
| `WHISPER_WEBHOOK_RETRY_COUNT` | `3` | Webhook retry attempts |
| `WHISPER_API_KEY_ENABLED` | `true` | Enforce API key auth |
| `WHISPER_API_KEYS` | — | Comma-separated valid keys: `key1,key2` |
| `WHISPER_UPLOAD_DIR` | `./uploads` | Upload directory |
| `WHISPER_OUTPUT_DIR` | `./outputs` | Result JSON directory |
| `WHISPER_TEMP_DIR` | `./temp` | Temp file directory |

---

## 12. Running in Google Colab

The notebook `speech2text_colab.ipynb` automates the entire Colab setup:

1. Installs all Python dependencies in the correct order (torch first, then WhisperX with `--no-deps`).
2. Clones or pulls the repo to `/content/speech2text`.
3. Starts Redis in the background.
4. Sets `WHISPER_HF_TOKEN` from Colab Secrets.
5. Starts the FastAPI server and the RQ worker.
6. Exposes the API via `ngrok` for external access.

To use on Colab:

1. Open `speech2text_colab.ipynb` in Google Colab.
2. Set the `WHISPER_HF_TOKEN` secret in Colab Secrets (key icon in the left sidebar).
3. Run all cells in order.
4. Copy the `ngrok` URL and use it as your API base URL.

---

## 13. Common Issues & Fixes

### All segments show "Speaker 1" even though multiple speakers exist

**Cause:** Whisper creates segments by content/sentence boundaries. If two speakers alternate, their words land in the same segment and the majority vote always wins for Speaker 1.

**Fix (automatic):** The `resegment_by_speakers()` step splits segments at word-level speaker changes.

**Fix (improve detection):** Pass `min_speakers=2` to force pyannote to find at least two speakers:
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.mp4" \
  -F "min_speakers=2"
```

---

### `UnpicklingError` / `WeightsUnpickler` when loading pyannote

**Cause:** PyTorch 2.6+ changed the default `torch.load` behaviour to `weights_only=True`. pyannote checkpoints contain OmegaConf objects not in the safe allowlist.

**Fix (automatic):** The code patches `torch.load` to force `weights_only=False` before importing pyannote. Check the logs for `"torch.load weights_only=False patch applied"`.

---

### `crop` error / duration mismatch on `.mp4` files

**Cause:** pyannote's internal `Audio.crop()` reads container duration from metadata, which can be longer than the actual decoded samples in `.mp4`/`.mkv`/`.webm` files.

**Fix (automatic):** Audio is pre-loaded with librosa into an in-memory dict `{"waveform": tensor, "sample_rate": 16000}` and passed to the pipeline directly, bypassing pyannote's file I/O.

---

### Diarization is enabled but `speaker_count: 0` in stats

**Causes and checks:**
1. `WHISPER_HF_TOKEN` not set → see `"WHISPER_HF_TOKEN is NOT SET"` warning in logs.
2. Model terms not accepted on HuggingFace → accept at:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
3. Audio is too short (< 5 s) → pyannote may return no turns.
4. Audio is single-speaker → expected, pass `min_speakers=1` to confirm.

---

### WhisperX alignment skipped

**Causes:**
- `task=translate` — alignment is always skipped because translated (English) text doesn't correspond to the source-language audio.
- WhisperX not installed — install with: `pip install --no-deps git+https://github.com/m-bain/whisperX.git@v3.8.5`
- Language not supported by WhisperX's wav2vec2 models — word timestamps fall back to Whisper's built-in estimates.

---

### Out of memory (CUDA OOM)

**Fixes:**
- Reduce `batch_size` in config (try `8` or `4`).
- Switch to a smaller model: `model.size: medium` or `small`.
- Switch to `compute_type: int8` to halve VRAM usage.
- On Colab T4 (16 GB), `large-v2` with `batch_size: 8` and `float16` is safe.
