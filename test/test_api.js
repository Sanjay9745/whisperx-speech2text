/**
 * test_api.js — Speech-to-Text API end-to-end test client
 *
 * Usage:
 *   node test_api.js
 *
 * All settings can be overridden with environment variables:
 *   API_URL        Base URL of the API (default: http://localhost:8000)
 *   API_KEY        x-api-key header value
 *   AUDIO_URL      Public audio URL to transcribe
 *   AUDIO_FILE     Local file path to upload instead of a URL
 *   WEBHOOK_URL    Optional URL that receives the completed result
 *   POLL_INTERVAL  Polling interval in ms (default: 3000)
 *   MAX_WAIT_SEC   Give up polling after this many seconds (default: 600)
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import FormData from 'form-data';
import axios from 'axios';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ============================================================================
// Configuration — edit here or override with env vars
// ============================================================================
const API_URL        = process.env.API_URL        || 'http://localhost:8000';
const API_KEY        = process.env.API_KEY        || 'change-me-key-1';
// NOTE: diarization requires ≥ ~10 s of audio with ≥ 2 speakers to show speaker turns.
// sample-3s.mp3 is too short — use a longer multi-speaker sample as the default.
const AUDIO_URL      = process.env.AUDIO_URL      || 'https://www2.cs.uic.edu/~i101/SoundFiles/StarWars60.wav';
const AUDIO_FILE     = process.env.AUDIO_FILE     || null;   // e.g. './sample.mp3'
const WEBHOOK_URL    = process.env.WEBHOOK_URL    || null;   // set to your webhook server URL
const POLL_INTERVAL  = parseInt(process.env.POLL_INTERVAL  || '3000',  10);
const MAX_WAIT_SEC   = parseInt(process.env.MAX_WAIT_SEC   || '600',   10);

// ============================================================================
// Helpers
// ============================================================================

/** Shared auth headers */
function authHeaders(extra = {}) {
  return { 'x-api-key': API_KEY, ...extra };
}

/** Sleep for ms milliseconds */
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/** Pretty-print JSON */
function pp(obj) {
  return JSON.stringify(obj, null, 2);
}

// ============================================================================
// Step 1 — Submit transcription job
// ============================================================================
async function submitJob() {
  const form = new FormData();

  if (AUDIO_FILE && fs.existsSync(AUDIO_FILE)) {
    form.append('file', fs.createReadStream(AUDIO_FILE));
    console.log(`[submit] audio source : FILE  → ${AUDIO_FILE}`);
  } else if (AUDIO_URL) {
    form.append('audio_url', AUDIO_URL);
    console.log(`[submit] audio source : URL   → ${AUDIO_URL}`);
  } else {
    console.error('[submit] ERROR: No audio source. Set AUDIO_URL or AUDIO_FILE.');
    process.exit(1);
  }

  const metadata = {
    test_run_id : `test-${Date.now()}`,
    client      : 'test_api.js',
    submitted_at: new Date().toISOString(),
  };
  form.append('metadata', JSON.stringify(metadata));

  if (WEBHOOK_URL) {
    form.append('webhook_url', WEBHOOK_URL);
    console.log(`[submit] webhook url  : ${WEBHOOK_URL}`);
  }

  console.log(`[submit] POST ${API_URL}/transcribe`);
  const t0 = Date.now();

  const resp = await axios.post(`${API_URL}/transcribe`, form, {
    headers: { ...form.getHeaders(), ...authHeaders() },
    maxContentLength: Infinity,
    maxBodyLength   : Infinity,
  });

  console.log(`[submit] ${resp.status} ${resp.statusText}  (${Date.now() - t0} ms)`);
  console.log(`[submit] response:\n${pp(resp.data)}`);

  const jobId = resp.data?.job_id;
  if (!jobId) throw new Error('API did not return a job_id');
  return jobId;
}

// ============================================================================
// Step 2 — Poll /status/:id until terminal state
// ============================================================================
async function pollUntilDone(jobId) {
  const deadline = Date.now() + MAX_WAIT_SEC * 1000;
  let lastProgress = -1;

  console.log(`\n[poll] Watching job ${jobId} (max ${MAX_WAIT_SEC}s) …`);

  while (Date.now() < deadline) {
    const resp = await axios.get(`${API_URL}/status/${jobId}`, {
      headers: authHeaders(),
    });

    const { status, progress, error } = resp.data;

    if (progress !== lastProgress) {
      console.log(`[poll] status=${status}  progress=${progress ?? '?'}%`);
      lastProgress = progress;
    }

    if (status === 'completed') {
      return 'completed';
    }

    if (status === 'failed') {
      console.error(`[poll] Job failed: ${error || '(no error detail)'}`);
      return 'failed';
    }

    await sleep(POLL_INTERVAL);
  }

  throw new Error(`Timed out waiting for job ${jobId} after ${MAX_WAIT_SEC}s`);
}

// ============================================================================
// Step 3 — Fetch and display result
// ============================================================================
async function fetchResult(jobId) {
  const resp = await axios.get(`${API_URL}/result/${jobId}`, {
    headers: authHeaders(),
  });

  const result = resp.data?.result ?? resp.data;
  console.log('\n============================================================');
  console.log(' TRANSCRIPTION RESULT');
  console.log('============================================================');
  console.log(`  language : ${result.language ?? 'unknown'}`);
  console.log(`  segments : ${(result.segments ?? []).length}`);
  console.log(`  words    : ${(result.words ?? []).length}`);
  if (result.speakers?.length) {
    const spkSet = new Set(result.speakers.map(s => s.speaker));
    console.log(`  speakers : ${[...spkSet].join(', ')}`);
  }
  console.log('\n  Full text:');
  console.log(`  ${result.formatted_text ?? result.text ?? '(empty)'}`);
  console.log('============================================================\n');
  return result;
}

// ============================================================================
// Main
// ============================================================================
async function main() {
  console.log('============================================================');
  console.log(' Speech-to-Text API — E2E Test');
  console.log(`  API URL : ${API_URL}`);
  console.log('============================================================\n');

  try {
    const jobId  = await submitJob();
    const status = await pollUntilDone(jobId);

    if (status === 'completed') {
      await fetchResult(jobId);
    }
  } catch (err) {
    console.error('\n[ERROR]', err.response?.data ?? err.message);
    process.exit(1);
  }
}

main();
