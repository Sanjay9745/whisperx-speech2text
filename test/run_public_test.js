#!/usr/bin/env node
/**
 * run_public_test.js — Full end-to-end test against a PUBLIC API URL
 *
 * What this script does:
 *   1. Starts the local webhook receiver (Express, port 4000)
 *   2. Opens a localtunnel (free, no account) to expose the webhook publicly
 *   3. Submits a transcription job to the public API URL
 *   4. Polls status until completed / failed / timed-out
 *   5. Fetches and pretty-prints the result
 *   6. Shows every webhook payload that was delivered
 *   7. Cleans up and exits
 *
 * ── Quick start ─────────────────────────────────────────────────────────────
 *   npm install          # first time only
 *   node run_public_test.js --api https://xxxx.trycloudflare.com --key colab-test-key
 *
 * ── All options (env vars OR --flags) ───────────────────────────────────────
 *   --api   / API_URL        Public API base URL (required)
 *   --key   / API_KEY        x-api-key value            (default: colab-test-key)
 *   --audio / AUDIO_URL      Audio URL to transcribe    (default: sample speech wav)
 *   --file  / AUDIO_FILE     Local audio file instead of URL
 *   --wh    / WEBHOOK_PORT   Local webhook port         (default: 4000)
 *   --wait  / MAX_WAIT_SEC   Max seconds to poll        (default: 600)
 *   --poll  / POLL_INTERVAL  Poll interval ms           (default: 3000)
 *   --no-webhook             Skip webhook server entirely
 */

import fs            from 'fs';
import path          from 'path';
import { fileURLToPath } from 'url';
import http          from 'http';
import { createRequire } from 'module';

import FormData from 'form-data';
import axios    from 'axios';
import express  from 'express';
import localtunnel from 'localtunnel';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ── ANSI colours ─────────────────────────────────────────────────────────────
const C = {
  reset  : '\x1b[0m',
  bold   : '\x1b[1m',
  cyan   : '\x1b[36m',
  green  : '\x1b[32m',
  yellow : '\x1b[33m',
  red    : '\x1b[31m',
  grey   : '\x1b[90m',
  magenta: '\x1b[35m',
};
const col  = (c, s) => `${c}${s}${C.reset}`;
const info = (...a) => console.log(col(C.cyan,   '[info ]'), ...a);
const ok   = (...a) => console.log(col(C.green,  '[ ok  ]'), ...a);
const warn = (...a) => console.log(col(C.yellow, '[warn ]'), ...a);
const err  = (...a) => console.log(col(C.red,    '[error]'), ...a);
const step = (n, s) => console.log(`\n${col(C.bold + C.magenta, `── Step ${n}:`)} ${col(C.bold, s)}`);
const sep  = ()     => console.log(col(C.grey, '─'.repeat(64)));

// ── Parse CLI args ────────────────────────────────────────────────────────────
function parseArgs() {
  const args = process.argv.slice(2);
  const get  = (flag, env, def) => {
    const i = args.indexOf(flag);
    return (i !== -1 && args[i + 1]) ? args[i + 1]
         : process.env[env] ?? def;
  };
  return {
    apiUrl       : get('--api',   'API_URL',       null),
    apiKey       : get('--key',   'API_KEY',        'colab-test-key'),
    audioUrl     : get('--audio', 'AUDIO_URL',
                    'https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav'),
    audioFile    : get('--file',  'AUDIO_FILE',     null),
    webhookPort  : parseInt(get('--wh',   'WEBHOOK_PORT',   '4000'), 10),
    maxWaitSec   : parseInt(get('--wait', 'MAX_WAIT_SEC',   '600'),  10),
    pollInterval : parseInt(get('--poll', 'POLL_INTERVAL',  '3000'), 10),
    noWebhook    : args.includes('--no-webhook'),
  };
}

// ── Helpers ───────────────────────────────────────────────────────────────────
const sleep = ms => new Promise(r => setTimeout(r, ms));

function authHeaders(apiKey, extra = {}) {
  return { 'x-api-key': apiKey, ...extra };
}

// ── Webhook server ────────────────────────────────────────────────────────────
function startWebhookServer(port) {
  const receivedPayloads = [];
  const app = express();
  app.use(express.json({ limit: '50mb' }));

  app.post('/webhook', (req, res) => {
    const payload    = req.body;
    const receivedAt = new Date().toISOString();
    receivedPayloads.push({ receivedAt, payload });

    const jobId  = payload?.job_id ?? '?';
    const status = payload?.status ?? '?';
    ok(`Webhook received  job_id=${col(C.cyan, jobId)}  status=${col(C.green, status)}`);

    if (status === 'completed' && payload?.result) {
      const r = payload.result;
      console.log(`         language=${r.language}  segments=${(r.segments ?? []).length}`);
      const preview = (r.text ?? '').slice(0, 120);
      console.log(`         text: ${col(C.grey, preview)}${r.text?.length > 120 ? '…' : ''}`);
    } else if (status === 'failed') {
      err(`Job failure reported via webhook: ${payload?.error}`);
    }
    res.status(200).json({ received: true });
  });

  app.get('/webhooks', (_req, res) =>
    res.json({ count: receivedPayloads.length, webhooks: receivedPayloads }));
  app.get('/health', (_req, res) => res.json({ status: 'ok' }));

  return new Promise((resolve, reject) => {
    const server = app.listen(port, () => {
      ok(`Webhook server listening on http://localhost:${port}/webhook`);
      resolve({ server, receivedPayloads });
    });
    server.on('error', reject);
  });
}

// ── localtunnel ───────────────────────────────────────────────────────────────
async function openTunnel(port) {
  info(`Opening localtunnel for port ${port} …`);
  const tunnel = await localtunnel({ port });
  ok(`Tunnel URL: ${col(C.bold, tunnel.url)}`);
  tunnel.on('error', e => warn(`Tunnel error: ${e.message}`));
  return tunnel;
}

// ── Submit ────────────────────────────────────────────────────────────────────
async function submitJob(cfg, webhookPublicUrl) {
  const form = new FormData();

  if (cfg.audioFile && fs.existsSync(cfg.audioFile)) {
    form.append('file', fs.createReadStream(cfg.audioFile));
    info(`Audio source: FILE → ${cfg.audioFile}`);
  } else {
    form.append('audio_url', cfg.audioUrl);
    info(`Audio source: URL  → ${cfg.audioUrl}`);
  }

  const metadata = {
    test_run_id : `pub-test-${Date.now()}`,
    client      : 'run_public_test.js',
    submitted_at: new Date().toISOString(),
  };
  form.append('metadata', JSON.stringify(metadata));

  if (webhookPublicUrl) {
    form.append('webhook_url', webhookPublicUrl + '/webhook');
    info(`Webhook URL: ${webhookPublicUrl}/webhook`);
  }

  info(`POST ${cfg.apiUrl}/transcribe`);
  const t0 = Date.now();

  const resp = await axios.post(`${cfg.apiUrl}/transcribe`, form, {
    headers: { ...form.getHeaders(), ...authHeaders(cfg.apiKey) },
    maxContentLength: Infinity,
    maxBodyLength   : Infinity,
  });

  const elapsed = Date.now() - t0;
  ok(`HTTP ${resp.status}  (${elapsed} ms)`);

  const jobId = resp.data?.job_id;
  if (!jobId) throw new Error('API response did not include job_id');
  ok(`Job ID: ${col(C.bold, jobId)}`);
  return jobId;
}

// ── Poll ──────────────────────────────────────────────────────────────────────
async function pollUntilDone(cfg, jobId) {
  const deadline = Date.now() + cfg.maxWaitSec * 1000;
  let lastLabel  = '';

  info(`Polling every ${cfg.pollInterval / 1000}s  (max ${cfg.maxWaitSec}s) …`);

  while (Date.now() < deadline) {
    const resp = await axios.get(`${cfg.apiUrl}/status/${jobId}`, {
      headers: authHeaders(cfg.apiKey),
    });
    const { status, progress, error: jobErr } = resp.data;
    const label = `status=${status}  progress=${progress ?? '?'}%`;

    if (label !== lastLabel) {
      info(label);
      lastLabel = label;
    }

    if (status === 'completed') return 'completed';
    if (status === 'failed') {
      err(`Job failed: ${jobErr ?? '(no detail)'}`);
      return 'failed';
    }

    await sleep(cfg.pollInterval);
  }

  throw new Error(`Timed out after ${cfg.maxWaitSec}s`);
}

// ── Fetch result ──────────────────────────────────────────────────────────────
async function fetchResult(cfg, jobId) {
  const resp = await axios.get(`${cfg.apiUrl}/result/${jobId}`, {
    headers: authHeaders(cfg.apiKey),
  });
  return resp.data?.result ?? resp.data;
}

// ── Print result ──────────────────────────────────────────────────────────────
function printResult(result) {
  sep();
  console.log(col(C.bold, ' TRANSCRIPTION RESULT'));
  sep();
  console.log(`  language : ${col(C.cyan,  result.language ?? '?')}`);
  console.log(`  segments : ${col(C.cyan,  (result.segments ?? []).length)}`);
  console.log(`  words    : ${col(C.cyan,  (result.words    ?? []).length)}`);
  if ((result.speakers ?? []).length) {
    const spkSet = new Set(result.speakers.map(s => s.speaker).filter(Boolean));
    console.log(`  speakers : ${col(C.cyan, [...spkSet].join(', ') || '—')}`);
  }
  console.log(`\n  ${col(C.bold, 'Full transcript:')}`);
  const text = (result.formatted_text ?? result.text ?? '').trim();
  console.log(`  ${col(C.grey, text || '(empty)')}`);

  if ((result.segments ?? []).length) {
    console.log(`\n  ${col(C.bold, 'Segments:')}`);
    for (const seg of result.segments) {
      const spk = seg.speaker ? col(C.magenta, ` [${seg.speaker}]`) : '';
      const ts  = col(C.grey, `[${seg.start?.toFixed(2)}s → ${seg.end?.toFixed(2)}s]`);
      console.log(`  ${ts}${spk}  ${seg.text ?? ''}`);
    }
  }
  sep();
}

// ── Main ──────────────────────────────────────────────────────────────────────
async function main() {
  const cfg = parseArgs();

  console.log('\n' + col(C.bold + C.magenta, '══ Speech-to-Text Public API Test ══'));

  // ── Validate required args ─────────────────────────────────────────────────
  if (!cfg.apiUrl) {
    err('No API URL provided.');
    console.log(`\nUsage:\n  node run_public_test.js --api <PUBLIC_URL> [--key <API_KEY>]\n`);
    console.log(`Examples:\n  node run_public_test.js --api https://xxxx.trycloudflare.com --key colab-test-key`);
    console.log(`  API_URL=https://xxxx.trycloudflare.com node run_public_test.js\n`);
    process.exit(1);
  }

  console.log(`  API URL  : ${col(C.bold, cfg.apiUrl)}`);
  console.log(`  API Key  : ${cfg.apiKey}`);
  console.log(`  Audio    : ${cfg.audioFile ?? cfg.audioUrl}`);
  console.log(`  Webhook  : ${cfg.noWebhook ? 'disabled' : `port ${cfg.webhookPort}`}`);
  console.log();

  // ── Test connection ────────────────────────────────────────────────────────
  step(1, 'Verify API connection');
  try {
    const r = await axios.get(`${cfg.apiUrl}/health`, {
      headers: authHeaders(cfg.apiKey), timeout: 10_000,
    });
    ok(`API is up — ${JSON.stringify(r.data)}`);
  } catch (e) {
    err(`Cannot reach API at ${cfg.apiUrl}: ${e.message}`);
    err('Make sure the server is running and the URL is correct.');
    process.exit(1);
  }

  // ── Webhook server + tunnel ────────────────────────────────────────────────
  let server = null;
  let tunnel = null;
  let receivedPayloads = [];
  let webhookBaseUrl = null;

  if (!cfg.noWebhook) {
    step(2, 'Start local webhook server + public tunnel');
    try {
      ({ server, receivedPayloads } = await startWebhookServer(cfg.webhookPort));
      tunnel = await openTunnel(cfg.webhookPort);
      webhookBaseUrl = tunnel.url;
    } catch (e) {
      warn(`Could not start webhook tunnel: ${e.message}`);
      warn('Continuing without webhook — results will still be polled and fetched.');
      if (server) server.close();
      tunnel = null;
    }
  } else {
    step(2, 'Webhook skipped (--no-webhook)');
  }

  // ── Submit ─────────────────────────────────────────────────────────────────
  step(3, 'Submit transcription job');
  let jobId;
  try {
    jobId = await submitJob(cfg, webhookBaseUrl);
  } catch (e) {
    err(`Submit failed: ${e.response?.data ? JSON.stringify(e.response.data) : e.message}`);
    cleanup(server, tunnel);
    process.exit(1);
  }

  // ── Poll ───────────────────────────────────────────────────────────────────
  step(4, 'Poll status');
  let finalStatus;
  try {
    finalStatus = await pollUntilDone(cfg, jobId);
  } catch (e) {
    err(e.message);
    cleanup(server, tunnel);
    process.exit(1);
  }

  // ── Result ─────────────────────────────────────────────────────────────────
  if (finalStatus === 'completed') {
    step(5, 'Fetch & display result');
    try {
      const result = await fetchResult(cfg, jobId);
      printResult(result);

      // Save result JSON
      const outFile = path.join(__dirname, `result_${jobId}.json`);
      fs.writeFileSync(outFile, JSON.stringify(result, null, 2));
      ok(`Result saved to: ${outFile}`);
    } catch (e) {
      err(`Could not fetch result: ${e.message}`);
    }
  }

  // ── Webhook summary ────────────────────────────────────────────────────────
  if (!cfg.noWebhook && server) {
    step(6, 'Webhook delivery summary');
    // Give 3 extra seconds for any in-flight webhook to arrive
    if (receivedPayloads.length === 0) {
      info('Waiting 3 s for any late webhook delivery…');
      await sleep(3000);
    }
    if (receivedPayloads.length > 0) {
      ok(`${receivedPayloads.length} webhook payload(s) received:`);
      for (const { receivedAt, payload } of receivedPayloads) {
        console.log(`  ${col(C.grey, receivedAt)}  job_id=${payload?.job_id}  status=${payload?.status}`);
      }
    } else {
      warn('No webhook payloads received (tunnel may have been unreachable from the server).');
    }
  }

  cleanup(server, tunnel);
  console.log(`\n${col(C.bold + C.green, '✅ Test complete.')}\n`);
  process.exit(finalStatus === 'completed' ? 0 : 1);
}

function cleanup(server, tunnel) {
  try { if (tunnel) tunnel.close(); } catch (_) {}
  try { if (server) server.close(); } catch (_) {}
}

main().catch(e => {
  err('Unhandled error:', e.message);
  process.exit(1);
});
