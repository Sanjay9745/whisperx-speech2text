/**
 * webhook_server.js — Webhook receiver + interactive HTML UI
 *
 * Starts an Express server that:
 *   1. Receives POST /webhook payloads from the Speech-to-Text API
 *   2. Serves an interactive HTML UI at / and /ui
 *   3. Optionally exposes itself via localtunnel (free public HTTPS URL)
 *
 * ── Quick start ─────────────────────────────────────────────────────────────
 *   node webhook_server.js                      # Run locally on :4000
 *   node webhook_server.js --tunnel             # + localtunnel for public URL
 *   PORT=5000 node webhook_server.js --tunnel   # Custom port + tunnel
 *
 * ── Usage ────────────────────────────────────────────────────────────────────
 *   Local:     http://localhost:4000/
 *   Webhook:   http://localhost:4000/webhook
 *
 *   Public (with --tunnel):
 *   UI:        https://xxxx.localtunneltunnel.net/
 *   Webhook:   https://xxxx.localtunneltunnel.net/webhook
 *
 * ── Env vars ────────────────────────────────────────────────────────────────
 *   PORT           Port to listen on           (default: 4000)
 *   SECRET         x-webhook-secret validation (optional)
 *   --tunnel       Open localtunnel public URL (requires installed localtunnel)
 */

import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import localtunnel from 'localtunnel';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const PORT   = parseInt(process.env.PORT   || '4000', 10);
const SECRET = process.env.SECRET || null;
const TUNNEL = process.argv.includes('--tunnel');

const app = express();
app.use(express.json({ limit: '50mb' }));
app.use(express.static(path.join(__dirname, '.')));  // Serve static files (index.html, etc.)

// ---------------------------------------------------------------------------
// GET /  or /ui — serve the interactive HTML tester
// ---------------------------------------------------------------------------
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});
app.get('/ui', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// ---------------------------------------------------------------------------
const receivedWebhooks = [];

// ---------------------------------------------------------------------------
// POST /webhook  — main receiver
// ---------------------------------------------------------------------------
app.post('/webhook', (req, res) => {
  // Optional secret validation
  if (SECRET) {
    const provided = req.headers['x-webhook-secret'];
    if (provided !== SECRET) {
      console.warn(`[webhook] Rejected request — wrong secret (got: ${provided})`);
      return res.status(401).json({ error: 'Invalid webhook secret' });
    }
  }

  const payload = req.body;
  const receivedAt = new Date().toISOString();
  const entry = { receivedAt, payload };
  receivedWebhooks.push(entry);

  console.log('\n' + '='.repeat(60));
  console.log(` WEBHOOK RECEIVED  #${receivedWebhooks.length}  at ${receivedAt}`);
  console.log('='.repeat(60));
  console.log(`  job_id  : ${payload?.job_id ?? '(none)'}`);
  console.log(`  status  : ${payload?.status ?? '(none)'}`);

  if (payload?.status === 'completed' && payload?.result) {
    const r = payload.result;
    console.log(`  language: ${r.language ?? 'unknown'}`);
    console.log(`  segments: ${(r.segments ?? []).length}`);
    console.log(`  text    : ${(r.text ?? '').slice(0, 200)}${r.text?.length > 200 ? '…' : ''}`);
  } else if (payload?.status === 'failed') {
    console.log(`  error   : ${payload?.error ?? '(no error detail)'}`);
  }

  if (payload?.metadata && Object.keys(payload.metadata).length) {
    console.log(`  metadata: ${JSON.stringify(payload.metadata)}`);
  }
  console.log('='.repeat(60) + '\n');

  res.status(200).json({ received: true });
});

// ---------------------------------------------------------------------------
// GET /webhooks  — list all received webhook payloads (for programmatic use)
// ---------------------------------------------------------------------------
app.get('/webhooks', (_req, res) => {
  res.json({ count: receivedWebhooks.length, webhooks: receivedWebhooks });
});

// ---------------------------------------------------------------------------
// GET /health
// ---------------------------------------------------------------------------
app.get('/health', (_req, res) => res.json({ status: 'ok' }));

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
const server = app.listen(PORT, async () => {
  console.log('============================================================');
  console.log(' Webhook Receiver + Test UI Server');
  console.log(`  listening on  : http://localhost:${PORT}`);
  console.log(`  UI            : http://localhost:${PORT}/`);
  console.log(`  webhook URL   : http://localhost:${PORT}/webhook`);
  console.log(`  view received : http://localhost:${PORT}/webhooks`);
  if (SECRET) console.log(`  secret        : set (header x-webhook-secret)`);
  console.log('');

  // ── Open localtunnel if --tunnel flag ───────────────────────────────────
  if (TUNNEL) {
    try {
      console.log(' Opening localtunnel (public HTTPS URL) …');
      const tunnel = await localtunnel({ port: PORT });
      console.log(`  Public URL    : ${tunnel.url}`);
      console.log(`  Public UI     : ${tunnel.url}/`);
      console.log(`  Public webhook: ${tunnel.url}/webhook`);
      console.log('');
      console.log('  Use this webhook URL in your API calls:');
      console.log(`    webhook_url: ${tunnel.url}/webhook`);
      console.log('============================================================\n');
      tunnel.on('error', e => console.error('Tunnel error:', e.message));
      tunnel.on('close', () => console.log('\nTunnel closed.'));
    } catch (e) {
      console.error(`  ❌ localtunnel failed: ${e.message}`);
      console.log('  Install it with: npm install localtunnel');
      console.error('============================================================\n');
    }
  } else {
    console.log('  (use --tunnel flag to expose via localtunnel)\n');
    console.log('============================================================\n');
  }
});
