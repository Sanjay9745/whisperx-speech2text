/**
 * webhook_server.js — Local webhook receiver for testing
 *
 * Starts a tiny Express server that listens for POST requests from the
 * Speech-to-Text API and pretty-prints every payload it receives.
 *
 * Usage:
 *   node webhook_server.js
 *
 * Then pass the URL shown in the console as the webhook_url when you
 * call /transcribe, or set it in test_api.js:
 *   WEBHOOK_URL=http://localhost:4000/webhook node test_api.js
 *
 * On Colab you need a tunnel (e.g. ngrok) to expose this port:
 *   !npm install -g localtunnel
 *   !lt --port 4000
 *   # Use the printed URL as your WEBHOOK_URL
 *
 * Env vars:
 *   PORT     Port to listen on (default: 4000)
 *   SECRET   If set, reject requests that don't include x-webhook-secret header
 */

import express from 'express';

const PORT   = parseInt(process.env.PORT   || '4000', 10);
const SECRET = process.env.SECRET || null;

const app = express();
app.use(express.json({ limit: '50mb' }));

// ---------------------------------------------------------------------------
// Track received webhooks in memory (for manual inspection in the console)
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
app.listen(PORT, () => {
  console.log('============================================================');
  console.log(' Webhook Receiver Server');
  console.log(`  listening on  : http://localhost:${PORT}`);
  console.log(`  webhook URL   : http://localhost:${PORT}/webhook`);
  console.log(`  view received : http://localhost:${PORT}/webhooks`);
  if (SECRET) console.log(`  secret        : set (header x-webhook-secret)`);
  console.log('');
  console.log('  Set WEBHOOK_URL in your test environment:');
  console.log(`    WEBHOOK_URL=http://localhost:${PORT}/webhook node test_api.js`);
  console.log('============================================================\n');
});
