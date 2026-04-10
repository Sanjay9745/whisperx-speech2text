# Webhook Server + HTML UI

The webhook server provides both a receiver for transcription job callbacks and an interactive HTML UI for testing.

## Quick Start

### Local Testing
```bash
npm run webhook-server        # Start at http://localhost:4000
```

Visit `http://localhost:4000` in your browser. You'll see:
- **Job Submission Form** — Enter API URL, API key, audio file/URL
- **Job Status Polling** — Real-time updates as the job progresses
- **Result Viewer** — Transcript, segments, word timestamps when ready
- **Webhook Viewer** — See all received webhook payloads

### Public Testing (with localtunnel)
```bash
npm run webhook-public        # Start + open public tunnel
npm run ui                    # Alias: same as webhook-public
```

This will:
1. Start the webhook server on port 4000
2. Open a free localtunnel connection
3. Print public URLs like `https://abcd-efgh.localtunnel.net`
4. Display both local and public webhook URLs

**Example output:**
```
============================================================
 Webhook Receiver + Test UI Server
  listening on  : http://localhost:4000
  UI            : http://localhost:4000/
  webhook URL   : http://localhost:4000/webhook

 Opening localtunnel (public HTTPS URL) …
  Public URL    : https://abcd-1234.localtunnel.net
  Public UI     : https://abcd-1234.localtunnel.net/
  Public webhook: https://abcd-1234.localtunnel.net/webhook

  Use this webhook URL in your API calls:
    webhook_url: https://abcd-1234.localtunnel.net/webhook
============================================================
```

## Configuration

### Environment Variables
```bash
PORT=5000 npm run webhook-server     # Custom port
SECRET=my-secret npm run webhook-server  # Validate webhook signature
```

### Using with the API

When submitting a job to the Speech-to-Text API, include the webhook URL:

```bash
curl -X POST https://your-api.trycloudflare.com/transcribe \
  -H "X-API-Key: colab-test-key" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://example.com/audio.mp3",
    "webhook_url": "https://xxxx.localtunnel.net/webhook",
    "language": "en"
  }'
```

The server will:
1. Receive the job response
2. Poll the status endpoint
3. Receive the webhook callback when complete
4. Display the result in the browser

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serve HTML UI |
| `/ui` | GET | Serve HTML UI (alias) |
| `/webhook` | POST | Receive job completion callbacks |
| `/webhooks` | GET | List all received webhooks |
| `/health` | GET | Health check |

## npm Scripts

```bash
npm run webhook-server        # Start webhook server locally
npm run webhook-public        # Start + localtunnel tunnel
npm run ui                    # Alias for webhook-public
npm run test:public           # Run automated public URL test
npm run test                  # Run basic API test
```

## Testing Against Public API

1. **In one terminal**, start the webhook server with public tunnel:
   ```bash
   npm run webhook-public
   ```
   Copy the public webhook URL from the output.

2. **In another terminal**, run the public test:
   ```bash
   WEBHOOK_URL=<public_webhook_url> node run_public_test.js \
     --api <public_api_url> \
     --key colab-test-key
   ```

3. **Or in your browser**, visit the public UI URL and submit a job manually.

## Webhook Payload Format

When a transcription job completes, the server sends a webhook payload:

```json
{
  "job_id": "abc-123-def",
  "status": "completed",
  "result": {
    "text": "Full transcript...",
    "segments": [
      {
        "start": 0.5,
        "end": 2.3,
        "text": "Segment text"
      }
    ],
    "word_timestamps": [...]
  }
}
```

The HTML UI automatically displays this when received.

## Troubleshooting

### "localtunnel failed: getaddrinfo ENOTFOUND"
- Ensure you have internet connectivity
- Run `npm install localtunnel` if not already installed

### "Cannot find module 'localtunnel'"
```bash
npm install
```

### Port already in use
```bash
PORT=5000 npm run webhook-server
```

### Want to see all received webhooks
```bash
curl http://localhost:4000/webhooks | jq
```

## HTML UI Features

- **Dark theme** for comfortable viewing
- **Settings stored in browser** (localStorage) — no need to re-enter
- **Real-time status polling** — updates every 1-2 seconds
- **Progress indicators** — visual feedback on job state
- **Transcript display** — with segments and word-level timestamps
- **Copy to clipboard** — easy sharing of results
- **Download JSON** — save results for archival
- **Webhook viewer** — inspect received payloads in real-time
