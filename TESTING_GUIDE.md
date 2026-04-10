# End-to-End Testing Guide

This guide walks through a complete test of the Speech-to-Text API deployment, from Colab to public testing.

## Architecture Overview

```
┌─ Google Colab ──────────────────────────────────────┐
│                                                      │
│  1. FastAPI server (port 8000)  ←──── cloudflared   │
│  2. RQ worker (Redis queue)          tunnel         │
│  3. Results saved to /content/outputs/              │
│                                                      │
└──────────────────────────┬──────────────────────────┘
                           │
                    PUBLIC HTTPS URL
                  (from cloudflared tunnel)
                           │
                           ↓
┌─ Local Machine ──────────────────────────────────────┐
│                                                      │
│  1. Webhook server (port 4000)  ←──── localtunnel   │
│  2. HTML UI for job submission        tunnel        │
│  3. Real-time status polling                        │
│  4. Result display + download                       │
│                                                      │
└──────────────────────────────────────────────────────┘
```

## Phase 1: Colab Setup (5-10 minutes)

### Step 1: Start Colab
1. Open `speech2text_colab.ipynb` in Google Colab
2. Run cells in order: **1 → 2 → 3 → 4 → 5 → 6 → 7**
   - Cell 1: Mount Google Drive
   - Cell 2: Clone/update repo
   - Cell 3: Install requirements (installs fixed versions)
   - Cell 4: Setup environment variables
   - Cell 5: Check imports
   - Cell 6: Start Redis
   - Cell 7: Start API + Worker (starts cloudflared tunnel)

### Step 2: Get Public API URL
After cell 7 runs, you'll see output like:
```
================== API ==================
API listening on http://localhost:8000
=== Cloudflared Tunnel ===
Your URL is: https://abc123def456.trycloudflare.com

✅ API is publicly accessible at:
   https://abc123def456.trycloudflare.com
```

**Copy this URL** — you'll need it for testing.

### Step 3: Verify API Health
Run cell 8 to test the API locally:
```bash
curl -X GET https://abc123def456.trycloudflare.com/health \
  -H "X-API-Key: colab-test-key"
```

Expected response:
```json
{"status": "ok"}
```

### Step 4: Check Job Results
Run cell 9 (Output Browser) to list all saved transcription results in `/content/outputs/`.

## Phase 2: Local Webhook Server (5 minutes)

In a local terminal, start the webhook server with public tunnel:

```bash
cd test
npm run webhook-public
```

You'll see output like:
```
============================================================
 Webhook Receiver + Test UI Server
  listening on  : http://localhost:4000
  UI            : http://localhost:4000/
  webhook URL   : http://localhost:4000/webhook

 Opening localtunnel (public HTTPS URL) …
  Public URL    : https://xyz789abc123.localtunnel.net
  Public UI     : https://xyz789abc123.localtunnel.net/
  Public webhook: https://xyz789abc123.localtunnel.net/webhook

  Use this webhook URL in your API calls:
    webhook_url: https://xyz789abc123.localtunnel.net/webhook
============================================================
```

**Copy the public webhook URL** for the next step.

## Phase 3: Browser-Based Testing (10 minutes)

### Step 1: Open HTML UI
1. Visit the public webhook URL in your browser:
   ```
   https://xyz789abc123.localtunnel.net/
   ```

2. You'll see a dark-themed form with fields:
   - **API URL**: Paste the Colab public API URL
   - **API Key**: `colab-test-key`
   - **Audio File**: URL to an audio file (or paste base64)
   - **Webhook URL**: Pre-filled with the public webhook URL
   - **Language**: Select language (e.g., `en`)

### Step 2: Submit a Job
1. Fill in the form:
   ```
   API URL:      https://abc123def456.trycloudflare.com
   API Key:      colab-test-key
   Audio File:   https://example.com/sample.mp3
   Webhook URL:  https://xyz789abc123.localtunnel.net/webhook
   Language:     en
   ```

2. Click **Submit Job**

3. You'll immediately see:
   - Job ID displayed
   - Status label showing `queued`
   - A progress indicator

### Step 3: Monitor Job Progress
The UI automatically polls the API status endpoint every 1-2 seconds. You'll see:
- Status transitions: `queued` → `processing` → `completed`
- Elapsed time
- Progress bar filling

### Step 4: View Results
When the job completes, you'll see:
- **Full transcript** at the top
- **Segments** table with timings and text
- **Word timestamps** (if enabled)
- Buttons to:
  - Copy transcript to clipboard
  - Download result as JSON

### Step 5: Check Webhook Delivery
In the terminal where the webhook server is running, you'll see:
```
============================================================
 WEBHOOK RECEIVED  #1  at 2024-01-15T10:30:45.123Z
============================================================
  job_id  : abc-123-def
  status  : completed
  result  :
    text (length): 145 chars
    segments    : 12 segments
```

The webhook payload is also displayed in the HTML UI under **Received Webhooks**.

## Phase 4: Automated Public Testing (5 minutes)

For hands-free testing, use the automated script:

```bash
cd test
npm run test:public -- \
  --api https://abc123def456.trycloudflare.com \
  --key colab-test-key
```

The script will:
1. ✅ Start a local webhook server
2. ✅ Open a localtunnel for the webhook
3. ✅ Submit a test job to the public API
4. ✅ Poll for job completion
5. ✅ Fetch and display the result
6. ✅ Show webhook delivery summary

Example output:
```
[test:public] Starting webhook server...
[test:public] Webhook server listening on http://localhost:4000
[test:public] Opening localtunnel...
[test:public] Public webhook URL: https://abcd-1234.localtunnel.net/webhook

[api] Submitting job...
  Audio:  https://example.com/audio.mp3
  Key:    colab-test-key
  API:    https://abc123def456.trycloudflare.com
  
[api] Job created: job-id-12345

[poll] Status: queued (0s)
[poll] Status: processing (2s)
[poll] Status: processing (5s)
[poll] Status: completed (8s)

[result] Transcript (145 chars):
  "The quick brown fox jumps over the lazy dog..."

[webhook] Received 1 webhook(s):
  #1 at 2024-01-15T10:30:45.123Z
     Status: completed
     Segments: 12
     Words: 25
     
✅ All tests passed!
```

## Troubleshooting

### "API is not accessible"
- ✅ Verify Colab cell 7 is running (cloudflared tunnel active)
- ✅ Check the public URL is correct (no typos)
- ✅ Try the health endpoint: `curl -H "X-API-Key: colab-test-key" <api-url>/health`

### "Job never completes"
- ✅ Check Colab worker (cell 7 — look for worker logs)
- ✅ Verify Redis is running (cell 6)
- ✅ Check job in queue: Visit Colab cell 11 to diagnose

### "Webhook not received"
- ✅ Verify webhook URL is public (from localtunnel, not localhost)
- ✅ Check firewall/network blocking
- ✅ View webhook server terminal for any errors
- ✅ Try accessing public webhook URL directly: `curl <webhook-url>/health`

### "Cannot find module errors"
```bash
cd test
npm install
```

### "Port already in use"
- Colab port 8000 in use: Change in cell 7 (e.g., `PORT=8001`)
- Webhook port 4000 in use: `PORT=5000 npm run webhook-server`

## Typical Timeline

| Phase | Duration | What's Happening |
|-------|----------|-----------------|
| Colab setup | 5 min | Install deps, start API/worker |
| Webhook server | 1 min | Start tunnel, get public URL |
| Job submission | <1 sec | API queues job |
| Processing | 5-15 sec | Worker transcribes audio |
| Webhook delivery | <1 sec | API sends result callback |
| Total | **10-20 min** | From start to result |

## Success Criteria

✅ You have successfully tested the full pipeline when:

1. **Colab is running**
   - API responds at public URL
   - Worker is processing jobs
   - Results saved to `/content/outputs/`

2. **Webhook server is public**
   - UI accessible at public URL
   - Can submit jobs from HTML form
   - Status updates in real-time

3. **Full pipeline works**
   - Job starts from public UI
   - Colab API processes it
   - Webhook delivered to public URL
   - Result displays in browser

4. **Data transfer works**
   - Transcript properly decoded
   - Segments with correct timestamps
   - Word-level timestamps present (if enabled)

## Next Steps

### Customization
- Modify audio preprocessing in `app/transcriber/chunker.py`
- Adjust VAD settings in `app/transcriber/vad.py`
- Tweak Whisper parameters in `app/transcriber/whisper.py`

### Deployment
- Host webhook server on a permanent server (not local machine)
- Use a reverse proxy (nginx) for multiple APIs
- Set up monitoring/alerting for job failures

### Integration
- Call the public API from your own application
- Parse webhook payloads for downstream processing
- Store results in your database

## Support

For issues, check:
1. Colab cell execution order (ensure 1→2→3→...→7)
2. Webhook server running with `--tunnel` flag
3. Public URLs are accessible in browser (no firewall)
4. Audio files are accessible (not behind authentication)

Good luck! 🚀
