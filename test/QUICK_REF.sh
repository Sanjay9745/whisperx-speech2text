#!/bin/bash
# quick-ref.sh - Quick Reference for Testing the Speech-to-Text API
#
# This is a reference of common commands. Copy/paste as needed.
#

# ============================================================================
# PHASE 1: COLAB SETUP
# ============================================================================
# 1. In Google Colab notebook (speech2text_colab.ipynb):
#    Run these cells in order:
#    Cell 1: !apt-get update && pip install --upgrade google-colab
#    Cell 2: git clone/pull repo
#    Cell 3: pip install -r requirements.txt
#    Cell 4: export env variables
#    Cell 5: python -c "import nemo; print('✅ nemo imports')"
#    Cell 6: redis-server --daemonize yes
#    Cell 7: Start API + Worker (cloudflared tunnel auto-opens)
#
# COLAB OUTPUTS TO COPY:
#    - Public API URL: https://abc123...trycloudflare.com
#    - Cloudflare tunnel ID for API reference

# ============================================================================
# PHASE 2: LOCAL WEBHOOK SERVER
# ============================================================================

# Start webhook server locally (for development)
npm run webhook-server
# → Listens on http://localhost:4000

# Start webhook server with public tunnel (for testing)
npm run webhook-public
# → Listens on http://localhost:4000
# → Also exposed via https://xxxx.localtunnel.net

# Equivalent commands:
cd test && node webhook_server.js                    # Local only
cd test && node webhook_server.js --tunnel            # With tunnel
PORT=5000 node webhook_server.js --tunnel             # Custom port

# ============================================================================
# PHASE 3: TEST THE API
# ============================================================================

# 1. Health check (verify API is accessible)
curl -X GET https://abc123def456.trycloudflare.com/health \
  -H "X-API-Key: colab-test-key"
# Expected: {"status": "ok"}

# 2. Submit a transcription job
curl -X POST https://abc123def456.trycloudflare.com/transcribe \
  -H "X-API-Key: colab-test-key" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://example.com/sample.mp3",
    "webhook_url": "https://xxxx.localtunnel.net/webhook",
    "language": "en",
    "word_timestamps": true
  }'
# Expected: {"job_id": "uuid...", "status": "queued"}

# 3. Check job status
JOB_ID="uuid-from-previous-step"
curl -X GET "https://abc123def456.trycloudflare.com/status/${JOB_ID}" \
  -H "X-API-Key: colab-test-key"
# Expected: {"job_id": "...", "status": "completed", ...}

# 4. Fetch completed result
curl -X GET "https://abc123def456.trycloudflare.com/result/${JOB_ID}" \
  -H "X-API-Key: colab-test-key" | jq
# Expected: {"text": "...", "segments": [...], "word_timestamps": [...]}

# ============================================================================
# PHASE 4: BROWSER-BASED TESTING
# ============================================================================

# Open the HTML UI in your browser
# Visit: https://xxxx.localtunnel.net/
#
# Use the form to:
#  1. Enter API URL (from Colab)
#  2. Enter API key (colab-test-key)
#  3. Enter audio file URL
#  4. Enter webhook URL (pre-filled)
#  5. Click Submit
#  6. Watch status updates
#  7. View result when complete

# View all received webhooks in the terminal where webhook server is running
# Or visit: http://localhost:4000/webhooks (JSON)

# ============================================================================
# PHASE 5: AUTOMATED END-TO-END TEST
# ============================================================================

# Run the full automated test
npm run test:public -- \
  --api https://abc123def456.trycloudflare.com \
  --key colab-test-key

# Or with environment variables
WEBHOOK_URL="https://xxxx.localtunnel.net/webhook" \
node test/run_public_test.js \
  --api https://abc123def456.trycloudflare.com \
  --key colab-test-key

# ============================================================================
# USEFUL DIAGNOSTICS
# ============================================================================

# Check webhook server health
curl http://localhost:4000/health
# Expected: {"status":"ok"}

# View webhook server logs
curl http://localhost:4000/webhooks | jq
# Shows all received webhook payloads

# Check API health
curl -H "X-API-Key: colab-test-key" https://api-url/health

# Get detailed job info (from Colab terminal):
# python -c "
# from app.config import redis_conn
# from rq import Queue
# q = Queue(connection=redis_conn)
# for job in q.get_job_ids():
#     j = q.fetch_job(job)
#     print(f'{j.id}: {j.get_status()}')"

# Monitor worker logs (in Colab cell 7 terminal)
# Watch for: 🔄 Processing job / ✅ Job saved

# ============================================================================
# TROUBLESHOOTING COMMANDS
# ============================================================================

# Check if ports are in use (Windows PowerShell):
# netstat -ano | findstr :8000
# netstat -ano | findstr :4000

# Kill process on port (Windows PowerShell):
# Stop-Process -Id <PID> -Force

# Restart Redis (Colab):
# killall redis-server ; sleep 1 ; redis-server --daemonize yes

# Check Colab disk usage:
# df -h
# du -sh /content/outputs

# Restart Colab session:
# Runtime → Factory reset → Clear all outputs and code

# ============================================================================
# SAMPLE TEST AUDIO FILES (Use these URLs)
# ============================================================================

# Audio with clear speech:
# https://commondatastorage.googleapis.com/codesearch/audio/16000/3.wav

# Sample from SQuAD (if available):
# https://example.com/sample.wav

# Create test audio (ffmpeg):
# echo "Hello world, this is a test." | ffmpeg -f lavfi -i anullsrc=r=16000:cl=mono -c:a pcm_s16le -t 5 test.wav

# ============================================================================
# TYPICAL WORKFLOW
# ============================================================================

# Terminal 1: Start Colab (in notebook)
#   Cells 1-7 (5-10 min)
#   → Copy public API URL

# Terminal 2: Start webhook server
cd test && npm run webhook-public
#   → Copy public webhook URL

# Browser: Open webhook UI
# Visit: https://<webhook-tunnel>/
#   → Fill form with API URL from Colab
#   → Webhook URL pre-filled
#   → Submit job
#   → Watch progress
#   → View result

# Or Terminal 3: Run automated test
npm run test:public -- --api <api-url> --key colab-test-key
#   → Full test in one command
#   → Results printed to console

# ============================================================================
# EXPECTED TIMINGS
# ============================================================================

# Colab setup:                  5-10 min
# Webhook server startup:       1 min
# HTML UI load:                 <1 sec
# Job submission:               <1 sec
# Job processing (in Colab):    5-15 sec (depends on audio length)
# Webhook delivery:             <1 sec
# Result display:               <1 sec (auto-polls)
# ─────────────────────────────────────
# Total end-to-end:            10-20 min

# ============================================================================
