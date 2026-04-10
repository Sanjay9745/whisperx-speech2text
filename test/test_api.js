import fs from 'fs';
import FormData from 'form-data';
import axios from 'axios';
import path from 'path';

// ==========================================
// CONFIGURATION
// ==========================================

const API_URL = 'http://localhost:8000'; // Make sure your FastAPI server is running here
const API_KEY = 'your_api_key_here';     // Place your API key here (if enabled in Python app)

// Provide either a URL or a FILE_PATH to transcribe.
// Set to null to use the other option.
const AUDIO_URL = 'https://download.samplelib.com/mp3/sample-3s.mp3';
const AUDIO_FILE_PATH = null; // e.g., './sample.mp3'

// Webhook to receive updates on transcription
const WEBHOOK_URL = 'https://webhook.site/your-webhook-id'; // Change this or use null

// Arbitrary metadata attached to the request
const METADATA = {
  userId: 'user-789',
  referenceId: 'test-job-001',
  environment: 'development'
};

// ==========================================
// TEST SCRIPT
// ==========================================

async function runTest() {
  console.log('=== Starting Speech-to-Text API Test ===');
  
  const formData = new FormData();
  
  // 1. Add audio source (File or URL)
  if (AUDIO_FILE_PATH && fs.existsSync(AUDIO_FILE_PATH)) {
    formData.append('file', fs.createReadStream(AUDIO_FILE_PATH));
    console.log(`[Config] Using AUDIO FILE: ${AUDIO_FILE_PATH}`);
  } else if (AUDIO_URL) {
    formData.append('audio_url', AUDIO_URL);
    console.log(`[Config] Using AUDIO URL: ${AUDIO_URL}`);
  } else {
    console.error('ERROR: You must provide either AUDIO_URL or a valid AUDIO_FILE_PATH.');
    process.exit(1);
  }

  // 2. Add metadata
  formData.append('metadata', JSON.stringify(METADATA));
  console.log(`[Config] Appended Metadata:`, METADATA);

  // 3. Add webhook
  if (WEBHOOK_URL) {
    formData.append('webhook_url', WEBHOOK_URL);
    console.log(`[Config] Using Webhook URL: ${WEBHOOK_URL}`);
  }

  // Set HTTP headers including auth and multipart form boundary
  const headers = {
    ...formData.getHeaders(),
    'x-api-key': API_KEY // Header used by `APIKeyMiddleware`
  };

  try {
    console.log(`\n[Request] Sending POST request to ${API_URL}/transcribe`);
    const startTime = Date.now();

    const response = await axios.post(`${API_URL}/transcribe`, formData, {
      headers: headers,
      maxContentLength: Infinity,
      maxBodyLength: Infinity
    });

    const elapsedMs = Date.now() - startTime;
    console.log(`[Response] Status: ${response.status} ${response.statusText} (${elapsedMs}ms)`);
    console.log('[Response] Body:', JSON.stringify(response.data, null, 2));

    // Wait and poll for status if we have a job ID
    if (response.data && response.data.job_id) {
      await pollJobStatus(response.data.job_id);
    }
    
  } catch (error) {
    console.error('[Error] Request failed!');
    if (error.response) {
      console.error('Status:', error.response.status);
      console.error('Data:', error.response.data);
    } else {
      console.error(error.message);
    }
  }
}

async function pollJobStatus(jobId) {
  console.log(`\n=== Polling Job Status (${jobId}) ===`);
  const pollIntervalMs = 3000;
  
  while (true) {
    try {
      const response = await axios.get(`${API_URL}/status/${jobId}`, {
        headers: { 'x-api-key': API_KEY }
      });
      
      const state = response.data.state;
      console.log(`[Job Status] ${state}`);
      
      if (state === 'finished' || state === 'failed') {
          console.log('\n=== Job Completed ===');
          // Fetch final result
          if (state === 'finished') {
              const resultRes = await axios.get(`${API_URL}/result/${jobId}`, {
                  headers: { 'x-api-key': API_KEY }
              });
              console.log('[Result] Transcription Data:');
              console.log(JSON.stringify(resultRes.data, null, 2));
          }
          break;
      }
      
      // Wait before polling again
      await new Promise(resolve => setTimeout(resolve, pollIntervalMs));
      
    } catch (err) {
      console.error('[Error] Status poller failed:');
      console.error(err.response ? err.response.data : err.message);
      break;
    }
  }
}

runTest();
