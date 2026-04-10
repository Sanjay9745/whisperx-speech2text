import gradio as gr
import requests
import time
import os

API_URL = "http://localhost:8000"
# Look for a common test key if there is one, or just empty if auth is disabled
API_KEY = os.environ.get("API_KEY", "") 

def transcribe_audio(audio_path, wait_for_result=True):
    if not audio_path:
        return "Please upload an audio file."
    
    headers = {"X-API-KEY": API_KEY} if API_KEY else {}
    
    try:
        # Step 1: Submit the audio file
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f)}
            response = requests.post(f"{API_URL}/transcribe", files=files, headers=headers)
        
        if response.status_code != 200:
            return f"Error submitting file: {response.text}"
        
        job_data = response.json()
        job_id = job_data.get("job_id")
        
        if not wait_for_result:
            return f"Job submitted! Job ID: {job_id}"
        
        # Step 2: Poll for results
        status_url = f"{API_URL}/status/{job_id}"
        result_url = f"{API_URL}/result/{job_id}"
        
        while True:
            status_res = requests.get(status_url, headers=headers)
            if status_res.status_code == 200:
                status = status_res.json().get("status")
                if status == "finished":
                    break
                elif status == "failed":
                    return "Job failed during processing."
            time.sleep(2)
            
        # Step 3: Fetch the results
        result_res = requests.get(result_url, headers=headers)
        if result_res.status_code == 200:
            result_data = result_res.json()
            # Try to format the output nicely
            try:
                text = ""
                for seg in result_data.get("segments", []):
                    speaker = seg.get("speaker", "UNKNOWN")
                    text += f"[{seg.get('start', 0.0):.2f} - {seg.get('end', 0.0):.2f}] {speaker}: {seg.get('text', '')}\n"
                return text if text else "Finished processing, but no text output."
            except Exception as e:
                return str(result_data)
        else:
            return f"Error fetching results: {result_res.text}"
            
    except requests.exceptions.ConnectionError:
        return "Cannot connect to the backend server. Make sure FastAPI is running (check restart.sh)."
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Define Gradio Interface
with gr.Blocks(title="Speech-to-Text Colab UI") as demo:
    gr.Markdown("# 🎙️ Speech-to-Text Transcription Tester")
    gr.Markdown("Upload an audio file to transcribe it using the locally running FastAPI backend.")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Upload Audio")
            submit_btn = gr.Button("Transcribe", variant="primary")
            
        with gr.Column():
            output_text = gr.Textbox(label="Transcription Result", lines=15)
            
    submit_btn.click(fn=transcribe_audio, inputs=[audio_input], outputs=[output_text])

if __name__ == "__main__":
    print("Starting Gradio UI...")
    # Share=True creates a public URL accessible from Colab
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
