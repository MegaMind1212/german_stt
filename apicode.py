import os
import re
import logging
import base64
import requests
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import ffmpeg
import torch
import torchaudio
from transformers import pipeline

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit
app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'mp4', 'wav', 'ogg', 'm4a'}

# Create upload folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Whisper model (using small instead of tiny for better accuracy)
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device=device
    )
    logging.info("Whisper model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load Whisper model: {str(e)}")
    pipe = None

# External API configurations
INDOSCRIBE_API_URL = "https://enabe-node-function-lbfmjph4ja-el.a.run.app"
INDOSCRIBE_LANGUAGES = {"hi-IN", "mr-IN", "gu-IN"}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def convert_to_wav(input_path, output_path):
    """Convert any audio file to WAV format using ffmpeg"""
    try:
        (
            ffmpeg
            .input(input_path)
            .output(output_path, ac=1, ar=16000, acodec='pcm_s16le')
            .overwrite_output()
            .run(quiet=True)
        return True
    except ffmpeg.Error as e:
        logging.error(f"FFmpeg error: {e.stderr.decode('utf8')}")
        return False

def transcribe_with_whisper(file_path, language):
    """Transcribe audio using Whisper model"""
    try:
        # Convert to WAV if needed
        if not file_path.lower().endswith('.wav'):
            wav_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp.wav')
            if convert_to_wav(file_path, wav_path):
                file_path = wav_path
            else:
                raise RuntimeError("Audio conversion failed")

        # Load audio file
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Get numpy array
        audio = waveform.squeeze().numpy()
        
        # Get language code (e.g., "en-IN" -> "en")
        whisper_lang = language.split('-')[0]
        
        # Transcribe
        result = pipe(audio, generate_kwargs={"language": whisper_lang})
        return result["text"]
    
    except Exception as e:
        logging.error(f"Whisper transcription error: {str(e)}")
        raise RuntimeError(f"Transcription failed: {str(e)}")

def transcribe_with_indoscribe(file_path, language):
    """Transcribe audio using IndoScribe API"""
    try:
        # Convert to WAV first
        wav_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_indoscribe.wav')
        if not convert_to_wav(file_path, wav_path):
            raise RuntimeError("Audio conversion failed for IndoScribe")
        
        # Read and encode audio
        with open(wav_path, "rb") as audio_file:
            audio_content = base64.b64encode(audio_file.read()).decode("utf-8")
        
        # Prepare payload
        payload = {
            "type": "Indo-Scribe",
            "action": "STT",
            "code": language,
            "audio": audio_content
        }
        
        # Call API
        response = requests.post(INDOSCRIBE_API_URL, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result.get("transcribedText", "")
        else:
            raise RuntimeError(f"IndoScribe API error: {response.status_code}")
    
    except Exception as e:
        logging.error(f"IndoScribe error: {str(e)}")
        raise RuntimeError(f"IndoScribe service failed: {str(e)}")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Main API endpoint for transcription"""
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    # Validate file
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400
    
    # Get language (default to English if not provided)
    language = request.form.get('language', 'en-US')
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Choose transcription method based on language
        if language in INDOSCRIBE_LANGUAGES:
            transcription = transcribe_with_indoscribe(file_path, language)
        else:
            transcription = transcribe_with_whisper(file_path, language)
        
        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return jsonify({
            "status": "success",
            "transcription": transcription,
            "language": language
        })
    
    except Exception as e:
        # Clean up in case of error
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": pipe is not None})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))