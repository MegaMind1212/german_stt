import os
import re
import unicodedata
import logging
import base64
import requests
import json
import time
import tempfile
from flask import Flask, request, render_template, send_from_directory, jsonify
import torch
import torchaudio
from transformers import pipeline
from moviepy.editor import VideoFileClip

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

# Use /tmp for Vercel compatibility
UPLOAD_FOLDER = '/tmp/Uploads'
TEMP_FOLDER = '/tmp/Temp'
DOWNLOAD_FOLDER = '/tmp/Downloads'
for folder in [UPLOAD_FOLDER, TEMP_FOLDER, DOWNLOAD_FOLDER]:
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
            logging.info(f"Created folder: {folder}")
        except Exception as e:
            logging.error(f"Failed to create folder {folder}: {str(e)}")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMP_FOLDER'] = TEMP_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

# Load Whisper model (tiny for speed)
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",
        device=device
    )
    logging.info("Whisper model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load Whisper model: {str(e)}")

# IndoScribe API URL
INDOSCRIBE_API_URL = "https://enabe-node-function-lbfmjph4ja-el.a.run.app"

# Verbal punctuation mapping
punctuation_map = {
    "ausrufezeichen": "!",
    "fragezeichen": "?",
    "punkt": ".",
    "komma": ",",
    "semikolon": ";",
    "doppelpunkt": ":"
}

# Simple dictionary of common German nouns
german_nouns = {
    "buch": "Buch",
    "musik": "Musik",
    "haus": "Haus",
    "auto": "Auto",
    "baum": "Baum",
    "freund": "Freund",
    "schule": "Schule",
    "lehrer": "Lehrer",
    "kind": "Kind",
    "stadt": "Stadt"
}

# Languages to use IndoScribe for
INDOSCRIBE_LANGUAGES = {"hi-IN", "mr-IN", "gu-IN"}

def sanitize_filename(filename):
    """Remove special characters and emojis from filename, ensuring compatibility."""
    filename = unicodedata.normalize('NFKD', filename).encode('ascii', 'ignore').decode('ascii')
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    filename = re.sub(r'_+', '_', filename)  # Replace multiple underscores with a single one
    base, ext = os.path.splitext(filename)
    if not ext.lower() in ('.mp3', '.mp4', '.wav'):
        ext = '.mp4' if filename.lower().endswith('.mp4') else '.mp3'
    return base + ext

def capitalize_german_nouns(text):
    """Capitalize German nouns using a dictionary."""
    words = text.split()
    result = []
    for word in words:
        lower_word = word.lower()
        if lower_word in german_nouns:
            result.append(german_nouns[lower_word])
        else:
            result.append(word)
    return " ".join(result)

def handle_verbal_punctuation(text):
    """Replace verbal punctuation with symbols."""
    for verbal, symbol in punctuation_map.items():
        text = re.sub(r'\b' + verbal + r'\b', symbol, text, flags=re.IGNORECASE)
    return text

def is_valid_audio_file(file_path):
    """Check if the file exists and is non-empty."""
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        return False, f"File does not exist: {file_path}"
    if os.path.getsize(file_path) == 0:
        return False, f"File is empty: {file_path}"
    if file_path.lower().endswith('.wav'):
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)
                if not header.startswith(b'RIFF'):
                    logging.warning(f"WAV file {file_path} has unexpected header, proceeding anyway")
        except Exception as e:
            logging.warning(f"Header check failed for {file_path}: {str(e)}, proceeding anyway")
    return True, ""

def convert_to_wav(file_path, temp_folder):
    """Convert audio file to WAV format."""
    file_path = os.path.abspath(file_path)  # Use absolute path
    logging.info(f"Converting file to WAV: {file_path}, exists: {os.path.exists(file_path)}")
    
    # Verify the file exists before proceeding
    is_valid, error = is_valid_audio_file(file_path)
    if not is_valid:
        raise ValueError(error)

    try:
        if file_path.lower().endswith('.wav'):
            return file_path
        
        wav_path = os.path.abspath(os.path.join(temp_folder, "converted_audio.wav"))
        logging.info(f"Target WAV path: {wav_path}")

        if file_path.lower().endswith('.mp4'):
            logging.info(f"Extracting audio from MP4: {file_path}")
            video = VideoFileClip(file_path)
            if video.audio is None:
                raise RuntimeError("MP4 file has no audio track")
            video.audio.write_audiofile(wav_path, codec='pcm_s16le', fps=16000, logger=None)
            video.close()
        else:  # MP3 or other formats
            logging.info(f"Converting audio file to WAV: {file_path}")
            audio, sample_rate = torchaudio.load(file_path)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                audio = resampler(audio)
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            torchaudio.save(wav_path, audio, 16000, format="wav")

        # Verify the WAV file was created
        if not os.path.exists(wav_path):
            raise RuntimeError(f"Converted WAV file does not exist: {wav_path}")
        if os.path.getsize(wav_path) == 0:
            raise RuntimeError(f"Converted WAV file is empty: {wav_path}")
        logging.info(f"Converted audio to WAV: {wav_path}, size: {os.path.getsize(wav_path)} bytes")
        return wav_path
    except Exception as e:
        logging.error(f"Failed to convert audio to WAV: {str(e)}")
        raise RuntimeError(f"Failed to convert audio to WAV: {str(e)}")

def transcribe_with_whisper(file_path, temp_folder, language, detect_language=True):
    """Transcribe audio using Whisper, optionally detecting the language.

    Args:
        file_path (str): Path to the audio file.
        temp_folder (str): Temporary folder for WAV conversion.
        language (str): Language code (e.g., 'hi-IN').
        detect_language (bool): Whether to detect the language automatically.

    Returns:
        tuple: (transcription, detected_language)
    """
    logging.info(f"Attempting to transcribe with Whisper: {file_path}")
    original_file_path = file_path

    if file_path.lower().endswith('.mp4'):
        wav_path = convert_to_wav(file_path, temp_folder)
        file_path = wav_path
    else:
        file_path = os.path.abspath(file_path)

    is_valid, error = is_valid_audio_file(file_path)
    if not is_valid:
        raise ValueError(error)

    try:
        logging.info(f"Loading audio file: {file_path}, exists: {os.path.exists(file_path)}")
        waveform, sample_rate = torchaudio.load(file_path)
        logging.info(f"Audio loaded: {file_path}, sample rate: {sample_rate}, shape: {waveform.shape}")
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {str(e)}")

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        logging.info(f"Resampled audio to 16kHz")

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        logging.info(f"Converted audio to mono")

    audio = waveform.squeeze().numpy()

    try:
        whisper_lang = language.split('-')[0]  # e.g., "hi-IN" -> "hi"
        detected_lang = whisper_lang

        if detect_language:
            # First pass: detect the language
            result = pipe(audio, generate_kwargs={"task": "transcribe", "return_timestamps": True})
            detected_lang = result.get("language", whisper_lang)
            logging.info(f"Detected language: {detected_lang}")
            if detected_lang != whisper_lang:
                logging.warning(f"Selected language {whisper_lang} does not match detected language {detected_lang}, using detected language")
                whisper_lang = detected_lang
        
        # Second pass: transcribe with the correct language
        result = pipe(audio, generate_kwargs={"language": whisper_lang, "return_timestamps": True})
        transcription = result["text"]
        logging.info(f"Whisper transcription completed: {transcription}")
        return transcription, detected_lang
    except Exception as e:
        raise RuntimeError(f"Whisper transcription failed: {str(e)}")
    finally:
        if original_file_path.lower().endswith('.mp4') and file_path.lower().endswith('.wav'):
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Cleaned up WAV: {file_path}")
        if os.path.exists(original_file_path):
            os.remove(original_file_path)
            logging.info(f"Cleaned up original: {original_file_path}")

def transcribe_with_indoscribe(file_path, temp_folder, language):
    """Transcribe audio using IndoScribe API.

    Args:
        file_path (str): Path to the audio file.
        temp_folder (str): Temporary folder for WAV conversion.
        language (str): Language code (e.g., 'hi-IN').

    Returns:
        str: Transcription text or None if failed.
    """
    logging.info(f"Attempting to transcribe with IndoScribe: {file_path}")
    original_file_path = file_path

    # Ensure the file is in WAV format for IndoScribe
    wav_path = convert_to_wav(file_path, temp_folder)
    file_path = wav_path

    try:
        # Test API accessibility
        try:
            test_response = requests.get(INDOSCRIBE_API_URL, timeout=5)
            logging.info(f"IndoScribe API test response: {test_response.status_code}")
        except requests.exceptions.RequestException as e:
            logging.warning(f"IndoScribe API not accessible: {str(e)}, returning None")
            return None

        # Convert audio file to base64
        with open(file_path, "rb") as audio_file:
            audio_content = base64.b64encode(audio_file.read()).decode("utf-8")
        logging.info(f"Audio file converted to base64, length: {len(audio_content)}")

        # Create payload for IndoScribe API
        payload = {
            "type": "Indo-Scribe",
            "action": "STT",
            "code": language,
            "audio": audio_content
        }

        # Send POST request to IndoScribe API
        logging.info(f"Sending request to IndoScribe API for language: {language}")
        response = requests.post(INDOSCRIBE_API_URL, json=payload, timeout=30)
        logging.info(f"IndoScribe API response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            logging.info(f"IndoScribe API response: {result}")
            transcription = result.get("transcribedText", "")
            if not transcription:
                logging.warning(f"IndoScribe returned empty transcription")
                return None
            logging.info(f"IndoScribe transcription completed: {transcription}")
            return transcription
        else:
            logging.warning(f"IndoScribe API failed: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        logging.error(f"IndoScribe transcription failed: {str(e)}")
        return None
    finally:
        if original_file_path.lower().endswith('.mp4') and file_path.lower().endswith('.wav'):
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Cleaned up WAV: {file_path}")
        if os.path.exists(original_file_path):
            os.remove(original_file_path)
            logging.info(f"Cleaned up original: {original_file_path}")

def select_better_transcription(indoscribe_transcription, whisper_transcription):
    """Select the better transcription based on length.

    Args:
        indoscribe_transcription (str): Transcription from IndoScribe.
        whisper_transcription (str): Transcription from Whisper.

    Returns:
        str: The better transcription.
    """
    if not indoscribe_transcription:
        return whisper_transcription
    if not whisper_transcription:
        return indoscribe_transcription
    
    # Simple heuristic: longer transcription is likely better
    indoscribe_len = len(indoscribe_transcription)
    whisper_len = len(whisper_transcription)
    logging.info(f"Comparing transcriptions - IndoScribe length: {indoscribe_len}, Whisper length: {whisper_len}")
    
    if indoscribe_len > whisper_len:
        logging.info("Selected IndoScribe transcription")
        return indoscribe_transcription
    else:
        logging.info("Selected Whisper transcription")
        return whisper_transcription

def transcribe_audio(file_path, temp_folder, language):
    """Transcribe audio using the appropriate vendor with fallback.

    Args:
        file_path (str): Path to the audio file.
        temp_folder (str): Temporary folder for WAV conversion.
        language (str): Language code (e.g., 'hi-IN').

    Returns:
        tuple: (transcription, detected_language)
    """
    # Always get Whisper transcription with language detection
    whisper_transcription, detected_lang = transcribe_with_whisper(file_path, temp_folder, language, detect_language=True)
    
    if language in INDOSCRIBE_LANGUAGES:
        indoscribe_transcription = transcribe_with_indoscribe(file_path, temp_folder, language)
        transcription = select_better_transcription(indoscribe_transcription, whisper_transcription)
        return transcription, detected_lang
    return whisper_transcription, detected_lang

def post_process_transcription(transcription, language):
    """Apply language-specific post-processing."""
    transcription = handle_verbal_punctuation(transcription)
    if language.startswith("de"):
        transcription = capitalize_german_nouns(transcription)
    return transcription

# HTML template for the web interface with language selection
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Language Speech-to-Text</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #333; }
        .container { max-width: 600px; margin: auto; }
        .result { margin-top: 20px; padding: 10px; border: 1px solid #ccc; }
        .error { color: red; }
        .download-btn { margin-top: 10px; padding: 5px 10px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 5px; }
        .download-btn:hover { background-color: #45a049; }
        select, input[type="file"], input[type="submit"] { margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Multi-Language Speech-to-Text</h1>
        <p>Upload an MP3, MP4, or WAV file and select the language for transcription.</p>
        <form method="post" enctype="multipart/form-data" action="/transcribe">
            <select name="language" required>
                <option value="en-IN">English (India)</option>
                <option value="en-US">English (US)</option>
                <option value="de-DE">German</option>
                <option value="hi-IN">Hindi</option>
                <option value="mr-IN">Marathi</option>
                <option value="gu-IN">Gujarati</option>
                <option value="fr-FR">French</option>
                <option value="es-ES">Spanish</option>
            </select>
            <br>
            <input type="file" name="audio" accept=".mp3,.mp4,.wav" required>
            <input type="submit" value="Transcribe">
        </form>
        {% if transcription %}
        <div class="result">
            <h3>Transcription:</h3>
            <p>{{ transcription }}</p>
            {% if download_link %}
            <a href="{{ download_link }}" class="download-btn" download>Download Transcription</a>
            {% endif %}
        </div>
        {% endif %}
        {% if error %}
        <div class="error">
            <p>Error: {{ error }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return render_template('index.html', error="No audio file provided")
    file = request.files['audio']
    language = request.form.get('language')
    if not language:
        return render_template('index.html', error="Please select a language")
    if file.filename == '':
        return render_template('index.html', error="No file selected")
    if file and file.filename.lower().endswith(('.mp3', '.mp4', '.wav')):
        # Sanitize the filename before saving
        safe_filename = sanitize_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file_path = os.path.abspath(file_path)  # Use absolute path
        logging.info(f"Attempting to save file: {file_path}")
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                file_content = file.read()
                f.write(file_content)
            logging.info(f"File saved: {file_path}, size: {os.path.getsize(file_path)} bytes")
            
            # Add a short delay to ensure the file is fully written
            time.sleep(0.1)
            
            # Verify the file exists after saving
            if not os.path.isfile(file_path):
                raise ValueError(f"Failed to save file to {file_path}")
            if os.path.getsize(file_path) == 0:
                raise ValueError(f"Saved file {file_path} is empty")
            
            transcription, detected_lang = transcribe_audio(file_path, app.config['TEMP_FOLDER'], language)
            transcription = post_process_transcription(transcription, language)

            transcription_filename = f"transcription_{safe_filename}.txt"
            transcription_path = os.path.join(app.config['DOWNLOAD_FOLDER'], transcription_filename)
            with open(transcription_path, 'w', encoding='utf-8') as f:
                f.write(transcription)
            logging.info(f"Transcription saved to {transcription_path}")

            return render_template(
                'index.html',
                transcription=transcription,
                download_link=f"/download/{transcription_filename}"
            )
        except Exception as e:
            logging.error(f"Error during processing: {str(e)}")
            return render_template('index.html', error=f"Error: {str(e)}")
    return render_template('index.html', error="Invalid file format, please upload an MP3, MP4, or WAV")

@app.route('/api/transcribe', methods=['POST'])
def api_transcribe():
    """
    API endpoint to transcribe audio files into text.

    Method: POST
    Content-Type: multipart/form-data

    Parameters (form-data):
        - audio (file): The audio file to transcribe (MP3, MP4, or WAV).
        - language (string): The language code for transcription (e.g., 'en-US', 'hi-IN', 'de-DE').

    Responses:
        - 200: Successful transcription
            {
                "status": "success",
                "transcription": "transcribed text",
                "detected_language": "language code (e.g., 'de')",
                "selected_language": "language code (e.g., 'de-DE')"
            }
        - 400: Invalid request (e.g., missing file, invalid format)
            {
                "status": "error",
                "error": "error message"
            }
        - 500: Server error (e.g., transcription failure)
            {
                "status": "error",
                "error": "error message"
            }

    Example:
        curl -X POST -F "audio=@audio.mp4" -F "language=de-DE" http://your-vercel-app.vercel.app/api/transcribe
    """
    if 'audio' not in request.files:
        return jsonify({"status": "error", "error": "No audio file provided"}), 400
    file = request.files['audio']
    language = request.form.get('language')
    if not language:
        return jsonify({"status": "error", "error": "Language parameter is required"}), 400
    if file.filename == '':
        return jsonify({"status": "error", "error": "No file selected"}), 400
    if file and file.filename.lower().endswith(('.mp3', '.mp4', '.wav')):
        # Sanitize the filename before saving
        safe_filename = sanitize_filename(file.filename)
        
        # Use a temporary file to avoid race conditions
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(safe_filename)[1], dir=app.config['UPLOAD_FOLDER']) as temp_file:
            file_path = temp_file.name
            logging.info(f"Attempting to save file: {file_path}")
            file_content = file.read()
            temp_file.write(file_content)
            temp_file.close()
        
        try:
            logging.info(f"File saved: {file_path}, size: {os.path.getsize(file_path)} bytes")
            
            # Verify the file exists after saving
            if not os.path.isfile(file_path):
                raise ValueError(f"Failed to save file to {file_path}")
            if os.path.getsize(file_path) == 0:
                raise ValueError(f"Saved file {file_path} is empty")
            
            # Add file size limit for Vercel (4MB max on free tier)
            MAX_FILE_SIZE = 4 * 1024 * 1024  # 4MB
            if os.path.getsize(file_path) > MAX_FILE_SIZE:
                raise ValueError("File size exceeds 4MB limit (Vercel free tier restriction)")
            
            transcription, detected_lang = transcribe_audio(file_path, app.config['TEMP_FOLDER'], language)
            transcription = post_process_transcription(transcription, language)
            
            return jsonify({
                "status": "success",
                "transcription": transcription,
                "detected_language": detected_lang,
                "selected_language": language
            }), 200
        except Exception as e:
            logging.error(f"Error during processing: {str(e)}")
            return jsonify({"status": "error", "error": str(e)}), 500
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Cleaned up temporary file: {file_path}")
    return jsonify({"status": "error", "error": "Invalid file format, please upload an MP3, MP4, or WAV"}), 400

@app.route('/download/<filename>')
def download_file(filename):
    """Serve the transcription file for download."""
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    with open('templates/index.html', 'w') as f:
        f.write(HTML_TEMPLATE)
    # For Vercel, the app is run via WSGI, so this is for local testing only
    try:
        app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    except Exception as e:
        logging.error(f"Failed to start Flask app: {str(e)}")