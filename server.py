from flask import Flask, render_template, request, jsonify, session
from speech import get_intervals, detect_clapping_cheering, calculate_intensities, get_reactive_lines, transcribe_audio_with_reaction_focus
from waitress import serve
import os
from werkzeug.utils import secure_filename
import tempfile
from pydub import AudioSegment
import json
from datetime import datetime
import threading
from functools import partial
import librosa
import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 60 * 1024 * 1024  # 60MB max file size
app.secret_key = 'aK8xs#p2Qn9v$mL5'

ALLOWED_EXTENSIONS = {'mp3', 'wav'}

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
@app.route("/index")
def index():
    print("Rendering index page.")
    return render_template('index.html')

def process_audio(filepath, filename):
    print(f"Processing audio file: {filename}")
    try:
        # Convert to WAV if MP3 and log the progress
        if filename.endswith('.mp3'):
            print("Converting MP3 to WAV format...")
            wav_path = os.path.join(app.config['UPLOAD_FOLDER'], 'speech.wav')
            audio = AudioSegment.from_mp3(filepath)
            audio.export(wav_path, format="wav")
            os.remove(filepath)
            filepath = wav_path
            print("Conversion complete.")

        # Load and process audio in smaller chunks
        print("Loading audio with librosa...")
        y, sr = librosa.load(filepath, sr=16000)  # Force 16kHz sampling rate
        print("Audio loaded.")

        # Process audio
        print("Detecting clapping and cheering intervals...")
        intervals, _, _ = detect_clapping_cheering(y, sr)
        print(f"Intervals detected: {intervals}")

        print("Calculating intensities...")
        intensities = calculate_intensities(intervals, y, sr)
        intensities = [float(intensity * 100) for intensity in intensities]
        print("Intensities calculated.")

        # Only transcribe portions with reactions to save time
        print("Starting transcription...")
        transcript = transcribe_audio_with_reaction_focus(filepath, intervals)
        print("Transcription complete.")

        print("Extracting reactive lines...")
        reactive_lines = get_reactive_lines(transcript, intervals)
        print(f"Raw reactive lines: {reactive_lines}")  # Debug print

        # Clean upS
        os.remove(filepath)
        print("Temporary file cleaned up.")

        # Keep timestamps in original format
        processed_reactive_lines = [
            {
                'start': start,  # Keep original timestamp string
                'end': end,      # Keep original timestamp string
                'text': text
            }
            for start, end, text in reactive_lines
        ]

        results = {
            'filename': filename,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'reactive_lines': processed_reactive_lines,
            'intervals': [[float(start), float(end)] for start, end in intervals],
            'intensities': intensities,
            'status': 'complete'
        }
        
        print("Processed results:", json.dumps(results, indent=2))  # Debug print
        print("Audio processing complete.")
        
        return results
        
    except Exception as e:
        print(f"Error during audio processing: {e}")
        return {'error': str(e), 'status': 'error'}

@app.route("/analyze", methods=['POST'])
def analyze_speech():
    logging.debug("Received request to /analyze endpoint.")
    logging.debug(f"Max content length set to: {app.config['MAX_CONTENT_LENGTH']} bytes")
    if 'file' not in request.files:
        print("No file part in request.")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        print("No selected file.")
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"File saved to {filepath}")
            
            # Start processing
            results = process_audio(filepath, filename)
            
            if 'error' in results:
                print(f"Error in results: {results['error']}")
                return jsonify({'error': results['error']}), 500
            
            return jsonify(results)
            
        except Exception as e:
            print(f"Exception during file analysis: {e}")
            return jsonify({'error': str(e)}), 500
            
    print("Invalid file type.")
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == "__main__":
    print("Server starting... Navigate to http://localhost:8000")
    app.config['DEBUG'] = True
    serve(app, host="0.0.0.0", port=8000, threads=4)