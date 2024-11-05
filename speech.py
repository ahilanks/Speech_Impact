from dotenv import load_dotenv
import requests
import csv
import os
import numpy as np
import pandas as pd
import librosa
#import tensorflow as tf
import tensorflow_hub as hub
from pydub import AudioSegment
import sys
from openai import OpenAI
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("API_KEY"))

def detect_clapping_cheering(y, sr):
    print("Loading YAMNet model from local directory...")
    try:
        # Specify the path to your local model directory
        model_path = '/Users/ahilankaruppusami/Downloads/yamnet-tensorflow2-yamnet-v1'
        model = hub.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)

    # Load class map from the local CSV file
    # Assuming the CSV is in the 'assets' directory of the model
    class_map_path = os.path.join(model_path, 'assets', 'yamnet_class_map.csv')
    if not os.path.exists(class_map_path):
        print(f"Class map CSV not found at {class_map_path}")
        sys.exit(1)
    class_map_df = pd.read_csv(class_map_path)

    # Ensure the audio is mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Resample to 16 kHz (required by YAMNet)
    if sr != 16000:
        print("Resampling audio to 16 kHz...")
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Run the model
    print("Running YAMNet model...")
    scores, embeddings, spectrogram = model(y)
    scores = scores.numpy()

    class_names = ['Applause', 'Cheering', 'Clapping', 'Crowd', "Chatter"]
    class_indices = []
    for name in class_names:
        idx = class_map_df[class_map_df['display_name'] == name]['index'].values
        if idx.size > 0:
            class_indices.append(idx[0])
        else:
            print(f"Warning: Class '{name}' not found in class map.")

    weights = {
        'clapping': 0.3,
        'cheering': 0.1,
        'clapping2': 0.2,
        'crowd': 0.1,
        "chatter": 0.1
    }

    class_scores = [scores[:, idx] for idx in class_indices]

    # Combine scores
    combined_scores = (weights['clapping'] * class_scores[0] +
                       weights['cheering'] * class_scores[1] +
                       weights['clapping2'] * class_scores[2] +
                       weights['crowd'] * class_scores[3] +
                       weights["chatter"] * class_scores[4])

    # Define a threshold
    threshold = np.mean(combined_scores) + (0.5 * np.std(combined_scores))

    # Detect where the combined score exceeds the threshold
    detected_frames = combined_scores > threshold

    # Calculate the timestamps for each frame
    frame_duration = 0.48  # Adjust if necessary
    times = np.arange(len(detected_frames)) * frame_duration

    # Extract intervals
    intervals = get_intervals(detected_frames, times)

    return intervals, y, sr

def convert_mp3_to_wav(mp3_filename, wav_filename):
    print("Converting MP3 to WAV...")
    audio = AudioSegment.from_mp3(mp3_filename)
    audio.export(wav_filename, format="wav")
    print("Conversion complete.")

def load_audio(wav_filename):
    print("Loading audio file...")
    y, sr = librosa.load(wav_filename, sr=None)
    print(f"Audio loaded. Sample rate: {sr} Hz")
    return y, sr

def get_intervals(detected_frames, times):
    print("Extracting intervals...")
    intervals = []
    in_interval = False
    frame_duration = times[1] - times[0] if len(times) > 1 else 0.48
    for i, detected in enumerate(detected_frames):
        if detected and not in_interval:
            start_time = times[i]
            in_interval = True
        elif not detected and in_interval:
            end_time = times[i]
            intervals.append((start_time, end_time))
            in_interval = False
    if in_interval:
        intervals.append((start_time, times[len(detected_frames)-1] + frame_duration))
    print(f"Found {len(intervals)} intervals.")
    return intervals

def calculate_intensities(intervals, y, sr):
    print("Calculating intensities...")
    intensities = []
    for start_time, end_time in intervals:
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        interval_audio = y[start_sample:end_sample]
        intensity = np.mean(np.abs(interval_audio))
        intensities.append(intensity)
    return intensities

def format_time(seconds):
    milliseconds = int((seconds - int(seconds)) * 1000)
    total_seconds = int(seconds)
    minutes, sec = divmod(total_seconds, 60)
    hours, min = divmod(minutes, 60)
    return f"{hours:02d}:{min:02d}:{sec:02d}:{milliseconds:03d}"

def generate_csv(intervals, intensities, csv_filename):
    print(f"Generating CSV file: {csv_filename}...")
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['start_time', 'end_time', 'intensity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for (start_time, end_time), intensity in zip(intervals, intensities):
            writer.writerow({
                'start_time': format_time(start_time),
                'end_time': format_time(end_time),
                'intensity': intensity
            })
    print("CSV file generated.")

def split_audio(file_path, chunk_length_ms):
    """Split audio into chunks of specified length."""
    audio = AudioSegment.from_wav(file_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunks.append(chunk)
    return chunks

def transcribe_chunk(chunk, chunk_index):
    """Transcribe a single chunk of audio."""
    chunk_filename = f"chunk_{chunk_index}.wav"
    chunk.export(chunk_filename, format="wav")
    
    with open(chunk_filename, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            response_format="verbose_json"
        )
    os.remove(chunk_filename)  # Clean up the temporary file
    return transcription

def transcribe_large_audio(file_path, chunk_length_ms=60000):
    """Transcribe a large audio file by splitting it into smaller chunks."""
    # Step 1: Split the audio file
    chunks = split_audio(file_path, chunk_length_ms)

    # Step 2: Transcribe each chunk and combine results
    full_transcript = {'segments': []}
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i+1}/{len(chunks)}...")
        chunk_transcription = transcribe_chunk(chunk, i)
        
        # Offset each segment by the chunk start time
        chunk_start_time = (i * chunk_length_ms) / 1000  # Convert ms to seconds
        for segment in chunk_transcription.segments:
            segment.start += chunk_start_time
            segment.end += chunk_start_time
            full_transcript['segments'].append(segment)
    
    return full_transcript

def get_reactive_lines(transcript, intervals):
    """Overlay intervals of reactions onto the transcript to get corresponding lines."""
    reactive_lines = []

    for segment in transcript['segments']:
        segment_start = segment.start
        segment_end = segment.end
        text = segment.text

        # Check if segment overlaps with any reaction interval
        for (start, end) in intervals:
            if (segment_start <= end) and (segment_end >= start):
                reactive_lines.append((format_time(segment_start), format_time(segment_end), text))
                break

    return reactive_lines

def plot_intensities_over_time(intervals, intensities, speech_name):

    """Plot the intensities of crowd reactions over the time of the speech with a smooth, curved line and data points."""
    # Create time points at each interval midpoint for smoother transitions
    mid_times = [(start + end) / 2 for start, end in intervals]
    
    # Convert lists to numpy arrays for spline interpolation
    
    times = np.array(mid_times)
    intensity_values = np.array(intensities)

    # Generate smooth time points for the spline curve
    smooth_times = np.linspace(times.min(), times.max(), 500)
    spline = make_interp_spline(times, intensity_values, k=3)  # k=3 for cubic spline
    smooth_intensities = spline(smooth_times)

    # Plotting the smooth intensity graph with data points
    plt.figure(figsize=(12, 6))
    plt.plot(smooth_times, smooth_intensities, linestyle='-', linewidth=2, label="Intensity")
    plt.scatter(times, intensity_values, color='red', marker='o', label="Data Points")  # Plot actual data points
    plt.title(f"Intensity of Crowd Reactions Over Time: {speech_name} ")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Intensity")
    plt.grid(True)
    plt.legend()
    plt.show()

    return times, intensity_values


def main(filename):
    file = filename
    wav_filename = "speech.wav"

    # Step 1: Convert MP3 to WAV for compatibility
    convert_mp3_to_wav(file, wav_filename)

    chunk_length_ms = 100000  # 4 minutes per chunk (~20MB for a WAV file)

    # Transcribe large audio
    transcript = transcribe_large_audio(wav_filename, chunk_length_ms)

    # Print combined transcript with timestamps
    print("Combined Transcription with Timestamps:")
    for segment in transcript['segments']:
        print(f"[{segment.start:.2f}s - {segment.end:.2f}s]: {segment.text}")

    # Step 3: Load the audio, detect clapping/cheering intervals, and calculate intensities
    y, sr = load_audio(wav_filename)
    intervals, _, _ = detect_clapping_cheering(y, sr)
    intensities = calculate_intensities(intervals, y, sr)

    intensities = [intensity * 100 for intensity in intensities]

    # Step 4: Overlay intervals onto transcript
    reactive_lines = get_reactive_lines(transcript, intervals)

    # Output results with intensity
    print("Lines that elicited crowd reactions with intensity:")
    for ((start, end, line), intensity) in zip(reactive_lines, intensities):
        print(f"[{start} - {end}]: {line}; Intensity: {intensity:.2f}")

    plot_intensities_over_time(intervals, intensities, re.search(r"/([^/]+)\.mp3$", file).group(1))

if __name__ == "__main__":
    mp3_filename_s = '/Users/ahilankaruppusami/Downloads/barackobama2004dncARXE-[AudioTrimmer.com].mp3'
    mp3_filename_l = "/Users/ahilankaruppusami/Downloads/barackobama2004dncARXE.mp3"
    mp3_filename_n = "/Users/ahilankaruppusami/Coding_Projects/Measuring Speech Impact/ObamaSpeeches/Address to the Illinois General Assembly.mp3"
    main(mp3_filename_s)