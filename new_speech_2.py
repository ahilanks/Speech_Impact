from dotenv import load_dotenv
import os
from openai import OpenAI
from pydub import AudioSegment
import sys
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import csv

file_path = '/Users/ahilankaruppusami/Downloads/khanna town hall.mp3'

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
    frame_duration = 0.25  # More granular timing
    times = np.arange(len(detected_frames)) * frame_duration

    # Extract intervals
    intervals = get_intervals(detected_frames, times)

    return intervals, y, sr

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
    
    # Consolidate overlapping intervals
    if intervals:
        print(f"Found {len(intervals)} raw intervals, consolidating overlaps...")
        # Sort intervals by start time
        intervals.sort(key=lambda x: x[0])
        
        consolidated = []
        current = intervals[0]
        
        for next_interval in intervals[1:]:
            # If current and next interval overlap or are very close (within 2 seconds)
            if next_interval[0] <= current[1] + 2:
                # Merge them by taking the earlier start and later end
                current = (current[0], max(current[1], next_interval[1]))
            else:
                # No overlap, so add current to results and move to next
                consolidated.append(current)
                current = next_interval
        
        # Don't forget to add the last interval
        consolidated.append(current)
        
        print(f"Consolidated to {len(consolidated)} intervals.")
        return consolidated
    
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
        intensities.append(intensity * 100)  # Scale for readability
    return intensities

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

def extract_pre_clapping_chunks(audio_file, intervals, chunk_duration=20):
    """
    Extract chunks of audio before each clapping interval
    
    Args:
        audio_file: Path to the audio file (wav format)
        intervals: List of (start_time, end_time) tuples for clapping intervals
        chunk_duration: Duration in seconds of the audio chunks to extract before each interval
        
    Returns:
        List of (chunk, start_time, end_time) tuples
    """
    print(f"Extracting {chunk_duration}-second chunks before {len(intervals)} clapping intervals...")
    
    audio = AudioSegment.from_file(audio_file, format="wav")
    chunks = []
    
    for i, (start_time, end_time) in enumerate(intervals):
        # Calculate the start time of the chunk (ensuring it doesn't go below 0)
        chunk_start_time = max(0, start_time - chunk_duration)
        chunk_end_time = start_time
        
        # Convert to milliseconds for pydub
        chunk_start_ms = int(chunk_start_time * 1000)
        chunk_end_ms = int(chunk_end_time * 1000)
        
        # Extract the chunk
        chunk = audio[chunk_start_ms:chunk_end_ms]
        
        chunks.append((chunk, chunk_start_time, chunk_end_time))
        print(f"Extracted chunk {i+1}: {chunk_start_time:.2f}s - {chunk_end_time:.2f}s")
    
    return chunks

def transcribe_chunk(chunk, start_time, end_time):
    """Transcribe a single chunk of audio."""
    chunk_filename = f"chunk_{start_time:.2f}_{end_time:.2f}.wav"
    chunk.export(chunk_filename, format="wav")
    
    print(f"Transcribing chunk {chunk_filename}...")
    with open(chunk_filename, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-transcribe", 
            file=audio_file,
            response_format="json",
            timestamp_granularities=["word"]
        )
    
    os.remove(chunk_filename)  # Clean up the temporary file
    return transcription

def main():
    # Define the output WAV filename
    wav_filename = "speech.wav"

    # Step 1: Convert MP3 to WAV for compatibility
    convert_mp3_to_wav(file_path, wav_filename)

    # Step 2: Load the audio and detect clapping/cheering intervals
    y, sr = load_audio(wav_filename)
    intervals, processed_y, processed_sr = detect_clapping_cheering(y, sr)
    
    # Step 3: Calculate intensities for each interval
    intensities = calculate_intensities(intervals, processed_y, processed_sr)
    
    # Step 3.5: Save clapping intervals and intensities to a separate CSV
    with open('clapping_intervals.csv', 'w', newline='') as csvfile:
        fieldnames = ['start_time', 'end_time', 'intensity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for (start_time, end_time), intensity in zip(intervals, intensities):
            writer.writerow({
                'start_time': start_time,
                'end_time': end_time,
                'intensity': intensity
            })
    print(f"Clapping intervals saved to clapping_intervals.csv")
    
    # Step 4: Extract 20-second chunks before each clapping interval
    chunks = extract_pre_clapping_chunks(wav_filename, intervals, chunk_duration=20)
    
    # Step 5: Transcribe each chunk
    results = []
    for i, ((chunk, start_time, end_time), intensity) in enumerate(zip(chunks, intensities)):
        transcription = transcribe_chunk(chunk, start_time, end_time)
        print(f"\nChunk {i+1} ({start_time:.2f}s - {end_time:.2f}s):")
        print(f"Transcription: {transcription}")
        print(f"Reaction intensity: {intensity:.2f}")
        
        results.append({
            'start_time': start_time,
            'end_time': end_time,
            'transcription': transcription,
            'intensity': intensity
        })
    
    # Step 6: Save results to a CSV file
    with open('pre_clapping_transcriptions.csv', 'w', newline='') as csvfile:
        fieldnames = ['start_time', 'end_time', 'transcription', 'intensity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\nTranscriptions saved to pre_clapping_transcriptions.csv")

if __name__ == "__main__":
    main()



