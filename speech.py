from dotenv import load_dotenv
import requests
import csv
import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from pydub import AudioSegment
import sys
from openai import OpenAI
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import re
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')  # Download the punkt tokenizer data
nltk.download('punkt_tab') 

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
            model="gpt-4o-transcribe", 
            file=audio_file,
            response_format="json"
        )
    os.remove(chunk_filename)  # Clean up the temporary file
    return transcription.text

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
        for segment in chunk_transcription['segments']:
            segment['start'] += chunk_start_time 
            segment['end'] += chunk_start_time
            full_transcript['segments'].append(segment)
    
    return full_transcript

def process_transcript_to_sentences(transcript):
    """Convert transcript segments into sentences with timing information."""
    # Combine all text first
    full_text = ' '.join(segment['text'] for segment in transcript['segments'])
    sentences = sent_tokenize(full_text)
    
    sentence_timings = []
    current_pos = 0
    
    for sentence in sentences:
        sentence_start = None
        sentence_end = None
        
        # Find the segments that contain this sentence
        for segment in transcript['segments']:
            segment_text = segment['text']
            if sentence in segment_text:
                # Found exact match
                sentence_start = segment['start']
                sentence_end = segment['end']
                break
            
            # Handle cases where sentence spans multiple segments
            segment_pos = segment_text.find(sentence[current_pos:])
            if segment_pos != -1:
                if sentence_start is None:
                    sentence_start = segment['start']
                sentence_end = segment['end']
                current_pos += len(segment_text)
        
        if sentence_start is not None and sentence_end is not None:
            sentence_timings.append({
                'text': sentence.strip(),
                'start': sentence_start,
                'end': sentence_end
            })
    
    return sentence_timings

def get_reactive_lines(transcript, intervals):
    """
    Overlay intervals of reactions onto the transcript sentences to get corresponding lines.
    Each returned element in reactive_lines is a tuple of:
       (start_time_str, end_time_str, text)

    Where 'text' includes up to 3 pre-context lines (skipping any with "thank you"),
    the current line with the clapping portion bolded, and 1 post-context line (again skipping any "thank you").
    """
    # Regex pattern for variations of "thank you"
    thank_you_pattern = re.compile(r'\b(thank you|thanks|thank you very much|thank you so much)\b', re.IGNORECASE)

    # Convert transcript into sentences (assuming you already have this helper)
    # Each sentence is a dict like: {'start': float, 'end': float, 'text': str}
    sentences = process_transcript_to_sentences(transcript)

    reactive_lines = []  # will hold tuples: (start_time_str, end_time_str, text_block)

    # Keep track of processed sentence indices to avoid duplicates
    processed_indices = set()

    def bold_overlap(text, sent_start, sent_end, clap_start, clap_end):
        """
        Bold the portion of `text` that overlaps [clap_start, clap_end].
        Uses fraction-of-duration approach to map times to word indices.
        """
        words = text.split()
        duration = sent_end - sent_start
        if duration <= 0:
            return text  # avoid division by zero or negative durations

        # Helper to map a time to a word index
        def time_to_word_index(t):
            fraction = (t - sent_start) / duration
            fraction = max(0.0, min(fraction, 1.0))  # clamp to [0,1]
            return int(len(words) * fraction)

        start_idx = time_to_word_index(clap_start)
        end_idx   = time_to_word_index(clap_end)
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        for i in range(start_idx, min(end_idx, len(words))):
            words[i] = f"**{words[i]}**"

        return " ".join(words)

    # Check each sentence to see if it overlaps with any clapping interval
    for i, sentence in enumerate(sentences):
        sent_start = sentence['start']
        sent_end   = sentence['end']
        sent_text  = sentence['text']

        # Skip if the sentence contains "thank you"
        if thank_you_pattern.search(sent_text):
            continue

        # Check for overlap with any of the given intervals
        for (clap_start, clap_end) in intervals:
            # Overlap occurs if they are not disjoint
            if not (sent_end < clap_start or sent_start > clap_end):
                # If we haven't processed this sentence yet, build the context
                if i not in processed_indices:
                    processed_indices.add(i)

                    # -------------------
                    # 1) Collect up to 2 previous lines (skipping those with "thank you")
                    # -------------------
                    pre_context = []
                    needed = 2
                    j = i - 1
                    while j >= 0 and needed > 0:
                        txt_j = sentences[j]['text']
                        if not thank_you_pattern.search(txt_j):
                            pre_context.append(txt_j)
                            needed -= 1
                        j -= 1
                    pre_context.reverse()  # because we collected backwards

                    # -------------------
                    # 2) Bold the overlap in this sentence
                    # -------------------
                    bolded_text = bold_overlap(
                        sent_text,
                        sent_start,
                        sent_end,
                        clap_start,
                        clap_end
                    )

                    # -------------------
                    # 3) Grab 1 post context line (skip if "thank you")
                    # -------------------
                    post_context = ""
                    if i + 1 < len(sentences):
                        next_text = sentences[i+1]['text']
                        if not thank_you_pattern.search(next_text):
                            post_context = next_text

                    # -------------------
                    # 4) Merge everything into one string
                    # -------------------
                    # Example format (you can change the parentheses/text):
                    #   "Pre-line1. Pre-line2. Pre-line3 (Pre-context). bolded_text (Clapping overlap). (Post-context) post_line"
                    final_text = ""

                    if pre_context:
                        final_text += " ".join(pre_context) + " "  # Join pre_context into a single string

                    # Current line
                    final_text += bolded_text + " "  # Add a space for separation

                    # Post context if present
                    if post_context:
                        final_text += post_context

                    # -------------------
                    # 5) Append to the results as a 3-tuple
                    # -------------------
                    reactive_lines.append((
                        format_time(sent_start),
                        format_time(sent_end),
                        final_text.strip()  # Strip to remove any trailing spaces
                    ))

                # Once we handle one interval overlap for sentence i,
                # we break so we don't add it again for a second overlap
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

    # Print combined transcript with timestamps by sentences
    print("Combined Transcription with Timestamps (by sentences):")
    sentences = process_transcript_to_sentences(transcript)
    for sentence in sentences:
        print(f"[{sentence['start']:.2f}s - {sentence['end']:.2f}s]: {sentence['text']}")

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
    mp3_filename_s = '/Users/ahilankaruppusami/Downloads/khanna town hall.mp3'
    mp3_filename_l = "/Users/ahilankaruppusami/Downloads/barackobama2004dncARXE.mp3"
    mp3_filename_n = "/Users/ahilankaruppusami/Coding_Projects/Measuring Speech Impact/ObamaSpeeches/Address to the Illinois General Assembly.mp3"
    main(mp3_filename_s)