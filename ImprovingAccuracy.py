from groq import Groq
import os
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_silence

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def remove_silence(input_file, output_file, silence_thresh=-40, min_silence_len=1000, keep_silence=200):
    try:
        # Load audio file
        audio = AudioSegment.from_wav(input_file)
        
        # Split audio on silence
        chunks = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence
        )
        
        # Combine non-silent chunks
        if chunks:
            # Add small silence between chunks to prevent abrupt cuts
            combined = AudioSegment.empty()
            for i, chunk in enumerate(chunks):
                combined += chunk
                if i < len(chunks) - 1:  # Don't add silence after last chunk
                    combined += AudioSegment.silent(duration=100)  # 100ms silence between chunks
            
            # Export cleaned audio
            combined.export(output_file, format="wav")
            print(f"Silence removed. Output saved to: {output_file}")
            return True
        else:
            print("No audio chunks found after silence removal")
            return False
            
    except Exception as e:
        print(f"Error removing silence: {e}")
        return False

def split_audio_into_segments(audio_file, segment_duration=30000):
    """
    Split audio file into segments of specified duration (in milliseconds)
    
    Args:
        audio_file (str): Path to input audio file
        segment_duration (int): Duration of each segment in milliseconds (default: 30000 = 30 seconds)
    
    Returns:
        list: List of segment file paths
    """
    try:
        # Load audio file
        audio = AudioSegment.from_wav(audio_file)
        total_duration = len(audio)
        
        # Create segments directory
        segments_dir = "segments"
        os.makedirs(segments_dir, exist_ok=True)
        
        segments = []
        segment_count = 0
        
        # Split audio into segments
        for start_time in range(0, total_duration, segment_duration):
            end_time = min(start_time + segment_duration, total_duration)
            segment = audio[start_time:end_time]
            
            # Only save non-empty segments
            if len(segment) > 1000:  # At least 1 second of audio
                segment_count += 1
                segment_file = f"{segments_dir}/segment_{segment_count:03d}.wav"
                segment.export(segment_file, format="wav")
                segments.append(segment_file)
                print(f"Created segment {segment_count}: {segment_file} ({len(segment)/1000:.1f}s)")
        
        print(f"Split audio into {len(segments)} segments")
        return segments
        
    except Exception as e:
        print(f"Error splitting audio: {e}")
        return []

def transcribe_segments(segments, client):
    """
    Transcribe each audio segment and combine results
    
    Args:
        segments (list): List of segment file paths
        client: Groq client instance
    
    Returns:
        str: Combined transcription text
    """
    all_transcriptions = []
    
    for i, segment_file in enumerate(segments, 1):
        print(f"\nProcessing segment {i}/{len(segments)}: {segment_file}")
        
        try:
            with open(segment_file, "rb") as file:
                result = client.audio.translations.create(
                    file=(segment_file, file.read()),
                    model="whisper-large-v3",
                    response_format="verbose_json",
                    temperature=0.1,
                )
            
            transcription = result.text.strip()
            all_transcriptions.append(transcription)
            print(f"Segment {i} transcription: {transcription[:100]}...")
            
        except Exception as e:
            print(f"Error transcribing segment {i}: {e}")
            all_transcriptions.append(f"[Error transcribing segment {i}]")
    
    # Combine all transcriptions
    combined_text = " ".join(all_transcriptions)
    return combined_text
# Process audio file with 30-second segments
input_file = "aaa.wav"
cleaned_file = "cleanedFiles/aaa.wav"

# Create output directory if it doesn't exist
os.makedirs("cleanedFiles", exist_ok=True)

# Remove silence first
if remove_silence(input_file, cleaned_file):
    # Use cleaned file for segmentation
    file_to_process = cleaned_file
    print("Using cleaned audio file")
else:
    # Fallback to original file if cleaning fails
    file_to_process = input_file
    print("Using original audio file")

# Split audio into 30-second segments
print(f"\nSplitting {file_to_process} into 30-second segments...")
segments = split_audio_into_segments(file_to_process, segment_duration=30000)

if segments:
    # Transcribe each segment
    print(f"\nTranscribing {len(segments)} segments...")
    combined_transcription = transcribe_segments(segments, client)
    
    print("\n" + "="*50)
    print("FINAL TRANSCRIPTION RESULT:")
    print("="*50)
    print(combined_transcription)
    
    # Clean up segment files (optional)
    import shutil
    try:
        shutil.rmtree("segments")
        print("\nCleaned up temporary segment files")
    except:
        pass
else:
    print("No segments created. Check your audio file.")