import subprocess
import os
from pathlib import Path
from tqdm import tqdm
from diarization import Diarization
from stt import STT


diarization = Diarization()
stt = STT()

def slice_audio_segment(audio_path: str, start_time: float, end_time: float, output_path: str):
    """
    Slice audio segment using ffmpeg
    """
    cmd = [
        'ffmpeg', '-y',  # -y to overwrite existing files
        '-i', audio_path,
        '-ss', str(start_time),  # start time
        '-to', str(end_time),    # end time
        '-c', 'copy',            # copy codec for faster processing
        '-loglevel', 'quiet',    # suppress ffmpeg output for cleaner progress
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error slicing audio: {e}")
        return False

def RunPipeline(audio_path: str, show_progress: bool = True):
    """
    Main pipeline function to diarize audio, slice segments, and transcribe each segment
    """
    print(f"🎵 Processing audio: {audio_path}")
    
    # Step 1: Diarize the audio
    print("\n📊 Step 1: Performing speaker diarization...")
    diarization_result = diarization.diarize(audio_path, show_progress=show_progress)
    
    # Count total segments first for progress tracking
    segments = list(diarization_result.itertracks(yield_label=True))
    total_segments = len(segments)
    print(f"Found {total_segments} speaker segments")
    
    # Step 2: Create output directory for segments
    audio_name = Path(audio_path).stem
    segments_dir = f"{audio_name}_segments"
    os.makedirs(segments_dir, exist_ok=True)
    
    # Step 3: Process segments with progress bar
    results = []
    
    print(f"\n🔄 Step 2: Processing {total_segments} segments...")
    
    # Create progress bar for segment processing
    if show_progress:
        progress_bar = tqdm(segments, desc="Processing segments", unit="segment")
    else:
        progress_bar = segments
    
    for i, (turn, _, speaker) in enumerate(progress_bar):
        start_time = turn.start
        end_time = turn.end
        duration = end_time - start_time
        
        # Update progress description
        if show_progress:
            progress_bar.set_description(f"Processing segment {i+1}/{total_segments} (Speaker {speaker})")
        else:
            print(f"Processing segment {i+1}/{total_segments}: {start_time:.1f}s - {end_time:.1f}s (Speaker {speaker}, Duration: {duration:.1f}s)")
        
        # Create output path for this segment
        segment_filename = f"segment_{i+1:03d}_{speaker}_{start_time:.1f}s-{end_time:.1f}s.wav"
        segment_path = os.path.join(segments_dir, segment_filename)
        
        # Slice the audio segment using ffmpeg
        if slice_audio_segment(audio_path, start_time, end_time, segment_path):
            # Transcribe the segment
            try:
                # Show transcription progress for current segment
                if not show_progress:
                    print(f"  🔤 Transcribing segment {i+1}...")
                
                transcription = stt.transcribe(segment_path)
                
                segment_result = {
                    'segment_id': i + 1,
                    'speaker': speaker,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'audio_file': segment_path,
                    'transcription': transcription['text'],
                    'language': transcription.get('language', 'unknown')
                }
                
                results.append(segment_result)
                
                if not show_progress:
                    print(f"  ✓ Transcribed: {transcription['text'][:100]}...")
                
            except Exception as e:
                error_msg = f"✗ Error transcribing segment {i+1}: {e}"
                if show_progress:
                    tqdm.write(error_msg)
                else:
                    print(f"  {error_msg}")
        else:
            error_msg = f"✗ Error slicing segment {i+1}"
            if show_progress:
                tqdm.write(error_msg)
            else:
                print(f"  {error_msg}")
    
    if show_progress:
        progress_bar.close()
    
    print(f"\n🎉 Completed processing {len(results)}/{total_segments} segments successfully!")
    print(f"📁 Segments saved in: {segments_dir}/")
    
    return results


if __name__ == "__main__":
    # Example usage
    audio_file = "arabicSample.wav"  # Change this to your audio file
    
    if os.path.exists(audio_file):
        print("="*60)
        print("🎬 AUDIO PROCESSING PIPELINE")
        print("="*60)
        
        results = RunPipeline(audio_file, show_progress=True)
        
        # Print summary
        print("\n" + "="*60)
        print("📋 PIPELINE RESULTS SUMMARY")
        print("="*60)
        
        for result in results:
            print(f"\n🎤 Segment {result['segment_id']} (Speaker {result['speaker']}):")
            print(f"  ⏰ Time: {result['start_time']:.1f}s - {result['end_time']:.1f}s")
            print(f"  ⏱️  Duration: {result['duration']:.1f}s")
            print(f"  🌍 Language: {result['language']}")
            print(f"  💬 Text: {result['transcription']}")
    else:
        print(f"❌ Audio file '{audio_file}' not found. Please check the path.")
    