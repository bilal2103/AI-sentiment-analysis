import os
import shutil
from groq import Groq
from dotenv import load_dotenv
from pydub.silence import split_on_silence, detect_silence
from pydub import AudioSegment
from DiarizationService import Diarization

class GroqSTT:
    def __init__(self):
        load_dotenv()
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.diarization = Diarization()
    
    def find_closest_diarization_boundary(self, diarization_result, target_time_seconds):
        """
        Find the end time of the diarization segment that contains or is closest to the target time.
        This ensures we don't cut off mid-sentence by extending to the speaker's utterance end.
        """
        # First, check if target falls within an active diarization segment
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            if turn.start <= target_time_seconds <= turn.end:
                print(f"  → Target {target_time_seconds:.1f}s is within segment [{turn.start:.1f}s - {turn.end:.1f}s] ({speaker})")
                print(f"  → Extending to segment end: {turn.end:.1f}s")
                return turn.end * 1000  # Return end time in milliseconds
        
        # Target is in a gap - find the segment that ends closest BEFORE the target
        # or the segment that starts closest AFTER the target
        closest_segment_end = None
        closest_speaker = None
        min_diff_before = float('inf')
        
        next_segment_start = None
        next_speaker = None
        min_diff_after = float('inf')
        
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            # Check segments that end before target
            if turn.end <= target_time_seconds:
                diff = target_time_seconds - turn.end
                if diff < min_diff_before:
                    min_diff_before = diff
                    closest_segment_end = turn.end
                    closest_speaker = speaker
            
            # Check segments that start after target
            if turn.start > target_time_seconds:
                diff = turn.start - target_time_seconds
                if diff < min_diff_after:
                    min_diff_after = diff
                    next_segment_start = turn.start
                    next_speaker = speaker
        
        # If there's a segment starting soon after target (within 2 seconds), use that start as boundary
        # Otherwise, use the end of the previous segment
        if next_segment_start is not None and min_diff_after <= 2.0:
            print(f"  → Target {target_time_seconds:.1f}s is in gap, next segment starts at {next_segment_start:.1f}s ({next_speaker})")
            return next_segment_start * 1000
        elif closest_segment_end is not None:
            print(f"  → Target {target_time_seconds:.1f}s is in gap, previous segment ended at {closest_segment_end:.1f}s ({closest_speaker})")
            return closest_segment_end * 1000
        
        print(f"  → No diarization segments found, using target time: {target_time_seconds:.1f}s")
        return target_time_seconds * 1000  # Fallback to target time if no segments found
    
    def SplitAndTranscribe(self, audio_file, segment_duration):
        try:
            segmentThreshold = 30000    #segment_duration will >= segmentThreshold
            audio = AudioSegment.from_wav(audio_file)
            total_duration = len(audio)
            segments_dir = "segments"
            
            # Clean up old segment files to prevent stale data issues
            if os.path.exists(segments_dir):
                shutil.rmtree(segments_dir)
            os.makedirs(segments_dir, exist_ok=True)
            
            # First, diarize the entire audio to get speaker boundaries
            print("Diarizing audio to find optimal segment boundaries...")
            full_diarization = self.diarization.diarize(audio_file)
            
            segments = []
            segment_count = 0
            start_time = 0
            transcriptionSegmentCount = 0
            
            while start_time < total_duration:
                # Calculate expected end time for this segment
                expected_end_time = start_time + segment_duration
                
                # Check if remaining audio is less than threshold
                remaining_duration = total_duration - start_time
                print(f"\n[Segment Planning] Start: {start_time/1000:.1f}s | Expected end: {expected_end_time/1000:.1f}s | Remaining: {remaining_duration/1000:.1f}s")
                
                if remaining_duration <= segmentThreshold:
                    # Process remaining audio as final segment
                    end_time = total_duration
                    is_last_segment = True
                    print(f"  → Remaining duration ({remaining_duration/1000:.1f}s) <= threshold ({segmentThreshold/1000:.1f}s), processing as final segment")
                elif expected_end_time >= total_duration:
                    # This will be the last segment
                    end_time = total_duration
                    is_last_segment = True
                    print(f"  → Expected end exceeds total duration, processing as final segment")
                else:
                    # Find the closest diarization boundary to the expected end time
                    print(f"  → Searching for diarization boundary near {expected_end_time/1000:.1f}s...")
                    end_time = self.find_closest_diarization_boundary(
                        full_diarization, 
                        expected_end_time / 1000  # Convert to seconds for diarization
                    )
                    is_last_segment = False
                    
                    # Ensure we don't go backwards or create too small segments
                    if end_time <= start_time + segmentThreshold / 2:
                        print(f"  → Boundary {end_time/1000:.1f}s too close to start, using expected end time instead")
                        end_time = expected_end_time
                    # Ensure we don't exceed total duration
                    if end_time > total_duration:
                        end_time = total_duration
                        is_last_segment = True
                        print(f"  → Boundary exceeds total duration, capping at {total_duration/1000:.1f}s")
                
                segment_count += 1
                segment = audio[start_time:end_time]
                segment_file = f"{segments_dir}/segment_{segment_count:03d}.wav"
                segment.export(segment_file, format="wav")
                print(f"✓ Created segment {segment_count}: {segment_file} | Duration: {len(segment)/1000:.1f}s | Range: [{start_time/1000:.1f}s - {end_time/1000:.1f}s]")
                
                print(f"  📝 Transcribing segment {segment_count}...")
                transcriptionSegments = self.transcribe_segment(segment_file)
                print(f"  ✓ Transcription complete: {len(transcriptionSegments)} utterances found")
                for seg in transcriptionSegments:
                    segments.append({
                        "id": transcriptionSegmentCount,
                        "start": seg["start"] + start_time/1000,
                        "end": seg["end"] + start_time/1000,
                        "text": seg["text"]
                    })
                    transcriptionSegmentCount += 1
                
                # Move to next segment starting at the diarization boundary
                start_time = end_time
                
                if is_last_segment:
                    break
                    
            print(f"\n✓ Transcription complete: {len(segments)} total utterances from {segment_count} audio segments")
            return segments, full_diarization
            
        except Exception as e:
            print(f"Error splitting audio: {e}")
            raise e

    def transcribe_segment(self, segment):
        try:
            with open(segment, "rb") as file:
                result = self.client.audio.translations.create(
                    file=(segment, file.read()),
                        prompt = "Please transcribe this call recording between a customer care representative of SEDER group, and a troubled customer.",
                        model="whisper-large-v3",
                        response_format="verbose_json",
                        temperature=0,
                    )
                
                transcriptionSegments = list(result.segments)
                filteredSegments = []
                discarded_count = 0
                
                # Log each segment with its metadata and filter noisy ones
                for i, seg in enumerate(transcriptionSegments):
                    # Extract values based on whether seg is dict or object
                    if isinstance(seg, dict):
                        avg_logprob = seg.get('avg_logprob', 0)
                        no_speech_prob = seg.get('no_speech_prob', 'N/A')
                        compression_ratio = seg.get('compression_ratio', 'N/A')
                        temperature = seg.get('temperature', 'N/A')
                        start = seg.get('start', 0)
                        end = seg.get('end', 0)
                        text = seg.get('text', '')
                    else:
                        avg_logprob = getattr(seg, 'avg_logprob', 0)
                        no_speech_prob = getattr(seg, 'no_speech_prob', 'N/A')
                        compression_ratio = getattr(seg, 'compression_ratio', 'N/A')
                        temperature = getattr(seg, 'temperature', 'N/A')
                        start = getattr(seg, 'start', 0)
                        end = getattr(seg, 'end', 0)
                        text = getattr(seg, 'text', '')
                    
                    # Format values for display
                    avg_logprob_str = f"{avg_logprob:.4f}" if isinstance(avg_logprob, (int, float)) else str(avg_logprob)
                    no_speech_prob_str = f"{no_speech_prob:.4f}" if isinstance(no_speech_prob, (int, float)) else str(no_speech_prob)
                    compression_ratio_str = f"{compression_ratio:.2f}" if isinstance(compression_ratio, (int, float)) else str(compression_ratio)
                    
                    # Check if segment is noisy (avg_logprob < -0.9)
                    is_noisy = isinstance(avg_logprob, (int, float)) and avg_logprob < -0.9
                    
                    if is_noisy:
                        discarded_count += 1
                        print(f"    [{i+1}] ⚠️ DISCARDED (noisy) [{start:.1f}s - {end:.1f}s] \"{text}\"")
                        print(f"        avg_logprob: {avg_logprob_str} | no_speech_prob: {no_speech_prob_str} | compression_ratio: {compression_ratio_str} | temp: {temperature}")
                    else:
                        filteredSegments.append(seg)
                        print(f"    [{i+1}] [{start:.1f}s - {end:.1f}s] \"{text}\"")
                        print(f"        avg_logprob: {avg_logprob_str} | no_speech_prob: {no_speech_prob_str} | compression_ratio: {compression_ratio_str} | temp: {temperature}")
                
                if discarded_count > 0:
                    print(f"    ⚠️ Discarded {discarded_count} noisy segment(s) with avg_logprob < -0.9")
                
                return filteredSegments
        except Exception as e:
            print(f"Error transcribing segment: {e}")
            raise e
    
    def transcribe(self, audio_path: str, splitDuration):
        """
        Returns: (transcription_segments, diarization_result)
        """
        try:
            segments, diarization_result = self.SplitAndTranscribe(audio_path, splitDuration)
            return segments, diarization_result
        except Exception as e:
            print(f"Error transcribing with Groq: {e}")
            raise e