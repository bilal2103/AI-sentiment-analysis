import os
from groq import Groq
from dotenv import load_dotenv
from pydub.silence import split_on_silence, detect_silence
from pydub import AudioSegment

class GroqSTT:
    def __init__(self):
        load_dotenv()
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    def SplitAndTranscribe(self, audio_file, segment_duration):
        try:
            segmentThreshold = 30000    #segment_duration will >= segmentThreshold
            buffer = 500
            audio = AudioSegment.from_wav(audio_file)
            total_duration = len(audio)
            segments_dir = "segments"
            os.makedirs(segments_dir, exist_ok=True)
            
            segments = []
            segment_count = 0
            totalSegments = total_duration // segment_duration + (1 if total_duration % segment_duration > 0 else 0)
            lastSegmentDuration = total_duration % segment_duration
            start_time = 0
            transcriptionSegmentCount = 0
            while segment_count < totalSegments:
                end_time = min(start_time + segment_duration, total_duration)
                if segment_count == totalSegments-2 and lastSegmentDuration > 0 and lastSegmentDuration < segmentThreshold:
                    segment = audio[start_time: total_duration]
                    segment_count = totalSegments
                else:
                    segment_count += 1
                    segment = audio[start_time:end_time]
                segment_file = f"{segments_dir}/segment_{segment_count:03d}.wav"
                segment.export(segment_file, format="wav")
                print(f"Created segment {segment_count}: {segment_file} ({len(segment)/1000:.1f}s). Start time: {start_time/1000:.1f}s, End time: {end_time/1000:.1f}s")
                transcriptionSegments = self.transcribe_segment(segment_file)
                for segment in transcriptionSegments[:-1] if segment_count != totalSegments else transcriptionSegments:
                    segments.append({
                        "id": transcriptionSegmentCount,
                        "start": segment["start"] + start_time/1000,
                        "end": segment["end"] + start_time/1000,
                        "text": segment["text"]
                    })
                    transcriptionSegmentCount += 1
                start_time += (transcriptionSegments[-1]["start"] * 1000 - buffer)
            print(f"Split audio into {len(segments)} segments")
            return segments
            
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
                        temperature=0.1,
                    )
                
                transcriptionSegments = list(result.segments)
                print(transcriptionSegments)
                return transcriptionSegments
        except Exception as e:
            print(f"Error transcribing segment: {e}")
            raise e
    def transcribe(self, audio_path: str, splitDuration):
        try:
            segments = self.SplitAndTranscribe(audio_path, splitDuration)
            return segments
        except Exception as e:
            print(f"Error transcribing with Groq: {e}")
            raise e