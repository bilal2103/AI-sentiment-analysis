import os
import shutil
from DiarizationService import Diarization
from STTService import GroqSTT
import json
from LLMService import LLMService
from pydub import AudioSegment

diarization = Diarization()
stt = GroqSTT()
llm = LLMService()

def Segment(segment):
    turn, _, speaker = segment
    return {
        "start": turn.start,
        "stop": turn.end,
        "speaker": speaker
    }

def UseIOU(transcriptionSegments, segments):
    def ComputeIOU(whisper_segment, pyannote_segment):
        whisper_start, whisper_end = whisper_segment["start"], whisper_segment["end"]
        pyannote_start, pyannote_end = pyannote_segment["start"], pyannote_segment["stop"]

        intersection_start = max(whisper_start, pyannote_start)
        intersection_end = min(whisper_end, pyannote_end)
        intersection = max(0, intersection_end - intersection_start)

        union_start = min(whisper_start, pyannote_start)
        union_end = max(whisper_end, pyannote_end)
        union = union_end - union_start

        iou = intersection / union if union > 0 else 0
        return iou
    mapping = {}
    for transcriptionSegment in transcriptionSegments:
        bestIOU = -1
        bestSegment = None
        for diarizationSegment in segments:
            iou = ComputeIOU(transcriptionSegment, diarizationSegment)
            if iou > bestIOU:
                bestIOU = iou
                bestSegment = diarizationSegment
        mapping[transcriptionSegment["id"]] = bestSegment
    return mapping
def RunPipeline(audio_path: str):
    # Create cleanedFiles directory if it doesn't exist
    os.makedirs("cleanedFiles", exist_ok=True)
    
    cleanedAudioPath = f"cleanedFiles/{audio_path.split('/')[-1]}"
    audio = AudioSegment.from_wav(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(cleanedAudioPath, format="wav")
    transcriptionResult = stt.transcribe(cleanedAudioPath, task="translate")
    transcriptionSegments = list(transcriptionResult.segments)
    diarization_result = diarization.diarize(cleanedAudioPath)
    diarizationSegments = [Segment(segment) for segment in list(diarization_result.itertracks(yield_label=True))]
    mapping = UseIOU(transcriptionSegments, diarizationSegments)
    script = []
    for key, value in mapping.items():
        text = transcriptionSegments[key]["text"]
        startTime = transcriptionSegments[key]["start"]
        endTime = transcriptionSegments[key]["end"]
        speaker = value["speaker"]
        script.append({
            "text": text,
            "speaker": speaker,
            "start": startTime,
            "end": endTime
        })
    with open("script.json", "w", encoding="utf-8") as f:
        json.dump(script, f, indent=4)
    response = llm.SummarizeAndAnalyze(script)
    return response


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Please provide an audio file path as argument")
        print("Usage: python pipeline.py <audio_file>")
        sys.exit(1)
        
    audio_file = sys.argv[1]
    audio_file = audio_file
    response = RunPipeline(audio_file)
    try:
        responseDict = json.loads(response)
        print("===================================\nSummary:\n===================================")
        print(responseDict["summary"])
        print("===================================\nSentiment Analysis:\n===================================")
        print(f"Representative: {responseDict['sentimentAnalysis']['representative']}")
        print(f"Customer: {responseDict['sentimentAnalysis']['customer']}")
        print("===================================\n===================================")
    except Exception as e:
        print(f"Error parsing response: {e}")
    
    # Clean up temporary files
    try:
        shutil.rmtree("cleanedFiles")
        print("Temporary files cleaned up successfully")
    except Exception as e:
        print(f"Error cleaning up temporary files: {e}")
    
        
    