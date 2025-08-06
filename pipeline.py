import os
import shutil
from DiarizationService import Diarization
from STTService import GroqSTT
import json
from LLMService import LLMService
from pydub import AudioSegment
import re

diarization = Diarization()
stt = GroqSTT()
llm = LLMService()

def extract_json_from_response(response):
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass
    
    json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(json_pattern, response, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    json_like_pattern = r'\{.*\}'
    matches = re.findall(json_like_pattern, response, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    return None

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
    os.makedirs("cleanedFiles", exist_ok=True)
    
    cleanedAudioPath = f"cleanedFiles/{audio_path.split('/')[-1]}"
    audio = AudioSegment.from_wav(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(cleanedAudioPath, format="wav")
    transcriptionResult = stt.transcribe(cleanedAudioPath, task="translate")
    transcriptionSegments = list(transcriptionResult.segments)
    diarization_result = diarization.diarize(cleanedAudioPath)
    diarizationSegments = [Segment(segment) for segment in list(diarization_result.itertracks(yield_label=True))]
    speakerCounts = {}
    for segment in diarizationSegments:
        if segment["speaker"] not in speakerCounts:
            speakerCounts[segment["speaker"]] = 1
        else:
            speakerCounts[segment["speaker"]] += 1
    if len(speakerCounts) > 2:
        print("More than 2 speakers detected by pyannote. Ensuring there are only 2 speakers...")
        print(f"Speaker counts: {speakerCounts}")
        
        # Sort speakers by count (descending) and keep only top 2
        sorted_speakers = sorted(speakerCounts.items(), key=lambda x: x[1], reverse=True)
        speakers_to_keep = [speaker for speaker, count in sorted_speakers[:2]]
        speakers_to_remove = [speaker for speaker, count in sorted_speakers[2:]]
        
        print(f"Keeping speakers: {speakers_to_keep}")
        print(f"Removing speakers: {speakers_to_remove}")
        
        # Remove segments from speakers we don't want to keep
        segments_removed = 0
        segments_to_remove = []
        
        for segment in diarizationSegments:
            if segment["speaker"] in speakers_to_remove:
                segments_to_remove.append(segment)
                segments_removed += 1
        
        # Remove segments (iterate backwards to avoid index issues)
        for segment in segments_to_remove:
            diarizationSegments.remove(segment)
            print(f"Removing segment: {segment}")
        
        print(f"Removed {segments_removed} segments from {len(speakers_to_remove)} speakers")
        print(f"Remaining segments: {len(diarizationSegments)}")

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
    models = ["llama-3.3-70b-versatile", "meta-llama/llama-guard-4-12b","llama-3.1-8b-instant","gemma2-9b-it"]
    for model in models:
        response = llm.SummarizeAndAnalyze(script, model)
        print(f"===================================\nSummary for model {model}:\n===================================")
        
        # Try to extract JSON from the response
        responseDict = extract_json_from_response(response)
        
        if responseDict:
            try:
                print(responseDict["summary"])
                print("===================================\nSentiment Analysis:\n===================================")
                print(f"Representative: {responseDict['sentimentAnalysis']['representative']}")
                print(f"Customer: {responseDict['sentimentAnalysis']['customer']}")
                print("===================================\n===================================")
            except KeyError as e:
                print(f"JSON structure is missing expected key: {e}")
                print("Available keys:", list(responseDict.keys()))
                print("Raw JSON content:")
                print(json.dumps(responseDict, indent=2))
        else:
            print(f"Could not extract valid JSON from response. Printing raw response:")
            print("="*50)
            print(response)
            print("="*50)
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
    
    
    # Clean up temporary files
    try:
        shutil.rmtree("cleanedFiles")
        print("Temporary files cleaned up successfully")
    except Exception as e:
        print(f"Error cleaning up temporary files: {e}")
    
        
    
