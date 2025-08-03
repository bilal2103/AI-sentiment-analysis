import os
import shutil
from DiarizationService import Diarization
from STTService import STT, GroqSTT
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
def obtainSlice(filePath, startTime, endTime, outputFilePath):
    if endTime - startTime < 0.5:
        return False
    audio = AudioSegment.from_wav(filePath)
    audio = audio[(startTime - 0.5)*1000:(endTime + 0.5)*1000]
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(outputFilePath, format="wav")
    return True

def merge_segments(segments):
    # merge adjacent segments if they have the same speaker and are close enough
    merged = []
    for segment in segments:
        if not merged:
            merged.append(segment)
            continue
        if merged[-1]["speaker"] == segment["speaker"]:
                merged[-1]["stop"] = segment["stop"]
                continue
        merged.append(segment)
    return merged
def RunPipeline(audio_path: str):
    diarization_result = diarization.diarize(audio_path)
    segments = list(diarization_result.itertracks(yield_label=True))
    segments = [Segment(segment) for segment in segments]
    
    segments = merge_segments(segments)
    nonOverlappingSegments = []
    for segment in segments:
        if len(nonOverlappingSegments) == 0:
            nonOverlappingSegments.append(segment)
            continue
        if segment['speaker'] == nonOverlappingSegments[-1]['speaker']:
            print("HOW CAN THIS BE?")
            exit()
        if segment['start'] < nonOverlappingSegments[-1]['stop']:
            boundary = nonOverlappingSegments[-1]['stop']
            nonOverlappingSegments[-1]['stop'] = segment['start'] - 0.5
            segment['start'] = boundary + 0.5
        nonOverlappingSegments.append(segment)
    
    i = 0
    script = []
    totalSegments = len(nonOverlappingSegments)
    os.makedirs("tempFiles", exist_ok=True)

    print(f"Total segments: {totalSegments}")
    i = 0
    for segment in nonOverlappingSegments:
        outputFilePath = f"tempFiles/{segment['speaker']}_{i}.wav"
        result = obtainSlice(audio_path, segment['start'], segment['stop'], outputFilePath)
        if not result:
            print(f"Excluding {outputFilePath}")
            i += 1
            continue
        print(f"Translating {outputFilePath}")
        transcription = stt.transcribe(outputFilePath, task="translate")
        if len(transcription) > 0:
            dialogue = {
                "text": transcription,
                "speaker": segment['speaker'],
            }
            script.append(dialogue)
        i += 1
    with open("script_cleaned.json", "w", encoding="utf-8") as f:
        json.dump(script, f, indent=4)
    print("Script saved to script_clean.json")
    return script


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Please provide an audio file path as argument")
        print("Usage: python pipeline.py <audio_file>")
        sys.exit(1)
        
    audio_file = sys.argv[1]
    audio_file = "SampleAudios/" + audio_file
    script = RunPipeline(audio_file)
    audio = AudioSegment.from_wav(audio_file)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export("tempFiles/complete.wav", format="wav")
    completeScript = stt.transcribe("tempFiles/complete.wav", task="translate")
    print("Complete script: ", completeScript)
    response = llm.SummarizeAndAnalyze(script)
    print("Response raw: ", response)
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
        shutil.rmtree("tempFiles")
        print("Temporary files cleaned up successfully")
    except Exception as e:
        print(f"Error cleaning up temporary files: {e}")
    
        
    