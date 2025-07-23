import os
import shutil
from DiarizationService import Diarization
from STTService import STT
from pydub import AudioSegment
import json
from LLMService import LLMService

diarization = Diarization()
stt = STT()
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
    audio = audio[startTime*1000:endTime*1000]
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
            print("Found over laping segments: ", end = "")
            print(nonOverlappingSegments[-1], end = ", ")
            print(segment)
            boundary = nonOverlappingSegments[-1]['stop']
            nonOverlappingSegments[-1]['stop'] = segment['start'] - 0.1
            segment['start'] = boundary + 0.1
        nonOverlappingSegments.append(segment)
    
    i = 0
    script = []
    totalSegments = len(nonOverlappingSegments)
    os.makedirs("tempFiles", exist_ok=True)

    print(f"Total segments: {totalSegments}")
    i = 0
    debugging = []
    for segment in nonOverlappingSegments:
        outputFilePath = f"tempFiles/{segment['speaker']}_{i}.wav"
        result = obtainSlice(audio_path, segment['start'], segment['stop'], outputFilePath)
        if not result:
            print(f"Excluding {outputFilePath}")
            i += 1
            continue
        print(f"Transcribing {outputFilePath}")
        transcriptionSegments, info = stt.transcribe(outputFilePath)
        transcriptionSegments = list(transcriptionSegments)
        if len(transcriptionSegments) > 0:
            # Convert Segment objects to dictionaries for JSON serialization
            segments_dict = []
            dialogue = {
                "text": "",
                "speaker": segment['speaker'],
            }
            for transcriptionSegment in transcriptionSegments:
                dialogue['text'] += transcriptionSegment.text
                segments_dict.append({
                    "id": transcriptionSegment.id,
                    "seek": transcriptionSegment.seek,
                    "start": transcriptionSegment.start,
                    "end": transcriptionSegment.end,
                    "text": transcriptionSegment.text,
                    "tokens": transcriptionSegment.tokens,
                    "temperature": transcriptionSegment.temperature,
                    "avg_logprob": transcriptionSegment.avg_logprob,
                    "compression_ratio": transcriptionSegment.compression_ratio,
                    "no_speech_prob": transcriptionSegment.no_speech_prob
                })
            
            # Convert TranscriptionInfo to dictionary
            info_dict = {
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "duration_after_vad": info.duration_after_vad,
                "all_language_probs": info.all_language_probs
            }
            debugging.append({
                "segments": segments_dict,
                "info": info_dict
            })
            script.append(dialogue)
                
        i += 1
    with open("debugging.json", "w") as f:
        json.dump(debugging, f)
    with open("script.json", "w") as f:
        json.dump(script, f)
    print("Script saved to script.json")
    return script


if __name__ == "__main__":
    audio_file = "SampleAudios/recording4.wav"
    script = RunPipeline(audio_file)
    response = llm.SummarizeAndAnalyze(script)
    try:
        responseDict = json.loads(response)
        print(responseDict["summary"])
        print(responseDict["sentimentAnalysis"])
    except Exception as e:
        print(f"Error parsing response: {e}")
    
    # Clean up temporary files
    try:
        shutil.rmtree("tempFiles")
        print("Temporary files cleaned up successfully")
    except Exception as e:
        print(f"Error cleaning up temporary files: {e}")
    
        
    