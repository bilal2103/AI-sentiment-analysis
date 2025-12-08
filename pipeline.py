import os
import shutil
from STTService import GroqSTT
import json
from LLMService import LLMService
from pydub import AudioSegment
from MongoService import MongoService
import re
from pydub.silence import split_on_silence

stt = None
llm = LLMService()
mongo = MongoService.GetInstance()

def PreProcessAudio(input_file, output_file, silence_thresh=-40, min_silence_len=1000, keep_silence=200):
    try:
        audio = AudioSegment.from_wav(input_file)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(output_file, format="wav")
        chunks = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence
        )
        if chunks:
            combined = AudioSegment.empty()
            for i, chunk in enumerate(chunks):
                combined += chunk
                if i < len(chunks) - 1:
                    combined += AudioSegment.silent(duration=500)
            combined.export(output_file, format="wav")
            print(f"Silence removed. Output saved to: {output_file}")
        else:
            print("No audio chunks found after silence removal")
            
    except Exception as e:
        print(f"Error removing silence: {e}")
        return False


    
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
def RunPipeline(audioFile):
    global stt
    stt = GroqSTT()
    os.makedirs("cleanedFiles", exist_ok=True)
    
    if hasattr(audioFile, 'filename') and hasattr(audioFile, 'file'):
        filename = audioFile.filename
        cleanedAudioPath = f"cleanedFiles/{filename.split('/')[-1]}"
        
        with open(cleanedAudioPath, "wb") as buffer:
            content = audioFile.file.read()
            buffer.write(content)
        
        audioFile.file.seek(0)
    else:
        raise ValueError("Invalid audio file")

    PreProcessAudio(cleanedAudioPath, cleanedAudioPath)
    # transcribe() now returns both transcription segments and diarization result
    segmentDuration = 80000
    insertedId = None
    transcriptionSegments, diarization_result = stt.transcribe(cleanedAudioPath, segmentDuration)
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

    insertedId = mongo.InsertTranscript(script, filename)
    return str(insertedId)

def GetScores(transcript, language, subject):
    response = llm.ScoreCall(transcript, subject)
    responseDict = extract_json_from_response(response)
    scoresDict = responseDict["scores"]
    if language and language == "arabic":
        for score_item in scoresDict:
            for criteriaKey, criteriaValue in score_item.items():
                criteriaValue["reasoning"] = llm.TranslateToArabic(criteriaValue["reasoning"])
        if "behaviorSummary" in responseDict:
            responseDict["behaviorSummary"] = llm.TranslateToArabic(responseDict["behaviorSummary"])
    totalScore = 0
    for score_item in scoresDict:
        for criteriaKey, criteriaValue in score_item.items():
            print(f"Criteria: {criteriaKey}")
            print(f"Score: {criteriaValue['score']}")
            print(f"Reasoning: {criteriaValue['reasoning']}")
            totalScore += criteriaValue["score"]
    print(f"Total score: {totalScore}")
    result = {
        "scoresDict": scoresDict,
        "totalScore": totalScore,
    }
    if subject == "customer":
        result["behaviorSummary"] = responseDict["behaviorSummary"]
    return result

def GetSummary(transcript, language):
    response = llm.SummarizeAndAnalyze(transcript)
    
    responseDict = extract_json_from_response(response)
    
    if responseDict:
        try:
            if language and language == "arabic":
                responseDict["summary"] = llm.TranslateToArabic(responseDict["summary"])
                responseDict["mainIssue"] = llm.TranslateToArabic(responseDict["mainIssue"])
                responseDict["sentimentAnalysis"]["representative"]["sentiment"] = llm.TranslateToArabic(responseDict["sentimentAnalysis"]["representative"]["sentiment"])
                responseDict["sentimentAnalysis"]["representative"]["reasoning"] = llm.TranslateToArabic(responseDict["sentimentAnalysis"]["representative"]["reasoning"])
                responseDict["sentimentAnalysis"]["customer"]["sentiment"] = llm.TranslateToArabic(responseDict["sentimentAnalysis"]["customer"]["sentiment"])
                responseDict["sentimentAnalysis"]["customer"]["reasoning"] = llm.TranslateToArabic(responseDict["sentimentAnalysis"]["customer"]["reasoning"])
                
            print(responseDict["summary"])
            print("===================================\nSentiment Analysis:\n===================================")
            print(f"Representative: {responseDict['sentimentAnalysis']['representative']}")
            print(f"Customer: {responseDict['sentimentAnalysis']['customer']}")
            print("===================================\n===================================")
            print(f"Main Issue: {responseDict['mainIssue']}")
            print("===================================\n===================================")

        except KeyError as e:
            print(f"JSON structure is missing expected key: {e}")
            print("Available keys:", list(responseDict.keys()))
            print("Raw JSON content:")
            print(json.dumps(responseDict, indent=2))
    return responseDict
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Please provide an audio file path as argument")
        print("Usage: python pipeline.py <audio_file>")
        sys.exit(1)
        
    audio_file = sys.argv[1]
    audio_file = audio_file
    response, totalScore = RunPipeline(audio_file)
    
    
        
    
