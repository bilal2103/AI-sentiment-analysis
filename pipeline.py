import os
import shutil
from STTService import GroqSTT
import json
from LLMService import LLMService
from pydub import AudioSegment
from MongoService import MongoService
import re
from pydub.silence import split_on_silence
from DiarizationService import Diarization
stt = None
llm = LLMService()
mongo = MongoService.GetInstance()
diarization = Diarization()
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



def UseIOU(transcriptionSegments, segments):
    def ComputeIOU(whisper_segment, pyannote_segment):
        whisper_start, whisper_end = whisper_segment["start"], whisper_segment["end"]
        pyannote_start, pyannote_end = pyannote_segment["start"], pyannote_segment["end"]

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
def CleanupTempFiles(cleanedAudioPath=None):
    """
    Safely clean up temporary files and directories.
    
    Args:
        cleanedAudioPath: Path to the specific cleaned audio file to remove (optional)
    """
    try:
        # Clean up the specific cleaned audio file if provided
        if cleanedAudioPath and os.path.exists(cleanedAudioPath):
            os.remove(cleanedAudioPath)
            print(f"✓ Cleaned up: {cleanedAudioPath}")
        
        # Clean up segments directory if it exists
        if os.path.exists("segments"):
            shutil.rmtree("segments")
            print("✓ Cleaned up: segments/")
        
        # Clean up cleanedFiles directory if empty or remove all files
        if os.path.exists("cleanedFiles"):
            files = os.listdir("cleanedFiles")
            if len(files) == 0:
                shutil.rmtree("cleanedFiles")
                print("✓ Cleaned up: cleanedFiles/ (directory was empty)")
            else:
                # Remove all files in cleanedFiles
                for file in files:
                    file_path = os.path.join("cleanedFiles", file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"✓ Cleaned up: {file_path}")
                # Remove directory if now empty
                if len(os.listdir("cleanedFiles")) == 0:
                    shutil.rmtree("cleanedFiles")
                    print("✓ Cleaned up: cleanedFiles/")
                    
    except Exception as e:
        print(f"⚠️ Warning: Error during cleanup: {e}")
        # Don't raise exception - cleanup failures shouldn't break the pipeline

def RunPipeline(audioFile, representativeId):
    global stt
    stt = GroqSTT()
    os.makedirs("cleanedFiles", exist_ok=True)
    
    cleanedAudioPath = None
    
    try:
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
        segmentDuration = 80000
        insertedId = None
        diarizationSegments = diarization.diarize(cleanedAudioPath, representativeId)
        transcriptionSegments = stt.transcribe(diarizationSegments, cleanedAudioPath, segmentDuration)
        
        

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
        
        # Clean up temporary files after successful processing
        print("\n" + "="*60)
        print("CLEANING UP TEMPORARY FILES")
        print("="*60)
        CleanupTempFiles(cleanedAudioPath)
        print("="*60 + "\n")
        
        return str(insertedId)
        
    except Exception as e:
        # Clean up temporary files even if processing fails
        print("\n⚠️ Error occurred, cleaning up temporary files...")
        CleanupTempFiles(cleanedAudioPath)
        raise e

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

    
    
        
    
