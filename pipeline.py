import os
from diarization import Diarization
from stt import STT
from pydub import AudioSegment

diarization = Diarization()
stt = STT()

def obtainSlice(filePath, startTime, endTime, outputFilePath):
    audio = AudioSegment.from_wav(filePath)
    audio = audio[startTime*1000:endTime*1000]
    audio.export(outputFilePath, format="wav")

def RunPipeline(audio_path: str):

    diarization_result = diarization.diarize(audio_path)
    segments = list(diarization_result.itertracks(yield_label=True))
    i = 0
    totalSegments = len(segments)
    os.makedirs("tempFiles", exist_ok=True)
    speakerWiseSegments = []
    while i < len(segments):
        turn, _, speaker = segments[i]
        currentSpeaker = speaker
        j = i+1
        startTime = turn.start
        endTime = turn.end
        while j < len(segments):
            nextTurn, _, nextSpeaker = segments[j]
            if nextSpeaker == currentSpeaker:
                j += 1
                endTime = nextTurn.end
            else:
                break
        speakerWiseSegments.append((startTime, endTime, speaker))
        i = j
    print(f"Total segments: {totalSegments}")
    print(f"Speaker wise segments: {len(speakerWiseSegments)}")
    i = 0
    for startTime, endTime, speaker in speakerWiseSegments:
        outputFilePath = f"tempFiles/{speaker}_{i}.wav"
        obtainSlice(audio_path, startTime, endTime, outputFilePath)
        segments, info = stt.transcribe(outputFilePath)
        print(f"speaker {speaker}:", end = "")
        for segment in segments:
            print(f"{segment.text}", end = " ")
        print()
        i += 1

def CompareTimestamps(audio_path):
    diarization_result = diarization.diarize(audio_path)
    diarization_segments = list(diarization_result.itertracks(yield_label=True))
    totalSegments = len(diarization_segments)
    transcription_segments, info = stt.transcribe(audio_path)
    transcription_segments = list(transcription_segments)  # Convert generator to list
    print(f"Total segments: {totalSegments}")
    print(f"Transcription segments: {len(transcription_segments)}")
    print("Transcription info: ", info)
    i = 0
    for i in range(min(totalSegments, len(transcription_segments))):
        turn, _, speaker = diarization_segments[i]
        transcriptionSegment = transcription_segments[i]
        print(f"Segment {i}: {turn.start} - {turn.end} - {speaker}")
        print(f"Transcription segment {i}: {transcriptionSegment.start} - {transcriptionSegment.end}, {transcriptionSegment.text}")
        i += 1
    while i < totalSegments:
        turn, _, speaker = diarization_segments[i]
        print(f"Segment {i}: {turn.start} - {turn.end} - {speaker}")
        i += 1
    while i < len(transcription_segments):
        transcriptionSegment = transcription_segments[i]
        print(f"Transcription segment {i}: {transcriptionSegment.start} - {transcriptionSegment.end}, {transcriptionSegment.text}")
        i += 1

if __name__ == "__main__":
    audio_file = "SampleAudios/recording2.wav"
    
    #if os.path.exists(audio_file):
        #RunPipeline(audio_file)
    CompareTimestamps(audio_file)
    
        
    