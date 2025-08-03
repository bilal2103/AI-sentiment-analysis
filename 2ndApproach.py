from STTService import GroqSTT
import json
from DiarizationService import Diarization
from pydub import AudioSegment
from scipy.optimize import linear_sum_assignment
stt = GroqSTT()
def Segment(segment):
    turn, _, speaker = segment
    return {
        "start": turn.start,
        "stop": turn.end,
        "speaker": speaker
    }
# result = stt.transcribe("SampleAudios/urduSample.wav", task="translate")
# segments = list(result.segments)
# with open("result.json", "w") as f:
#     json.dump(segments, f, indent=4)

# diarization = Diarization()
# result = diarization.diarize("SampleAudios/urduSample.wav")
# segments = list(result.itertracks(yield_label=True))
# segments = [Segment(segment) for segment in segments]
# with open("diarization.json", "w") as f:
#     json.dump(segments, f, indent=4)
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

    #segments = merge_segments(segments)

# i = 0
# for segment in segments:
#     segment["id"] = i
#     i += 1

# with open("diarization.json", "w") as f:
#     json.dump(segments, f, indent=4)
def ComputeDistance(segment_A, segment_B):
        startDiff = abs(segment_A["start"] - segment_B["start"])
        endDiff = abs(segment_A["stop"] - segment_B["end"])
        return startDiff + endDiff
def create_cost_matrix(whisper_segments, pyannote_segments):
    cost_matrix = []
    
    PENALTY_COST = 1000
    
    for i, whisper_seg in enumerate(whisper_segments):
        row = []
        for j, pyannote_seg in enumerate(pyannote_segments):
            
            whisper_is_dummy = whisper_seg.get('is_dummy', False)
            pyannote_is_dummy = pyannote_seg.get('is_dummy', False)
            
            if whisper_is_dummy or pyannote_is_dummy:
                cost = PENALTY_COST
            else:
                cost = ComputeDistance(pyannote_seg, whisper_seg )
            row.append(cost)
        cost_matrix.append(row)
    
    return cost_matrix
def balance_segments_for_hungarian(whisper_segments, pyannote_segments):
    whisper_count = len(whisper_segments)
    pyannote_count = len(pyannote_segments)
    print(f"Whisper count: {whisper_count}, Pyannote count: {pyannote_count}")
    if whisper_count == pyannote_count:
        return whisper_segments, pyannote_segments
    
    if whisper_count < pyannote_count:
        # Add dummy whisper segments
        dummy_count = pyannote_count - whisper_count
        for i in range(dummy_count):
            dummy_segment = {"id": whisper_count + i,"start": -1,  "end": -1,   "text": f"DUMMY_WHISPER_{i}", "is_dummy": True}
            whisper_segments.append(dummy_segment)
    else:
        # Add dummy pyannote segments
        dummy_count = whisper_count - pyannote_count
        for i in range(dummy_count):
            dummy_segment = {"id": pyannote_count + i,"start": -1,  "stop": -1,   "speaker": f"DUMMY_SPEAKER_{i}", "is_dummy": True}
            pyannote_segments.append(dummy_segment)
    
    return whisper_segments, pyannote_segments
def HungarianAlgorithm(transcriptionSegments, segments):
    if len(transcriptionSegments) != len(segments):
        transcriptionSegments, segments = balance_segments_for_hungarian(transcriptionSegments, segments)
    costMatrix = create_cost_matrix(transcriptionSegments, segments)
    row_ind, col_ind = linear_sum_assignment(costMatrix)
    return row_ind, col_ind

 
def UseDistance(transcriptionSegments, segments):
    mapping = {}
    iouMapping = {}
    for transcriptionSegment in transcriptionSegments:
        minDistance = float("inf")
        minSegment = None
        distances = []
        for diarizationSegment in segments:
            distance = ComputeDistance(diarizationSegment, transcriptionSegment)
            distances.append((distance, diarizationSegment["id"]))
            if distance < minDistance:
                minDistance = distance
                minSegment = diarizationSegment

        distances.sort(key=lambda x: x[0])
        top3Distances = distances[:3]
        print(f"{transcriptionSegment['text']} - {top3Distances}")
        if f"dSegment_{minSegment['id']}_{minSegment['speaker']}" not in mapping:
            mapping[f"dSegment_{minSegment['id']}_{minSegment['speaker']}"] = [transcriptionSegment["text"]]
        else:
            mapping[f"dSegment_{minSegment['id']}_{minSegment['speaker']}"].append(transcriptionSegment["text"])
        iouMapping[transcriptionSegment["id"]] = minDistance

    with open("usingDistance.json", "w") as f:
        json.dump(mapping, f, indent=4)
    return mapping, iouMapping
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
    iouMapping = {}
    for transcriptionSegment in transcriptionSegments:
        bestIOU = -1
        bestSegment = None
        for diarizationSegment in segments:
            iou = ComputeIOU(transcriptionSegment, diarizationSegment)
            if iou > bestIOU:
                bestIOU = iou
                bestSegment = diarizationSegment
        print(f"{transcriptionSegment['text']} - {bestSegment['id']} - {bestIOU}")
        iouMapping[transcriptionSegment["id"]] = bestIOU
        if f"dSegment_{bestSegment['id']}_{bestSegment['speaker']}" not in mapping:
            mapping[f"dSegment_{bestSegment['id']}_{bestSegment['speaker']}"] = [transcriptionSegment["text"]]
        else:
            mapping[f"dSegment_{bestSegment['id']}_{bestSegment['speaker']}"].append(transcriptionSegment["text"])
    
    with open("usingIOU.json", "w") as f:
        json.dump(mapping, f, indent=4)
    return mapping, iouMapping
def AnalyzeMapping(mapping, iouMapping, trueMapping):
    invalidMatching = 0
    for key,value in mapping.items():
        speaker = key[-2:]
        for segment in transcriptionSegments:
            if segment["text"] in value:
                segmentIndex = segment["id"]
                break
        trueSpeaker = trueMapping[str(segmentIndex)]
        if trueSpeaker != speaker:
            invalidMatching += 1
            print(f"{key} - {value} - {trueSpeaker} - {speaker}", end = "")
            print(f" - {iouMapping[segmentIndex]}")
            iouMapping.pop(segmentIndex)
    
    print("Printing iou mapping of correct matching")
    for key, value in iouMapping.items():
        print(f"{key} - {value}")
    correctMatching = len(mapping) - invalidMatching
    print(f"Total mapping: {len(mapping)}")
    print(f"Invalid matching: {invalidMatching}")
    print(f"Accuracy: {correctMatching / len(mapping)}")

def analyze_hungarian_results(row_ind, col_ind, whisper_segments, pyannote_segments, trueMapping):
    print("=== HUNGARIAN ALGORITHM RESULTS ===")
    print(f"Total assignments: {len(row_ind)}")
    print()
    
    real_matches = 0
    dummy_matches = 0
    incorrectMatches = 0
    for i in range(len(row_ind)):
        whisper_idx = row_ind[i]
        pyannote_idx = col_ind[i]
        
        whisper_seg = whisper_segments[whisper_idx]
        pyannote_seg = pyannote_segments[pyannote_idx]
        
        if whisper_seg.get('is_dummy', False):
            print(f"UNMATCHED: Pyannote segment {pyannote_idx} (Speaker: {pyannote_seg['speaker']}) has no good whisper match")
            dummy_matches += 1
        else:
            print(f"MATCH {real_matches + 1}:")
            print(f"  Whisper {whisper_idx}: '{whisper_seg['text'].strip()}'")
            print(f"  → Speaker: {pyannote_seg['speaker']}")
            print(f"  Time: {whisper_seg['start']:.1f}-{whisper_seg['end']:.1f}s ↔ {pyannote_seg['start']:.1f}-{pyannote_seg['stop']:.1f}s")
            print()
            if pyannote_seg['speaker'][-2:] != trueMapping[str(whisper_seg['id'])]:
                incorrectMatches += 1
            real_matches += 1
    
    print(f"=== SUMMARY ===")
    print(f"Real matches: {real_matches}")
    print(f"Unmatched pyannote segments: {dummy_matches}")
    print(f"Total whisper segments processed: {len([w for w in whisper_segments if not w.get('is_dummy', False)])}")
    print(f"Incorrect matches: {incorrectMatches}")
    print(f"Accuracy: {(real_matches - incorrectMatches) / (real_matches)}")
if __name__ == "__main__":
    with open("diarization.json", "r") as f:
        segments = json.load(f)
    with open("result.json", "r") as f:
        transcriptionSegments = json.load(f)
    #mapping, iouMapping = UseDistance(transcriptionSegments, segments)
    #mapping, iouMapping = UseIOU(transcriptionSegments, segments)
    with open("trueMapping_urduSample.json", "r") as f:
        trueMapping = json.load(f)
    row_ind, col_ind = HungarianAlgorithm(transcriptionSegments, segments)
    analyze_hungarian_results(row_ind, col_ind, transcriptionSegments, segments, trueMapping)
