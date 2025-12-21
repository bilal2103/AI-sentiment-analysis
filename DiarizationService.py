import os
import torch
import warnings
import json
from dotenv import load_dotenv
from pyannote.audio import Pipeline, Model, Inference
from pyannote.core import Segment
import numpy as np
import torchaudio
from EmbeddingService import EmbeddingService
# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*")
warnings.filterwarnings("ignore", message=".*custom_fwd.*")
warnings.filterwarnings("ignore", message=".*ReproducibilityWarning.*")


def CustomSegment(segment):
    turn, _, speaker = segment
    return {
        "start": turn.start,
        "end": turn.end,
        "speaker": speaker
    }
class Diarization:
    def __init__(self):
        load_dotenv()
        os.environ['SPEECHBRAIN_CACHE_STRATEGY'] = 'copy'
        
        # Enable TF32 for better performance if supported (suppresses the warning)
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except:
                pass
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=os.getenv("HF_TOKEN"))
        self.pipeline = self.pipeline.to(device)
        
        self.embedding_model = Model.from_pretrained("pyannote/embedding", use_auth_token=os.getenv("HF_TOKEN"))
        self.inference = Inference(self.embedding_model, window="whole")
        
        # Load Abdullah's reference embedding
        try:
            with open("abdullah_embedding.json", "r") as f:
                self.abdullah_embedding = np.array(json.load(f))
            with open("fatima_embedding.json", "r") as f:
                self.fatima_embedding = np.array(json.load(f))
            print("✓ Abdullah and Fatima embeddings loaded successfully!")
        except FileNotFoundError:
            print("⚠️ abdullah_embedding.json or fatima_embedding.json not found. Similarity scores will not be calculated.")
            self.abdullah_embedding = None
            self.fatima_embedding = None
    
    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        return dot_product / (norm1 * norm2)

    def GetAveragedEmbeddings(self, speaker_embeddings):
        speaker_averaged_embeddings = {}
        for speaker, embeddings in speaker_embeddings.items():
            if len(embeddings) > 0:
                # Average all embeddings for this speaker
                averaged_embedding = np.mean(embeddings, axis=0)
                speaker_averaged_embeddings[speaker] = averaged_embedding
        return speaker_averaged_embeddings
    def EnsureTwoSpeakers(self, diarizationSegments):
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
            
            # Remove segments from the list
            for segment in segments_to_remove:
                diarizationSegments.remove(segment)
                print(f"Removing segment: {segment}")
            
            print(f"Removed {segments_removed} segments from {len(speakers_to_remove)} speakers")
            print(f"Remaining segments: {len(diarizationSegments)}")
    
    def GetSpeakerCloserToRepresentative(self, speaker_averaged_embeddings, representativeEmbedding):
        #speaker_averaged_embeddings will contain exactly two speakers
        closestSimilarity = 0
        closestSpeaker = None
        for speaker, embedding in speaker_averaged_embeddings.items():
            similarity = self.cosine_similarity(embedding, representativeEmbedding)
            if similarity > closestSimilarity:
                closestSimilarity = similarity
                closestSpeaker = speaker
        return closestSpeaker
            
        return speaker_similarities
    def GetRepresentativeEmbedding(self, representativeId):
        embeddingService = EmbeddingService.GetInstance()
        embedding = embeddingService.get_embedding_by_id(representativeId)
        return embedding
    
    def diarize(self, audio_path: str, representativeId: str):
        representativeEmbedding = self.GetRepresentativeEmbedding(representativeId)
        if representativeEmbedding is None:
            raise Exception(f"Representative embedding with ID {representativeId} not found")
        print(f"Starting diarization for: {audio_path}")
        waveform, sample_rate = torchaudio.load(audio_path)
        audio_duration = waveform.shape[1] / sample_rate
        print(f"Audio duration: {audio_duration:.2f}s")
        
        diarization_result = self.pipeline(audio_path)
        print("✓ Diarization completed!")
        diarizationSegments = [CustomSegment(segment) for segment in list(diarization_result.itertracks(yield_label=True))]
        self.EnsureTwoSpeakers(diarizationSegments)
        # Print all segments with timing information
        print("\n" + "="*60)
        print("DIARIZATION SEGMENTS:")
        print("="*60)
        for segment in diarizationSegments:
            print(f"{segment['start']:>7.2f}s - {segment['end']:>7.2f}s | {segment['speaker']}")
        print("="*60 + "\n")
        
        print("Extracting speaker embeddings...")
        
        # Extract embeddings for each segment and group by speaker
        segments_with_embeddings = []
        speaker_embeddings = {}  # Dictionary to collect embeddings per speaker
        
        for segment in diarizationSegments:
            # Skip segments shorter than 2 seconds for better embedding quality
            segment_duration = segment['end'] - segment['start']
            if segment_duration < 2.0:
                print(f"Skipping short segment: {segment_duration:.3f}s for speaker {segment['speaker']}")
                continue
            
            try:
                # Clamp segment boundaries to audio duration
                segment_start = max(0.0, segment['start'])
                segment_end = min(audio_duration, segment['end'])
                
                # Skip if the clamped segment is too short
                if segment_end - segment_start < 2.0:
                    print(f"Skipping segment after clamping: {segment_end - segment_start:.3f}s for speaker {segment['speaker']}")
                    continue
                
                excerpt = Segment(segment_start, segment_end)
                embedding = self.inference.crop(audio_path, excerpt)
                
                if segment['speaker'] not in speaker_embeddings:
                    speaker_embeddings[segment['speaker']] = []
                speaker_embeddings[segment['speaker']].append(embedding)
                
            except RuntimeError as e:
                print(f"Error processing segment {segment['start']:>7.2f}s - {segment['end']:>7.2f}s: {e}")
        print("✓ Embedding extraction completed!")
        
        speaker_averaged_embeddings = self.GetAveragedEmbeddings(speaker_embeddings)
        representativeEmbeddingVector = np.array(representativeEmbedding['embedding_vector'])
        speaker_similarities = self.GetSpeakerCloserToRepresentative(speaker_averaged_embeddings, representativeEmbeddingVector)
        for segment in diarizationSegments:
            if segment['speaker'] == speaker_similarities:
                segment['speaker'] = "Representative"
            else:
                segment['speaker'] = "Customer"
        
        return diarizationSegments  
    