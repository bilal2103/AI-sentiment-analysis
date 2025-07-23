from faster_whisper import WhisperModel
import soundfile as sf
import numpy as np
import torch

class STT:
    # Simplified prompt - complex prompts can sometimes cause hallucinations
    prompt = "Transcribe the following call recording clip accurately."
    # Use only low temperatures to reduce hallucinations
    temperature = tuple(np.arange(0, 0.4 + 1e-6, 0.2))  # [0, 0.2, 0.4]
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        # Stricter parameters to reduce hallucinations
        self.args = {
            "language": "en",
            "temperature": self.temperature,
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -0.5,
            "no_speech_threshold": 0.8,  # Increased from 0.6 to be more strict
            "condition_on_previous_text": False,  # Disable to prevent context bleeding
            "initial_prompt": self.prompt,
            "word_timestamps": True,  # Enable for better detection
        }
        self.model = WhisperModel("large-v2", device=device, compute_type="float16")    

    def transcribe(self, audio_path: str):
        segments, info = self.model.transcribe(audio_path, **self.args)
        # Filter out segments with high no_speech_prob
        filtered_segments = []
        for segment in segments:
            if segment.no_speech_prob < 0.8:  # Additional filtering
                filtered_segments.append(segment)
        return filtered_segments, info
    
    def translate(self, audio_path: str) -> str:
        result = self.model.transcribe(audio_path, task="translate", language="en")
        return result


# Transcribe the audio file
# result = model.transcribe("arabicSample.wav")

# # Print the detected language and confidence
# print(f"Detected language: {result['language']}")
# print(f"Text: {result['text']}")

# result = model.transcribe("arabicSample.wav", task="translate", language="en")
# print(f"Detected language: {result['language']}")
# print(f"Text: {result['text']}")