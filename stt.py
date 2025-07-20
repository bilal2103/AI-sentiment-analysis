from faster_whisper import WhisperModel
import soundfile as sf
import numpy as np
import torch

class STT:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        self.model = WhisperModel("large-v2", device=device)    

    def transcribe(self, audio_path: str) -> str:
        result = self.model.transcribe(audio_path, word_timestamps=True)
        return result
    
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