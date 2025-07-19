import whisper
import soundfile as sf
import numpy as np
import torch

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the model (options: tiny, base, small, medium, large)
model = whisper.load_model("large", device=device)


# Transcribe the audio file
result = model.transcribe("arabicSample.wav")

# Print the detected language and confidence
print(f"Detected language: {result['language']}")
print(f"Text: {result['text']}")

result = model.transcribe("arabicSample.wav", task="translate", language="en")
print(f"Detected language: {result['language']}")
print(f"Text: {result['text']}")