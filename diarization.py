import os
import torch
from dotenv import load_dotenv
load_dotenv()
os.environ['SPEECHBRAIN_CACHE_STRATEGY'] = 'copy'

from pyannote.audio import Pipeline

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Replace this with your Hugging Face token
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=os.getenv("HF_TOKEN"))

# Move pipeline to GPU if available
pipeline = pipeline.to(device)

# Run diarization on your audio file
diarization = pipeline("recording2.wav")

with open("diarization.txt", "w") as f:
    # Print diarization segments
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        f.write(f"{turn.start:.1f}s - {turn.end:.1f}s: Speaker {speaker}\n")