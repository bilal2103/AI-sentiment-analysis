import os
import torch
import warnings
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from tqdm import tqdm

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*")
warnings.filterwarnings("ignore", message=".*custom_fwd.*")
warnings.filterwarnings("ignore", message=".*ReproducibilityWarning.*")

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
        
        print("Loading diarization model... (this may take a moment)")
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=os.getenv("HF_TOKEN"))
        self.pipeline = self.pipeline.to(device)
        print("✓ Diarization model loaded successfully!")

    def diarize(self, audio_path: str, show_progress: bool = True) -> str:
        print(f"Starting diarization for: {audio_path}")
        
        if show_progress:
            print("🔄 Processing audio... (this may take several minutes for long files)")
            
        # For now, run without progress hook to avoid compatibility issues
        # We can show a simple spinner or progress message instead
        diarization = self.pipeline(audio_path)
        
        print("✓ Diarization completed!")
        return diarization

# Run diarization on your audio file
# diarization = Diarization()
# diarization.diarize("recording2.wav")

# with open("diarization.txt", "w") as f:
#     # Print diarization segments
#     for turn, _, speaker in diarization.itertracks(yield_label=True):
#         f.write(f"{turn.start:.1f}s - {turn.end:.1f}s: Speaker {speaker}\n")