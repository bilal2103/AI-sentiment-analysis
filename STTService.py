import os
from groq import Groq
from dotenv import load_dotenv

class GroqSTT:
    def __init__(self):
        load_dotenv()
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    def transcribe(self, audio_path: str, task="transcribe"):
        try:
            with open(audio_path, "rb") as file:
                if task == "translate":
                    result = self.client.audio.translations.create(
                        file=(audio_path, file.read()),
                        prompt = "Please transcribe this call recording between a customer care representative of SEDER group, and a troubled customer.",
                        model="whisper-large-v3",
                        response_format="verbose_json",
                        temperature=0.0,
                    )
                else:
                    result = self.client.audio.transcriptions.create(
                        file=(audio_path, file.read()),
                        prompt="Please transcribe this call recording between a customer care representative of SEDER group, and a troubled customer. The representative is either a sir or a madam. When someone says 'mom' as a respectful address, please transcribe it as 'ma'am'.",
                        model="whisper-large-v3-turbo",
                        response_format="verbose_json",
                        temperature=0.0,
                    )
                # Return the full result to access segments with timestamps
                return result
                
        except Exception as e:
            print(f"Error transcribing with Groq: {e}")
            return [], None