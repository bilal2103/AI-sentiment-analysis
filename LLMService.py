from groq import Groq
from dotenv import load_dotenv
import os
import json


load_dotenv()



class LLMService:
    groq_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL")
    def __init__(self):
        self.groq_client = Groq(api_key=self.groq_key)

    def SummarizeAndAnalyze(self, script):
        try:
            with open("Prompts.json", "r") as f:
                prompts = json.load(f)
        except Exception as e:
            print(f"Error loading Prompts.json: {e}")
            raise e
        
        messages = [
            {"role": "system", "content": f"{prompts['roleAssigning']}\n{prompts['transcriptGuide']}\n{prompts['outputFormat']}"},
            {"role": "user", "content": f"Here's the call recording's transcription: {script}"}
        ]
        response = self.groq_client.chat.completions.create(
            model=self.groq_model,
            messages=messages
        )
        return response.choices[0].message.content


