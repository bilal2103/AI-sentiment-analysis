from groq import Groq
from dotenv import load_dotenv
import os
import yaml


load_dotenv()



class LLMService:
    groq_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL")
    def __init__(self):
        self.groq_client = Groq(api_key=self.groq_key)

    def SummarizeAndAnalyze(self, script):
        try:
            with open("prompts.yaml", "r") as f:
                prompts = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading prompts.yaml: {e}")
            raise e
        
        messages = [
            {"role": "system", "content": f"{prompts['roleAssigning']}\n-------------\n{prompts['transcriptGuide']}\n-------------\n{prompts['outputFormat']}"},
            {"role": "user", "content": f"Here's the call recording's transcription: {script}"}
        ]
        response = self.groq_client.chat.completions.create(
            model=self.groq_model,
            messages=messages,
            temperature=0.0
        )
        return response.choices[0].message.content
    
    def ScoreCall(self, script, subject):
        try:
            with open("prompts.yaml", "r") as f:
                prompts = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading prompts.yaml: {e}")
            raise e
        if subject == "representative":
            messages = [
                {"role": "system", "content": f"{prompts['scoringPromptRoleAssigning']}\n-------------\n{prompts['scoringGuideRepresentative']}\n-------------\n{prompts['scoringOutputFormat']}\n{prompts['sampleOutputRepresentative']}"},
                {"role": "user", "content": f"Here's the call recording's transcription: {script}"}
            ]
        else:
            messages = [
                {"role": "system", "content": f"{prompts['scoringPromptRoleAssigning']}\n-------------\n{prompts['scoringGuideCustomer']}\n-------------\n{prompts['scoringOutputFormat']}\n{prompts['sampleOutputCustomer']}"},
                {"role": "user", "content": f"Here's the call recording's transcription: {script}"}
            ]
        response = self.groq_client.chat.completions.create(
            model=self.groq_model,
            messages=messages,
            temperature=0.0
        )
        return response.choices[0].message.content
    
    def TranslateToArabic(self, text):
        messages = [
            {
                "role": "system",
                "content": "You are expert in translating text in English to Arabic. In your output, please only provide the translated text and nothing else."
            },
            {
                "role": "user",
                "content": f"Translate the following text to Arabic: {text}"
            }
        ]
        response = self.groq_client.chat.completions.create(
            model=self.groq_model,
            messages=messages,
            temperature=0.0
        )
        return response.choices[0].message.content