#!/usr/bin/env python3
import os
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pipeline import RunPipeline, GetSummary, GetScores
from MongoService import MongoService
import traceback
load_dotenv()

app = FastAPI(
    title="AI Sentiment Analysis API",
    description="A FastAPI server for AI-powered sentiment analysis with speech processing",
    version="1.0.0"
)
mongoService = MongoService.GetInstance()

REQUIRED_ENV_VARS = [
    "HF_TOKEN",
    "GROQ_API_KEY", 
    "GROQ_MODEL",
    "MONGO_CONNECTION_STRING",
    "DATABASE_NAME"
]


@app.get("/")
async def root():
    return {
        "message": "AI Sentiment Analysis API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/status")
async def get_status() -> Dict[str, Any]:
    status_info = {
        "api_status": "healthy",
        "environment_variables": {},
        "missing_variables": [],
        "all_variables_set": True
    }
    
    # Check each required environment variable
    for var_name in REQUIRED_ENV_VARS:
        var_value = os.getenv(var_name)
        
        if var_value is None or var_value.strip() == "":
            status_info["environment_variables"][var_name] = "NOT_SET"
            status_info["missing_variables"].append(var_name)
            status_info["all_variables_set"] = False
        else:
            # Don't expose the actual values for security reasons
            status_info["environment_variables"][var_name] = "SET"
    
    # Update overall status if variables are missing
    if not status_info["all_variables_set"]:
        status_info["api_status"] = "configuration_incomplete"
    
    return status_info

@app.post("/process")
async def process_audio(audioFile: UploadFile = File(...), representativeId: str = Form(...)):
    if representativeId is None or representativeId.strip() == "":
        raise HTTPException(status_code=400, detail="Representative ID is required")
    if audioFile is None or audioFile.file is None:
        raise HTTPException(status_code=400, detail="Invalid audio file")
    
    # Check if it's a valid audio file (more flexible content type checking)
    valid_audio_types = ["audio/wav", "audio/wave", "audio/x-wav", "application/octet-stream"]
    if audioFile.content_type not in valid_audio_types:
        raise HTTPException(status_code=400, detail=f"Invalid audio file type: {audioFile.content_type}. Expected audio file.")
    
    try:
        insertedId = RunPipeline(audioFile, representativeId)
        return {
            "transcriptId": insertedId
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transcript/{transcriptId}")
async def get_transcript(transcriptId: str):
    transcript = mongoService.GeTranscript(transcriptId)
    return {
        "transcript": transcript["transcript"],
        "filename": transcript["filename"]
    }

@app.get("/summary/{transcriptId}")
async def get_summary(transcriptId: str, language: str = "english"):
    if language not in ["english", "arabic"]:
        raise HTTPException(status_code=400, detail="Language must be either english or arabic")
    transcript = mongoService.GeTranscript(transcriptId)
    if transcript is None:
        raise HTTPException(status_code=404, detail="Transcript not found")
    transcript = transcript["transcript"]
    summary = GetSummary(transcript, language)
    return {
        "summary": summary["summary"],
        "mainIssue": summary["mainIssue"],
        "sentimentAnalysis": summary["sentimentAnalysis"],
    }

@app.get("/score/{transcriptId}")
async def get_score(transcriptId: str, subject: str, language: str = "english"):
    if language not in ["english", "arabic"]:
        raise HTTPException(status_code=400, detail="Language must be either english or arabic")
    if subject not in ["representative", "customer"]:
        raise HTTPException(status_code=400, detail="Subject must be either representative or customer")
    transcript = mongoService.GeTranscript(transcriptId)
    if transcript is None:
        raise HTTPException(status_code=404, detail="Transcript not found")
    transcript = transcript["transcript"]
    scores = GetScores(transcript, language, subject)
    return {
        "scores": scores
    }
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or default to 8000
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
