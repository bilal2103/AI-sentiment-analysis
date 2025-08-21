#!/usr/bin/env python3
import os
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pipeline import RunPipeline

load_dotenv()

app = FastAPI(
    title="AI Sentiment Analysis API",
    description="A FastAPI server for AI-powered sentiment analysis with speech processing",
    version="1.0.0"
)

REQUIRED_ENV_VARS = [
    "HF_TOKEN",
    "GROQ_API_KEY", 
    "GROQ_MODEL"
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
async def process_audio(audioFile: UploadFile = File(...), language: str = "english"):
    if audioFile is None or audioFile.file is None or audioFile.content_type != "audio/wav":
        raise HTTPException(status_code=400, detail="Invalid audio file")
    
    if language not in ["english", "arabic"]:
        raise HTTPException(status_code=400, detail="Language must be either english or arabic")
    
    try:
        response = RunPipeline(audioFile, language)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
