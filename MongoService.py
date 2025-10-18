from pymongo import MongoClient
from dotenv import load_dotenv
import os
from bson import ObjectId
load_dotenv()

class MongoService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.client = MongoClient(os.getenv("MONGO_CONNECTION_STRING"))
        self.db = self.client[os.getenv("DATABASE_NAME")]
        self.transcriptCollection = self.db["transcripts"]
        self._initialized = True

    @classmethod
    def GetInstance(cls):
        return cls()

    def InsertTranscript(self, transcript, filename):
        result = self.transcriptCollection.insert_one({
            "transcript": transcript,
            "filename": filename
        })
        return result.inserted_id
    
    def GeTranscript(self, transcriptId):
        return self.transcriptCollection.find_one({"_id": ObjectId(transcriptId)})
    