from pymongo import MongoClient
from dotenv import load_dotenv
import os
import json
import numpy as np
from typing import Optional, List, Dict
from bson import ObjectId

load_dotenv()

class EmbeddingService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.client = MongoClient(os.getenv("MONGO_CONNECTION_STRING"))
        self.db = self.client[os.getenv("DATABASE_NAME")]
        self.embedding_collection = self.db["embedding_vectors"]
        self._initialized = True

    @classmethod
    def GetInstance(cls):
        return cls()

    def insert_embedding(self, speaker_name: str, embedding_vector: np.ndarray, metadata: Optional[Dict] = None) -> str:
        embedding_list = embedding_vector.tolist() if isinstance(embedding_vector, np.ndarray) else embedding_vector
        
        document = {
            "speaker_name": speaker_name,
            "embedding_vector": embedding_list,
            "vector_dimension": len(embedding_list),
            "metadata": metadata or {}
        }
        
        result = self.embedding_collection.insert_one(document)
        print(f"✓ Inserted embedding for {speaker_name} with ID: {result.inserted_id}")
        return str(result.inserted_id)

    def get_embedding_by_id(self, embedding_id: str) -> Optional[Dict]:
        return self.embedding_collection.find_one({"_id": ObjectId(embedding_id)})

    def get_all_embeddings(self) -> List[Dict]:
        return list(self.embedding_collection.find())

    def update_embedding(self, speaker_name: str, embedding_vector: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        embedding_list = embedding_vector.tolist() if isinstance(embedding_vector, np.ndarray) else embedding_vector
        
        update_data = {
            "embedding_vector": embedding_list,
            "vector_dimension": len(embedding_list)
        }
        
        if metadata:
            update_data["metadata"] = metadata
        
        result = self.embedding_collection.update_one(
            {"speaker_name": speaker_name},
            {"$set": update_data},
            upsert=True
        )
        
        print(f"✓ Updated embedding for {speaker_name}")
        return result.acknowledged

    def upload_from_json_file(self, speaker_name: str, json_file_path: str, metadata: Optional[Dict] = None) -> str:
        try:
            with open(json_file_path, 'r') as f:
                embedding_vector = json.load(f)
            
            embedding_array = np.array(embedding_vector)
            
            return self.insert_embedding(speaker_name, embedding_array, metadata)
        except FileNotFoundError:
            print(f"❌ Error: File {json_file_path} not found")
            raise
        except json.JSONDecodeError:
            print(f"❌ Error: Invalid JSON in {json_file_path}")
            raise


def upload_initial_embeddings():
    service = EmbeddingService.GetInstance()
    
    print("Uploading initial embedding vectors...")
    print("=" * 60)
    
    # Upload Abdullah's embedding
    try:
        abdullah_id = service.upload_from_json_file(
            speaker_name="Abdullah",
            json_file_path="abdullah_embedding.json",
            metadata={
                "description": "Reference embedding for speaker Abdullah",
                "source": "abdullah_embedding.json"
            }
        )
        print(f"Abdullah embedding uploaded with ID: {abdullah_id}")
    except Exception as e:
        print(f"❌ Failed to upload Abdullah embedding: {e}")
    
    # Upload Fatima's embedding
    try:
        fatima_id = service.upload_from_json_file(
            speaker_name="Fatima",
            json_file_path="fatima_embedding.json",
            metadata={
                "description": "Reference embedding for speaker Fatima",
                "source": "fatima_embedding.json"
            }
        )
        print(f"Fatima embedding uploaded with ID: {fatima_id}")
    except Exception as e:
        print(f"❌ Failed to upload Fatima embedding: {e}")
    
    print("=" * 60)
    print("✓ Initial embedding upload completed!")


if __name__ == "__main__":
    service = EmbeddingService.GetInstance()
    
    print("\nRetrieving all embeddings from database...")
    print("=" * 60)
    
    all_embeddings = service.get_all_embeddings()
    for embedding in all_embeddings:
        print(f"Speaker: {embedding['speaker_name']}")
        print(f"Vector dimension: {embedding['vector_dimension']}")
        print(f"ID: {embedding['_id']}")
        print(f"Metadata: {embedding.get('metadata', {})}")
        print("-" * 60)
    
    print(f"\nTotal embeddings in database: {len(all_embeddings)}")
