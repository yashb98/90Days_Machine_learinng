import asyncio
import numpy as np
from typing import List, Dict, Optional
from google import genai
from google.genai import types
import config
from services.location_service import LocationService, GeoPose


class MemoryService:
    def __init__(self, location_service: LocationService):
        self.loc_service = location_service
        self.client = None

        # In-memory "Database" for demo purposes
        # Format: [{"text": str, "vector": np.array, "pose": GeoPose}]
        self.memory_db: List[Dict] = []

        # Initialize Vertex AI Client for Embeddings
        try:
            self.client = genai.Client(
                vertexai=True,
                project=config.GOOGLE_CLOUD_PROJECT,
                location=config.GOOGLE_CLOUD_LOCATION
            )
            print("✅ Memory Service (Vertex AI Embeddings) Initialized")
        except Exception as e:
            print(f"❌ Memory Service Init Error: {e}")

    async def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Fetches vector embedding from Vertex AI (text-embedding-004)."""
        if not self.client:
            return None

        try:
            # Using the new Unified SDK for Embeddings
            response = self.client.models.embed_content(
                model="text-embedding-004",
                contents=text
            )
            return np.array(response.embeddings[0].values)
        except Exception as e:
            print(f"⚠️ Embedding Error: {e}")
            return None

    async def store_memory(self, text: str, pose: GeoPose):
        """Saves a new observation to the spatial DB."""
        vector = await self._get_embedding(text)
        if vector is not None:
            self.memory_db.append({
                "text": text,
                "vector": vector,
                "pose": pose
            })
            print(
                f"💾 Memory Stored: '{text[:20]}...' at {pose.lat}, {pose.lng}")

    async def recall(self, query: str, current_pose: GeoPose, radius_meters: float = 50.0) -> str:
        """
        Retrieves relevant info based on:
        1. Location (must be within radius)
        2. Semantic Similarity (vector cosine similarity)
        """
        if not self.memory_db:
            return ""

        # 1. Spatial Filtering
        nearby_memories = [
            m for m in self.memory_db
            if self.loc_service.is_within_radius(current_pose, m["pose"], radius_meters)
        ]

        if not nearby_memories:
            return ""

        # 2. Semantic Search (if we have candidates)
        query_vector = await self._get_embedding(query)
        if query_vector is None:
            return ""

        best_score = -1.0
        best_text = ""

        for mem in nearby_memories:
            # Cosine Similarity
            score = np.dot(mem["vector"], query_vector) / (
                np.linalg.norm(mem["vector"]) * np.linalg.norm(query_vector)
            )
            if score > best_score:
                best_score = score
                best_text = mem["text"]

        # Threshold to avoid irrelevant hallucinations
        if best_score > 0.6:
            return f"Context: {best_text}"

        return ""
