import os
from sentence_transformers import SentenceTransformers
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdran_client.models import Distance, VectorParams, PointStruct
from groq import Groq


load_dotenv()

Qdrant_api = os.getenv("QDRANT_API_KEY")
Qdrant_URL = os.getenv("QDRANT_URL")

client = QdrantClient(url = Qdrant_URL, api_key = Qdrant_api)


