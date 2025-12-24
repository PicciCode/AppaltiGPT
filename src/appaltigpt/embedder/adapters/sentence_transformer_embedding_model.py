import asyncio
from typing import List
from sentence_transformers import SentenceTransformer
from appaltigpt.embedder.ports.embedding_model_port import EmbeddingModelPort

class SentenceTransformerEmbeddingModel(EmbeddingModelPort):
    def __init__(self, model_name: str = "google/embeddinggemma-300m"):
        self.model = SentenceTransformer(model_name)

    async def encode_queries(self, texts: List[str]) -> List[List[float]]:

        res = await asyncio.to_thread(self.model.encode, texts, prompt_name="query")
        return [embedding.tolist() for embedding in res]

    async def encode_documents(self, texts: List[str]) -> List[List[float]]:

        res = await asyncio.to_thread(self.model.encode, texts)
        return [embedding.tolist() for embedding in res]
