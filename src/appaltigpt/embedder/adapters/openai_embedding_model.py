from appaltigpt.embedder.ports.embedding_model_port import EmbeddingModelPort
from openai import AsyncOpenAI
from typing import List, Optional

class OpenAIEmbeddingModel(EmbeddingModelPort):
    def __init__(self, client: Optional[AsyncOpenAI] = None, model: str = "text-embedding-3-small", model_size: int = 1536):
        self.client = client if client else AsyncOpenAI()
        self.model = model
        self.model_size = model_size

    async def encode_queries(self, texts: List[str]) -> List[List[float]]:
        res = await self.client.embeddings.create(input=texts, model=self.model)
        return [embedding.embedding for embedding in res.data]

    async def encode_documents(self, texts: List[str]) -> List[List[float]]:
        res = await self.client.embeddings.create(input=texts, model=self.model)
        return [embedding.embedding for embedding in res.data]
