import asyncio
from typing import List, Any
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from appaltigpt.chunkizer.ports.vector_store_port import VectorStorePort
from appaltigpt.embedder.ports.embedding_model_port import EmbeddingModelPort
from appaltigpt.chunkizer.schema import QdrantChunk


class QdrantVectorWriter(VectorStorePort):
    def __init__(self, client: QdrantClient, settings: Any, embedding_model: EmbeddingModelPort = None):
        self.client = client
        self.collection_name = settings.collection_name
        self.embedding_model = embedding_model


        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedding_model.model_size, distance=Distance.COSINE)     
            )

    async def upsert(self, chunks: List[QdrantChunk]) -> None:
        if not chunks:
            return
        
        texts = [c.content for c in chunks]
        
        

        embeddings = await self.embedding_model.encode_documents(texts)

        def _upsert_qdrant():
            points = []
            for chunk, vector in zip(chunks, embeddings,strict=True):
                points.append(PointStruct(
                    id=chunk.id,
                    vector=vector,
                    payload=chunk.model_dump()
                ))

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        

        await asyncio.to_thread(_upsert_qdrant)
