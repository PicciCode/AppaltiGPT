import asyncio
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from appaltigpt.retrieval.vector_search.ports.search_port import VectorSearchPort

class QdrantVectorReader(VectorSearchPort):
    def __init__(self, client: QdrantClient, settings: Any):
        self.client = client
        self.collection_name = settings.collection_name

    async def search(self, vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        def _search_qdrant():
            # Usiamo query_points che è l'API unificata più recente
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=limit
            )
            return response.points

        search_result = await asyncio.to_thread(_search_qdrant)

        results = []
        for point in search_result:
            results.append({
                "id": point.id,
                "score": point.score,
                "payload": point.payload,
            })
            
        return results
