import asyncio
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from appaltigpt.retrieval.ports.keyword_search_port import KeywordSearchPort

class QdrantKeywordReader(KeywordSearchPort):
    
    def __init__(self, client: QdrantClient, settings: Any, content_field: str = "content"):
        self.client = client
        self.collection_name = settings.collection_name
        self.content_field = content_field

    async def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        
        def _scroll_qdrant():
            search_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=self.content_field,
                        match=models.MatchText(text=query)
                    )
                ]
            )
            
            response, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=search_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            return response

        points = await asyncio.to_thread(_scroll_qdrant)

        results = []
        for i, point in enumerate(points):
            fake_score = 1.0 - (i * 0.01) 
            
            results.append({
                "id": point.id,
                "score": fake_score, 
                "payload": point.payload,
            })
            
        return results
