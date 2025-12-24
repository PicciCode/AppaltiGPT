from typing import List, Dict, Any
from appaltigpt.embedder.ports.embedding_model_port import EmbeddingModelPort
from appaltigpt.retrieval.vector_search.ports.search_port import VectorSearchPort

class SemanticSearchService:
    def __init__(self, embedder: EmbeddingModelPort, searcher: VectorSearchPort):
        self.embedder = embedder
        self.searcher = searcher
    async def retrieve(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        vectors = await self.embedder.encode_queries([query])
        
        if not vectors:
            raise ValueError("Errore nella generazione dell'embedding per la query")
        
        query_vector = vectors[0]
        
        results = await self.searcher.search(query_vector, limit=limit)
        
        return results

