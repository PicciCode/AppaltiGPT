from typing import List, Dict, Any
from ...embedder.ports.embedding_model_port import EmbeddingModelPort
from .ports.generator_port import HypotheticalDocumentGeneratorPort
from appaltigpt.retrieval.vector_search.ports.search_port import VectorSearchPort

class HyDERetrieverService:
    
    def __init__(
        self, 
        generator: HypotheticalDocumentGeneratorPort,
        embedder: EmbeddingModelPort,
        searcher: VectorSearchPort
    ):
        self.generator = generator
        self.embedder = embedder
        self.searcher = searcher

    async def retrieve(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        hypothetical_doc = await self.generator.generate(query)
        
        vectors = await self.embedder.encode_documents([hypothetical_doc])
        if not vectors:
            raise ValueError("Errore nella generazione dell'embedding per il documento ipotetico")
        
        query_vector = vectors[0]
        
        results = await self.searcher.search(query_vector, limit=limit)
        
        return results

    async def retrieve_with_explanation(self, query: str, limit: int = 5) -> Dict[str, Any]:
        hypothetical_doc = await self.generator.generate(query)
        vectors = await self.embedder.encode_documents([hypothetical_doc])
        query_vector = vectors[0]
        results = await self.searcher.search(query_vector, limit=limit)
        
        return {
            "results": results,
            "hypothetical_document": hypothetical_doc
        }

