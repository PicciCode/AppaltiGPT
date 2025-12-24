from typing import List, Dict, Any
from .ports.generator_port import RagGeneratorPort



class RagService:
    def __init__(self, retriever, generator: RagGeneratorPort):
        self.retriever = retriever
        self.generator = generator
        
    async def answer(self, query: str) -> Dict[str, Any]:
        """
        Esegue la pipeline RAG completa: Retrieve -> Generate.
        Restituisce la risposta e le fonti.
        """

        docs = await self.retriever.retrieve(query, limit=5)
        

        answer_text = await self.generator.generate_response(query, docs)
        
        return {
            "answer": answer_text,
            "sources": docs
        }

