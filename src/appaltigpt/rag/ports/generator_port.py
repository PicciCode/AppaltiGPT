from typing import List, Protocol, Dict, Any

class RagGeneratorPort(Protocol):
    """Porta per la generazione della risposta RAG (interfaccia verso LLM)."""
    
    async def generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Genera una risposta basata sulla query e sul contesto fornito.
        """
        ...

