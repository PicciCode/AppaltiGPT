from typing import List, Protocol, Any, Dict

class KeywordSearchPort(Protocol):
    """Porta per la ricerca basata su parole chiave (Lexical Search / BM25)."""
    
    async def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Cerca i documenti basandosi su keyword matching.
        
        Args:
            query: La stringa di ricerca originale.
            limit: Numero massimo di risultati.
            
        Returns:
            List[Dict[str, Any]]: Lista di risultati standardizzati (id, score, payload, etc.)
        """
        ...

