from typing import List, Protocol, Any, Dict

class VectorSearchPort(Protocol):
    
    async def search(self, vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        ...

