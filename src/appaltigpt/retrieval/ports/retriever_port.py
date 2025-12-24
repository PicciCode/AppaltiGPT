from typing import List, Protocol, Dict, Any

class RetrieverPort(Protocol):
    
    async def retrieve(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:...

