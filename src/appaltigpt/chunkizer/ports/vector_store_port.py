from typing import List, Protocol
from ..schema import QdrantChunk

class VectorStorePort(Protocol):
    async def upsert(self, chunks: List[QdrantChunk]) -> None: ...

