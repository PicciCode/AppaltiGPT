from typing import Protocol

class HypotheticalDocumentGeneratorPort(Protocol):
    
    async def generate(self, query: str) -> str: ...

