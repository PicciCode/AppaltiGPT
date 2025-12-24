from typing import List, Protocol
from pathlib import Path

class DocumentConverterPort(Protocol):
    async def convert(self, file_path: Path) -> List[str]: ...

