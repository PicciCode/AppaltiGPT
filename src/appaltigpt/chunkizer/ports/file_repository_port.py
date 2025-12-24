from typing import List, Protocol
from pathlib import Path

class FileRepositoryPort(Protocol):
    def list_files(self, folder: Path, extension: str) -> List[Path]: ...

