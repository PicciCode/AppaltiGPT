import os
from pathlib import Path
from typing import List
from ..ports.file_repository_port import FileRepositoryPort

class LocalFileRepository(FileRepositoryPort):
    def list_files(self, folder: Path, extension: str) -> List[Path]:
        return [
            folder / f 
            for f in os.listdir(folder) 
            if f.endswith(extension)
        ]

