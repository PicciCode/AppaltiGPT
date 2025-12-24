import base64
import asyncio
import re
from pathlib import Path
from typing import List, Union
from mistralai import Mistral
from ..ports.document_converter_port import DocumentConverterPort

class MistralDocumentConverter(DocumentConverterPort):
    def __init__(self, client_or_key: Union[Mistral, str]):
        if isinstance(client_or_key, str):
            self.client = Mistral(api_key=client_or_key)
        else:
            self.client = client_or_key

    async def convert(self, file_path: Path) -> List[str]:
        def _convert():
            with open(file_path, "rb") as pdf_file:
                base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
            
            ocr_response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{base64_pdf}"
                },
                table_format="markdown",
                include_image_base64=False 
            )
            
            full_text = "\n\n".join([page.markdown for page in ocr_response.pages])
            
            segments = re.split(r'(?=^#{1,2}\s)', full_text, flags=re.MULTILINE)
            
            return [s.strip() for s in segments if s.strip()]

        return await asyncio.to_thread(_convert)
