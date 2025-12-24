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
            
            # Uniamo tutte le pagine
            full_text = "\n\n".join([page.markdown for page in ocr_response.pages])
            
            # Dividiamo per Headers (H1, H2, H3)
            # Regex: inizio riga, da 1 a 3 #, spazio, resto della riga
            segments = re.split(r'(?m)^(#{1,3}\s+.*$)', full_text)
            
            structured_segments = []
            
            # Gestiamo eventuale testo prima del primo header
            if segments and segments[0].strip():
                 structured_segments.append(segments[0])
            
            # Iteriamo a coppie (Header, Content)
            for i in range(1, len(segments), 2):
                header = segments[i]
                content = segments[i+1] if i+1 < len(segments) else ""
                structured_segments.append(f"{header}\n{content}")
            
            # Fallback se non ci sono header o qualcosa è andato storto
            if not structured_segments:
                return [full_text]
                
            return structured_segments

        return await asyncio.to_thread(_convert)
