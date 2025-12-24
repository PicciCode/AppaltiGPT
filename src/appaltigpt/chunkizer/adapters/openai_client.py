from pathlib import Path
import openai
from ..ports.ai_client_port import AIClientPort
from ..schema import RagDocument, RagAnalysis

class OpenAIClient(AIClientPort):
    def __init__(self, client: openai.AsyncOpenAI, model: str = "gpt-5.2"):
        self.client = client
        self.model = model

    async def upload_file(self, file_path: Path) -> str:
        with open(file_path, 'rb') as f:
            file = await self.client.files.create(
                file=f,
                purpose='assistants'
            )
        return file.id

    async def analyze_document(self, file_id: str, prompt: str) -> RagDocument:

        response = await self.client.responses.parse(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_file", "file_id": file_id}
                    ]
                }
            ],
            text_format=RagDocument,
        )
        return response.output_parsed

    async def analyze_text(self, text: str, prompt: str) -> RagAnalysis:
        response = await self.client.responses.parse(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"{prompt}\n\nDOCUMENT CONTENT PART:\n{text}"}
                    ]
                }
            ],
            text_format=RagAnalysis,
        )
        return response.output_parsed

