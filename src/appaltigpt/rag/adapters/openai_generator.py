import openai
from typing import List, Dict, Any
from ..ports.generator_port import RagGeneratorPort

class OpenAIRagGenerator(RagGeneratorPort):
    def __init__(self, client: openai.AsyncOpenAI, model: str = "gpt-4o"):
        self.client = client
        self.model = model
        
    async def generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:

        context_text = "\n\n".join([
            f"Documento (Score: {doc.get('hybrid_score', doc.get('score', 0)):.2f}):\n{doc['payload']['content']}"
            for doc in context
        ])
        
        system_prompt = """Sei un esperto assistente legale specializzato in appalti pubblici italiani.
Rispondi alla domanda dell'utente basandoti ESCLUSIVAMENTE sul contesto fornito.
Se il contesto non contiene le informazioni necessarie, dillo chiaramente.
Cita i riferimenti ai documenti quando possibile.
"""

        user_prompt = f"""
Contesto:
{context_text}

Domanda: {query}
"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0 
        )
        
        return response.choices[0].message.content or ""

