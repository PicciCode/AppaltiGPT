from openai import AsyncOpenAI
from ..ports.generator_port import HypotheticalDocumentGeneratorPort

class OpenAIHypotheticalGenerator(HypotheticalDocumentGeneratorPort):
    def __init__(self,client: AsyncOpenAI | None = None, model: str = "gpt-5.2"):
        self.client = client if client else AsyncOpenAI()
        self.model = model
        self.prompt_template = """Scrivi un passaggio di testo che risponda alla seguente domanda.
Il passaggio deve essere plausibile e contenere le parole chiave e i concetti tecnici rilevanti, anche se le informazioni specifiche sono inventate.
Questo testo verrà utilizzato per la ricerca semantica.

Domanda: {query}
Passaggio:"""

    async def generate(self, query: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Sei un assistente esperto in appalti pubblici. Il tuo compito è generare risposte ipotetiche per migliorare la ricerca semantica."},
                {"role": "user", "content": self.prompt_template.format(query=query)}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content or ""

