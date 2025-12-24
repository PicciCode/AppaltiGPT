import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import openai
    from dotenv import load_dotenv
    import os
    import asyncio

    load_dotenv()

    client = openai.AsyncOpenAI()
    return client, os


@app.cell
def _(client, os):
    from appaltigpt.chunkizer.adapters import (
        LocalFileRepository,
        OpenAIClient,
        MistralDocumentConverter,
    )
    from appaltigpt.chunkizer.services import ChunkingService
    from root_folders import DOCS_FOLDER

    from appaltigpt.chunkizer.adapters.qdrant_vector_store import QdrantVectorStore
    from qdrant_client import QdrantClient
    from appaltigpt.chunkizer.adapters.openai_embedding_model import OpenAIEmbeddingModel

    embedding_model=OpenAIEmbeddingModel()
    qclient = QdrantClient(host="localhost", port=6333)
    vector_store = QdrantVectorStore(client=qclient, embedding_model=embedding_model)

    file_repo = LocalFileRepository()
    ai_client = OpenAIClient(client)

    # Initialize Mistral Converter
    mistral_api_key = os.environ.get("MISTRAL_API_KEY")
    if not mistral_api_key:
        raise ValueError("MISTRAL_API_KEY not found in environment")

    converter = MistralDocumentConverter(api_key=mistral_api_key)

    service = ChunkingService(file_repo, ai_client, converter=converter,vector_store=vector_store)
    return (converter,)


@app.cell
async def _(converter):
    conv= await converter.convert('/Users/carlo/Desktop/SideQuests/AppaltiGPT/docs/CAPITOLATO_TECNICO_RETTIFICATO.pdf')
    return (conv,)


@app.cell
def _(conv):
    conv
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
