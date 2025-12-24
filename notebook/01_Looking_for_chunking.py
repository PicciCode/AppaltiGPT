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
    from appaltigpt.embedder.adapters.openai_embedding_model import OpenAIEmbeddingModel

    embedding_model=OpenAIEmbeddingModel()
    qclient = QdrantClient(host="localhost", port=6333)
    vector_store = QdrantVectorStore(client=qclient, embedding_model=embedding_model)

    file_repo = LocalFileRepository()
    ai_client = OpenAIClient(client)


    mistral_api_key = os.environ.get("MISTRAL_API_KEY")
    if not mistral_api_key:
        raise ValueError("MISTRAL_API_KEY not found in environment")

    converter = MistralDocumentConverter(api_key=mistral_api_key)

    service = ChunkingService(file_repo, ai_client, converter=converter,vector_store=vector_store)
    return DOCS_FOLDER, embedding_model, service, vector_store


@app.cell
async def _(embedding_model):
    res = await embedding_model.encode_documents('ciao')
    return


@app.cell
async def _(DOCS_FOLDER, service):
    results = await service.process_documents(DOCS_FOLDER)
    return (results,)


@app.cell
def _(results, service):
    from appaltigpt.chunkizer.schema_converter import rag_document_to_qdrant_chunks
    from uuid import uuid4

    qdrant_chunks=[]

    for k,v in results.items():
        id=str(uuid4())
        qdrant_chunks.extend(service._to_qdrant_chunks(doc=v,filename=k))
    return (qdrant_chunks,)


@app.cell
async def _(embedding_model, qdrant_chunks):
    embeddings= await  embedding_model.encode_documents(texts=[c.content for c in qdrant_chunks])
    return (embeddings,)


@app.cell
def _(qdrant_chunks):
    ids = [p.id for p in qdrant_chunks]
    print(len(ids), len(set(ids)))
    return


@app.cell
def _(embeddings, qdrant_chunks):
    from qdrant_client.models import PointStruct, VectorParams, Distance

    points=[]
    for chunk, vector in zip(qdrant_chunks, embeddings,strict=True):
        points.append(PointStruct(
            id=chunk.id,
            vector=vector,
            payload=chunk.model_dump()
        ))
    return (points,)


@app.cell
def _(points):
    points[0]
    return


@app.cell
def _(points, vector_store):
    vector_store.client.upsert(collection_name=vector_store.collection_name,points=points)
    return


@app.cell
def _(vector_store):
    vector_store.client.get_collection(vector_store.collection_name)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
