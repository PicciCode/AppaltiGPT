import asyncio
from pathlib import Path
import streamlit as st
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from settings import Settings
from mistralai import Mistral

# Core Imports for Retrieval
from appaltigpt.embedder.adapters.openai_embedding_model import OpenAIEmbeddingModel
from appaltigpt.retrieval.hybrid.service import HybridRetrieverService
from appaltigpt.retrieval.semantic.service import SemanticSearchService
from appaltigpt.retrieval.HyDE.service import HyDERetrieverService
from appaltigpt.retrieval.HyDE.adapters.openai_generator import OpenAIHypotheticalGenerator
from appaltigpt.vector_store.qdrant.reader import QdrantVectorReader
from appaltigpt.vector_store.qdrant.keyword import QdrantKeywordReader

# Core Imports for Ingestion (Chunking)
from appaltigpt.chunkizer.services.chunking import ChunkingService
from appaltigpt.chunkizer.adapters.local_file_repository import LocalFileRepository
from appaltigpt.chunkizer.adapters.openai_client import OpenAIClient as OpenAIChunkingClient
from appaltigpt.chunkizer.adapters.mistral_document_converter import MistralDocumentConverter
from appaltigpt.vector_store.qdrant.writer import QdrantVectorWriter


@st.cache_resource
def get_settings() -> Settings:
    return Settings()

@st.cache_resource
def get_qdrant_client(url: str, api_key: str = None) -> QdrantClient:
    return QdrantClient(url=url, api_key=api_key)

@st.cache_resource
def get_openai_client(api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=api_key)

async def bootstrap_database(settings: Settings, qdrant_client: QdrantClient, openai_client: AsyncOpenAI):

    collection_name = settings.collection_name
    
    # Check esistenza collezione
    exists = await asyncio.to_thread(qdrant_client.collection_exists, collection_name)
    if not exists:
        st.warning(f"La collezione '{collection_name}' non esiste. Inizio il processo di ingestione documenti...")
        
        # Setup Ingestion Pipeline
        # 1. File Repo
        # Assumiamo che la cartella docs sia nella root del progetto
        docs_path = Path("docs") 
        file_repo = LocalFileRepository()
        
        # 2. AI Client (Wrapper per Chunking)
        ai_client_adapter = OpenAIChunkingClient(openai_client)
        
        # 3. Converter (Mistral)
        mistral_client = Mistral(api_key=settings.mistral_api_key)
        converter = MistralDocumentConverter(mistral_client)
        
        # 4. Vector Writer
        # Serve l'embedder model anche per il writer
        embedder = OpenAIEmbeddingModel(
            client=openai_client,
            model=settings.embedding_model_openai,
            model_size=settings.embedding_model_size_openai
        )
        vector_writer = QdrantVectorWriter(qdrant_client, settings, embedder)
        
        # 5. Service
        chunking_service = ChunkingService(
            file_repo=file_repo,
            ai_client=ai_client_adapter,
            converter=converter,
            vector_store=vector_writer
        )
        
        # Esegui Ingestione
        with st.spinner("Indicizzazione documenti in corso (potrebbe richiedere alcuni minuti)..."):
            await chunking_service.process_documents(docs_path)
            
        st.success("Ingestione completata! Database pronto.")
    else:
        # Potremmo controllare se è vuota con qdrant_client.count(collection_name).count == 0
        pass

@st.cache_resource
def get_retriever_service() -> HybridRetrieverService:
    """
    Costruisce e restituisce il servizio di retrieval completo.
    Combina 3 strategie (Tri-Hybrid):
    1. Semantic Search (Embedding standard della query)
    2. HyDE (Embedding del documento ipotetico)
    3. Keyword Search (Filtro testuale / BM25)
    
    Usa Reciprocal Rank Fusion (RRF) per unire i risultati.
    """
    settings = get_settings()
    
    # 1. Clients
    qdrant_client = get_qdrant_client(url="http://localhost:6333") 
    openai_client = get_openai_client(api_key=settings.openai_api_key)
    
    # --- SETUP SERVICES ---
    
    # 2. Adapters & Base Components
    embedder = OpenAIEmbeddingModel(
        client=openai_client, 
        model=settings.embedding_model_openai,
        model_size=settings.embedding_model_size_openai
    )
    
    vector_reader = QdrantVectorReader(
        client=qdrant_client,
        settings=settings
    )
    
    keyword_reader = QdrantKeywordReader(
        client=qdrant_client,
        settings=settings
    )
    
    hyde_generator = OpenAIHypotheticalGenerator(
        client=openai_client,
        model="gpt-4o"
    )
    
    # 3. Strategie di Retrieval Singole
    
    # A. Semantic Standard
    semantic_service = SemanticSearchService(
        embedder=embedder, 
        searcher=vector_reader
    )
    
    # B. HyDE
    hyde_service = HyDERetrieverService(
        generator=hyde_generator,
        embedder=embedder,
        searcher=vector_reader
    )
    
    # C. Keyword (è già keyword_reader)
    
    # 4. Hybrid Service (Fusione a 3 vie)
    hybrid_service = HybridRetrieverService(
        retrievers=[semantic_service, hyde_service, keyword_reader],
        rrf_k=60
    )
    
    return hybrid_service
