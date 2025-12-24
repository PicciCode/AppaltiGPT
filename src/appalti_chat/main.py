import streamlit as st
import asyncio
from appalti_chat.dependencies import (
    get_retriever_service, 
    get_openai_client, 
    get_settings, 
    get_qdrant_client, 
    bootstrap_database
)
from appaltigpt.rag.adapters.openai_generator import OpenAIRagGenerator
from appaltigpt.rag.service import RagService

st.set_page_config(page_title="AppaltiGPT Chat", layout="wide")

st.title("🤖 AppaltiGPT Chat")

# --- BOOTSTRAP DATABASE ---
# Verifica se la collezione esiste e popola se necessario
settings = get_settings()
qdrant = get_qdrant_client(url="http://localhost:6333")
openai_client_bs = get_openai_client(settings.openai_api_key)

# Esegui bootstrap (check veloce se esiste, ingestion lunga se non esiste)
if "db_checked" not in st.session_state:
    with st.spinner("Verifica integrità Database e Documenti..."):
        asyncio.run(bootstrap_database(settings, qdrant, openai_client_bs))
    st.session_state.db_checked = True


# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Fai una domanda sugli appalti..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Setup Services (Lazy Load)
        retriever = get_retriever_service()
        # Settings e Client sono già caricati sopra
        
        generator = OpenAIRagGenerator(openai_client_bs, model="gpt-4o")
        rag_service = RagService(retriever, generator)
        
        async def run_rag():
            result = await rag_service.answer(prompt)
            return result

        with st.spinner("Analisi dei documenti in corso..."):
            # Run async loop
            result = asyncio.run(run_rag())
        
        answer = result["answer"]
        sources = result["sources"]
        
        # Display Answer
        message_placeholder.markdown(answer)
        
        # Display Sources (Expandable)
        with st.expander("Fonti Consultate"):
            for idx, doc in enumerate(sources):
                score = doc.get('hybrid_score', doc.get('score', 0))
                st.markdown(f"**Documento {idx+1} (Rilevanza: {score:.2f})**")
                st.caption(f"File: {doc['payload'].get('filename', 'N/A')}")
                st.text(doc['payload']['content'][:300] + "...")
                st.divider()

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
