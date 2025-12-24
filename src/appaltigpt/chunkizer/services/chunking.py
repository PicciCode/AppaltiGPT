import asyncio
import uuid
from pathlib import Path
from typing import Dict, Optional, List
from appaltigpt.chunkizer.ports.file_repository_port import FileRepositoryPort
from appaltigpt.chunkizer.ports.ai_client_port import AIClientPort
from appaltigpt.chunkizer.ports.document_converter_port import DocumentConverterPort
from appaltigpt.chunkizer.ports.vector_store_port import VectorStorePort
from appaltigpt.chunkizer.schema import RagDocument, RagAnalysis, QdrantChunk
from appaltigpt.chunkizer.prompt import PROMPT_TEMPLATE

class ChunkingService:
    def __init__(self, 
                 file_repo: FileRepositoryPort, 
                 ai_client: AIClientPort, 
                 converter: Optional[DocumentConverterPort] = None,
                 vector_store: Optional[VectorStorePort] = None):
        self.file_repo = file_repo
        self.ai_client = ai_client
        self.converter = converter
        self.vector_store = vector_store
        self.batch_size = 15 

    def _to_qdrant_chunks(self, doc: RagDocument, filename: str) -> List[QdrantChunk]:
        qdrant_chunks = []
        for chunk in doc.chunks:
            
            namespace = uuid.uuid4()
            chunk_key = (
                f"{filename}|"
                f"{chunk.chunk_id}|"
                f"{chunk.pages.start}-{chunk.pages.end}|"
                f"{hash(chunk.markdown_content)}"
            )

            chunk_uuid = uuid.uuid5(namespace, chunk_key)
            
            q_chunk = QdrantChunk(
                id=chunk_uuid,
                document_id=filename,
                document_title=doc.document_title,
                source=filename,
                language=doc.language,
                
                chunk_id=chunk.chunk_id,
                chapter_title=chunk.chapter_title,
                chapter_level=chunk.chapter_level,
                
                page_start=chunk.pages.start,
                page_end=chunk.pages.end,
                
                content=chunk.markdown_content,
                
                summary=chunk.summary,
                keywords=chunk.keywords,
                entities=chunk.entities,
                
                topic=chunk.rag_notes.topic,
                use_cases=chunk.rag_notes.use_cases,
                retrieval_hints=chunk.rag_notes.retrieval_hints
            )
            qdrant_chunks.append(q_chunk)
        return qdrant_chunks

    async def process_documents(self, folder: Path) -> Dict[str, RagDocument]:
        """
        Processes all PDF documents in the given folder:
        1. Lists files
        2. Converts to Markdown Pages (Mistral)
        3. Batches pages and Analyzes/Chunks them (OpenAI)
        4. Merges results
        5. (Optional) Indexes chunks into Vector Store
        
        Returns a dictionary mapping filename to RagDocument
        """

        files = self.file_repo.list_files(folder, '.pdf')
        if not files:
            return {}
        
        if self.converter:

            print("Converting documents...")
            conversion_tasks = [self.converter.convert(f) for f in files]
            documents_pages = await asyncio.gather(*conversion_tasks)
            
            results = []
            
            for file, pages in zip(files, documents_pages,strict=True):
                print(f"Analyzing {file.name} ({len(pages)} pages)...")
                
                analysis_tasks = []
                
                for i in range(0, len(pages), self.batch_size):
                    batch = pages[i : i + self.batch_size]
                    start_page = i + 1
                    end_page = i + len(batch)
                    
                    batch_text = "\n\n".join(batch)
                    
                    batch_prompt = (
                        f"CONTEXT INFO:\n"
                        f"This content segment represents pages {start_page} to {end_page} of the original document.\n"
                        f"When assigning page numbers to chunks, strict adherence to this range is required.\n"
                        f"--------------------------------------------------\n\n"
                        f"{PROMPT_TEMPLATE}"
                    )
                    
                    analysis_tasks.append(
                        self.ai_client.analyze_text(batch_text, batch_prompt)
                    )
                
                batch_analyses: List[RagAnalysis] = await asyncio.gather(*analysis_tasks)
                
                merged_chunks = []
                doc_title = batch_analyses[0].document_title if batch_analyses else "Unknown"
                doc_lang = batch_analyses[0].language if batch_analyses else "it"
                
                for analysis in batch_analyses:
                    merged_chunks.extend(analysis.chunks)
                
                doc = RagDocument(
                    document_title=doc_title,
                    language=doc_lang,
                    chunks=merged_chunks,
                    markdown_document="\n\n".join(pages)
                )
                
                if self.vector_store:
                    print(f"Indexing {file.name} to Vector Store...")
                    q_chunks = self._to_qdrant_chunks(doc, file.name)
                    print(f"Number of chunks: {len(q_chunks)}")
                    await self.vector_store.upsert(q_chunks)
                    
                results.append(doc)
            
        else:

            print("Uploading documents...")
            upload_tasks = [self.ai_client.upload_file(f) for f in files]
            file_ids = await asyncio.gather(*upload_tasks)
            
            print("Analyzing documents...")
            analysis_tasks = []
            for fid in file_ids:
                analysis_tasks.append(
                    self.ai_client.analyze_document(fid, PROMPT_TEMPLATE)
                )
            results = await asyncio.gather(*analysis_tasks)
            

        
        return {
            f.name: result 
            for f, result in zip(files, results,strict=True)
        }
