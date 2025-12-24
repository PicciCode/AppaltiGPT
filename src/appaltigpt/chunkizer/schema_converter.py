from typing import List
import uuid
from appaltigpt.chunkizer.schema import RagDocument,QdrantChunk


def rag_document_to_qdrant_chunks(
    rag_document: RagDocument,
    document_id: str,
    source: str,
) -> List[QdrantChunk]:
    chunks = []

    for chunk in rag_document.chunks:
        chunks.append(
            QdrantChunk(
                id=str(uuid.uuid4()),

                document_id=document_id,
                document_title=rag_document.document_title,
                source=source,
                language=rag_document.language,

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
                retrieval_hints=chunk.rag_notes.retrieval_hints,
            )
        )

    return chunks