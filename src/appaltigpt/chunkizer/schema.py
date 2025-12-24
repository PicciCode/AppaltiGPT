from pydantic import BaseModel, Field
import uuid
from typing import List


class Pages(BaseModel):
    start: int = Field(..., ge=1, description="Starting page (inclusive)")
    end: int = Field(..., ge=1, description="Ending page (inclusive)")


class RagNotes(BaseModel):
    topic: str = Field(..., description="High-level topic of the chunk")
    use_cases: List[str] = Field(
        ..., description="Typical RAG query intents for this chunk"
    )
    retrieval_hints: List[str] = Field(
        ..., description="Terms or phrases useful for semantic retrieval"
    )


class Chunk(BaseModel):
    chunk_id: str = Field(..., description="Stable chunk identifier (e.g., ch_01)")
    chapter_title: str = Field(..., description="Title of the chapter or section")
    chapter_level: int = Field(
        ..., ge=1, le=6, description="Markdown heading level (# = 1, ## = 2, etc.)"
    )
    pages: Pages
    markdown_content: str = Field(
        ..., description="Markdown content of the chapter only"
    )
    keywords: List[str] = Field(
        ..., description="Lowercase keywords, no duplicates"
    )
    entities: List[str] = Field(
        ..., description="Named entities or formal identifiers"
    )
    summary: str = Field(
        ..., description="Concise summary (max 3–4 sentences)"
    )
    rag_notes: RagNotes


class RagAnalysis(BaseModel):
    """Output from the LLM analysis"""
    document_title: str = Field(
        default="Unknown Document Title", description="Inferred title of the document"
    )
    
    language: str = Field(
        default="it", description="Document language (e.g., Italian, English)"
    )
    
    chunks: List[Chunk] = Field(
        ..., description="List of chapter-based chunks"
    )


class RagDocument(RagAnalysis):
    """Full Document model combining analysis and original content"""
    markdown_document: str = Field(
        ..., description="Full reconstructed document in Markdown"
    )

    
class QdrantChunk(BaseModel):
    id: uuid.UUID  

    document_id: str          
    document_title: str
    source: str               
    language: str

    chunk_id: str
    chapter_title: str
    chapter_level: int

    page_start: int
    page_end: int

    content: str

    summary: str
    keywords: List[str]
    entities: List[str]

    topic: str
    use_cases: List[str]
    retrieval_hints: List[str]
