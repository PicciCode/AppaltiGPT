from appaltigpt.chunkizer.schema import RagAnalysis
import json

RAG_ANALYSIS_SCHEMA = RagAnalysis.model_json_schema()
RAG_ANALYSIS_SCHEMA = json.dumps(RAG_ANALYSIS_SCHEMA)



PROMPT_TEMPLATE=f"""You are an expert document parser and RAG-oriented content engineer.

TASK:
Analyze the document segment provided below.
Split it into coherent chunks based on chapters or major sections, and return a structured JSON output optimized for Retrieval-Augmented Generation (RAG).

INPUT:
A segment of the document content is provided as text/markdown below.
The segment corresponds to a specific range of pages (specified in the context).

REQUIREMENTS:
- Do NOT invent or hallucinate content.
- Preserve original meaning, terminology, and structure.
- Identify chapters or major logical sections using headings and semantic structure.
- Create exactly one chunk per chapter or major section found in this segment.
- If a chapter spans across the end of this segment, chunk what is available.
- IMPORTANT: Use the absolute page numbers provided in the context for the "pages" field.
- Prefer semantic coherence over chunk size.
- Each chunk must be self-contained.
- Extract meaningful keywords (technical terms, domain concepts).
- Extract named entities when relevant (e.g., standards, organizations, methods, tools).
- Produce concise summaries (maximum 3–4 sentences).
- Output MUST be valid JSON.
- Output MUST strictly follow the schema below.
- Do NOT include explanations, comments, or text outside the JSON object.

OUTPUT SCHEMA (STRICT):

{RAG_ANALYSIS_SCHEMA}
"""