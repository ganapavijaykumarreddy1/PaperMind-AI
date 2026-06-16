from __future__ import annotations

from dataclasses import dataclass
import re
import logging
from typing import Iterable, List, Dict, Tuple, Any

import faiss
import numpy as np
from pypdf import PdfReader
from google import genai
from google.genai import types

from prompts import CHAT_PROMPT
from utils import get_embedding_model, get_max_embedding_chunks

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 6
EMBEDDING_BATCH_SIZE = 100

@dataclass
class VectorStore:
    index: faiss.IndexFlatIP
    chunks: list[str]
    pages: list[int]
    embedding_model: str
    total_chunks: int
    embedded_chunks: int
    metadata: dict  # Map of paper_title -> metadata_dict
    chunk_sources: list[dict] = None  # List of {"title": str, "page": int}

def _clean_metadata_value(value) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    cleaned = cleaned.strip("\x00")
    return cleaned or None

def _extract_year(text: str, fallback: str | None = None) -> str | None:
    match = re.search(r"\b(?:19|20)\d{2}\b", text)
    if match:
        return match.group(0)
    return fallback

def _looks_like_author_line(line: str) -> bool:
    author_indicators = (",", " and ", ";", "university", "institute", "department")
    lower_line = line.lower()
    if re.search(r"\b(abstract|introduction|keywords|doi|arxiv)\b", lower_line):
        return False
    return any(indicator in lower_line for indicator in author_indicators) or bool(
        re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z]\.)?(%s+)?" % r"[A-Z][a-z]+", line)
    )

def _extract_title_from_lines(lines: list[str]) -> str | None:
    title_lines = []
    for line in lines[:4]:
        if re.search(r"\b(abstract|keywords|introduction)\b", line, re.IGNORECASE):
            break
        if _looks_like_author_line(line) and title_lines:
            break
        title_lines.append(line)

    title = " ".join(title_lines).strip()
    return title if title else None

def extract_metadata(reader: PdfReader, pages: list[dict]) -> dict:
    """Extract lightweight paper metadata from PDF metadata and first-page text."""
    pdf_metadata = reader.metadata or {}
    first_page_text = pages[0]["raw_text"] if pages else ""
    first_lines = [
        line.strip()
        for line in first_page_text.splitlines()
        if 8 <= len(line.strip()) <= 180
    ]

    title = _clean_metadata_value(pdf_metadata.get("/Title"))
    if not title and first_lines:
        title = _extract_title_from_lines(first_lines)

    authors = _clean_metadata_value(pdf_metadata.get("/Author"))
    if not authors and first_lines:
        author_candidates = [
            line for line in first_lines[1:8]
            if _looks_like_author_line(line)
        ]
        authors = author_candidates[0] if author_candidates else None

    creation_date = _clean_metadata_value(pdf_metadata.get("/CreationDate"))
    mod_date = _clean_metadata_value(pdf_metadata.get("/ModDate"))
    year = _extract_year(
        first_page_text,
        fallback=_extract_year(creation_date or "") or _extract_year(mod_date or ""),
    )

    return {
        "title": title or "Unknown",
        "authors": authors or "Unknown",
        "year": year or "Unknown",
        "pages": len(reader.pages),
    }

def extract_pages(reader: PdfReader) -> list[dict]:
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        text = " ".join(raw_text.split())
        if text:
            pages.append({"page": page_number, "text": text, "raw_text": raw_text})
    if not pages:
        raise ValueError("No readable text found in this PDF. Scanned PDFs may need OCR.")
    return pages

def split_text(pages: list[dict], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> tuple[list[str], list[int]]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks: list[str] = []
    page_numbers: list[int] = []
    step = chunk_size - chunk_overlap

    for page in pages:
        text = page["text"]
        page_number = page["page"]

        for start in range(0, len(text), step):
            chunk = text[start : start + chunk_size].strip()
            if len(chunk) >= 80:
                chunks.append(chunk)
                page_numbers.append(page_number)

    if not chunks:
        raise ValueError("The extracted text is too short to create useful chunks.")

    return chunks, page_numbers

def _embed_texts(
    client: genai.Client,
    texts: Iterable[str],
    task_type: str,
    model: str | None = None,
) -> np.ndarray:
    import time
    embedding_model = model or get_embedding_model()
    text_list = list(texts)
    vectors = []

    for start in range(0, len(text_list), EMBEDDING_BATCH_SIZE):
        batch = text_list[start : start + EMBEDDING_BATCH_SIZE]
        
        retries = 4
        delay = 10
        for attempt in range(retries):
            try:
                result = client.models.embed_content(
                    model=embedding_model,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                vectors.extend(embedding.values for embedding in result.embeddings)
                break
            except Exception as e:
                err_str = str(e).upper()
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    if attempt < retries - 1:
                        logger.warning(f"Gemini Embedding Rate Limit (429) hit. Retrying in {delay} seconds (attempt {attempt + 1}/{retries})...")
                        time.sleep(delay)
                        delay *= 2
                        continue
                raise e

    array = np.array(vectors, dtype="float32")
    faiss.normalize_L2(array)
    return array

def build_vector_store(client: genai.Client, pdf_file) -> VectorStore:
    """Build the initial vector store for a single uploaded PDF."""
    pdf_file.seek(0)
    reader = PdfReader(pdf_file)
    pages = extract_pages(reader)
    metadata = extract_metadata(reader, pages)
    
    paper_title = metadata.get("title", "Unknown")
    chunks, page_numbers = split_text(pages)
    
    total_chunks = len(chunks)
    max_chunks = get_max_embedding_chunks()
    chunks = chunks[:max_chunks]
    page_numbers = page_numbers[:max_chunks]
    
    embeddings = _embed_texts(
        client,
        chunks,
        task_type="RETRIEVAL_DOCUMENT"
    )
    
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    chunk_sources = [{"title": paper_title, "page": page} for page in page_numbers]
    
    return VectorStore(
        index=index,
        chunks=chunks,
        pages=page_numbers,
        embedding_model=get_embedding_model(),
        total_chunks=total_chunks,
        embedded_chunks=len(chunks),
        metadata={paper_title: metadata},
        chunk_sources=chunk_sources
    )

def add_paper_to_store(client: genai.Client, vector_store: VectorStore, pdf_file) -> VectorStore:
    """Parse, embed, and append a new paper into the existing VectorStore index."""
    pdf_file.seek(0)
    reader = PdfReader(pdf_file)
    pages = extract_pages(reader)
    metadata = extract_metadata(reader, pages)
    
    paper_title = metadata.get("title", "Unknown")
    if paper_title in vector_store.metadata:
        # Avoid double indexing the same paper title
        return vector_store

    chunks, page_numbers = split_text(pages)
    total_chunks = len(chunks)
    max_chunks = get_max_embedding_chunks()
    chunks = chunks[:max_chunks]
    page_numbers = page_numbers[:max_chunks]
    
    embeddings = _embed_texts(
        client,
        chunks,
        task_type="RETRIEVAL_DOCUMENT",
        model=vector_store.embedding_model
    )
    
    # Append to existing index
    vector_store.index.add(embeddings)
    
    # Append to lists
    start_idx = len(vector_store.chunks)
    vector_store.chunks.extend(chunks)
    vector_store.pages.extend(page_numbers)
    
    if vector_store.chunk_sources is None:
        # Fallback populate for legacy chunks
        vector_store.chunk_sources = [{"title": list(vector_store.metadata.keys())[0], "page": p} for p in vector_store.pages[:start_idx]]
        
    for page in page_numbers:
        vector_store.chunk_sources.append({"title": paper_title, "page": page})
        
    vector_store.metadata[paper_title] = metadata
    vector_store.total_chunks += total_chunks
    vector_store.embedded_chunks += len(chunks)
    
    return vector_store

def retrieve_context(
    client: genai.Client,
    vector_store: VectorStore,
    question: str,
    top_k: int = TOP_K,
) -> list[dict]:
    """Retrieve top matched chunks mapped to their source paper details."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    query_embedding = _embed_texts(
        client,
        [question],
        task_type="RETRIEVAL_QUERY",
        model=vector_store.embedding_model,
    )

    safe_top_k = min(top_k, len(vector_store.chunks))
    scores, indices = vector_store.index.search(query_embedding, safe_top_k)

    results = []
    for score, index in zip(scores[0], indices[0]):
        if index == -1 or index >= len(vector_store.chunks):
            continue
        
        # Determine source
        source_title = "Unknown"
        page_num = vector_store.pages[index]
        
        if vector_store.chunk_sources and index < len(vector_store.chunk_sources):
            source = vector_store.chunk_sources[index]
            source_title = source["title"]
            page_num = source["page"]
        else:
            # Fallback
            if vector_store.metadata:
                source_title = list(vector_store.metadata.keys())[0]

        results.append({
            "text": vector_store.chunks[index],
            "page": page_num,
            "paper_title": source_title,
            "score": float(score)
        })

    return results

def chunks_to_context(chunks: list[dict]) -> str:
    """Format matching chunks as reference blocks for Gemini."""
    return "\n\n".join(
        f"[Source: page {chunk['page']}, paper \"{chunk['paper_title']}\", relevance {chunk['score']:.3f}]\n{chunk['text']}"
        for chunk in chunks
    )

def answer_question(
    client: genai.Client,
    vector_store: VectorStore,
    question: str,
    model_name: str,
) -> tuple[str, list[dict]]:
    """Query FAISS, build prompt, and call Gemini to output cited RAG answer."""
    retrieved_chunks = retrieve_context(client, vector_store, question)
    context = chunks_to_context(retrieved_chunks)
    
    answer = generate_response(
        client=client,
        model_name=model_name,
        prompt=CHAT_PROMPT.format(context=context, question=question),
    )
    
    return answer, retrieved_chunks

def generate_response(client: genai.Client, model_name: str, prompt: str) -> str:
    response = client.models.generate_content(model=model_name, contents=prompt)
    text = getattr(response, "text", None)
    if text:
        return text.strip()
    raise RuntimeError("Gemini returned an empty response.")

def generate_from_full_context(
    client: genai.Client,
    vector_store: VectorStore,
    prompt_template: str,
    model_name: str,
    focus: str | None = None
) -> str:
    """Select representative chunks from all papers and call Gemini."""
    # Group chunks by paper
    papers_chunks: dict[str, list[tuple[str, int]]] = {}
    
    # Initialize
    for title in vector_store.metadata.keys():
        papers_chunks[title] = []
        
    for idx, chunk in enumerate(vector_store.chunks):
        if vector_store.chunk_sources and idx < len(vector_store.chunk_sources):
            src = vector_store.chunk_sources[idx]
            title = src["title"]
            page = src["page"]
        else:
            title = list(vector_store.metadata.keys())[0]
            page = vector_store.pages[idx]
            
        if title in papers_chunks:
            papers_chunks[title].append((chunk, page))
            
    # For each paper, collect the first 8 chunks (introductory text)
    selected_context_blocks = []
    for title, chunk_list in papers_chunks.items():
        for chunk, page in chunk_list[:8]:
            selected_context_blocks.append(f"[Source: page {page}, paper \"{title}\"]\n{chunk}")
            
    context = "\n\n".join(selected_context_blocks)
    
    if focus is not None:
        # literature review format
        prompt = prompt_template.format(context=context, focus_guidelines=focus)
    else:
        # general summary or gap analysis format
        prompt = prompt_template.format(context=context)
        
    return generate_response(
        client=client,
        model_name=model_name,
        prompt=prompt,
    )
