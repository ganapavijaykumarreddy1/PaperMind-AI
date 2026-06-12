from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import faiss
import numpy as np
from pypdf import PdfReader
from google import genai
from google.genai import types

from prompts import CHAT_PROMPT
from utils import get_embedding_model, get_max_embedding_chunks


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
EMBEDDING_BATCH_SIZE = 100


@dataclass
class RetrievedChunk:
    text: str
    page: int
    score: float


@dataclass
class VectorStore:
    index: faiss.IndexFlatIP
    chunks: list[str]
    pages: list[int]
    embedding_model: str
    total_chunks: int
    embedded_chunks: int
    metadata: dict


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
        re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+)\b", line)
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


def extract_text(pdf_file) -> list[dict]:
    """Extract page-wise text from an uploaded PDF file."""
    return extract_pages(PdfReader(pdf_file))


def split_text(pages: list[dict], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> tuple[list[str], list[int]]:
    """Split extracted PDF text into overlapping chunks."""
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
    embedding_model = model or get_embedding_model()
    text_list = list(texts)
    vectors = []

    for start in range(0, len(text_list), EMBEDDING_BATCH_SIZE):
        batch = text_list[start : start + EMBEDDING_BATCH_SIZE]
        result = client.models.embed_content(
            model=embedding_model,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        vectors.extend(embedding.values for embedding in result.embeddings)

    array = np.array(vectors, dtype="float32")
    faiss.normalize_L2(array)
    return array


def create_vector_store(client: genai.Client, chunks: list[str], page_numbers: list[int]) -> VectorStore:
    """Create a FAISS vector store from text chunks."""
    embedding_model = get_embedding_model()
    embeddings = _embed_texts(
        client,
        chunks,
        task_type="RETRIEVAL_DOCUMENT",
        model=embedding_model,
    )
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return VectorStore(
        index=index,
        chunks=chunks,
        pages=page_numbers,
        embedding_model=embedding_model,
        total_chunks=len(chunks),
        embedded_chunks=len(chunks),
        metadata={},
    )


def build_vector_store(client: genai.Client, pdf_file) -> VectorStore:
    pdf_file.seek(0)
    reader = PdfReader(pdf_file)
    pages = extract_pages(reader)
    metadata = extract_metadata(reader, pages)
    chunks, page_numbers = split_text(pages)
    total_chunks = len(chunks)
    max_chunks = get_max_embedding_chunks()
    chunks = chunks[:max_chunks]
    page_numbers = page_numbers[:max_chunks]
    vector_store = create_vector_store(client, chunks, page_numbers)
    vector_store.total_chunks = total_chunks
    vector_store.embedded_chunks = len(chunks)
    vector_store.metadata = metadata
    return vector_store


def retrieve_context(
    client: genai.Client,
    vector_store: VectorStore,
    question: str,
    top_k: int = TOP_K,
) -> list[RetrievedChunk]:
    """Retrieve the most relevant chunks for a user question."""
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
        if index == -1:
            continue
        results.append(
            RetrievedChunk(
                text=vector_store.chunks[index],
                page=vector_store.pages[index],
                score=float(score),
            )
        )

    return results


def chunks_to_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[Source: page {chunk.page}, relevance {chunk.score:.3f}]\n{chunk.text}"
        for chunk in chunks
    )


def _format_page_references(chunks: list[RetrievedChunk]) -> str:
    pages = sorted({chunk.page for chunk in chunks})
    if not pages:
        return ""
    return ", ".join(f"p. {page}" for page in pages)


def add_citation_footer(answer: str, chunks: list[RetrievedChunk]) -> str:
    """Ensure every answer exposes the retrieved page references."""
    page_references = _format_page_references(chunks)
    if not page_references:
        return answer

    if "Evidence pages:" in answer:
        return answer

    return f"{answer}\n\n**Evidence pages:** {page_references}"


def generate_response(client: genai.Client, model_name: str, prompt: str) -> str:
    response = client.models.generate_content(model=model_name, contents=prompt)
    text = getattr(response, "text", None)

    if text:
        return text.strip()

    raise RuntimeError("Gemini returned an empty response. Try again with a shorter prompt.")


def answer_question(
    client: genai.Client,
    vector_store: VectorStore,
    question: str,
    model_name: str,
) -> tuple[str, list[dict]]:
    retrieved_chunks = retrieve_context(client, vector_store, question)
    context = chunks_to_context(retrieved_chunks)
    answer = generate_response(
        client=client,
        model_name=model_name,
        prompt=CHAT_PROMPT.format(context=context, question=question),
    )
    answer = add_citation_footer(answer, retrieved_chunks)
    sources = [
        {"page": chunk.page, "score": chunk.score, "text": chunk.text}
        for chunk in retrieved_chunks
    ]
    return answer, sources


def generate_from_full_context(
    client: genai.Client,
    vector_store: VectorStore,
    prompt_template: str,
    model_name: str,
) -> str:
    """Generate feature outputs using broad paper context within MVP limits."""
    selected_chunks = zip(vector_store.chunks[:24], vector_store.pages[:24])
    context = "\n\n".join(
        f"[Source: page {page}]\n{chunk}"
        for chunk, page in selected_chunks
    )
    return generate_response(
        client=client,
        model_name=model_name,
        prompt=prompt_template.format(context=context),
    )
