from __future__ import annotations

import os
from html import escape
from io import BytesIO
from typing import Optional

from google import genai
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
import streamlit as st


DEFAULT_CHAT_MODEL = "gemini-2.5-flash"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_MAX_EMBEDDING_CHUNKS = 90


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read configuration from Streamlit secrets first, then environment."""
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


def configure_gemini() -> tuple[genai.Client, str]:
    """Create a Gemini client and return it with the selected chat model name."""
    api_key = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Gemini API key is missing. Add GEMINI_API_KEY to Streamlit secrets "
            "or your local .env file."
        )

    client = genai.Client(api_key=api_key)
    model_name = get_secret("GEMINI_MODEL", DEFAULT_CHAT_MODEL) or DEFAULT_CHAT_MODEL
    return client, model_name


def get_embedding_model() -> str:
    return get_secret("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL) or DEFAULT_EMBEDDING_MODEL


def get_max_embedding_chunks() -> int:
    value = get_secret("MAX_EMBEDDING_CHUNKS", str(DEFAULT_MAX_EMBEDDING_CHUNKS))
    try:
        return max(1, int(value or DEFAULT_MAX_EMBEDDING_CHUNKS))
    except ValueError:
        return DEFAULT_MAX_EMBEDDING_CHUNKS


def format_source_list(sources: list[dict]) -> str:
    if not sources:
        return "No source chunks available."

    lines = []
    for index, source in enumerate(sources, start=1):
        page = source.get("page", "unknown")
        score = source.get("score")
        score_text = f", score: {score:.3f}" if isinstance(score, float) else ""
        lines.append(f"{index}. Page {page}{score_text}")
    return "\n".join(lines)


def format_source_cards(sources: list[dict]) -> list[str]:
    """Create short source previews suitable for Streamlit captions."""
    cards = []
    for source in sources:
        page = source.get("page", "unknown")
        score = source.get("score")
        text = str(source.get("text") or "")
        preview = text[:280] + ("..." if len(text) > 280 else "")
        score_text = f" · relevance {score:.3f}" if isinstance(score, float) else ""
        cards.append(f"Page {page}{score_text}\n\n{preview}")
    return cards


def safe_file_name(file_name: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in file_name)


def build_markdown_export(title: str, metadata: dict, content: str) -> bytes:
    markdown = [
        f"# {title}",
        "",
        "## Paper Metadata",
        f"- **Title:** {metadata.get('title') or 'Unknown'}",
        f"- **Authors:** {metadata.get('authors') or 'Unknown'}",
        f"- **Year:** {metadata.get('year') or 'Unknown'}",
        f"- **Pages:** {metadata.get('pages') or 'Unknown'}",
        "",
        "## Generated Content",
        content,
        "",
    ]
    return "\n".join(markdown).encode("utf-8")


def build_pdf_export(title: str, metadata: dict, content: str) -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, title=title)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(escape(title), styles["Title"]),
        Spacer(1, 12),
        Paragraph("<b>Paper Metadata</b>", styles["Heading2"]),
        Paragraph(f"<b>Title:</b> {escape(metadata.get('title') or 'Unknown')}", styles["BodyText"]),
        Paragraph(f"<b>Authors:</b> {escape(metadata.get('authors') or 'Unknown')}", styles["BodyText"]),
        Paragraph(f"<b>Year:</b> {escape(metadata.get('year') or 'Unknown')}", styles["BodyText"]),
        Paragraph(f"<b>Pages:</b> {escape(str(metadata.get('pages') or 'Unknown'))}", styles["BodyText"]),
        Spacer(1, 12),
    ]

    for line in content.splitlines():
        clean_line = line.strip()
        if not clean_line:
            story.append(Spacer(1, 8))
        elif clean_line.startswith("#"):
            story.append(Paragraph(escape(clean_line.lstrip("#").strip()), styles["Heading2"]))
        elif clean_line.startswith(("- ", "* ")):
            story.append(Paragraph(f"&bull; {escape(clean_line[2:])}", styles["BodyText"]))
        else:
            story.append(Paragraph(escape(clean_line.replace("|", " | ")), styles["BodyText"]))

    document.build(story)
    return buffer.getvalue()
