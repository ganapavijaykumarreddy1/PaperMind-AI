from __future__ import annotations

import os
from html import escape
from io import BytesIO
from typing import Optional, Any

from google import genai
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
import streamlit as st

DEFAULT_CHAT_MODEL = "gemini-2.5-flash"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_MAX_EMBEDDING_CHUNKS = 80  # Capped at 80 to fit within free-tier rate limits (100 RPM)

def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)

def configure_gemini() -> tuple[genai.Client, str]:
    api_key = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Gemini API key is missing. Add GEMINI_API_KEY to your local .env file."
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
        paper = source.get("paper_title", "Unknown")
        score = source.get("score")
        score_text = f", relevance: {score:.3f}" if isinstance(score, float) else ""
        lines.append(f"{index}. Page {page} in \"{paper}\"{score_text}")
    return "\n".join(lines)

def format_source_cards(sources: list[dict]) -> list[str]:
    cards = []
    for source in sources:
        page = source.get("page", "unknown")
        paper = source.get("paper_title", "Unknown")
        score = source.get("score")
        text = str(source.get("text") or "")
        preview = text[:280] + ("..." if len(text) > 280 else "")
        score_text = f"relevance {score:.3f}" if isinstance(score, float) else ""
        
        card_html = f"""
        <div class="source-card">
            <div class="source-card-header">
                <span class="source-paper-title">📄 {escape(paper)}</span>
                <span class="source-page-badge">Page {page}</span>
            </div>
            <p class="source-text">"{escape(preview)}"</p>
            {f'<div class="source-card-footer"><span style="color: #38BDF8; font-weight: 600;">{score_text}</span></div>' if score_text else ''}
        </div>
        """
        cards.append(card_html)
    return cards

def safe_file_name(file_name: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in file_name)

def build_markdown_export(title: str, metadata: dict[str, Any], content: str) -> bytes:
    """Export markdown report listing metadata for all papers inside workspace."""
    markdown = [
        f"# {title}",
        "",
        "## Active Papers in Workspace",
    ]
    
    # Check if multi-document metadata
    if metadata:
        for paper_title, details in metadata.items():
            markdown.append(f"- **Title**: {paper_title}")
            markdown.append(f"  - **Authors**: {details.get('authors', 'Unknown')}")
            markdown.append(f"  - **Year**: {details.get('year', 'Unknown')}")
            markdown.append(f"  - **Total Pages**: {details.get('pages', 0)}")
    else:
        markdown.append("No active papers listed.")
        
    markdown.extend([
        "",
        "## Analysis Report",
        content,
        "",
        "---",
        "*Report compiled by PaperMind AI*",
    ])
    return "\n".join(markdown).encode("utf-8")

def build_pdf_export(title: str, metadata: dict[str, Any], content: str) -> bytes:
    """Export styled PDF report listing metadata for all papers inside workspace."""
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
        title=title
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#1E1B4B'),
        alignment=0,
        spaceAfter=15
    )
    
    h2_style = ParagraphStyle(
        'CustomH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#4338CA'),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#374151'),
        spaceAfter=6
    )
    
    meta_style = ParagraphStyle(
        'CustomMeta',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#6B7280'),
        spaceAfter=10
    )

    story = [
        Paragraph(escape(title), title_style),
        Spacer(1, 8),
        Paragraph("<b>Workspace Papers Metadata</b>", h2_style)
    ]

    # Add papers list to PDF
    if metadata:
        for paper_title, details in metadata.items():
            p_title = escape(paper_title)
            p_auth = escape(details.get('authors', 'Unknown'))
            p_year = escape(details.get('year', 'Unknown'))
            p_pages = details.get('pages', 0)
            story.append(Paragraph(f"&bull; <b>Title:</b> {p_title} ({p_year})<br/>&nbsp;&nbsp;<b>Authors:</b> {p_auth} | <b>Pages:</b> {p_pages}", body_style))
    else:
        story.append(Paragraph("No papers in workspace.", body_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Report Content</b>", h2_style))

    # Parse and format content
    for line in content.splitlines():
        clean_line = line.strip()
        if not clean_line:
            story.append(Spacer(1, 6))
        elif clean_line.startswith("###"):
            text = clean_line.lstrip("#").strip()
            story.append(Paragraph(escape(text), h2_style))
        elif clean_line.startswith("##"):
            text = clean_line.lstrip("#").strip()
            story.append(Paragraph(escape(text), h2_style))
        elif clean_line.startswith("#"):
            text = clean_line.lstrip("#").strip()
            story.append(Paragraph(escape(text), title_style))
        elif clean_line.startswith(("- ", "* ")):
            text = clean_line[2:].strip()
            story.append(Paragraph(f"&bull; {escape(text)}", body_style))
        else:
            formatted_text = escape(clean_line)
            # Replace **bold** with <b>bold</b>
            import re
            formatted_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', formatted_text)
            formatted_text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', formatted_text)
            story.append(Paragraph(formatted_text, body_style))

    try:
        document.build(story)
        return buffer.getvalue()
    except Exception as e:
        buffer_err = BytesIO()
        doc_err = SimpleDocTemplate(buffer_err, pagesize=letter)
        story_err = [
            Paragraph("Error compiling report PDF", styles["Heading1"]),
            Paragraph(escape(str(e)), styles["BodyText"]),
        ]
        doc_err.build(story_err)
        return buffer_err.getvalue()
