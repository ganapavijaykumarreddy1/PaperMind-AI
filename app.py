from __future__ import annotations

import os
import re
import json
import logging
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from html import escape
from io import BytesIO
import streamlit as st
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

from prompts import (
    SUMMARY_PROMPT, 
    RESEARCH_GAP_PROMPT, 
    INTERVIEW_PROMPT, 
    CHAT_PROMPT, 
    LITERATURE_REVIEW_PROMPT,
    AGENT_COMPARISON_PROMPT,
    AGENT_SUMMARY_PROMPT,
    PPT_PROMPT
)
from rag import (
    answer_question, 
    build_vector_store, 
    add_paper_to_store,
    generate_from_full_context,
    retrieve_context
)
from utils import (
    build_markdown_export,
    build_pdf_export,
    configure_gemini,
    format_source_cards,
    format_source_list,
    safe_file_name,
)

load_dotenv()

st.set_page_config(
    page_title="PaperMind AI",
    page_icon="📄",
    layout="wide",
)

# --- Visual Enhancements CSS Injection ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');
    
    /* Global Background and Typography */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #17153B 0%, #0C0C1E 50%, #03001C 100%) !important;
        background-attachment: fixed !important;
        color: #F8FAFC !important;
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        color: #F8FAFC;
    }

    /* Custom layout styling */
    .hero {
        align-items: center;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        display: flex;
        gap: 1.25rem;
        padding: 1.5rem;
        margin-bottom: 2rem;
        background: rgba(15, 23, 42, 0.45);
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3), inset 0 0 0 1px rgba(255, 255, 255, 0.05);
        transition: border-color 0.3s ease;
    }
    .hero:hover {
        border-color: rgba(6, 182, 212, 0.2);
    }
    
    .logo-mark {
        align-items: center;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #06B6D4 100%);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        display: flex;
        flex: 0 0 auto;
        height: 4.8rem;
        justify-content: center;
        width: 4.8rem;
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.35);
    }
    
    .logo-copy {
        min-width: 0;
    }
    
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0;
        background: linear-gradient(135deg, #FFFFFF 0%, #E2E8F0 50%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .subtitle {
        font-size: 0.95rem;
        line-height: 1.6;
        margin-top: 0.35rem;
        color: #94A3B8;
        max-width: 54rem;
    }

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090916 0%, #02020a 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stSidebar"] .stMarkdown p {
        color: #CBD5E1 !important;
    }

    /* Styled Container Cards */
    .card-wrapper {
        background: rgba(15, 23, 42, 0.55) !important;
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .card-wrapper:hover {
        border-color: rgba(6, 182, 212, 0.25);
        box-shadow: 0 8px 32px 0 rgba(6, 182, 212, 0.05), inset 0 0 0 1px rgba(6, 182, 212, 0.05);
        transform: translateY(-2px);
    }

    /* Recommendations Badges */
    .recommend-badge {
        background: rgba(16, 185, 129, 0.12) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        color: #34D399 !important;
        padding: 0.25rem 0.65rem;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        display: inline-flex;
        align-items: center;
        gap: 4px;
        box-shadow: 0 2px 10px rgba(16, 185, 129, 0.1);
    }

    .low-rel-badge {
        background: rgba(148, 163, 184, 0.08) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        color: #94A3B8 !important;
        padding: 0.25rem 0.65rem;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }

    /* Custom Styled GAP scorecards */
    .gap-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin: 1.25rem 0;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
    }
    .gap-table th {
        background: rgba(79, 70, 229, 0.25) !important;
        color: #38BDF8 !important;
        font-weight: 700;
        padding: 10px 14px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        text-align: left;
        font-size: 0.85rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .gap-table td {
        padding: 10px 14px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        color: #E2E8F0;
        font-size: 0.82rem;
        line-height: 1.5;
    }
    .gap-table tr:last-child td {
        border-bottom: none;
    }
    .gap-table tr:hover {
        background: rgba(255, 255, 255, 0.03) !important;
    }

    /* Modern Tabs Design */
    div[data-testid="stTabs"] {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 8px 8px 0px 8px;
        border-radius: 14px;
        backdrop-filter: blur(8px);
        margin-bottom: 1.5rem;
    }
    div[data-testid="stTabs"] [role="tablist"] {
        gap: 8px;
        border-bottom: none !important;
    }
    div[data-testid="stTabs"] button[role="tab"] {
        background-color: transparent !important;
        color: #94A3B8 !important;
        border: 1px solid transparent !important;
        border-radius: 10px !important;
        padding: 8px 16px !important;
        height: auto !important;
        font-weight: 600 !important;
        font-family: 'Outfit', sans-serif !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div[data-testid="stTabs"] button[role="tab"]:hover {
        color: #F8FAFC !important;
        background-color: rgba(255, 255, 255, 0.03) !important;
        border-color: rgba(255, 255, 255, 0.08) !important;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #FFFFFF !important;
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.6) 0%, rgba(124, 58, 237, 0.6) 100%) !important;
        border-color: rgba(255, 255, 255, 0.15) !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25) !important;
    }
    div[data-testid="stTabs"] [role="tablist"] button[aria-selected="true"]::after {
        display: none !important;
    }
    div[data-testid="stTabs"] [role="tablist"] + div {
        background-color: transparent !important;
    }

    /* Streamlit expander styling */
    div[data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.35) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        margin-bottom: 0.75rem;
    }
    div[data-testid="stExpander"] summary {
        color: #CBD5E1 !important;
        font-weight: 600;
        font-size: 0.88rem;
    }
    div[data-testid="stExpander"] summary:hover {
        color: #38BDF8 !important;
    }

    /* Buttons visual polishing */
    div.stButton > button {
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        font-family: 'Outfit', sans-serif !important;
        padding: 8px 20px !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3) !important;
    }
    div.stButton > button:hover {
        border-color: #38BDF8 !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.5) !important;
        transform: translateY(-1px);
    }
    
    /* Secondary/Download Buttons */
    div.stDownloadButton > button {
        border-radius: 10px !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        background: rgba(56, 189, 248, 0.08) !important;
        color: #38BDF8 !important;
        font-weight: 600 !important;
        font-family: 'Outfit', sans-serif !important;
        padding: 8px 20px !important;
        transition: all 0.25s ease !important;
    }
    div.stDownloadButton > button:hover {
        background: #38BDF8 !important;
        color: #03001C !important;
        box-shadow: 0 4px 16px rgba(56, 189, 248, 0.35) !important;
        transform: translateY(-1px);
    }

    /* File Uploader glass styles */
    div[data-testid="stFileUploader"] section {
        background-color: rgba(15, 23, 42, 0.25) !important;
        border: 2px dashed rgba(255, 255, 255, 0.1) !important;
        border-radius: 14px !important;
        padding: 1.5rem !important;
        transition: all 0.3s ease;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: #6366F1 !important;
        background-color: rgba(99, 102, 241, 0.04) !important;
    }

    /* Input inputs styling */
    div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea {
        background-color: rgba(15, 23, 42, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #F8FAFC !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        transition: all 0.25s ease;
    }
    div[data-testid="stTextInput"] input:focus, div[data-testid="stTextArea"] textarea:focus {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2) !important;
    }

    /* Source Cards Styling */
    .source-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        transition: all 0.25s ease;
    }
    .source-card:hover {
        background: rgba(255, 255, 255, 0.04);
        border-color: rgba(56, 189, 248, 0.2);
        transform: translateY(-1px);
    }
    .source-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.6rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        padding-bottom: 0.4rem;
    }
    .source-paper-title {
        font-weight: 600;
        color: #F8FAFC;
        font-size: 0.85rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 75%;
    }
    .source-page-badge {
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #A5B4FC;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.15rem 0.5rem;
        border-radius: 6px;
    }
    .source-text {
        font-size: 0.82rem;
        color: #CBD5E1;
        line-height: 1.5;
        font-style: italic;
        margin: 0;
    }
    .source-card-footer {
        margin-top: 0.6rem;
        font-size: 0.75rem;
        text-align: right;
    }

    /* Style chat message container */
    [data-testid="stChatMessage"] {
        background-color: rgba(15, 23, 42, 0.35) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-radius: 14px !important;
        margin-bottom: 1rem !important;
        padding: 1rem !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    [data-testid="stChatMessageContent"] {
        font-size: 0.9rem !important;
        color: #E2E8F0 !important;
        line-height: 1.6 !important;
    }

    /* Toast message style */
    div[data-testid="stToast"] {
        background-color: #0F172A !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def initialize_state() -> None:
    defaults = {
        "vector_store": None,
        "indexed_files": [], # List of {"name": str, "bytes": bytes}
        "chat_history": [],
        "summary": None,
        "research_gaps": None,
        "literature_review": None,
        "interview_questions": None,
        "ppt_content": None,
        "agent_report": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_outputs() -> None:
    st.session_state.chat_history = []
    st.session_state.summary = None
    st.session_state.research_gaps = None
    st.session_state.literature_review = None
    st.session_state.interview_questions = None
    st.session_state.ppt_content = None
    st.session_state.agent_report = None

def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="logo-mark" aria-label="PaperMind AI logo">
                <svg width="55" height="55" viewBox="0 0 62 62" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="14" y="8" width="32" height="44" rx="7" fill="white" stroke="#4F46E5" stroke-width="2.8"/>
                    <path d="M38 8V18C38 20.2 39.8 22 42 22H46" stroke="#4F46E5" stroke-width="2.8" stroke-linecap="round"/>
                    <path d="M22 31C22 25.8 25.6 22 30.5 22C35.4 22 39 25.8 39 31C39 36.2 35.4 40 30.5 40C25.6 40 22 36.2 22 31Z" fill="#EEF2FF"/>
                    <circle cx="27" cy="29" r="2.5" fill="#06B6D4"/>
                    <circle cx="34" cy="27" r="2.5" fill="#4F46E5"/>
                    <circle cx="33" cy="35" r="2.5" fill="#7C3AED"/>
                    <path d="M29.2 29L32 27.8M28.6 31.1L31.4 33.6" stroke="#64748B" stroke-width="1.8" stroke-linecap="round"/>
                    <path d="M22 45H39" stroke="#94A3B8" stroke-width="2" stroke-linecap="round"/>
                </svg>
            </div>
            <div class="logo-copy">
                <p class="main-title">PaperMind AI</p>
                <p class="subtitle">
                    Multi-document research intelligence platform. Chat across your papers, build comparative matrices, uncover methodologies, score limitations, and run external search agents.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def require_papers() -> bool:
    if st.session_state.vector_store is None:
        st.info("Upload and index at least one research paper PDF from the sidebar to get started.")
        return False
    return True

def rebuild_store_from_files(client) -> None:
    """Reconstruct vector store cumulatively from the files saved in session state."""
    files = st.session_state.indexed_files
    if not files:
        st.session_state.vector_store = None
        return
        
    with st.spinner("Rebuilding FAISS index..."):
        try:
            # Index first file
            first_io = BytesIO(files[0]["bytes"])
            first_io.name = files[0]["name"]
            store = build_vector_store(client, first_io)
            
            # Index remaining files
            for idx in range(1, len(files)):
                next_io = BytesIO(files[idx]["bytes"])
                next_io.name = files[idx]["name"]
                store = add_paper_to_store(client, store, next_io)
                
            st.session_state.vector_store = store
        except Exception as e:
            st.error(f"Error rebuilding vector store: {e}")
            st.session_state.vector_store = None

# --- AI Research Agent Helper logic ---

def search_arxiv(query: str, max_results: int = 4) -> list[dict]:
    encoded_query = urllib.parse.quote(query)
    url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&max_results={max_results}"
    
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            xml_data = response.read()
    except Exception as e:
        logger.error(f"Error connecting to arXiv: {e}")
        return []

    try:
        root = ET.fromstring(xml_data)
    except Exception:
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []

    for entry in root.findall("atom:entry", ns):
        title_node = entry.find("atom:title", ns)
        summary_node = entry.find("atom:summary", ns)
        id_node = entry.find("atom:id", ns)
        published_node = entry.find("atom:published", ns)
        
        title = " ".join(title_node.text.strip().split()) if title_node is not None else "Unknown"
        abstract = " ".join(summary_node.text.strip().split()) if summary_node is not None else ""
        pdf_url = id_node.text.strip().replace("/abs/", "/pdf/") + ".pdf" if id_node is not None else ""
        published_date = published_node.text if published_node is not None else ""
        year = published_date[:4] if len(published_date) >= 4 else "Unknown"

        authors = []
        for author_node in entry.findall("atom:author", ns):
            name_node = author_node.find("atom:name", ns)
            if name_node is not None and name_node.text:
                authors.append(name_node.text.strip())
        authors_str = ", ".join(authors) if authors else "Unknown"

        papers.append({
            "title": title,
            "authors": authors_str,
            "year": year,
            "summary": abstract,
            "pdf_url": pdf_url
        })
    return papers

def run_agent_workflow(client, vector_store, query: str, model_name: str) -> dict:
    """Agent loop: search external arXiv, compare with workspace models, recommend."""
    # 1. Search
    external_papers = search_arxiv(query)
    if not external_papers:
        return {
            "summary": "Agent found no relevant papers on arXiv or was blocked by network issues.",
            "recommendations": []
        }
        
    # 2. Compare
    workspace_papers_list = "\n".join(f"- {title}" for title in vector_store.metadata.keys())
    recommendations = []
    evals_text = ""
    
    for paper in external_papers:
        prompt = AGENT_COMPARISON_PROMPT.format(
            workspace_papers_list=workspace_papers_list,
            new_title=paper["title"],
            new_authors=paper["authors"],
            new_year=paper["year"],
            new_abstract=paper["summary"][:1200]
        )
        
        comparison = "Unable to analyze connection."
        recommend = False
        reason = "Error during parsing."
        
        try:
            res_text = answer_question_raw(client, model_name, prompt)
            json_match = re.search(r"```json\s*([\s\S]*?)\s*```", res_text)
            if json_match:
                res_text = json_match.group(1)
            eval_data = json.loads(res_text.strip())
            comparison = eval_data.get("comparison", comparison)
            recommend = bool(eval_data.get("recommend", recommend))
            reason = eval_data.get("reason", reason)
        except Exception:
            pass
            
        recommendations.append({
            "title": paper["title"],
            "authors": paper["authors"],
            "year": paper["year"],
            "summary": paper["summary"],
            "pdf_url": paper["pdf_url"],
            "comparison": comparison,
            "recommend": recommend,
            "reason": reason
        })
        evals_text += f"\nPaper: \"{paper['title']}\"\nComparison: {comparison}\nRecommend: {recommend}\nReason: {reason}\n"

    # 3. Summarize
    summary_prompt = AGENT_SUMMARY_PROMPT.format(
        query=query,
        evaluations_summary=evals_text
    )
    try:
        summary = answer_question_raw(client, model_name, summary_prompt)
    except Exception:
        summary = f"Completed arXiv search for '{query}' and analyzed {len(recommendations)} papers."
        
    return {
        "summary": summary,
        "recommendations": recommendations
    }

def answer_question_raw(client, model_name: str, prompt: str) -> str:
    response = client.models.generate_content(model=model_name, contents=prompt)
    return getattr(response, "text", "").strip()

def render_styled_gap_tables(text: str):
    """Parse raw markdown tables in gap response and render using custom HTML table CSS."""
    lines = text.split('\n')
    output_blocks = []
    table_lines = []
    is_inside_table = False
    
    def compile_table_block(t_lines):
        if len(t_lines) <= 2:
            return ""
        header = t_lines[0]
        rows = t_lines[2:]
        
        parse_cells = lambda l: [c.strip() for c in l.split('|')[1:-1]]
        headers = parse_cells(header)
        
        html_table = '<table class="gap-table"><thead><tr>'
        for h in headers:
            html_table += f'<th>{escape(h)}</th>'
        html_table += '</tr></thead><tbody>'
        
        for r in rows:
            cells = parse_cells(r)
            html_table += '<tr>'
            for c in cells:
                formatted_cell = escape(c)
                # Bold highlight
                formatted_cell = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', formatted_cell)
                html_table += f'<td>{formatted_cell}</td>'
            html_table += '</tr>'
        html_table += '</tbody></table>'
        return html_table

    for line in lines:
        if line.strip().startswith('|'):
            is_inside_table = True
            table_lines.append(line)
        else:
            if is_inside_table:
                table_html = compile_table_block(table_lines)
                if table_html:
                    output_blocks.append(table_html)
                table_lines = []
                is_inside_table = False
                
            if line.strip():
                # Format normal text bold items
                formatted_line = escape(line)
                formatted_line = re.sub(r'\*\*(.*?)\*\*', r'<b style="color:#38BDF8">\1</b>', formatted_line)
                output_blocks.append(f'<p style="font-size:0.88rem; color:#E2E8F0; line-height:1.6">{formatted_line}</p>')
            else:
                output_blocks.append('<div style="height:8px"></div>')
                
    if is_inside_table:
        table_html = compile_table_block(table_lines)
        if table_html:
            output_blocks.append(table_html)
            
    st.markdown("".join(output_blocks), unsafe_allow_html=True)


# --- Streamlit Application UI layout ---

def main() -> None:
    initialize_state()
    render_header()

    try:
        client, model_name = configure_gemini()
    except RuntimeError as error:
        st.error(str(error))
        st.stop()

    # --- Sidebar Document Management ---
    with st.sidebar:
        st.header("Document Manager")
        uploaded_files = st.file_uploader(
            "Upload research papers (PDF)", 
            type=["pdf"], 
            accept_multiple_files=True
        )
        
        # Check for newly uploaded files to index
        if uploaded_files:
            for file in uploaded_files:
                # Check if this file is already in our indexed list
                if not any(f["name"] == file.name for f in st.session_state.indexed_files):
                    # Save bytes to session state
                    file_bytes = file.read()
                    st.session_state.indexed_files.append({"name": file.name, "bytes": file_bytes})
                    
                    with st.spinner(f"Indexing '{file.name}' into FAISS..."):
                        file_io = BytesIO(file_bytes)
                        file_io.name = file.name
                        try:
                            if st.session_state.vector_store is None:
                                st.session_state.vector_store = build_vector_store(client, file_io)
                            else:
                                st.session_state.vector_store = add_paper_to_store(client, st.session_state.vector_store, file_io)
                            st.toast(f"Paper '{file.name}' indexed successfully!")
                            reset_outputs()
                        except Exception as e:
                            st.error(f"Failed to process '{file.name}': {e}")
                            
        st.divider()
        
        # Display Uploaded Papers List
        if st.session_state.indexed_files:
            st.markdown("<p style='font-size:0.85rem; font-weight:700; color:#38BDF8; margin-top:10px; margin-bottom:10px; text-transform:uppercase; letter-spacing:0.05em;'>Active Workspace Papers</p>", unsafe_allow_html=True)
            store = st.session_state.vector_store
            
            for idx, file in enumerate(st.session_state.indexed_files):
                metadata = None
                if store and store.metadata:
                    keys = list(store.metadata.keys())
                    if idx < len(keys):
                        title_key = keys[idx]
                        metadata = store.metadata[title_key]
                
                with st.container(border=True):
                    col_info, col_btn = st.columns([5, 1])
                    with col_info:
                        if metadata:
                            title_display = metadata.get("title", file["name"])
                            authors_display = metadata.get("authors", "Unknown")
                            year_display = metadata.get("year", "Unknown")
                            pages_display = metadata.get("pages", 0)
                            
                            if len(title_display) > 40:
                                title_display = title_display[:37] + "..."
                            
                            st.markdown(
                                f"""
                                <p style='font-size:0.8rem; font-weight:600; margin:0; color:#F8FAFC' title='{escape(metadata.get("title", ""))}'>📄 {escape(title_display)}</p>
                                <p style='font-size:0.7rem; margin:2px 0 0 0; color:#94A3B8'>{escape(authors_display)} · {escape(year_display)} · {pages_display} pages</p>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            disp_name = file["name"]
                            if len(disp_name) > 30:
                                disp_name = disp_name[:27] + "..."
                            st.markdown(
                                f"<p style='font-size:0.8rem; font-weight:600; margin:0; color:#F8FAFC'>📄 {escape(disp_name)}</p><p style='font-size:0.7rem; margin:2px 0 0 0; color:#94A3B8'>Indexing details pending...</p>",
                                unsafe_allow_html=True
                            )
                    with col_btn:
                        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                        if st.button("🗑️", key=f"del_{idx}", help=f"Remove {file['name']}"):
                            st.session_state.indexed_files.pop(idx)
                            rebuild_store_from_files(client)
                            reset_outputs()
                            st.rerun()
                        
            if st.button("Clear Workspace", type="secondary"):
                st.session_state.indexed_files = []
                st.session_state.vector_store = None
                reset_outputs()
                st.rerun()
        else:
            st.caption("No papers uploaded yet.")
            
        st.sidebar.divider()
        st.sidebar.markdown(
            """
            <div style="padding: 10px; border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; background: rgba(255, 255, 255, 0.02); text-align: center;">
                <p style="font-size: 0.72rem; margin: 0; color: #94A3B8; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;">Project Owner</p>
                <p style="font-size: 0.88rem; margin: 3px 0 0 0; color: #F8FAFC; font-weight: 600;">Vijay Kumar Reddy Ganapa</p>
                <p style="font-size: 0.75rem; margin: 2px 0 0 0;"><a href="mailto:ganapavijaykumarreddy1@gmail.com" style="color: #38BDF8; text-decoration: none; font-weight: 500;">ganapavijaykumarreddy1@gmail.com</a></p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --- Central Tabs Panel ---
    tab_agent, tab_chat, tab_summary, tab_review, tab_gaps, tab_interview, tab_ppt = st.tabs(
        [
            "🤖 AI Research Agent",
            "💬 Chat with Workspace",
            "📝 Document Summary",
            "📖 Literature Review",
            "🔍 Research Gaps",
            "🎓 Interview Prep",
            "📊 Presentation Slides"
        ]
    )

    # Tab 1: AI Research Agent (Featured Highlight)
    with tab_agent:
        if require_papers():
            st.markdown("### AI Research Agent Discovery")
            st.caption("🔍 **Task**: Autonomously search arXiv XML for papers on a topic, cross-reference them with your workspace using Gemini, and suggest relevant articles to read.")
            
            agent_query = st.text_input(
                "Search query (e.g. 'Adversarial Defense MRI', 'Transformer efficiency'):",
                placeholder="Topic to search on arXiv..."
            )
            
            if st.button("Execute Agent Research", type="primary") and agent_query:
                # Simulated steps log
                steps_placeholder = st.empty()
                with steps_placeholder.container():
                    st.info("🤖 Agent: Triggering discovery query on arXiv XML API...")
                    
                import time
                time.sleep(1.2)
                with steps_placeholder.container():
                    st.info("🤖 Agent: Fetching xml metadata and extracting paper abstracts...")
                    
                # Search
                arxiv_results = search_arxiv(agent_query)
                if not arxiv_results:
                    st.error("No relevant articles found on arXiv for the topic.")
                    steps_placeholder.empty()
                else:
                    time.sleep(1.2)
                    with steps_placeholder.container():
                          st.info(f"🤖 Agent: Performing LLM comparison against {len(st.session_state.vector_store.metadata)} papers in workspace...")
                        
                    # Execute agent compare
                    try:
                        st.session_state.agent_report = run_agent_workflow(
                            client=client,
                            vector_store=st.session_state.vector_store,
                            query=agent_query,
                            model_name=model_name
                        )
                        st.success("Agent Research completed successfully!")
                        steps_placeholder.empty()
                    except Exception as e:
                        st.error(f"Agent analysis failed: {e}")
                        steps_placeholder.empty()

            if st.session_state.agent_report:
                report = st.session_state.agent_report
                st.markdown("#### Agent Synthesis Report")
                st.markdown(
                    f"<div class='card-wrapper'><p style='font-size:0.85rem; line-height:1.6; color:#E2E8F0'>{report['summary']}</p></div>", 
                    unsafe_allow_html=True
                )
                
                st.markdown("#### Evaluated arXiv Papers")
                for idx, rec in enumerate(report["recommendations"]):
                    badge_class = "recommend-badge" if rec["recommend"] else "low-rel-badge"
                    badge_text = "Recommend Add" if rec["recommend"] else "Low Relevance"
                    
                    st.markdown(
                        f"""
                        <div class="card-wrapper">
                            <div style="display:flex; justify-content:space-between; align-items:start; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:8px; margin-bottom:8px">
                                <div>
                                    <h5 style="margin:0; font-size:0.9rem; color:#FFFFFF">{rec['title']}</h5>
                                    <span style="font-size:0.75rem; color:#64748B">{rec['authors']} &bull; {rec['year']}</span>
                                </div>
                                <span class="{badge_class}">{badge_text}</span>
                            </div>
                            <p style="font-size:0.8rem; color:#94A3B8; font-style:italic; margin-bottom:8px"><b>Abstract:</b> {rec['summary']}</p>
                            <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; font-size:0.8rem; border-top:1px solid rgba(255,255,255,0.04); padding-top:8px">
                                <div>
                                    <b style="color:#06B6D4">Semantic Match:</b><br/>
                                    <span style="color:#CBD5E1">{rec['comparison']}</span>
                                </div>
                                <div>
                                    <b style="color:#4F46E5">Recommendation Reason:</b><br/>
                                    <span style="color:#CBD5E1">{rec['reason']}</span>
                                </div>
                              </div>
                              <div style="display:flex; justify-content:flex-end; margin-top:8px">
                                  <a href="{rec['pdf_url']}" target="_blank" style="font-size:0.75rem; color:#06B6D4; text-decoration:none">View PDF Link &rarr;</a>
                              </div>
                          </div>
                          """,
                          unsafe_allow_html=True
                      )

    # Tab 2: Grounded Multi-Doc Chat
    with tab_chat:
        if require_papers():
            st.markdown("### Workspace Chat Assistant")
            st.caption("💬 **Task**: Ask questions grounded strictly in the text of your uploaded documents. Gemini answers with precise page-level citations.")
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "sources" in message and message["sources"]:
                        with st.expander("Evidence Sources"):
                            for card in format_source_cards(message["sources"]):
                                st.markdown(card, unsafe_allow_html=True)

            question = st.chat_input("Ask a question about the papers...")
            if question:
                st.session_state.chat_history.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    with st.spinner("Searching FAISS context & generating answer..."):
                        try:
                            answer, sources = answer_question(
                                client,
                                st.session_state.vector_store,
                                question,
                                model_name,
                            )
                            st.markdown(answer)
                            if sources:
                                with st.expander("Evidence Sources"):
                                    for card in format_source_cards(sources):
                                        st.markdown(card, unsafe_allow_html=True)
                            st.session_state.chat_history.append({
                                "role": "assistant", 
                                "content": answer,
                                "sources": sources
                            })
                        except Exception as error:
                            st.error(f"Could not generate response: {error}")

    # Tab 3: Document Summary
    with tab_summary:
        if require_papers():
            st.markdown("### Document Summary")
            st.caption("📝 **Task**: Generates a unified overview summarizing the core objectives, methodology commonalities, and key findings across all workspace papers.")
            
            if st.button("Generate Summary", type="primary"):
                with st.spinner("Summarizing papers..."):
                    try:
                        st.session_state.summary = generate_from_full_context(
                            client=client,
                            vector_store=st.session_state.vector_store,
                            prompt_template=SUMMARY_PROMPT,
                            model_name=model_name
                        )
                    except Exception as error:
                        st.error(f"Failed to generate summary: {error}")
                        
            if st.session_state.summary:
                col_exp_md, col_exp_pdf, _ = st.columns([1, 1, 3])
                with col_exp_md:
                    st.download_button(
                        "Download Markdown",
                        data=build_markdown_export(
                          "Unified Document Summary",
                          st.session_state.vector_store.metadata,
                          st.session_state.summary,
                        ),
                        file_name="summary.md",
                        mime="text/markdown",
                        key="dl_summary_md"
                    )
                with col_exp_pdf:
                    st.download_button(
                        "Download PDF Report",
                        data=build_pdf_export(
                          "Unified Document Summary",
                          st.session_state.vector_store.metadata,
                          st.session_state.summary,
                        ),
                        file_name="summary.pdf",
                        mime="application/pdf",
                        key="dl_summary_pdf"
                    )
                st.divider()
                st.markdown(st.session_state.summary)

    # Tab 4: Literature Review
    with tab_review:
        if require_papers():
            st.markdown("### Literature Review Synthesis")
            st.caption("📖 **Task**: Synthesizes your workspace documents into a comparative review matrix, highlighting methodology differences and contributions.")
            
            col_in, col_btn = st.columns([4, 1])
            with col_in:
                focus_text = st.text_input(
                    "Optional synthesis focus (e.g., benchmark parameters, medical limitations):",
                    key="lit_focus"
                )
            with col_btn:
                st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
                trigger_review = st.button("Generate Review", type="primary")
                
            if trigger_review:
                with st.spinner("Synthesizing paper methodology blocks..."):
                    try:
                        st.session_state.literature_review = generate_from_full_context(
                            client=client,
                            vector_store=st.session_state.vector_store,
                            prompt_template=LITERATURE_REVIEW_PROMPT,
                            model_name=model_name,
                            focus=focus_text.strip() if focus_text else "Standard synthesis."
                        )
                    except Exception as error:
                        st.error(f"Failed to generate review: {error}")
                        
            if st.session_state.literature_review:
                col_exp_md, col_exp_pdf, _ = st.columns([1, 1, 3])
                with col_exp_md:
                    st.download_button(
                        "Download Markdown",
                        data=build_markdown_export(
                            "Literature Review Synthesis",
                            st.session_state.vector_store.metadata,
                            st.session_state.literature_review,
                        ),
                        file_name="literature_review.md",
                        mime="text/markdown",
                    )
                with col_exp_pdf:
                    st.download_button(
                        "Download PDF Report",
                        data=build_pdf_export(
                            "Literature Review Synthesis",
                            st.session_state.vector_store.metadata,
                            st.session_state.literature_review,
                        ),
                        file_name="literature_review.pdf",
                        mime="application/pdf",
                    )
                    
                st.divider()
                st.markdown(st.session_state.literature_review)

    # Tab 5: Research Gaps
    with tab_gaps:
        if require_papers():
            st.markdown("### Research Gaps & Limitations")
            st.caption("🔍 **Task**: Scans workspace papers to evaluate dataset limitations, unresolved computational constraints, and calculates gap feasibility scores.")
            
            if st.button("Discover Limitations & Gaps", type="primary"):
                with st.spinner("Evaluating paper conclusions..."):
                    try:
                        st.session_state.research_gaps = generate_from_full_context(
                            client=client,
                            vector_store=st.session_state.vector_store,
                            prompt_template=RESEARCH_GAP_PROMPT,
                            model_name=model_name
                        )
                    except Exception as error:
                        st.error(f"Failed to discover research gaps: {error}")
                        
            if st.session_state.research_gaps:
                col_exp_md, col_exp_pdf, _ = st.columns([1, 1, 3])
                with col_exp_md:
                    st.download_button(
                        "Download Markdown",
                        data=build_markdown_export(
                            "Research Gap Analysis",
                            st.session_state.vector_store.metadata,
                            st.session_state.research_gaps,
                        ),
                        file_name="research_gaps.md",
                        mime="text/markdown",
                    )
                with col_exp_pdf:
                    st.download_button(
                        "Download PDF Report",
                        data=build_pdf_export(
                            "Research Gap Analysis",
                            st.session_state.vector_store.metadata,
                            st.session_state.research_gaps,
                        ),
                        file_name="research_gaps.pdf",
                        mime="application/pdf",
                    )
                    
                st.divider()
                # Render using custom styled gap scorecard tables HTML converter!
                render_styled_gap_tables(st.session_state.research_gaps)

    # Tab 6: Technical Interview Prep
    with tab_interview:
        if require_papers():
            st.markdown("### Technical Interview Prep")
            st.caption("🎓 **Task**: Formulates technical interview questions and answers grounded in your workspace papers for study or defense prep.")
            
            if st.button("Generate Interview Questions", type="primary"):
                with st.spinner("Creating interview questions..."):
                    try:
                        st.session_state.interview_questions = generate_from_full_context(
                            client=client,
                            vector_store=st.session_state.vector_store,
                            prompt_template=INTERVIEW_PROMPT,
                            model_name=model_name
                        )
                    except Exception as error:
                        st.error(f"Failed to generate interview questions: {error}")
                        
            if st.session_state.interview_questions:
                col_exp_md, col_exp_pdf, _ = st.columns([1, 1, 3])
                with col_exp_md:
                    st.download_button(
                        "Download Markdown",
                        data=build_markdown_export(
                            "Technical Interview Questions",
                            st.session_state.vector_store.metadata,
                            st.session_state.interview_questions,
                        ),
                        file_name="interview_questions.md",
                        mime="text/markdown",
                        key="dl_interview_md"
                    )
                with col_exp_pdf:
                    st.download_button(
                        "Download PDF Report",
                        data=build_pdf_export(
                            "Technical Interview Questions",
                            st.session_state.vector_store.metadata,
                            st.session_state.interview_questions,
                        ),
                        file_name="interview_questions.pdf",
                        mime="application/pdf",
                        key="dl_interview_pdf"
                    )
                st.divider()
                st.markdown(st.session_state.interview_questions)

    # Tab 7: Presentation Content
    with tab_ppt:
        if require_papers():
            st.markdown("### Presentation Content Draft")
            st.caption("📊 **Task**: Drafts a structured outline of slide titles and bullet points based on your documents to kickstart your slide deck.")
            
            if st.button("Generate PPT Content Draft", type="primary"):
                with st.spinner("Drafting slide outlines..."):
                    try:
                        st.session_state.ppt_content = generate_from_full_context(
                            client=client,
                            vector_store=st.session_state.vector_store,
                            prompt_template=PPT_PROMPT,
                            model_name=model_name
                        )
                    except Exception as error:
                        st.error(f"Failed to generate slides: {error}")
                        
            if st.session_state.ppt_content:
                st.download_button(
                    "Download Presentation Draft (MD)",
                    data=st.session_state.ppt_content.encode("utf-8"),
                    file_name="presentation_slides.md",
                    mime="text/markdown"
                )
                st.divider()
                st.markdown(st.session_state.ppt_content)

if __name__ == "__main__":
    main()
