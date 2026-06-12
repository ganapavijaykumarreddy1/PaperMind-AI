from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from prompts import INTERVIEW_PROMPT, PPT_PROMPT, RESEARCH_GAP_PROMPT, SUMMARY_PROMPT
from rag import answer_question, build_vector_store, generate_from_full_context
from utils import (
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


def initialize_state() -> None:
    defaults = {
        "vector_store": None,
        "uploaded_file_name": None,
        "chat_history": [],
        "summary": None,
        "research_gaps": None,
        "interview_questions": None,
        "ppt_content": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_outputs() -> None:
    st.session_state.chat_history = []
    st.session_state.summary = None
    st.session_state.research_gaps = None
    st.session_state.interview_questions = None
    st.session_state.ppt_content = None


def render_header() -> None:
    st.markdown(
        """
        <style>
        .hero {
            align-items: center;
            border: 1px solid rgba(148, 163, 184, 0.35);
            border-radius: 18px;
            display: flex;
            gap: 1rem;
            padding: 1rem 1.15rem;
            margin-bottom: 1.2rem;
            background: rgba(255, 255, 255, 0.04);
        }
        .logo-mark {
            align-items: center;
            background: linear-gradient(135deg, #eef2ff 0%, #ecfeff 100%);
            border: 1px solid #c7d2fe;
            border-radius: 18px;
            display: flex;
            flex: 0 0 auto;
            height: 4.4rem;
            justify-content: center;
            width: 4.4rem;
        }
        .logo-copy {
            min-width: 0;
        }
        .main-title {
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin-bottom: 0;
        }
        .subtitle {
            font-size: 1.06rem;
            line-height: 1.65;
            margin-top: 0.25rem;
            max-width: 52rem;
        }
        </style>
        <div class="hero">
            <div class="logo-mark" aria-label="PaperMind AI logo">
                <svg width="62" height="62" viewBox="0 0 62 62" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="14" y="8" width="32" height="44" rx="7" fill="white" stroke="#4F46E5" stroke-width="2.4"/>
                    <path d="M38 8V18C38 20.2 39.8 22 42 22H46" stroke="#4F46E5" stroke-width="2.4" stroke-linecap="round"/>
                    <path d="M22 31C22 25.8 25.6 22 30.5 22C35.4 22 39 25.8 39 31C39 36.2 35.4 40 30.5 40C25.6 40 22 36.2 22 31Z" fill="#EEF2FF"/>
                    <circle cx="27" cy="29" r="2.5" fill="#06B6D4"/>
                    <circle cx="34" cy="27" r="2.5" fill="#4F46E5"/>
                    <circle cx="33" cy="35" r="2.5" fill="#7C3AED"/>
                    <path d="M29.2 29L32 27.8M28.6 31.1L31.4 33.6" stroke="#64748B" stroke-width="1.7" stroke-linecap="round"/>
                    <path d="M22 45H39" stroke="#94A3B8" stroke-width="2" stroke-linecap="round"/>
                </svg>
            </div>
            <div class="logo-copy">
                <p class="main-title">PaperMind AI</p>
                <p class="subtitle">
                    Research paper intelligence for summaries, citation-aware answers,
                    gap scoring, interview prep, and presentation-ready notes.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def require_paper() -> bool:
    if st.session_state.vector_store is None:
        st.info("Upload a research paper PDF from the sidebar to get started.")
        return False
    return True


def render_metadata_card(metadata: dict) -> None:
    st.markdown("**Metadata**")
    st.caption(f"Title: {metadata.get('title', 'Unknown')}")
    st.caption(f"Authors: {metadata.get('authors', 'Unknown')}")
    st.caption(f"Year: {metadata.get('year', 'Unknown')}")
    st.caption(f"Pages: {metadata.get('pages', 'Unknown')}")


def render_sources(sources: list[dict], title: str = "Retrieved sources") -> None:
    with st.expander(title):
        st.text(format_source_list(sources))
        for card in format_source_cards(sources):
            st.caption(card)


def render_summary_download() -> None:
    st.download_button(
        "Download Summary as PDF",
        data=build_pdf_export(
            "PaperMind AI Summary",
            st.session_state.vector_store.metadata,
            st.session_state.summary,
        ),
        file_name="papermind_summary.pdf",
        mime="application/pdf",
    )


def render_interview_download() -> None:
    st.download_button(
        "Download Interview Questions as PDF",
        data=build_pdf_export(
            "PaperMind AI Interview Questions",
            st.session_state.vector_store.metadata,
            st.session_state.interview_questions,
        ),
        file_name="papermind_interview_questions.pdf",
        mime="application/pdf",
    )


def process_uploaded_file(client, uploaded_file) -> None:
    incoming_name = safe_file_name(uploaded_file.name)
    if incoming_name == st.session_state.uploaded_file_name:
        return

    reset_outputs()
    with st.spinner("Reading paper and building FAISS index..."):
        try:
            st.session_state.vector_store = build_vector_store(client, uploaded_file)
            st.session_state.uploaded_file_name = incoming_name
            st.success("Paper indexed successfully.")
        except Exception as error:
            st.session_state.vector_store = None
            st.session_state.uploaded_file_name = None
            st.error(f"Could not process PDF: {error}")


def render_index_warning() -> None:
    vector_store = st.session_state.vector_store
    if vector_store is None:
        return

    if vector_store.embedded_chunks < vector_store.total_chunks:
        st.warning(
            "Indexed the first "
            f"{vector_store.embedded_chunks} of {vector_store.total_chunks} chunks "
            "to stay within Gemini free-tier embedding limits. Increase "
            "MAX_EMBEDDING_CHUNKS if your quota allows it."
        )


def render_sidebar(client, model_name: str) -> None:
    with st.sidebar:
        st.header("Upload PDF")
        uploaded_file = st.file_uploader("Choose a research paper", type=["pdf"])
        st.caption("Supported: text-based PDFs. Scanned PDFs may require OCR first.")

        st.divider()
        st.markdown("**Settings**")
        st.write("Model:", os.getenv("GEMINI_MODEL", model_name))
        st.write("Retrieval:", "Top-k = 4")
        st.write("Chunks:", "1000 chars, 200 overlap")
        st.caption("Free-tier friendly: indexes first 90 chunks by default.")

        if uploaded_file is not None:
            process_uploaded_file(client, uploaded_file)
            render_index_warning()

        if st.session_state.vector_store is not None:
            st.divider()
            st.success(f"Active paper: {st.session_state.uploaded_file_name}")
            render_metadata_card(st.session_state.vector_store.metadata)
            if st.button("Clear paper"):
                st.session_state.vector_store = None
                st.session_state.uploaded_file_name = None
                reset_outputs()
                st.rerun()


def render_overview() -> None:
    overview_left, overview_right = st.columns([2, 1])
    with overview_left:
        st.markdown(
            """
            Upload a PDF once, then ask grounded questions, export a Markdown
            summary, score research gaps, and download interview prep as a PDF.
            """
        )
    with overview_right:
        if st.session_state.vector_store is not None:
            vector_store = st.session_state.vector_store
            st.metric(
                "Indexed Chunks",
                f"{vector_store.embedded_chunks}/{vector_store.total_chunks}",
            )


def render_chat_tab(client, model_name: str) -> None:
    if not require_paper():
        return

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask anything about the paper...")
    if not question:
        return

    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            try:
                answer, sources = answer_question(
                    client,
                    st.session_state.vector_store,
                    question,
                    model_name,
                )
                st.markdown(answer)
                render_sources(sources, "Citation sources")
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            except Exception as error:
                st.error(f"Could not answer question: {error}")


def render_summary_tab(client, model_name: str) -> None:
    if not require_paper():
        return

    if st.button("Generate Summary", type="primary"):
        with st.spinner("Generating concise summary..."):
            try:
                st.session_state.summary = generate_from_full_context(
                    client,
                    st.session_state.vector_store,
                    SUMMARY_PROMPT,
                    model_name,
                )
            except Exception as error:
                st.error(f"Could not generate summary: {error}")

    if st.session_state.summary:
        st.markdown(st.session_state.summary)
        render_summary_download()


def render_gaps_tab(client, model_name: str) -> None:
    if not require_paper():
        return

    st.caption("Scores use a 1-5 scale for impact, novelty, and feasibility.")
    if st.button("Find Research Gaps", type="primary"):
        with st.spinner("Reviewing limitations and future work..."):
            try:
                st.session_state.research_gaps = generate_from_full_context(
                    client,
                    st.session_state.vector_store,
                    RESEARCH_GAP_PROMPT,
                    model_name,
                )
            except Exception as error:
                st.error(f"Could not generate research gaps: {error}")

    if st.session_state.research_gaps:
        st.markdown(st.session_state.research_gaps)


def render_interview_tab(client, model_name: str) -> None:
    if not require_paper():
        return

    if st.button("Generate Interview Questions", type="primary"):
        with st.spinner("Creating interview questions with answers..."):
            try:
                st.session_state.interview_questions = generate_from_full_context(
                    client,
                    st.session_state.vector_store,
                    INTERVIEW_PROMPT,
                    model_name,
                )
            except Exception as error:
                st.error(f"Could not generate interview questions: {error}")

    if st.session_state.interview_questions:
        st.markdown(st.session_state.interview_questions)
        render_interview_download()


def render_ask_tab(client, model_name: str) -> None:
    if not require_paper():
        return

    question = st.text_area("Ask a focused question based on the paper")
    if not st.button("Get Answer", type="primary"):
        return

    if not question.strip():
        st.warning("Please enter a question first.")
        return

    with st.spinner("Finding the most relevant paper sections..."):
        try:
            answer, sources = answer_question(
                client,
                st.session_state.vector_store,
                question,
                model_name,
            )
            st.markdown(answer)
            render_sources(sources)
        except Exception as error:
            st.error(f"Could not answer question: {error}")


def render_ppt_tab(client, model_name: str) -> None:
    if not require_paper():
        return

    st.caption("Stretch feature: quickly draft presentation slide content.")
    if st.button("Generate PPT Content"):
        with st.spinner("Drafting slide content..."):
            try:
                st.session_state.ppt_content = generate_from_full_context(
                    client,
                    st.session_state.vector_store,
                    PPT_PROMPT,
                    model_name,
                )
            except Exception as error:
                st.error(f"Could not generate PPT content: {error}")

    if st.session_state.ppt_content:
        st.markdown(st.session_state.ppt_content)


def main() -> None:
    initialize_state()
    render_header()

    try:
        client, model_name = configure_gemini()
    except RuntimeError as error:
        st.error(str(error))
        st.stop()

    render_sidebar(client, model_name)
    render_overview()

    tab_chat, tab_summary, tab_gaps, tab_interview, tab_ask, tab_ppt = st.tabs(
        [
            "Chat with Paper",
            "Summary",
            "Research Gaps",
            "Interview Questions",
            "Ask a Question",
            "PPT Content",
        ]
    )

    with tab_chat:
        render_chat_tab(client, model_name)
    with tab_summary:
        render_summary_tab(client, model_name)
    with tab_gaps:
        render_gaps_tab(client, model_name)
    with tab_interview:
        render_interview_tab(client, model_name)
    with tab_ask:
        render_ask_tab(client, model_name)
    with tab_ppt:
        render_ppt_tab(client, model_name)


if __name__ == "__main__":
    main()
