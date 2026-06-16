# PaperMind AI

**GenAI-powered Multi-Document Research Intelligence Platform**

PaperMind AI is a Streamlit-native application designed to accelerate scientific literature analysis. Upload multiple research papers (PDFs) into an active workspace, chat across documents with page-level grounding, compare methodologies in lit matrices, discover dataset limitations in scored tables, and run external search agents to recommend new literature.

---

## Key Features

1.  **Multi-Document Workspace**: Upload multiple PDFs. The app extracts metadata (title, authors, year, pages) and cumulatively indexes them into a local FAISS vector store.
2.  **AI Research Agent**: Autonomously queries the arXiv XML API on a topic, extracts metadata/abstracts, performs a semantic comparison against your workspace using Gemini, and recommends papers to add.
3.  **Grounded Chat**: Ask questions across all workspace papers. Gemini answers using strictly retrieved contexts and cites page numbers like `[p. X, "Paper Title"]` inside glowing glassmorphic cards.
4.  **Document Summary**: Generates a unified overview summarizing the core research objectives, methodology commonalities, and key findings of the indexed papers.
5.  **Literature Review Matrix**: Synthesizes workspace documents into a comparative review layout, highlighting architecture differences and open controversies, with support for custom focus prompts.
6.  **Research Gap Scorecard**: Evaluates methodology limitations and unresolved challenges, presenting them in a compact, styled scorecard table with novelty, feasibility, and impact metrics.
7.  **Technical Interview Prep**: Generates focused questions and answers grounded in your documents for thesis defense or study preparation.
8.  **Presentation Slides**: Outlines slide layouts (collective objectives, benchmarks, limitations) to jumpstart slide decks.
9.  **Rate-Limit Resilience**: Incorporates automatic exponential backoff retries (catching `429 RESOURCE_EXHAUSTED` errors) to handle large document ingestion smoothly in the Gemini free tier.
10. **Report Exports**: Compile summaries, reviews, gaps, and interview prep into raw Markdown or publication-quality PDFs.

---

## Design & Aesthetics

The interface is styled using a custom glassmorphic stylesheet:
*   **Theme**: Deep indigo-slate radial background canvas (`#17153B` to `#03001C`).
*   **Layout**: Clear sidebar manager with document cards, pill-shaped central tab navigations, and expandable evidence blocks.
*   **Typography**: Clean sans-serif weights powered by Google Fonts (*Inter* and *Outfit*).

---

## Tech Stack

*   **Core**: Python, Streamlit
*   **LLM & RAG**: Google GenAI SDK (Gemini 2.5 Flash, Gemini Embeddings)
*   **Vector Search**: Facebook AI Similarity Search (FAISS)
*   **PDF Extraction**: PyPDF
*   **Export Engine**: ReportLab (for custom-styled PDFs)

---

## Local Setup

1.  **Activate Virtual Environment**:
    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Create `.env` File**:
    ```env
    GEMINI_API_KEY=your_gemini_api_key_here
    GEMINI_MODEL=gemini-2.5-flash
    GEMINI_EMBEDDING_MODEL=gemini-embedding-001
    MAX_EMBEDDING_CHUNKS=80
    ```
4.  **Run Streamlit**:
    ```bash
    streamlit run app.py
    ```
