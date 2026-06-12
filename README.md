# PaperMind AI

**GenAI-powered Research Paper Intelligence Platform**

🔗 Live Demo: [![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://papermind-ai-gvkr.streamlit.app/)

📂 Source Code: https://github.com/<your-username>/PaperMind-AI

PaperMind AI is a Streamlit app for reading research papers faster. Upload a text-based PDF, ask grounded questions, generate summaries, identify research gaps, prepare interview answers, and export study material.

## Features

- Upload text-based research paper PDFs
- Extract paper metadata: title, authors, year, and page count
- Chunk paper text with page tracking
- Generate Gemini embeddings and store vectors locally with FAISS
- Retrieve top relevant chunks for RAG answers
- Generate citation-aware answers with page references
- Create concise paper summaries
- Identify limitations, open challenges, and future research gaps
- Score research gaps by impact, novelty, and feasibility
- Generate technical interview questions with answers
- Draft PPT-ready slide content
- Download summaries as PDF
- Download interview questions as PDF

## Architecture

```text
PDF Upload
  -> PyPDF Text Extraction
  -> Metadata Extraction
  -> Page-Aware Text Chunking
  -> Gemini Embeddings
  -> FAISS Vector Store
  -> Top-k Retrieval
  -> Gemini Response with Citations
  -> PDF Export
```

## Project Structure

```text
PaperMind-AI/
├── app.py
├── rag.py
├── prompts.py
├── utils.py
├── requirements.txt
└── README.md
```

## Local Setup

1. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Add your Gemini API key.

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
MAX_EMBEDDING_CHUNKS=90
```

4. Run the app.

```bash
streamlit run app.py
```

## Streamlit Community Cloud Deployment

1. Push `PaperMind-AI` to a GitHub repository.
2. Open Streamlit Community Cloud.
3. Create a new app and select `app.py`.
4. Add secrets in Streamlit Cloud:

```toml
GEMINI_API_KEY = "your_api_key_here"
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
MAX_EMBEDDING_CHUNKS = "90"
```

5. Deploy the app.

## Notes

- Designed for single-user demos, student projects, and research workflows.
- Scanned PDFs may not work because they require OCR.
- No authentication, database, or multi-user persistence is included by design.
- Large papers are capped to the first 90 chunks by default to avoid Gemini free-tier embedding limits. Increase `MAX_EMBEDDING_CHUNKS` if your quota allows it.
- Page citations are generated from retrieved text chunks, so citations depend on PDF text extraction quality.

## Tech Stack

- Python
- Streamlit
- Gemini 2.5 Flash
- Gemini Embeddings
- FAISS
- PyPDF
- ReportLab
