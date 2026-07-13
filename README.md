# Transformer Multimodal RAG

A multimodal Retrieval-Augmented Generation (RAG) pipeline over the **"Attention Is All You Need"** paper using the **Gemini API**. Extracts text, tables, and figures from the PDF, captions images via **Gemini Vision**, embeds everything into **FAISS**, and answers queries with grounded generation.

## Architecture

```
attention-is-all-you-need.pdf
         │
         ▼
┌──────────────────────────────────────┐
│          PDF Processor                │
│  ┌─────────┬──────────┬──────────┐   │
│  │  Text   │  Images  │  Tables  │   │
│  │ (fitz)  │ (fitz)   │(pdfplumb)│   │
│  └────┬────┴────┬─────┴────┬─────┘   │
└───────┼─────────┼──────────┼─────────┘
        │         │          │
        │    ┌────┘          │
        │    ▼               │
        │  Gemini Vision     │
        │  (caption image)   │
        │    │               │
        └────┼───────────────┘
             ▼
     ┌───────────────┐
     │  56 Chunks    │
     │ (text+img+tbl)│
     └───────┬───────┘
             ▼
     Gemini Embeddings
     (3072-dim vectors)
             ▼
     ┌───────────────┐
     │  FAISS Index  │
     │(cosine search)│
     └───────────────┘
             │
   Query ────┤
             ▼
     Retrieve top-k chunks
             │
             ▼
     Gemini Chat (2.5 Flash)
     (grounded answer)
             │
             ▼
         Final Answer
```

## Project Structure

```
transformer-multimodal-rag/
├── .env.example              # Template for API key (copy to .env)
├── .gitignore
├── requirements.txt
├── main.py                   # Entry point: --build or run demo queries
├── attention-is-all-you-need.pdf
├── vector_store/             # Built artifacts (gitignored)
│   ├── chunks.json           # 56 chunks with metadata
│   └── index.faiss           # FAISS index
└── src/
    ├── __init__.py
    ├── config.py             # Paths, model names, env loading
    ├── pdf_processor.py      # Extract text (w/ heading detection), images, tables
    ├── gemini_interface.py   # Vision, embeddings, answer generation via Gemini
    ├── vector_store.py       # FAISS index build, save, load, search
    └── rag_pipeline.py       # Orchestrates build & query flow
```

## Quick Start

### Prerequisites
- Python 3.10+
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey) (or Google Cloud with billing enabled)

### Setup

```bash
# 1. Clone
git clone https://github.com/NoumanZahid-85/transformer-multimodal-rag.git
cd transformer-multimodal-rag

# 2. Create virtual environment
python -m venv .venv

# 3. Activate
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Set API key
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux
# Then edit .env and paste your Gemini API key
```

### Build the Vector Store (first time only)

```bash
python main.py --build
```

This extracts text/images/tables from the PDF, captions images via Gemini Vision, generates embeddings, and saves the FAISS index.

### Run Demo Queries

```bash
python main.py
```

Runs 3 sample queries covering different modalities (table, figure, text).

## Demo Queries

| # | Query | Modality Tested | Expected Source |
|---|-------|----------------|-----------------|
| 1 | *"What BLEU score did the Transformer achieve on English-to-German translation?"* | **Table/Text** | Section 5 (Results), Table 2 |
| 2 | *"Explain the encoder-decoder architecture of the Transformer."* | **Figure/Image** | Figure 1 architecture diagram |
| 3 | *"What is multi-head attention and why is it useful?"* | **Text + Image** | Section 3.2.2, Figure 2 |

## How It Works

### Build Phase
1. **Text Extraction** — PyMuPDF reads each page, detects headings by font size/boldness, splits into ~1000-char chunks
2. **Image Extraction** — PyMuPDF extracts embedded images; Gemini Vision captions each with a technical description
3. **Table Extraction** — pdfplumber detects table structures; formatted as markdown-style text
4. **Embedding** — All chunks are embedded into 3072-dim vectors via `gemini-embedding-001`
5. **Indexing** — FAISS `IndexFlatIP` stores vectors for cosine similarity search

### Query Phase
1. Query is embedded using the same model
2. FAISS returns top-3 most similar chunks
3. Retrieved chunks are assembled as context with modality labels
4. Gemini 2.5 Flash generates a grounded answer citing sources

## API Reference

### `main.py`

```bash
python main.py              # Run 3 demo queries
python main.py --build      # Build/rebuild vector store from scratch
```

### `src/rag_pipeline.py`

```python
from src.rag_pipeline import build, query

build()                               # Build vector store
result = query("your question here")  # Returns dict with answer + context
```

## Notes
- Your API key is loaded from `.env` (gitignored) — never hardcode it
- The vector store is gitignored; run `--build` to recreate it
- The PDF is included in the repo for reproducibility
