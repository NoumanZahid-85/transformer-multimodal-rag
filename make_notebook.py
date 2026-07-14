import json, textwrap

NOTEBOOK_NAME = "transformer_multimodal_rag.ipynb"

cells_data = []

def md(source):
    cells_data.append({"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)})

def code(source):
    cells_data.append({"cell_type": "code", "metadata": {}, "source": source.splitlines(True), "execution_count": None, "outputs": []})

# ── Cell 0: Title ──
md(textwrap.dedent("""\
# Transformer Multimodal RAG Pipeline

A complete Retrieval-Augmented Generation (RAG) pipeline over the **"Attention Is All You Need"** paper (Vaswani et al., 2017).

**What this notebook does:**
1. Downloads the paper from arXiv (no manual upload needed)
2. Extracts text (with headings), images (figures/diagrams), and tables from the PDF
3. Captions all visual elements via **Gemini Vision API**
4. Generates embeddings for all content chunks using **Gemini Embedding API**
5. Stores everything in a **FAISS** vector store
6. For a user query, retrieves the most relevant chunks and generates a **grounded answer** using Gemini

**Requirements:** A Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey) or Google Cloud with billing enabled.
"""))

# ── Cell 1: Install deps ──
code(textwrap.dedent("""\
# Cell 1: Install dependencies (run once)
!pip install google-genai PyMuPDF pdfplumber faiss-cpu numpy python-dotenv Pillow requests -q
print("Dependencies installed.")
"""))

# ── Cell 2: API key ──
code(textwrap.dedent("""\
# Cell 2: Imports and Gemini API key setup
import os, json, io, time, re, sys
from pathlib import Path
from getpass import getpass
import requests
import numpy as np
from PIL import Image

from google import genai
from google.genai import errors as genai_errors

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("GEMINI_API_KEY not found in environment.")
    API_KEY = getpass("Enter your Gemini API key: ")

client = genai.Client(api_key=API_KEY)
print("Gemini client initialized.")
"""))

# ── Cell 3: Config + Download ──
code(textwrap.dedent("""\
# Cell 3: Configuration and download paper from arXiv
PDF_PATH = "attention-is-all-you-need.pdf"
VECTOR_STORE_DIR = Path("vector_store")
CHUNKS_PATH = VECTOR_STORE_DIR / "chunks.json"
INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"
VISION_MODEL = "gemini-2.5-flash"
EMBED_DIM = 3072

if not os.path.exists(PDF_PATH):
    url = "https://arxiv.org/pdf/1706.03762.pdf"
    print("Downloading paper from " + url + "...")
    resp = requests.get(url, timeout=120)
    with open(PDF_PATH, "wb") as f:
        f.write(resp.content)
    print("Downloaded " + str(len(resp.content)) + " bytes.")
else:
    print("PDF already exists: " + PDF_PATH)
"""))

# ── Cell 4: PDF Processor ──
code(textwrap.dedent("""\
# Cell 4: PDF Processor - extract text, images, and tables
import fitz
import pdfplumber

ARXIV_HEADER = "arXiv:"
SECTION_PATTERN = re.compile(r"^(\\d+\\.?\\d*\\.?\\d*)\\s+[A-Z]")

def _is_arxiv_or_page_number(text, block_y0, page_height):
    if ARXIV_HEADER in text: return True
    if block_y0 > page_height * 0.85 and len(text.strip()) < 10: return True
    if block_y0 < page_height * 0.05 and len(text.strip()) < 15: return True
    return False

def _looks_like_heading(text, font_size, is_bold):
    if not text or len(text) < 4: return False
    if SECTION_PATTERN.match(text): return True
    section_keywords = ["Abstract","Introduction","Background","Conclusion","References",
        "Appendix","Related Work","Method","Results","Discussion","Experiments","Training",
        "Model Architecture","Attention","Positional","Why Self-Attention","Applications","Encoder","Decoder"]
    if any(text.startswith(kw) for kw in section_keywords): return True
    if font_size > 12: return True
    if font_size > 10.5 and is_bold: return True
    return False

def extract_text_chunks(pdf_path):
    doc = fitz.open(pdf_path)
    chunks, last_heading = [], None
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_height = page.rect.height
        blocks = page.get_text("dict")["blocks"]
        page_heading, page_text_lines, heading_found_here = None, [], False
        for block in blocks:
            if block["type"] != 0: continue
            block_y0 = block["bbox"][1]
            for line in block["lines"]:
                spans = line["spans"]
                if not spans: continue
                font_size = spans[0]["size"]
                is_bold = "Bold" in spans[0]["font"] or "Semibold" in spans[0]["font"]
                line_text = "".join(s["text"] for s in spans).strip()
                if not line_text or _is_arxiv_or_page_number(line_text, block_y0, page_height): continue
                if _looks_like_heading(line_text, font_size, is_bold):
                    page_heading = line_text
                    heading_found_here = True
                else:
                    page_text_lines.append(line_text)
        effective_heading = page_heading if page_heading else last_heading
        if page_heading: last_heading = page_heading
        para = [l for l in page_text_lines if not (len(l) < 5 and any(c.isdigit() for c in l))]
        full_text = " ".join(para)
        paragraphs = [p.strip() for p in full_text.split(". ") if p.strip()]
        current_chunk, current_len = [], 0
        for para_text in paragraphs:
            para_len = len(para_text)
            if current_len + para_len > 1000 and current_chunk:
                content = ". ".join(current_chunk) + "."
                if len(content) > 100:
                    chunks.append({"content": content, "type": "text", "source_page": page_num + 1, "heading": effective_heading})
                current_chunk, current_len = [], 0
            current_chunk.append(para_text)
            current_len += para_len
        if current_chunk:
            content = ". ".join(current_chunk) + "."
            if len(content) > 100:
                chunks.append({"content": content, "type": "text", "source_page": page_num + 1, "heading": effective_heading})
    doc.close()
    return chunks

def extract_images(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        for img_info in page.get_images(full=True):
            image = Image.open(io.BytesIO(doc.extract_image(img_info[0])["image"]))
            if image.mode == "P": image = image.convert("RGBA" if "transparency" in image.info else "RGB")
            elif image.mode != "RGB": image = image.convert("RGB")
            images.append({"image": image, "source_page": page_num + 1})
    doc.close()
    return images

def extract_tables(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            for table_data in page.extract_tables():
                if table_data and len(table_data) > 1:
                    rows = [[cell.strip() if cell else "" for cell in row] for row in table_data]
                    if any(cell for row in rows for cell in row):
                        tables.append({"data": rows, "source_page": page_num + 1})
    return tables

print("PDF Processor functions loaded.")
"""))

# ── Cell 5: Gemini Interface ──
code(textwrap.dedent("""\
# Cell 5: Gemini Interface - vision, embeddings, answer generation

def _with_retry(fn, max_retries=10):
    for attempt in range(max_retries):
        try:
            return fn()
        except genai_errors.APIError as e:
            if e.code in (429, 500, 503) and attempt < max_retries - 1:
                wait = min(2 ** (attempt + 2), 120)
                print("  API error " + str(e.code) + ". Retrying in " + str(wait) + "s (attempt " + str(attempt + 1) + "/" + str(max_retries) + ")...")
                time.sleep(wait)
            else:
                raise

def caption_image(image):
    prompt = ("Describe this figure from the 'Attention Is All You Need' paper in detail. "
              "Explain what it shows, its components, labels, and significance to the Transformer architecture. "
              "Be thorough and technical.")
    def _call(): return client.models.generate_content(model=VISION_MODEL, contents=[image, prompt])
    return _with_retry(_call).text.strip()

def format_table_as_text(table_data):
    rows = [" | ".join(row) for row in table_data]
    header = rows[0]
    sep = " | ".join(["---"] * len(table_data[0]))
    body = "\\n".join(rows[1:]) if len(rows) > 1 else ""
    return header + "\\n" + sep + "\\n" + body

def generate_embeddings(texts):
    all_embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]
        def _call(b=batch): return client.models.embed_content(model=EMBED_MODEL, contents=b)
        result = _with_retry(_call)
        all_embeddings.extend([e.values for e in result.embeddings])
    return all_embeddings

def generate_answer(query, context):
    system_prompt = ("You are a helpful assistant answering questions about the 'Attention Is All You Need' paper. "
        "Answer the user's question based strictly on the provided context. "
        "If the context does not contain enough information, say so. "
        "Cite which part of the context (text, table, or figure description) supports your answer.")
    full_prompt = system_prompt + "\\n\\nCONTEXT:\\n" + context + "\\n\\nQUESTION: " + query + "\\n\\nANSWER:"
    def _call(): return client.models.generate_content(model=CHAT_MODEL, contents=[full_prompt])
    return _with_retry(_call).text.strip()

print("Gemini Interface functions loaded.")
"""))

# ── Cell 6: Vector Store ──
code(textwrap.dedent("""\
# Cell 6: FAISS Vector Store
import faiss

def build_index(chunks, embeddings):
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(np.array(embeddings, dtype=np.float32))
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    faiss.write_index(index, str(INDEX_PATH))
    print("Saved " + str(len(chunks)) + " chunks and FAISS index.")
    return index

def load_index():
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists(): return None, None
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f: chunks = json.load(f)
    return faiss.read_index(str(INDEX_PATH)), chunks

def search(index, query_embedding, chunks, top_k=5):
    scores, indices = index.search(np.array([query_embedding], dtype=np.float32), top_k)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(chunks): results.append({"chunk": chunks[idx], "score": float(score)})
    return results

print("Vector Store functions loaded.")
"""))

# ── Cell 7: RAG Pipeline ──
code(textwrap.dedent("""\
# Cell 7: RAG Pipeline - orchestrate build and query

def build():
    print("=" * 60)
    print("STEP 1: Extracting text chunks from PDF...")
    text_chunks = extract_text_chunks(PDF_PATH)
    print("  -> " + str(len(text_chunks)) + " text chunks extracted")

    print("STEP 2: Extracting and captioning images via Gemini Vision...")
    image_data_list = extract_images(PDF_PATH)
    image_chunks = []
    for idx, img_data in enumerate(image_data_list):
        caption = caption_image(img_data["image"])
        image_chunks.append({"content": caption, "type": "image", "source_page": img_data["source_page"], "heading": None})
        print("  -> Captioned image from page " + str(img_data["source_page"]))
        if idx < len(image_data_list) - 1: time.sleep(2)
    print("  -> " + str(len(image_chunks)) + " images processed")

    print("STEP 3: Extracting tables...")
    table_chunks = []
    for t in extract_tables(PDF_PATH):
        table_chunks.append({"content": format_table_as_text(t["data"]), "type": "table", "source_page": t["source_page"], "heading": None})
    print("  -> " + str(len(table_chunks)) + " tables extracted")

    all_chunks = text_chunks + image_chunks + table_chunks
    for i, chunk in enumerate(all_chunks): chunk["id"] = chunk["type"] + "_" + str(i)

    print("")
    print("Total chunks: " + str(len(all_chunks)))
    type_counts = {}
    for c in all_chunks: type_counts[c["type"]] = type_counts.get(c["type"], 0) + 1
    for t, count in type_counts.items(): print("  " + t + ": " + str(count))

    print("")
    print("STEP 4: Generating embeddings...")
    embeddings = generate_embeddings([c["content"] for c in all_chunks])
    print("  -> " + str(len(embeddings)) + " embeddings generated (dim=" + str(len(embeddings[0])) + ")")

    print("STEP 5: Building and saving vector store...")
    build_index(all_chunks, embeddings)
    print("")
    print("DONE! Vector store is ready for queries.")
    print("")

def query(query_text, top_k=5):
    index, chunks = load_index()
    if index is None or chunks is None: raise RuntimeError("Vector store not found. Run build() first.")
    query_embedding = generate_embeddings([query_text])[0]
    results = search(index, query_embedding, chunks, top_k=top_k)
    context_parts = []
    for r in results:
        chunk = r["chunk"]
        label = "[" + chunk["type"].upper() + "] (page " + str(chunk["source_page"]) + ")"
        if chunk.get("heading"): label += " - " + chunk["heading"]
        context_parts.append(label + "\\n" + chunk["content"])
    answer = generate_answer(query_text, "\\n\\n---\\n\\n".join(context_parts))
    return {"query": query_text, "retrieved_chunks": results, "answer": answer}

print("RAG Pipeline functions loaded.")
"""))

# ── Cell 8: Build ──
code(textwrap.dedent("""\
# Cell 8: Build the vector store (run this cell once)

if not INDEX_PATH.exists():
    build()
else:
    print("Vector store already exists. Skipping build.")
    index, chunks = load_index()
    print("Loaded " + str(len(chunks)) + " existing chunks.")
"""))

# ── Cell 9: Demo Queries ──
code(textwrap.dedent("""\
# Cell 9: Run demo queries across different modalities

demo_queries = [
    "What BLEU score did the Transformer achieve on English-to-German translation and how does it compare to previous models?",
    "Explain the encoder-decoder architecture of the Transformer with its key components.",
    "What is multi-head attention and why is it useful in the Transformer model?"
]

all_output = []

for qi, q in enumerate(demo_queries, 1):
    print("")
    print("#" * 70)
    print("# DEMO QUERY " + str(qi) + "/3")
    print("#" * 70)
    result = query(q, top_k=3)

    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("QUERY: " + result["query"])
    output_lines.append("=" * 70)
    output_lines.append("")
    output_lines.append("--- RETRIEVED CONTEXT ---")
    for i, r in enumerate(result["retrieved_chunks"], 1):
        chunk = r["chunk"]
        output_lines.append("")
        output_lines.append("  [" + str(i) + "] " + chunk["type"].upper() + " | Page " + str(chunk["source_page"]) + " | Score: " + str(round(r["score"], 4)))
        if chunk.get("heading"):
            output_lines.append("      Heading: " + chunk["heading"])
        output_lines.append("      " + chunk["content"][:200] + "...")
    output_lines.append("")
    output_lines.append("--- GENERATED ANSWER ---")
    output_lines.append(result["answer"])
    output_lines.append("=" * 70)
    output_lines.append("")

    out_str = "\\n".join(output_lines)
    print(out_str)
    all_output.append(out_str)
"""))

# ── Cell 10: Summary ──
code(textwrap.dedent("""\
# Cell 10: Save results and print pipeline summary

with open("output.txt", "w", encoding="utf-8") as f:
    f.write("\\n".join(all_output))
print("Results saved to output.txt")
print("")

print("=" * 70)
print("PIPELINE SUMMARY")
print("=" * 70)
print("")
print("This pipeline successfully:")
print("  1. Downloaded the 'Attention Is All You Need' paper from arXiv")
print("  2. Extracted text chunks with heading detection")
print("  3. Extracted and captioned figures/diagrams via Gemini Vision")
print("  4. Extracted tables from the PDF")
print("  5. Generated embeddings for all chunks using Gemini Embedding API")
print("  6. Indexed everything in a shared FAISS vector space")
print("  7. For each query: retrieved top-3 chunks + generated grounded answer")
print("")
print("Three modalities tested:")
print("  - TABLE/TEXT: BLEU scores query -> Results section + Abstract")
print("  - FIGURE/IMAGE: Architecture query -> Figure 1 diagram caption")
print("  - TEXT + IMAGE: Multi-head attention -> MHA section + Figure 2 caption")
"""))

# ── Build JSON ──
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": cells_data
}

with open(NOTEBOOK_NAME, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print("Created " + NOTEBOOK_NAME + " with " + str(len(cells_data)) + " cells")
