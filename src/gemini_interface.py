import time
from google import genai
from google.genai import errors as genai_errors

from src.config import GEMINI_API_KEY, CHAT_MODEL, VISION_MODEL, EMBED_MODEL

client = genai.Client(api_key=GEMINI_API_KEY)


def _with_retry(fn, max_retries=10):
    for attempt in range(max_retries):
        try:
            return fn()
        except genai_errors.APIError as e:
            if e.code in (429, 500, 503) and attempt < max_retries - 1:
                wait = min(2 ** (attempt + 2), 120)
                print(f"  API error {e.code}. Retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise


def caption_image(image):
    prompt = (
        "Describe this figure from the 'Attention Is All You Need' paper in detail. "
        "Explain what it shows, its components, labels, and significance to the Transformer architecture. "
        "Be thorough and technical."
    )
    def _call():
        return client.models.generate_content(
            model=VISION_MODEL,
            contents=[image, prompt]
        )
    response = _with_retry(_call)
    return response.text.strip()


def format_table_as_text(table_data):
    rows = []
    for row in table_data:
        rows.append(" | ".join(row))
    header = rows[0]
    sep = " | ".join(["---"] * len(table_data[0]))
    body = "\n".join(rows[1:]) if len(rows) > 1 else ""
    return f"{header}\n{sep}\n{body}"


def generate_embeddings(texts):
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        def _call(b=batch):
            return client.models.embed_content(
                model=EMBED_MODEL,
                contents=b
            )
        result = _with_retry(_call)
        all_embeddings.extend([e.values for e in result.embeddings])
    return all_embeddings


def generate_answer(query, context):
    system_prompt = (
        "You are a helpful assistant answering questions about the 'Attention Is All You Need' paper. "
        "Answer the user's question based strictly on the provided context. "
        "If the context does not contain enough information, say so. "
        "Cite which part of the context (text, table, or figure description) supports your answer."
    )
    full_prompt = f"{system_prompt}\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"
    def _call():
        return client.models.generate_content(
            model=CHAT_MODEL,
            contents=[full_prompt]
        )
    response = _with_retry(_call)
    return response.text.strip()
