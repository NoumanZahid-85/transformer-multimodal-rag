import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file. Create a .env file with GEMINI_API_KEY=your_key")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "attention-is-all-you-need.pdf"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
CHUNKS_PATH = VECTOR_STORE_DIR / "chunks.json"
INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"
VISION_MODEL = "gemini-2.5-flash"
EMBED_DIM = 3072
