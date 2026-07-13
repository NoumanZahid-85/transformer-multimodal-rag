import json
import numpy as np
import faiss

from src.config import CHUNKS_PATH, INDEX_PATH, VECTOR_STORE_DIR, EMBED_DIM


def build_index(chunks, embeddings):
    import os
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    embedding_matrix = np.array(embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embedding_matrix)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    faiss.write_index(index, str(INDEX_PATH))
    print(f"Saved {len(chunks)} chunks and FAISS index to {VECTOR_STORE_DIR}")
    return index


def load_index():
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        return None, None

    index = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks


def search(index, query_embedding, chunks, top_k=5):
    query_vector = np.array([query_embedding], dtype=np.float32)
    scores, indices = index.search(query_vector, top_k)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(chunks):
            chunk = chunks[idx]
            results.append({
                "chunk": chunk,
                "score": float(score)
            })
    return results
