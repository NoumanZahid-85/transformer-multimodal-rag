import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.rag_pipeline import build, query


def safe(text, max_len=200):
    s = str(text).encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    return s[:max_len]


def print_result(result):
    print("\n" + "=" * 70)
    print(f"QUERY: {result['query']}")
    print("=" * 70)

    print("\n--- RETRIEVED CONTEXT ---")
    for i, r in enumerate(result["retrieved_chunks"], 1):
        chunk = r["chunk"]
        modality = chunk["type"].upper()
        page = chunk["source_page"]
        print(f"\n  [{i}] {modality} | Page {page} | Score: {r['score']:.4f}")
        if chunk.get("heading"):
            print(f"      Heading: {safe(chunk['heading'], 80)}")
        print(f"      {safe(chunk['content'], 250)}...")

    print("\n--- GENERATED ANSWER ---")
    print(result["answer"])
    print("=" * 70 + "\n")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--build":
        build()
        return

    print("Checking vector store...")
    from src.vector_store import load_index
    index, chunks = load_index()
    if index is None:
        print("Vector store not found. Building...")
        build()
    else:
        print(f"Loaded existing vector store with {len(chunks)} chunks.\n")

    demo_queries = [
        "What BLEU score did the Transformer achieve on English-to-German translation and how does it compare to previous models?",
        "Explain the encoder-decoder architecture of the Transformer with its key components.",
        "What is multi-head attention and why is it useful in the Transformer model?"
    ]

    for i, q in enumerate(demo_queries, 1):
        print(f"\n{'#' * 70}")
        print(f"# DEMO QUERY {i}/3")
        print(f"{'#' * 70}")
        result = query(q, top_k=5)
        print_result(result)


if __name__ == "__main__":
    main()
