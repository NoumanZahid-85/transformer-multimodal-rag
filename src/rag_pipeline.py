import time
from src.pdf_processor import extract_text_chunks, extract_images, extract_tables
from src.gemini_interface import caption_image, format_table_as_text, generate_embeddings, generate_answer
from src.vector_store import build_index, load_index, search
from src.config import PDF_PATH


def build():
    print("=" * 60)
    print("STEP 1: Extracting text chunks from PDF...")
    text_chunks = extract_text_chunks(PDF_PATH)
    print(f"  -> {len(text_chunks)} text chunks extracted")

    print("STEP 2: Extracting and captioning images...")
    image_data_list = extract_images(PDF_PATH)
    image_chunks = []
    for idx, img_data in enumerate(image_data_list):
        caption = caption_image(img_data["image"])
        image_chunks.append({
            "content": caption,
            "type": "image",
            "source_page": img_data["source_page"],
            "heading": None
        })
        print(f"  -> Captioned image from page {img_data['source_page']}")
        if idx < len(image_data_list) - 1:
            time.sleep(2)
    print(f"  -> {len(image_chunks)} images processed")

    print("STEP 3: Extracting tables...")
    table_data_list = extract_tables(PDF_PATH)
    table_chunks = []
    for table_data in table_data_list:
        table_text = format_table_as_text(table_data["data"])
        table_chunks.append({
            "content": table_text,
            "type": "table",
            "source_page": table_data["source_page"],
            "heading": None
        })
    print(f"  -> {len(table_chunks)} tables extracted")

    all_chunks = text_chunks + image_chunks + table_chunks
    for i, chunk in enumerate(all_chunks):
        chunk["id"] = f"{chunk['type']}_{i}"

    print(f"\nTotal chunks: {len(all_chunks)}")
    type_counts = {}
    for c in all_chunks:
        type_counts[c["type"]] = type_counts.get(c["type"], 0) + 1
    for t, count in type_counts.items():
        print(f"  {t}: {count}")

    print("\nSTEP 4: Generating embeddings...")
    texts_to_embed = [c["content"] for c in all_chunks]
    embeddings = generate_embeddings(texts_to_embed)
    print(f"  -> {len(embeddings)} embeddings generated (dim={len(embeddings[0])})")

    print("STEP 5: Building and saving vector store...")
    build_index(all_chunks, embeddings)

    print("\nDONE! Vector store is ready for queries.\n")


def query(query_text, top_k=5):
    index, chunks = load_index()
    if index is None or chunks is None:
        raise RuntimeError("Vector store not found. Run build() first or call pipeline.build().")

    query_embedding = generate_embeddings([query_text])[0]
    results = search(index, query_embedding, chunks, top_k=top_k)

    context_parts = []
    for r in results:
        chunk = r["chunk"]
        label = f"[{chunk['type'].upper()}] (page {chunk['source_page']})"
        if chunk["heading"]:
            label += f" - {chunk['heading']}"
        context_parts.append(f"{label}\n{chunk['content']}")
    context = "\n\n---\n\n".join(context_parts)

    answer = generate_answer(query_text, context)

    return {
        "query": query_text,
        "retrieved_chunks": results,
        "context": context,
        "answer": answer
    }
