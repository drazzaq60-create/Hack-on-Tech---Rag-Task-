# Step 2: make the chunks searchable by meaning.
# Embed each chunk with a sentence-transformer and store the vectors in ChromaDB.
# Run this after ingest.py (plain chunks) or after contextualize.py with --contextual.

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from sentence_transformers import SentenceTransformer
import chromadb

COLLECTION = "pydocs"


def load_chunks(path):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def main():
    # pass --contextual to index the Gemini-contextualized chunks instead of the plain ones
    use_ctx = "--contextual" in sys.argv
    path = config.CONTEXTUAL_PATH if use_ctx else config.CHUNKS_PATH
    print(f"Indexing from: {path}" + ("  (contextual)" if use_ctx else "  (plain)"))
    chunks = load_chunks(path)
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metas = [
        {
            "section_title": c["section_title"],
            "url": c["url"],
            "heading_path": c["heading_path"],
        }
        for c in chunks
    ]
    print(f"Loaded {len(chunks)} chunks.")

    print(f"Embedding with {config.EMBED_MODEL} (first run downloads the model)...")
    model = SentenceTransformer(config.EMBED_MODEL)
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    ).tolist()

    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION)   # fresh rebuild each run
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    col.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metas)
    print(f"  Stored {col.count()} vectors in ChromaDB at {config.CHROMA_DIR}")
    print("\nIndex ready.")


if __name__ == "__main__":
    main()
