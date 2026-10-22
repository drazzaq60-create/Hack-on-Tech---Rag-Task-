# The retriever: given a question, find the chunks most likely to answer it.
#
# Two stages:
#   search_vector  - meaning search over the contextualized chunks (casts a wide net)
#   search_rerank  - a cross-encoder reads (question, chunk) together and keeps the best 5
# search_rerank is what the service uses.
#
# (I also built BM25 keyword search + hybrid fusion while developing, and measured
# they added nothing once the reranker was in place, so I removed them. That whole
# experiment and its numbers are written up in the README.)
#
# Quick try: python src/retriever.py "How do I write a for loop?"

import os
import sys
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

COLLECTION = "pydocs"

# Load these ONCE (they're reused for every question).
_model = SentenceTransformer(config.EMBED_MODEL)
_reranker = CrossEncoder(config.RERANK_MODEL)   # reads (question, chunk) together
_client = chromadb.PersistentClient(path=config.CHROMA_DIR)
try:
    _col = _client.get_collection(COLLECTION)
except Exception:
    raise RuntimeError(
        f"Vector index '{COLLECTION}' not found at {config.CHROMA_DIR}. "
        "Build it first:  python src/index.py --contextual"
    )

# The embedding + cross-encoder models aren't guaranteed thread-safe, and FastAPI
# runs sync endpoints in a threadpool, so guard inference with a lock.
_lock = threading.Lock()


def search_vector(query, k=config.TOP_K):
    """Return the top-k chunks by meaning similarity to the query."""
    with _lock:
        q_emb = _model.encode([query], normalize_embeddings=True).tolist()
    res = _col.query(
        query_embeddings=q_emb,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for i in range(len(res["ids"][0])):
        hits.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "section_title": res["metadatas"][0][i]["section_title"],
            "url": res["metadatas"][0][i]["url"],
            # cosine distance -> similarity score (higher = more relevant)
            "score": round(1 - res["distances"][0][i], 3),
        })
    return hits


def _rerank(query, candidates, k):
    """Cross-encoder reads (question, chunk) together and keeps the best k."""
    if not candidates:
        return []
    pairs = [[query, c["text"]] for c in candidates]
    with _lock:
        scores = _reranker.predict(pairs)   # higher = more relevant
    for c, s in zip(candidates, scores):
        c["score"] = round(float(s), 3)
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:k]


def search_rerank(query, k=config.TOP_K):
    """THE PRODUCTION PIPELINE. Vector casts a wide net (recall), reranker sorts it (precision)."""
    return _rerank(query, search_vector(query, config.CANDIDATES), k)


def _pretty(hits):
    for rank, h in enumerate(hits, 1):
        snippet = h["text"].replace("\n", " ")[:160]
        print(f"\n[{rank}] score={h['score']}  |  {h['section_title']}")
        print(f"    {h['url']}")
        print(f"    {snippet}...")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How do I write a for loop in Python?"
    print(f"Question: {question}")
    print("=" * 70)
    _pretty(search_rerank(question))
