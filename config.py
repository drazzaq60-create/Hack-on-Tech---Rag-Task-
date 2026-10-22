"""Central settings. Experiments = change one value here and re-run eval."""

import os

# --- Source ---
TUTORIAL_INDEX = "https://docs.python.org/3/tutorial/index.html"
BASE_URL = "https://docs.python.org/3/tutorial/"

# --- Paths (absolute, so the app runs from any working directory) ---
_ROOT = os.path.dirname(os.path.abspath(__file__))
CHUNKS_PATH = os.path.join(_ROOT, "data", "chunks.jsonl")
CONTEXTUAL_PATH = os.path.join(_ROOT, "data", "contextualized.jsonl")  # chunks + an LLM context line
CHROMA_DIR = os.path.join(_ROOT, "data", "chroma")

# --- Chunking (experiment: sweep CHUNK_SIZE 300 vs 800) ---
CHUNK_SIZE = 800          # characters per chunk
CHUNK_OVERLAP = 120       # overlap between consecutive chunks

# --- Models ---
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # baseline (fast, local)
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"    # cross-encoder reranker

# --- Retrieval ---
TOP_K = 5                 # passages returned / scored (Hit Rate@5)
CANDIDATES = 20           # retrieve this many before reranking

# --- Service guards ---
MAX_QUESTION_CHARS = 1000     # reject longer questions
RATE_LIMIT_PER_MIN = 30       # max /ask requests per client IP per minute

# --- LLM ---
GEMINI_MODEL = "gemini-3.6-flash"              # answer generation (quality matters, few calls)
CONTEXTUALIZE_MODEL = "gemini-3.5-flash-lite"  # bulk chunk labelling (cheap, generous free quota)
CONTEXTUALIZE_BATCH = 10                        # chunks per request (fewer calls = kinder to free tier)
