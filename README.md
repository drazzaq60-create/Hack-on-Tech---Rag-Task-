# Python Tutorial RAG

A retrieval-augmented generation (RAG) service that answers questions about Python
using the official [Python 3 Tutorial](https://docs.python.org/3/tutorial/) as its
**only** source of truth. It retrieves the relevant documentation, answers strictly
from it with inline citations, and refuses when the answer isn't in the docs.

## Summary

- **What:** a FastAPI `/ask` endpoint (plus a small web UI) that returns a grounded,
  cited answer, or an honest refusal.
- **Pipeline:** structure-aware chunking → Contextual Retrieval → vector search →
  cross-encoder reranker → Gemini (grounded + cited + refusal).
- **Result:** **Hit@5 = 1.00, MRR = 0.812** on a 20-question evaluation set.
- **Design principle:** build each stage, *measure* whether it helps, and keep only
  what earns its place. Retrieval was built and tested first: if it hands the LLM the
  wrong pages, no prompting can save the answer.

## Quickstart

The chunked and contextualized data is committed (`data/*.jsonl`), so no re-scraping
or Gemini quota is needed to run it.

```bash
python -m venv venv
venv\Scripts\activate                 # Windows  (source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

copy .env.example .env                 # then edit .env, paste your free Gemini key
                                       # (https://aistudio.google.com/app/apikey)

python src/index.py --contextual       # build the vector index (no API calls)
uvicorn src.app:app                     # serve -> http://127.0.0.1:8000
```

## How it works

```
question
   -> [ retrieve ]  vector search over the docs, then a cross-encoder reranks the top candidates
   -> [ generate ]  Gemini answers ONLY from those chunks, cites them, or refuses if absent
   -> answer + source links
```

RAG is two jobs: **retrieval** (find the right pages) and **generation** (write the
answer from them).

## Design choices

**Chunking (`src/ingest.py`).** How you cut the pages decides what search can find.
Instead of fixed-size slicing, I chunk by the docs' own `<section>` blocks, so a chunk
never mixes topics and carries real metadata (heading + URL anchor) for citations:

```json
{ "text": "...", "section_title": "4.2. for Statements",
  "url": "https://docs.python.org/3/tutorial/controlflow.html#for-statements",
  "heading_path": "More Control Flow Tools > for Statements" }
```

Long sections are split into overlapping windows on paragraph boundaries (never
mid-code), and code blocks are preserved as fenced blocks. Result: **468 chunks across
16 pages, 132 sections.** (Fixed one bug where pages decoded as non-UTF-8 produced
garbled characters and space-less headings, which would have poisoned every embedding.)

**Indexing & retrieval (`src/index.py`, `src/retriever.py`).** Each chunk is embedded
with `all-MiniLM-L6-v2` and stored in ChromaDB (cosine). A question retrieves the
nearest chunks (recall), then a cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) reads
each candidate *together with the question* and re-sorts for precision. Small, free,
local models by choice: strong architecture, modest-sized parts, no API bill.

## Evaluation and experiments

20 hand-written questions, each labelled with the section that should answer it
(`eval/questions.json`, scored by `eval/evaluate.py`). Metrics: **Hit@1/3/5** (was the
right section in the top 1/3/5?) and **MRR** (rewards ranking it higher). Every change
below is a measured number, not a vibe.

**Retrieval progression.** I compared meaning search, keyword search (BM25), their
fusion (Reciprocal Rank Fusion), and adding the reranker:

| Retriever               | Hit@1 | Hit@3 | Hit@5 | MRR   |
|-------------------------|-------|-------|-------|-------|
| Keyword only (BM25)     | 0.45  | 0.70  | 0.80  | 0.587 |
| Meaning only (baseline) | 0.65  | 0.85  | 0.90  | 0.743 |
| Hybrid (RRF)            | 0.65  | 0.80  | 0.90  | 0.737 |
| **Hybrid + reranker**   | **0.70** | **0.90** | **0.90** | **0.800** |

Hybrid *tied* the baseline (keyword search is the weaker partner here and an equal vote
mostly added noise). But with a wider pool of 20 candidates, the correct section was
present for **all 20 questions** (recall@20 = 100%), so the gap was ranking, not recall
— exactly what a reranker fixes. The reranker lifted Hit@1/Hit@3 and MRR (0.743 → 0.800).

**Experiment: chunk size.** Swept on the plain vector baseline:

| Chunk size     | Hit@1 | Hit@3 | Hit@5 | MRR   |
|----------------|-------|-------|-------|-------|
| **800 (kept)** | **0.65** | 0.85 | 0.90 | **0.743** |
| 300            | 0.60  | 0.90  | 0.90  | 0.733 |

300-char chunks slightly hurt Hit@1 and MRR — they fragment explanations, so the single
best chunk is less often the clear winner. Kept 800.

**Experiment: does BM25 earn its place?** With the reranker in play, "vector → reranker"
vs "hybrid → reranker" came out **identical, line for line**. Everything hybrid pulled
in, vector's own top-20 already had. So I **removed BM25 and hybrid entirely** (code,
index, and the `rank-bm25` dependency) — fewer moving parts, zero loss.

**Contextual Retrieval (the decisive lever).** Two questions still missed, with a shared
cause: the canonical section ("Class Definition Syntax", "for Statements") is short and
code-heavy, so chunks that *talk about* a topic out-scored the one that *shows* it. Fix
(Anthropic's Contextual Retrieval): before embedding, Gemini writes one plain sentence
per chunk describing what it explains, prepended to the text — so the class chunk now
starts with "Explains how to define a class using the `class` keyword". Run once over
all chunks (batched, cached to `data/contextualized.jsonl`).

| Pipeline                                    | Hit@1 | Hit@3 | Hit@5 | MRR   |
|---------------------------------------------|-------|-------|-------|-------|
| Meaning only (plain baseline)               | 0.65  | 0.85  | 0.90  | 0.743 |
| Hybrid + reranker (plain)                   | 0.70  | 0.90  | 0.90  | 0.800 |
| **Contextual Retrieval + reranker (final)** | 0.65  | **0.95** | **1.00** | **0.812** |

**Both misses rescued; Hit@5 = 1.00** ("define a class" went from missing to rank 1).

**Honest caveats.** (1) Hit@1 dipped 0.70 → 0.65: context prefixes make chunks read
slightly more alike, shuffling one exact-top position — a fair trade for perfect top-5
coverage, since the LLM sees all 5 anyway. (2) The reranker no longer moves the aggregate
once chunks are contextualized, only because a 20-question set capped at 1.00 has no room
left; I kept it (cheap, and it helps harder real questions). (3) 20 questions is a small
set — Hit@5 = 1.00 means "perfect on these 20", not "perfect forever".

## Three questions it gets wrong, and why

Hit@5 is 1.00 on the labelled set, so I probed harder questions to find real failures —
three different kinds:

1. **"What is the `__name__ == '__main__'` idiom?" — false refusal.** The answer *is* in
   the docs (§6.1.1), but the tokens pull toward the class/namespace sections while the
   real answer is a two-line mention in Modules, so retrieval scored everything near-zero
   and the service refused. *Fix:* query expansion, or indexing short idioms as their own
   chunks.
2. **"How do I remove duplicates from a list?" — right answer, wrong-ish source.**
   Correctly answers "use `set()`" but cites §5.6 Looping Techniques over the more
   canonical §5.4 Sets (which ranked 2nd); "list" pulls it lexically toward looping. A
   Hit@1 slip. *Fix:* a domain-tuned reranker, or boosting title-matching sections.
3. **"What is the difference between `is` and `==`?" — corpus limit.** The tutorial has
   no section on identity vs equality, so it correctly refuses rather than invent. *Fix:*
   add a broader corpus (e.g. the Language Reference).

The pattern: failures come from the source documents and query wording, not from the
pipeline inventing things. Two of the three correctly refuse instead of hallucinating.

## What I considered and skipped

- **HyDE** — would help the synonym gap, but Contextual Retrieval already closed the
  misses, so no measurable gain here.
- **A stronger embedding model** (BGE/E5/Nomic) — the biggest general lever; skipped to
  stay free and fully local.
- **ColBERT, GraphRAG, agentic multi-hop, fine-tuning** — real tools for huge, messy
  corpora; on a 16-page tutorial they'd be over-engineering.

Knowing which technique fits mattered more than stacking every trick.

## What I'd improve next

1. **Fix the false-refusal case** — index short idioms separately and/or add query
   expansion.
2. **A faithfulness check** — a cheap LLM-as-judge pass confirming the answer only used
   the cited passages (belt-and-braces over the existing grounding + refusal).
3. **A stronger embedding model** behind the same interface, with the eval re-run.
4. **Grow the corpus** (e.g. the Language Reference) so currently-refused questions can
   be answered.
5. **Streaming answers + a feedback button** to collect real labels and grow the eval set.

## The service

**Generation (`src/answer.py`).** Three rules make it a real RAG service: **grounding**
(answer only from retrieved chunks), **refusal** (say so if the answer isn't there, no
guessing), and **citations** (inline `[n]` linking the exact sections). Citations are
pinpoint: chunks are grouped by section (so each number is a distinct source) and only
the sections the answer actually cited are shown.

**Model failover (`src/llm.py`).** Every Gemini call tries models in order and falls
through to the next on a quota error (each free model has its own daily quota). Reused
from my own project, Illomra — simpler here since the calls are stateless.

**Hardening (`src/app.py`).** Input validation (rejects empty/over-long questions),
a per-IP rate limit, and graceful handling when the model quota is exhausted (a clear
"busy, try again" instead of a 500). Answer HTML is sanitized with DOMPurify and source
fields are escaped.

## API

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I handle exceptions with try and except?"}'
```

```json
{
  "question": "How do I handle exceptions with try and except?",
  "answer": "To handle exceptions, use a `try` statement [1, 3]. ...",
  "sources": [
    {"n": 1, "section_title": "8.3. Handling Exceptions",
     "url": "https://docs.python.org/3/tutorial/errors.html#handling-exceptions",
     "passage": "The try statement works as follows. First, the try clause ...",
     "cited": true}
  ],
  "refused": false,
  "model": "gemini-3.6-flash"
}
```

Each source includes the actual `passage` text it used. Out-of-scope questions refuse:

```json
{ "answer": "I couldn't find that in the Python tutorial.", "refused": true, "sources": [] }
```

## Project layout

```
config.py              central settings (chunk size, model names, top_k, guards)
src/ingest.py          step 1: scrape the tutorial, cut into structure-aware chunks
src/contextualize.py   step 2: Gemini writes a "what this is about" line per chunk
src/index.py           step 3: embed chunks into ChromaDB (plain, or --contextual)
src/retriever.py       retrieval: vector search -> cross-encoder reranker
src/llm.py             Gemini gateway with multi-model failover
src/answer.py          generation: grounded, cited answer (or refusal)
src/app.py             FastAPI service: POST /ask + the web UI
src/static/index.html  frontend: Markdown answers + pinpoint citations
eval/questions.json    20 questions + the section that should answer each
eval/evaluate.py       Hit@1/3/5 and MRR
data/                  committed chunks + contextualized chunks (vector DB is built locally)
```

## Reproduce or rebuild

```bash
python eval/evaluate.py                      # Hit@5 / MRR: baseline vs production
python src/retriever.py "How do I create a dictionary?"   # raw retrieval

# only to regenerate the data from scratch (uses Gemini quota):
python src/ingest.py                         # re-scrape + chunk
python src/contextualize.py                  # re-run the context lines
```

The Gemini key lives in `.env` (gitignored, never committed); `.env.example` is the
template.
