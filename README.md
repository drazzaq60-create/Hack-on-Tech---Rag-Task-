# Python Tutorial RAG

A question-answering service for Python, built on top of the official
[Python 3 Tutorial](https://docs.python.org/3/tutorial/). It is the **only** source
the service uses. You ask a question, it finds the relevant part of the docs, answers
using only that, and links back to the exact section. If the answer isn't in the docs,
it says so instead of guessing.

## Summary

- **What it is:** a FastAPI `/ask` endpoint (and a small web page) that returns a
  grounded, cited answer, or an honest "I couldn't find that".
- **How it works:** structure-aware chunking → Contextual Retrieval → vector search →
  a reranker → Gemini (answers only from the docs, with citations).
- **Result:** **Hit@5 = 1.00, MRR = 0.812** on a 20-question test set.
- **How I built it:** add one stage at a time, *measure* whether it actually helps, and
  keep only what earns its place. I built and tested retrieval first — if it finds the
  wrong pages, no amount of clever prompting fixes the answer.

## How to run

You need **Python 3.9 or newer** installed. The prepared data is already in the repo,
so you don't need to re-scrape anything or spend any Gemini quota to get it running.
Follow these steps in order.

**1. Download the project** and open its folder:

```bash
git clone https://github.com/drazzaq60-create/Hack-on-Tech---Rag-Task-.git
cd Hack-on-Tech---Rag-Task-
```

**2. Create and activate a virtual environment** (keeps the install clean):

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
```

**3. Install the dependencies:**

```bash
pip install -r requirements.txt
```

**4. Add your Gemini API key.** It's free — get one at
https://aistudio.google.com/app/apikey. Then copy the example file and paste your key
into it:

```bash
copy .env.example .env         # Windows   (use: cp .env.example .env on macOS/Linux)
```

Open the new `.env` file and set your key, so the line reads:

```
GEMINI_API_KEY=your_key_here
```

**5. Build the search index** (reads the data already in the repo, makes no API calls):

```bash
python src/index.py --contextual
```

**6. Start the service:**

```bash
uvicorn src.app:app
```

Now open **http://127.0.0.1:8000** in your browser and ask a question (for example,
"How do I define a class in Python?"). You can also send a question directly to the API
with `curl` — see the [API](#api) section below.

## How it works

```
question
   -> [ retrieve ]  search the docs, then a reranker re-sorts the best candidates
   -> [ generate ]  Gemini answers ONLY from those chunks, cites them, or refuses
   -> answer + source links
```

A RAG service has two parts: **retrieval** (find the right pages) and **generation**
(write the answer from them).

## Design choices

**Chunking (`src/ingest.py`).** How you cut the pages into pieces decides what the
search can find. Instead of cutting every N characters, I cut along the docs' own
`<section>` blocks. That way a chunk never mixes two topics, and each one keeps its
heading and URL so the answer can cite a real source:

```json
{ "text": "...", "section_title": "4.2. for Statements",
  "url": "https://docs.python.org/3/tutorial/controlflow.html#for-statements",
  "heading_path": "More Control Flow Tools > for Statements" }
```

Long sections are split into smaller overlapping pieces, always on paragraph breaks so
code is never cut in half. Code blocks are kept intact. Result: **468 chunks across 16
pages, 132 sections.** (I also fixed a bug where some pages decoded wrong and produced
garbled characters and missing spaces in headings — left unfixed, it would have quietly
spoiled every embedding.)

**Indexing and retrieval (`src/index.py`, `src/retriever.py`).** Each chunk is turned
into a vector with `all-MiniLM-L6-v2` and stored in ChromaDB. A question pulls the
closest chunks, then a reranker (`ms-marco-MiniLM-L-6-v2`) reads each candidate
*together with the question* and re-sorts them properly. I chose small, free, local
models on purpose: a strong design with light parts, and no API bill.

## Evaluation and experiments

I wrote 20 questions by hand and, for each, noted the section that should answer it
(`eval/questions.json`, scored by `eval/evaluate.py`). The metrics are **Hit@1/3/5**
(was the right section in the top 1/3/5?) and **MRR** (rewards ranking it higher). Every
change below is a real number, not a guess.

**Comparing retrievers.** I tested meaning search, keyword search (BM25), the two
combined (Reciprocal Rank Fusion), and adding the reranker:

| Retriever               | Hit@1 | Hit@3 | Hit@5 | MRR   |
|-------------------------|-------|-------|-------|-------|
| Keyword only (BM25)     | 0.45  | 0.70  | 0.80  | 0.587 |
| Meaning only (baseline) | 0.65  | 0.85  | 0.90  | 0.743 |
| Hybrid (RRF)            | 0.65  | 0.80  | 0.90  | 0.737 |
| **Hybrid + reranker**   | **0.70** | **0.90** | **0.90** | **0.800** |

Combining the two searches only *tied* the baseline — keyword search is the weaker one
here, and giving it an equal vote mostly added noise. But when I widened the pool to 20
candidates, the right section was present for **all 20 questions** (recall@20 = 100%). So
the problem was ranking, not finding — and that's exactly what a reranker fixes. It
lifted Hit@1, Hit@3, and MRR (0.743 → 0.800).

**Experiment: chunk size.** I tried smaller chunks on the plain baseline:

| Chunk size     | Hit@1 | Hit@3 | Hit@5 | MRR   |
|----------------|-------|-------|-------|-------|
| **800 (kept)** | **0.65** | 0.85 | 0.90 | **0.743** |
| 300            | 0.60  | 0.90  | 0.90  | 0.733 |

Smaller 300-character chunks slightly hurt Hit@1 and MRR — they break explanations into
fragments, so the single best chunk is less often a clear winner. I kept 800.

**Experiment: is keyword search worth keeping?** With the reranker in place, "vector →
reranker" and "hybrid → reranker" gave **identical results, line for line**. Everything
keyword search added, the vector search already had. So I **removed BM25 and hybrid
completely** — the code, the index, and the dependency. Fewer moving parts, no loss.

**Contextual Retrieval (the big win).** Two questions still missed, for the same reason:
the section that actually answers them ("Class Definition Syntax", "for Statements") is
short and full of code, so chunks that just *talk about* the topic beat the one that
*shows* it. The fix (Anthropic's Contextual Retrieval): before embedding, I have Gemini
write one plain sentence describing each chunk and add it to the front. Now the class
chunk starts with "Explains how to define a class using the `class` keyword" — the same
words a user would type. I run this once over all chunks and cache it
(`data/contextualized.jsonl`).

| Pipeline                                    | Hit@1 | Hit@3 | Hit@5 | MRR   |
|---------------------------------------------|-------|-------|-------|-------|
| Meaning only (plain baseline)               | 0.65  | 0.85  | 0.90  | 0.743 |
| Hybrid + reranker (plain)                   | 0.70  | 0.90  | 0.90  | 0.800 |
| **Contextual Retrieval + reranker (final)** | 0.65  | **0.95** | **1.00** | **0.812** |

**Both misses are fixed, and Hit@5 reached 1.00** ("define a class" went from missing
to rank 1).

**Honest notes.** (1) Hit@1 dropped a little (0.70 → 0.65): the added sentences make
chunks look slightly more alike, so one exact-top spot shuffled — a fair trade for
getting the right answer into the top 5 every time, since the LLM reads all 5 anyway.
(2) The reranker no longer changes the overall scores once chunks are contextualized,
simply because a 20-question set capped at 1.00 has no room left to show a difference; I
kept it because it's cheap and helps on harder real questions. (3) 20 questions is a
small set — 1.00 means "perfect on these 20", not "perfect on everything".

## Three questions it gets wrong, and why

Since Hit@5 is 1.00 on the test set, I probed harder questions to find real failures —
three different kinds:

1. **"What is the `__name__ == '__main__'` idiom?" — it refuses, but the answer is
   actually in the docs.** It's covered in §6.1.1, but the words pull toward the
   class/namespace sections, and the real answer is just a two-line mention in the
   Modules chapter, so it scored too low to surface. *Fix:* expand the query, or index
   short idioms as their own small chunks.
2. **"How do I remove duplicates from a list?" — right answer, slightly wrong source.**
   It correctly says "use `set()`" but cites §5.6 Looping Techniques instead of the more
   fitting §5.4 Sets (which came 2nd); the word "list" pulls it toward looping. *Fix:* a
   reranker tuned for this domain, or favouring sections whose title matches the question.
3. **"What is the difference between `is` and `==`?" — the docs don't cover it.** The
   tutorial has no section on this, so the service correctly refuses instead of making
   something up. *Fix:* add a broader source, like the Language Reference.

The common thread: failures come from the documents and the wording of the question, not
from the service inventing answers. In two of the three it refuses rather than guess —
which is exactly what I wanted.

## What I considered and skipped

- **HyDE** — would help with wording gaps, but Contextual Retrieval already fixed the
  misses, so it wouldn't add anything measurable here.
- **A stronger embedding model** (BGE / E5 / Nomic) — the biggest lever in general, but
  I kept MiniLM to stay free and fully local.
- **ColBERT, GraphRAG, multi-step agents, fine-tuning** — useful for large, messy data
  sets. On a 16-page tutorial they would be overkill.

Knowing which technique fits the problem mattered more than using every one I could name.

## What I'd improve next

1. **Fix the false refusal** — index short idioms separately, or expand the query so its
   synonyms reach the right section.
2. **Add a faithfulness check** — a small second LLM pass that confirms the answer only
   used the cited passages. An extra safety net on top of grounding and refusal.
3. **Try a stronger embedding model** behind the same interface, and re-run the eval to
   measure the gain.
4. **Add more documents** (like the Language Reference) so questions the tutorial doesn't
   cover can be answered instead of refused.
5. **Stream answers and add a feedback button** to collect real labels and grow the test
   set beyond the hand-written 20.

## The service

**Generation (`src/answer.py`).** Three rules make this a real RAG service, not just
"ask the model and hope": **grounding** (answer only from the retrieved chunks),
**refusal** (say so when the answer isn't there), and **citations** (inline `[n]` that
link the exact sections). Citations are precise: chunks are grouped by section, so each
number is a different source, and only the sections the answer actually used are shown.

**Model failover (`src/llm.py`).** Each Gemini call tries models in order and moves to
the next one if a model's free quota runs out (each model has its own daily limit). I
reused this idea from my own project, Illomra — it's simpler here because each call
stands alone, with no chat history to carry over.

**Hardening (`src/app.py`).** The service checks input (rejects empty or very long
questions), limits requests per client, and handles a used-up quota gracefully (a clear
"busy, try again" instead of a crash). Answers are sanitized before display and source
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

Each source includes the actual `passage` text it used. A question outside the tutorial
gets refused:

```json
{ "answer": "I couldn't find that in the Python tutorial.", "refused": true, "sources": [] }
```

## Project layout

```
config.py              central settings (chunk size, model names, top_k, guards)
src/ingest.py          step 1: scrape the tutorial and cut it into chunks
src/contextualize.py   step 2: Gemini writes a "what this is about" line per chunk
src/index.py           step 3: turn chunks into vectors in ChromaDB (plain, or --contextual)
src/retriever.py       retrieval: vector search -> reranker
src/llm.py             Gemini gateway with multi-model failover
src/answer.py          generation: grounded, cited answer (or refusal)
src/app.py             FastAPI service: POST /ask + the web page
src/static/index.html  frontend: Markdown answers + precise citations
eval/questions.json    20 questions + the section that should answer each
eval/evaluate.py       Hit@1/3/5 and MRR
data/                  chunks + contextualized chunks (the vector index is built locally)
```

## Reproduce or rebuild

```bash
python eval/evaluate.py                      # Hit@5 / MRR: baseline vs final
python src/retriever.py "How do I create a dictionary?"   # see raw retrieval

# only if you want to rebuild the data from scratch (uses Gemini quota):
python src/ingest.py                         # re-scrape and chunk
python src/contextualize.py                  # re-write the context lines
```

The Gemini key stays in `.env` (gitignored, never committed); `.env.example` is the
template.
