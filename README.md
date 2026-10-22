# Python Tutorial RAG, a question answering service over the official docs

This is a small retrieval augmented generation (RAG) service that answers
questions about Python using the official Python 3 Tutorial
(https://docs.python.org/3/tutorial/) as its only source of truth. Ask it
something like "how do I write a for loop?" and it finds the relevant part of
the docs, then answers from that, with a link back to the exact section.

The task was small, so I could have thrown together one script that scrapes a
page and stuffs it into an LLM. I didn't want to do that. I wanted to build it
the way a real retrieval system is built, measure whether each piece actually
helps, and be honest about where it still falls short. That write up is below.

## How it works, in one picture

```
question
   |
   v
[ retrieve ]  find the most relevant chunks of the docs
   |            (meaning search + keyword search, then a reranker re-sorts them)
   v
[ generate ]  hand those chunks to the LLM, which answers ONLY from them,
   |            cites the section, and says "I don't know" if the answer isn't there
   v
answer + source links
```

RAG is really two jobs bolted together: **retrieval** (find the right pages)
and **generation** (write the answer from them). I built and tested the
retrieval half first, because if retrieval hands the LLM the wrong pages, no
amount of clever prompting saves you. Garbage in, garbage out.

## Project layout

```
config.py              all the knobs in one place (chunk size, model names, top_k...)
src/ingest.py          step 1: scrape the tutorial, cut it into clean chunks
src/index.py           step 2: embed the chunks into ChromaDB (plain, or --contextual)
src/retriever.py       the retriever: vector search -> cross-encoder reranker
src/contextualize.py   step 7: Gemini writes a "what this is about" line per chunk
src/llm.py             the Gemini gateway (multi-model failover on quota errors)
src/answer.py          the generation half: grounded, cited answer (or a refusal)
src/app.py             the FastAPI service: POST /ask, /meta, and the web UI
src/static/index.html  the frontend: renders answers (Markdown) + pinpoint citations
eval/questions.json    20 test questions with the section that should answer each
eval/evaluate.py       step 4: score how often we find the right section
data/                  generated stuff (chunks, contextualized chunks, the vector DB)
```

## The build, step by step

### Step 1, ingest (scrape and chunk)

I only pull the tutorial pages, nothing else from docs.python.org. The important
decision here is how you cut the pages into pieces ("chunks"), because that
decides what the search can even find.

Instead of blindly chopping every 500 characters (which cuts sentences and code
in half), I chunk by the docs' own `<section>` blocks. Each section already has
a heading and a URL anchor, so every chunk carries real metadata with it:

```
{ "text": "...", "section_title": "4.2. for Statements",
  "url": "https://docs.python.org/3/tutorial/controlflow.html#for-statements",
  "heading_path": "More Control Flow Tools > for Statements" }
```

That `url` is what lets the final answer cite its source. Long sections get split
further into overlapping windows, but I break on paragraph boundaries so I never
slice a code example in half. Code blocks are kept as fenced ``` blocks so they
survive intact.

Result: **469 chunks across 16 pages, 132 unique sections.**

(One bug I hit and fixed: the pages came through with garbled characters like
"Â" and headings glued together with no spaces. The fix was forcing UTF-8
decoding and using a space separator when pulling heading text. Small thing, but
it would have quietly poisoned every embedding.)

### Step 2, index (make it searchable two ways)

I build two separate search indexes over the same chunks, on purpose:

- **Meaning search**: embed every chunk with `all-MiniLM-L6-v2` and store the
  vectors in ChromaDB. This finds chunks by *idea*, so "car" and "automobile"
  look similar to it.
- **Keyword search**: a BM25 index. This is dumb but literal, it matches *exact
  words*. Useless for synonyms, but it never gets distracted by vibes.

I built both so I could compare them fairly and later combine them.

### Step 3, retrieve (the baseline)

The first retriever is pure meaning search: embed the question, find the closest
chunks. This is the baseline everything else has to beat.

### Step 4, evaluate (so I'm not guessing)

I wrote 20 questions by hand and, for each, noted which section *should* answer
it. Then the scorer checks: did the correct section show up in the top results?

- **Hit@1 / Hit@3 / Hit@5**: was the right section in the top 1 / 3 / 5?
- **MRR**: rewards putting the right answer higher up (rank 1 beats rank 5).

Without this, "it feels better" is just a vibe. With it, every change below is a
number I can defend.

### Step 5, hybrid search (meaning + keyword together)

I combined the two searches using Reciprocal Rank Fusion (RRF): a chunk earns
points for ranking high on *either* list, and the totals decide the final order.

Honest result: on this corpus, **hybrid basically tied the meaning-only
baseline** (Hit@5 stayed 0.90, MRR barely moved). It rescued one question the
baseline missed, but dragged another down, because keyword search here is the
weaker partner and giving it an equal vote mostly added noise. That is a real
finding, not a failure. A naive hybrid is not automatically better, and on a
small clean corpus where meaning search is already strong, there isn't much room
left for keyword search to help.

So I didn't stop at hybrid. I used it as a wider *net* (pull 20 candidates
instead of 5) and let a reranker do the precise final sort. I checked that this
is even possible first: with 20 candidates, the correct section is in the pool
for **all 20 questions** (recall@20 = 100%). So the right answer was always
findable, it was just ranked too low. That's a ranking problem, and ranking
problems have a clean fix.

### Step 6, reranking (the real precision win)

A cross-encoder (`ms-marco-MiniLM-L-6-v2`) reads the question and each candidate
chunk *together* and scores how well it actually answers. The earlier searches
compare question and chunk separately, so they can be fooled by surface overlap.
The reranker reads both at once and catches "this chunk really does answer this."
It's slower, but it only runs on 20 chunks, so who cares.

This is where the numbers moved for real:

| Retriever                | Hit@1 | Hit@3 | Hit@5 | MRR   |
|--------------------------|-------|-------|-------|-------|
| Keyword only (BM25)      | 0.45  | 0.70  | 0.80  | 0.587 |
| Meaning only (baseline)  | 0.65  | 0.85  | 0.90  | 0.743 |
| Hybrid (RRF)             | 0.65  | 0.80  | 0.90  | 0.737 |
| **Hybrid + reranker**    | **0.70** | **0.90** | **0.90** | **0.800** |

The reranker pushed the best answer to rank 1 more often (Hit@1 and Hit@3 both
up, MRR up from 0.743 to 0.800). The gain looks modest because the baseline was
already strong, and I'd rather report an honest +0.06 than invent a dramatic one.

### Experiment: chunk size (300 vs 800 characters)

I also swept the chunk size on the plain (non-contextual) vector baseline, to see
whether smaller, tighter chunks would retrieve better:

| Chunk size | Hit@1 | Hit@3 | Hit@5 | MRR   |
|------------|-------|-------|-------|-------|
| **800 (kept)** | **0.65** | 0.85 | 0.90 | **0.743** |
| 300        | 0.60  | 0.90  | 0.90  | 0.733 |

Smaller 300-char chunks slightly *hurt* Hit@1 and MRR (same Hit@5), I think because
they chop explanations into fragments that each carry less context, so the single
best chunk is less often the clear winner. 800 kept more of each idea together, so
I kept it. Small effect, but a real one, and it's why the default is 800.

### A quick experiment: is BM25 actually pulling its weight?

Hybrid tied the baseline, so I asked a blunt question: once the reranker is in
place, does BM25 add anything at all? I ran two pipelines head to head, "vector
candidates into the reranker" vs "hybrid candidates into the reranker". They came
out **identical, line for line** (same Hit@5, same MRR, same ranks). Every answer
hybrid pulled in, vector's own top-20 already had. So I **removed BM25 and hybrid
entirely** (the code, the index file, and the `rank-bm25` dependency). Fewer moving
parts, zero loss. This write-up is the record of that experiment; the shipped code
keeps only the production path. Deleting something that doesn't earn its place is a
feature, not a regret.

### Step 7, Contextual Retrieval (the fix for the two misses)

The two failures above shared one root cause: the canonical section ("Class
Definition Syntax", "for Statements") is short and code-heavy, so chunks that just
*talk about* a topic out-scored the chunk that actually *shows* it.

So I applied **Contextual Retrieval** (Anthropic's idea): before embedding, I had
Gemini write one plain sentence for each chunk saying what it explains and what a
learner would search for, then glued that sentence to the front of the chunk. Now
the code-heavy class chunk literally starts with "Explains how to define a class
using the `class` keyword", exactly the words the question uses.

I ran this as a one-time batch over all 468 chunks (batched ~10 per request to
respect the free-tier quota), cached to `data/contextualized.jsonl`. Here's the
full picture, every stage, honest numbers:

| Pipeline                                   | Hit@1 | Hit@3 | Hit@5 | MRR   |
|--------------------------------------------|-------|-------|-------|-------|
| Meaning only (plain baseline)              | 0.65  | 0.85  | 0.90  | 0.743 |
| Hybrid + reranker (plain)                  | 0.70  | 0.90  | 0.90  | 0.800 |
| **Contextual Retrieval + reranker (final)**| 0.65  | **0.95** | **1.00** | **0.812** |

**Both misses are gone. Hit@5 hit 1.00**, the correct section is now in the top 5
for every one of the 20 questions, and MRR ticked up to 0.812. "Define a class"
went from missing entirely to rank 1. Contextual Retrieval was clearly the single
biggest lever in the whole project.

### Two things I'm not going to hide

1. **Hit@1 actually dropped a touch (0.70 to 0.65).** One question slid from rank
   1 to rank 2, because the context prefixes make chunks read a little more alike,
   so the *exact* top spot shuffled. I'll take that trade: perfect top-5 coverage
   for one lost position, and the service feeds the LLM all 5 chunks anyway, so the
   right answer is always in front of it.

2. **The reranker no longer moves the aggregate numbers** once the chunks are
   contextualized (contextual-vector and contextual-vector-plus-reranker tie). That
   is not because the reranker is useless, it's because a 20-question set maxed out
   at Hit@5 = 1.00 has no room left to show a difference. I kept the reranker
   anyway: stripping it out to win a point on a saturated toy benchmark would be
   overfitting, and it costs almost nothing (it only sorts 20 candidates). It earns
   its keep on harder, more ambiguous real questions this small set can't capture.

One caveat on all of this: 20 questions is a small, hand-written set, and Hit@5 =
1.00 means "perfect on these 20", not "perfect forever". It's an honest signal, not
a guarantee.

## Three questions it gets wrong, and why

Hit@5 is 1.00 on the 20-question set, so to find real failures I probed *harder*
questions outside that set. Three honest ones, each a different kind of failure:

1. **"What is the `__name__ == '__main__'` idiom?" → it refuses, but the answer IS
   in the docs (a false refusal).** The tutorial covers this in §6.1.1 "Executing
   modules as scripts", but retrieval scored every candidate so low (top relevance
   ~0.01) that the service gave up and refused. *Why:* the tokens `__name__` /
   `__main__` pull semantically toward the class/namespace sections (§9.2 Scopes and
   Namespaces), while the actual answer is a two-line mention buried in the Modules
   chapter, so it never surfaces. This is the honest downside of a conservative
   refusal rule: it sometimes declines a question it could have answered.
   *Fix:* query expansion, or indexing short idioms as their own small chunks.

2. **"How do I remove duplicates from a list?" → right answer, wrong-ish source.**
   It correctly answers "use `set()`", but cites **§5.6 Looping Techniques** as the
   top source instead of the more canonical **§5.4 Sets** (which ranked 2nd). *Why:*
   "remove duplicates from a **list**" leans lexically toward the list-looping
   section, and Looping Techniques does show `sorted(set(...))`, so it's defensible
   but not ideal. A Hit@1 slip, not a Hit@5 one. *Fix:* a reranker fine-tuned on
   this domain, or boosting sections whose title matches the query intent.

3. **"What is the difference between `is` and `==`?" → it refuses, and that's a
   corpus limit.** The Python Tutorial has no section that directly explains identity
   vs equality, so retrieval returns only weak, loosely-related chunks and the
   service refuses rather than inventing an answer. *Why:* RAG can only be as good as
   the documents it's given; if the corpus doesn't cover something, the honest move is
   to say so. *Fix:* nothing on the retrieval side; this would need a broader corpus
   (e.g. the Language Reference) added to the index.

The pattern across all three: failures come from the *source documents and the
query wording*, not from the pipeline silently making things up. In two of the three
it correctly refuses rather than hallucinate, which is the behavior I wanted.

## What I considered and deliberately did NOT do

I used small, free, local models (MiniLM for the embedder and reranker) so the
retrieval runs on a laptop with no API bill. That's a conscious trade: strong
*architecture*, modest-sized *parts*. Things I weighed and skipped on purpose:

- **HyDE** (draft a hypothetical answer, then search with it). Would help the
  synonym gap, but Contextual Retrieval already closed our misses, so adding HyDE
  would have been effort for no measurable gain on this corpus.
- **A stronger embedding model** (BGE, E5, Nomic...). The single biggest lever in
  general, since MiniLM is small and old. Skipped to stay free and fully local.
- **ColBERT, GraphRAG, agentic multi-hop, fine-tuning.** Real tools for huge messy
  corpora. On a 16-page tutorial they'd be over-engineering, and claiming them
  would signal the opposite of good judgment.

Knowing which technique fits the problem (and which doesn't) mattered more here
than stacking every trick I could name.

## What I'd improve next

In rough priority order, if I kept working on this:

1. **Fix the false-refusal case** (the `__name__ == '__main__'` miss above): index
   short idioms as their own small chunks, and/or add light query expansion so a
   question's synonyms reach the right section.
2. **A faithfulness check**: a cheap LLM-as-judge pass that confirms the answer only
   used what the cited passages actually say. Grounding + refusal already cover the
   common failure; this would be belt-and-braces and catch subtle drift.
3. **A stronger embedding model** (BGE / E5 / Nomic) behind the same interface, with
   the eval re-run to measure the lift. MiniLM was chosen for free + local, not for
   being the best.
4. **Grow the corpus** (e.g. the Python Language Reference) so questions the tutorial
   doesn't cover (`is` vs `==`) can be answered instead of refused.
5. **Streaming answers** over the API and a small per-answer feedback button, to start
   collecting real relevance labels and expand the eval set beyond the hand-written 20.

## The generation half (built)

Retrieval hands the top chunks to Gemini, which has to play by three rules. These
are what make it a real RAG service instead of "ask the model and hope":

- **Grounding** - it answers using only the retrieved chunks, not its own memory.
- **Refusal** - if the answer isn't in those chunks, it replies "I couldn't find
  that in the Python tutorial" instead of inventing something. Ask it the capital
  of France and that's exactly what you get.
- **Citations** - every answer points back to the tutorial sections it used, with
  real links, numbered [1], [2] inline.

Every Gemini call goes through `src/llm.py`, which tries models in order and falls
through to the next one on a quota error (each free model has its own quota). That
multi-model failover is an idea I reused from my own project, Illomra; it's simpler
here because these calls are stateless, so there's no chat history to carry across
a switch.

`src/app.py` wraps it in a small FastAPI service: `POST /ask` returns the answer
plus its sources as JSON, and `GET /` is the web UI.

The frontend (`src/static/index.html`) keeps it simple and readable:
- answers render as proper Markdown with syntax-highlighted code (not raw text),
- **citations are pinpoint**: chunks are grouped by section so each number is a
  distinct source, only the sections the answer actually cited are shown, and each
  links straight to that section's anchor on docs.python.org.

Answer HTML is sanitized with DOMPurify before rendering (and source fields are
escaped), so model or document text can't inject anything into the page. The
Hit@5 / MRR numbers live here in the README, where they belong, not bolted onto
every answer (they're a system-wide benchmark over labelled questions, not a
per-question score).

### Still could add (didn't, to keep scope honest)

- [ ] A faithfulness check (an LLM-as-judge pass confirming the answer only used
      what the sources actually say). The grounding prompt + refusal already cover
      the common failure; a judge would be belt-and-braces on top.

## Running it

The chunked and contextualized data is committed (`data/*.jsonl`), so you do **not**
need to re-scrape or spend any Gemini quota to get it running. Quick path:

```bash
python -m venv venv
venv\Scripts\activate              # Windows  (use: source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

# add your Gemini key (free: https://aistudio.google.com/app/apikey)
copy .env.example .env             # then edit .env and paste your key

python src/index.py --contextual   # build the vector index from the committed data (no API calls)
uvicorn src.app:app --reload       # serve it -> http://127.0.0.1:8000
```

Reproduce the evaluation, or rebuild the data from scratch:

```bash
python eval/evaluate.py            # Hit@5 / MRR for baseline vs production
python src/retriever.py "How do I create a dictionary?"   # see raw retrieval

# only if you want to regenerate the data yourself:
python src/ingest.py               # re-scrape + chunk     -> data/chunks.jsonl
python src/contextualize.py        # re-run Gemini labels  -> data/contextualized.jsonl  (uses quota)
```

The Gemini key lives in `.env` (gitignored, never committed). `.env.example` is the
template.

## Asking it questions (the service)

```bash
uvicorn src.app:app --reload
```

Then either open http://127.0.0.1:8000 in a browser (there's a simple ask box),
or POST to it:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I handle exceptions with try and except?"}'
```

You get back the grounded answer, whether it refused, which model answered, and
the source sections:

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

Each source carries the actual `passage` text it used (not just a link). Ask it
something outside the tutorial ("what is the capital of France?") and it refuses
instead of guessing:

```json
{ "answer": "I couldn't find that in the Python tutorial.", "refused": true, "sources": [] }
```
