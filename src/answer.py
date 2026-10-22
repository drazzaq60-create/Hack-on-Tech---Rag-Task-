# The generation half: turn retrieved chunks into a grounded, cited answer.
#
# Three rules the LLM must follow, and they're what make this a real RAG service
# instead of "ask the model and hope":
#   1. GROUNDING  - answer only from the chunks we retrieved, not from memory
#   2. REFUSAL    - if the answer isn't in those chunks, say so, don't invent
#   3. CITATIONS  - point back to the exact tutorial sections used, with links
#
# For clean citations we group the retrieved chunks by section (so [1] and [2] are
# never the same page) and flag which ones the answer actually cited. Every LLM
# call goes through src/llm.py, so a used-up model quietly falls through to the next.

import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src import llm
from src.retriever import search_rerank

REFUSAL = "I couldn't find that in the Python tutorial."

PROMPT = """You are a helpful assistant that answers questions about Python using ONLY the
tutorial excerpts provided below. Follow these rules exactly:

- Answer using only the information in the excerpts. Do not use outside knowledge.
- If the excerpts do not contain the answer, reply with exactly this and nothing else:
  {refusal}
- Cite the excerpts you used inline with their numbers, like [1] or [2].
- Format the answer in clean Markdown: short paragraphs, bullet lists where useful,
  and fenced ```python code blocks for any code.
- Be concise and practical.

Question: {question}

Excerpts:
{context}

Answer:"""


def _original_text(text):
    """Contextualized chunks are 'one-line context\\n\\noriginal'. For the passage we
    show the real doc text, not our internal context line (the LLM still gets the full text)."""
    parts = text.split("\n\n", 1)
    return parts[1] if len(parts) == 2 else text


def _group_by_section(hits, max_sources):
    """
    Collapse chunks that come from the same section into one source, so citations
    are pinpoint (one number per section). Keep each section's best score and merge
    its text for the model. Returns the top `max_sources` sections by score.
    """
    groups, order = {}, []
    for h in hits:
        u = h["url"]
        if u not in groups:
            groups[u] = {"section_title": h["section_title"], "url": u,
                         "texts": [h["text"]], "score": h["score"]}
            order.append(u)
        else:
            groups[u]["texts"].append(h["text"])
            groups[u]["score"] = max(groups[u]["score"], h["score"])
    ranked = sorted((groups[u] for u in order), key=lambda g: g["score"], reverse=True)
    return ranked[:max_sources]


def answer(question, k=config.TOP_K):
    question = (question or "").strip()[:config.MAX_QUESTION_CHARS]   # defensive cap
    if not question:
        return {"question": question, "answer": REFUSAL, "sources": [],
                "refused": True, "model": None}

    hits = search_rerank(question, max(k + 3, 8))
    sections = _group_by_section(hits, k)

    blocks, sources = [], []
    for i, g in enumerate(sections, 1):
        merged = "\n\n".join(g["texts"])[:3000]
        blocks.append(f"[{i}] (section: {g['section_title']})\n{merged}")
        passage = "\n\n".join(_original_text(t) for t in g["texts"])[:800]
        sources.append({
            "n": i,
            "section_title": g["section_title"],
            "url": g["url"],
            "passage": passage,          # the actual source doc text used
        })

    if not sources:
        return {"question": question, "answer": REFUSAL, "sources": [],
                "refused": True, "model": None}

    context = "\n\n".join(blocks)
    prompt = PROMPT.format(refusal=REFUSAL, question=question, context=context)
    text, model_used = llm.generate(prompt)
    text = text.strip()

    low = text.lower()
    refused = ("couldn't find" in low or "could not find" in low
               or REFUSAL.lower()[:25] in low)

    # which sources did the answer actually cite? (pinpoint, not "here's everything")
    # handle grouped citations too, e.g. "[1, 2]" or "[1,2]", not just "[2]"
    cited_nums = set()
    for grp in re.findall(r"\[([\d,\s]+)\]", text):
        for n in re.findall(r"\d+", grp):
            cited_nums.add(int(n))
    for s in sources:
        s["cited"] = s["n"] in cited_nums
    shown = [s for s in sources if s["cited"]] or sources

    return {
        "question": question,
        "answer": text,
        "sources": [] if refused else shown,
        "refused": refused,
        "model": model_used,
    }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How do I define a class in Python?"
    res = answer(q)
    print(f"\nQ: {res['question']}\n\n{res['answer']}")
    if res["sources"]:
        print("\nSources:")
        for s in res["sources"]:
            print(f"  [{s['n']}] {s['section_title']}  {s['url']}")
    print(f"\n(model: {res['model']}, refused: {res['refused']})")
