# Step 4: score how good the retrieval actually is, so I'm not just guessing.
#
# For each of the 20 test questions I check whether the section that SHOULD
# answer it showed up near the top:
#   Hit@1 / Hit@3 / Hit@5 : was the right section in the top 1 / 3 / 5?
#   MRR@5                 : 1/rank of the first correct hit (rewards ranking it high)
#
# A "hit" means the correct section name appears in a retrieved chunk's title.
# If a label never matches ANY section it prints a warning, that means I wrote
# the label wrong, not that retrieval failed. Run this after index.py.

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.append(ROOT)

import config
from src.retriever import search_vector, search_rerank


def load_questions():
    with open(os.path.join(HERE, "questions.json"), encoding="utf-8") as f:
        return json.load(f)


def all_section_titles():
    titles = []
    with open(config.CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            titles.append(json.loads(line)["section_title"].lower())
    return titles


def first_hit_rank(hits, correct):
    """Rank (1-based) of the first retrieved chunk whose section matches `correct`; 0 if none."""
    c = correct.lower()
    for rank, h in enumerate(hits, 1):
        if c in h["section_title"].lower():
            return rank
    return 0


def evaluate(retrieve_fn, k=config.TOP_K, name="vector"):
    questions = load_questions()
    titles = all_section_titles()
    n = len(questions)
    hit1 = hit3 = hit5 = 0
    mrr = 0.0
    bad_labels = []

    print(f"\n=== {name}  (k={k}, {n} questions) ===")
    for q in questions:
        correct = q["correct"]
        if not any(correct.lower() in t for t in titles):
            bad_labels.append(correct)       # label doesn't exist in corpus -> fix it
        hits = retrieve_fn(q["question"], k)
        rank = first_hit_rank(hits, correct)
        if rank == 1:
            hit1 += 1
        if 1 <= rank <= 3:
            hit3 += 1
        if 1 <= rank <= 5:
            hit5 += 1
        if rank:
            mrr += 1 / rank
        mark = "OK " if rank else "MISS"
        print(f"  [{mark}] rank={rank or '-'}  {q['question'][:58]}")

    print(f"\n  Hit@1={hit1/n:.2f}  Hit@3={hit3/n:.2f}  Hit@5={hit5/n:.2f}  MRR@5={mrr/n:.3f}")
    if bad_labels:
        print("  !! These labels were NOT found in any section (fix the label, not retrieval):")
        for b in bad_labels:
            print("     -", b)
    return {"hit1": hit1 / n, "hit3": hit3 / n, "hit5": hit5 / n, "mrr": mrr / n}


if __name__ == "__main__":
    # The baseline (meaning search) vs the production pipeline (+ reranker).
    # The fuller comparison (keyword, hybrid, recall ceiling) is written up in the README.
    evaluate(search_vector, name="VECTOR only (meaning) baseline")
    evaluate(search_rerank, name="VECTOR + reranker (production)")
