# Contextual Retrieval: give every chunk a one-line "what this is about" prefix.
#
# Why: some of our chunks are short and code-heavy (e.g. the class-definition
# syntax), so they embed badly and lose to chunks that just TALK about the topic.
# Fixing it: ask Gemini to write one plain sentence describing each chunk, then
# glue that sentence to the front before we re-embed. Now the chunk carries the
# words a user would actually search with. (This is Anthropic's Contextual
# Retrieval idea, applied to our exact failure.)
#
# I send chunks in small BATCHES (one request describes ~10 at once) because the
# free tier quota is tight, and 47 requests is a lot friendlier than 469. It's
# cached + resumable: results append to data/contextualized.jsonl as they're made,
# so if it stops you just run it again and it continues. It also backs off and
# retries when the free tier says "slow down".

import json
import os
import re
import sys
import time

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src import llm

# Prefer the high-quota model for this bulk job, but fall through to the rest.
CTX_MODELS = [config.CONTEXTUALIZE_MODEL] + [
    m for m in llm.FALLBACK_MODELS if m != config.CONTEXTUALIZE_MODEL
]

PROMPT = """These are snippets from the Python tutorial. For EACH one, write a single short
sentence (max 25 words) saying what it explains and what question a learner would
ask to find it. Use plain search words.

Return ONLY a JSON object mapping each id to its sentence, like:
{{"chunk_0001": "Explains ...", "chunk_0002": "Explains ..."}}

Snippets:
{snippets}"""


def load_chunks(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def already_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                done.add(json.loads(line)["id"])
    return done


def parse_json(text):
    """Gemini sometimes wraps JSON in ```json fences; strip them and parse."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    return json.loads(t)


def describe_batch(batch, max_retries=3):
    """One request -> {id: sentence} for the whole batch. Quota failover lives in llm."""
    snippets = "\n\n".join(
        f'id: {c["id"]}\nsection: {c["section_title"]}\ntext: {c["text"][:1000]}'
        for c in batch
    )
    prompt = PROMPT.format(snippets=snippets)
    for attempt in range(max_retries):
        # llm.generate handles quota failover across models + transient retries
        text, _model_used = llm.generate(prompt, models=CTX_MODELS)
        try:
            return parse_json(text)
        except Exception:
            time.sleep(2)   # rare JSON hiccup: ask again
    raise RuntimeError(f"gave up on batch starting {batch[0]['id']}")


def main():
    chunks = load_chunks(config.CHUNKS_PATH)
    done = already_done(config.CONTEXTUAL_PATH)
    todo = [c for c in chunks if c["id"] not in done]
    by_id = {c["id"]: c for c in chunks}
    print(f"{len(chunks)} chunks total, {len(done)} already done, {len(todo)} to do.")

    B = config.CONTEXTUALIZE_BATCH
    batches = [todo[i:i + B] for i in range(0, len(todo), B)]

    with open(config.CONTEXTUAL_PATH, "a", encoding="utf-8") as out:
        for batch in tqdm(batches, desc="contextualizing"):
            result = describe_batch(batch)
            for c in batch:
                ctx = str(result.get(c["id"], "")).strip().replace("\n", " ")
                if not ctx:                      # model skipped one -> fall back to the heading
                    ctx = f"About: {c['section_title']}."
                new = dict(c)
                new["context"] = ctx
                new["text"] = ctx + "\n\n" + c["text"]   # what we actually embed
                out.write(json.dumps(new, ensure_ascii=False) + "\n")
            out.flush()
            time.sleep(2.0)   # gentle pacing between batches

    print(f"\nDone. Contextualized chunks in {config.CONTEXTUAL_PATH}")


if __name__ == "__main__":
    main()
