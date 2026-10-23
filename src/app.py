# The service: a small FastAPI app so you can actually ASK the thing questions.
#
# POST /ask   {"question": "..."}  -> grounded answer + pinpoint sources (or a refusal)
# GET  /meta                        -> the model failover chain, for the UI footer
# GET  /health                      -> quick "is it up" check
# GET  /                            -> the web UI (src/static/index.html)
#
# Run it:  uvicorn src.app:app --reload   then open http://127.0.0.1:8000

import os
import sys
import time
from collections import defaultdict, deque

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src import llm
from src.answer import answer, REFUSAL

app = FastAPI(title="Python Tutorial RAG", version="1.0")

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML = os.path.join(HERE, "static", "index.html")

# Tiny in-memory per-IP rate limit (one process). Not a substitute for a real
# gateway, but it stops a single client burning the daily Gemini quota in seconds.
_hits = defaultdict(deque)


def _rate_limited(ip):
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= config.RATE_LIMIT_PER_MIN:
        return True
    q.append(now)
    return False


class Ask(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/meta")
def meta():
    """Small facts the UI shows: the model failover chain."""
    return {"models": llm.FALLBACK_MODELS}


@app.post("/ask")
def ask(body: Ask, request: Request):
    """Retrieve from the Python tutorial and answer, grounded, with citations."""
    ip = request.client.host if request.client else "unknown"
    if _rate_limited(ip):
        return JSONResponse(status_code=429,
                            content={"detail": "Too many requests, slow down a moment."})

    q = (body.question or "").strip()
    if not q:
        return JSONResponse(status_code=400,
                            content={"detail": "Question is empty."})
    if len(q) > config.MAX_QUESTION_CHARS:
        return JSONResponse(status_code=400,
                            content={"detail": f"Question too long (max {config.MAX_QUESTION_CHARS} characters)."})

    try:
        return answer(q)
    except RuntimeError:
        # every model's free quota is used up (or the LLM is unreachable)
        return JSONResponse(status_code=503, content={
            "question": q,
            "answer": "The answering service is busy right now (model quota reached). "
                      "Please try again in a little while.",
            "sources": [], "refused": True, "model": None,
            "metrics": {"grounded": False},
        })
    except Exception:
        return JSONResponse(status_code=500,
                            content={"detail": "Unexpected error handling the question."})


@app.get("/", response_class=HTMLResponse)
def home():
    with open(INDEX_HTML, encoding="utf-8") as f:
        return f.read()
