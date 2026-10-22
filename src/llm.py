# One small gateway for every Gemini call in the project.
#
# The free tier caps each model at its own small daily quota, so instead of dying
# when one model is used up, we just fall through to the next model and keep going.
# This is the multi-model failover idea from my Illomra project, but much simpler
# here: these calls are stateless (one prompt in, one answer out), so there's no
# chat history to carry across a switch. Each call still retries a model a couple
# of times on transient errors before moving on.

import os
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))

# Ordered so the model with the most generous free daily quota comes first
# (flash-lite comfortably handled hundreds of calls; the others cap around 20/day).
# Chaining them multiplies how much free headroom we have in a day.
FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
]


def _is_quota(err):
    m = str(err).lower()
    return "429" in m or "quota" in m or "resource" in m or "exhaust" in m


def generate(prompt, models=None, per_model_tries=2):
    """
    Ask the first model that answers. On a quota error, move to the next model.
    On a transient error, retry the same model with a short backoff first.
    Returns (text, model_name). Raises if every model is exhausted.
    """
    models = models or FALLBACK_MODELS
    last_err = None
    for name in models:
        delay = 2.0
        for _ in range(per_model_tries):
            try:
                r = _client.models.generate_content(model=name, contents=prompt)
                return r.text, name
            except Exception as e:
                last_err = e
                if _is_quota(e):
                    break            # this model's quota is used up -> next model
                time.sleep(delay)    # transient -> wait and retry same model
                delay *= 2
    raise RuntimeError(f"all models exhausted; last error: {str(last_err)[:200]}")
