# Container for the Python Tutorial RAG service.
# Builds the vector index at image-build time (no API calls), then serves FastAPI.
# Listens on $PORT (e.g. Railway) or 7860 locally.

FROM python:3.11-slim

# Run as a non-root user so the Chroma index files stay writable at runtime
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install Python dependencies first (better layer caching)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the project (chunks + contextualized data are committed, so no re-scraping)
COPY --chown=user . .

# Pre-download the embedding + reranker models and build the Chroma index now,
# so the container starts fast and makes zero API calls on boot.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" \
 && python src/index.py --contextual

# Listen on the port the host provides ($PORT), defaulting to 7860 for local runs.
EXPOSE 7860
CMD uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-7860}
