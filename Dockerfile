FROM python:3.14-slim AS backend

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app ./app
COPY data ./data
COPY docs/corpus ./docs/corpus

# Deterministic offline profile needs no credential to build the BM25 index.
RUN uv run python -m app.rag.ingest --rebuild

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
