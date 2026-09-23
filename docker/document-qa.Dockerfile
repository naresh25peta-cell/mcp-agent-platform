# document-qa MCP server — hybrid RAG retrieval (FAISS + BM25Plus), fully
# local/free, no API key needed. Talks MCP over stdio, so run with `-i`:
#   docker run -i --rm mcp-agent-platform-document-qa
#
# Build from the REPO ROOT (context matters — it needs src/, samples/, and
# the vendor/document-qa submodule):
#   git submodule update --init --recursive
#   docker build -f docker/document-qa.Dockerfile -t mcp-agent-platform-document-qa .
FROM python:3.11-slim

WORKDIR /app

# libgomp1: runtime dependency of faiss-cpu on slim base images
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock* ./
RUN pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --only main --no-root

COPY src ./src
COPY samples ./samples
COPY vendor/document-qa ./vendor/document-qa

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "mcp_agent_platform.servers.document_qa_server"]
