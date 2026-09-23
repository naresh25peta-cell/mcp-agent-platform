# clauseguard MCP server — read-only access to seeded NominationRules
# (SQLite), fully local/free, no API key needed. Talks MCP over stdio, so
# run with `-i`:
#   docker run -i --rm -v clauseguard-data:/app/data mcp-agent-platform-clauseguard
#
# Build from the REPO ROOT (context matters — it needs src/ and the
# vendor/clauseguard submodule):
#   git submodule update --init --recursive
#   docker build -f docker/clauseguard.Dockerfile -t mcp-agent-platform-clauseguard .
#
# Note: this image installs the same shared pyproject.toml as document-qa
# (including sentence-transformers/faiss), so it is larger than it strictly
# needs to be for a DB-only tool. Splitting per-server dependency groups is
# a good Phase 2 follow-up; kept simple for Phase 1.
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml poetry.lock* ./
RUN pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --only main --no-root

COPY src ./src
COPY vendor/clauseguard ./vendor/clauseguard

ENV PYTHONUNBUFFERED=1
VOLUME ["/app/data"]
ENTRYPOINT ["python", "-m", "mcp_agent_platform.servers.clauseguard_server"]
