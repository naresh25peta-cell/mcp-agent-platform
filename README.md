# mcp-agent-platform

An MCP-native platform that wraps two existing projects — [document-qa](https://github.com/naresh25peta-cell/document-qa)
(hybrid RAG retrieval) and [clauseguard](https://github.com/naresh25peta-cell/clauseguard)
(contract clause/risk assistant) — as [Model Context Protocol](https://modelcontextprotocol.io)
tool servers, and demonstrates acting as an MCP *client* against a
third-party server too.

## Phase 1 scope (this repo, as it stands)

This is Phase 1 of a larger roadmap. It deliberately covers just:

- **Two MCP servers**, each wrapping one existing project's *free, local-only*
  functionality — no code in either source project was modified; both are
  vendored in as git submodules under `vendor/` and imported directly.
- **One MCP client demo** connecting to the official filesystem reference
  server, to show this platform driving *someone else's* MCP server too.
- **Docker images** for both servers.
- **A CI-gated test suite** (GitHub Actions): tests run on every push, and
  both Docker images are built as part of CI too.

**Every tool in this repo runs at $0 cost.** Nothing here calls a paid LLM
API. Both source projects also have LLM-backed features (`document-qa`'s
`generate_answer`, `clauseguard`'s extraction/scoring tools) — those are
intentionally *not* wrapped in Phase 1, so nothing in this repo can ever
incur an API bill. A later phase can add them behind an explicit,
opt-in API key.

### `document-qa` MCP server — `src/mcp_agent_platform/servers/document_qa_server.py`

Wraps `document-qa`'s hybrid retrieval pipeline: semantic chunking, a FAISS
dense index, BM25Plus sparse search, and reciprocal-rank fusion. Embeddings
run locally via `sentence-transformers/all-MiniLM-L6-v2` (downloaded once on
first use, then cached — free, no account needed).

Tools: `list_sample_documents`, `search_documents`.

### `clauseguard` MCP server — `src/mcp_agent_platform/servers/clauseguard_server.py`

Wraps `clauseguard`'s `NominationRules` reference data: a plain SQLite read
over the project's own seeded sample data.

Tools: `list_nomination_rules`.

### Filesystem MCP client demo — `src/mcp_agent_platform/clients/filesystem_demo.py`

Connects to the official `@modelcontextprotocol/server-filesystem` reference
server (launched on demand via `npx`) and calls its `write_file`,
`list_directory`, and `read_text_file` tools against a sandboxed temp
directory. Chosen specifically because it needs no API key or account.

## Setup

```bash
git clone --recurse-submodules <this-repo-url>
cd mcp-agent-platform
# if you cloned without --recurse-submodules:
git submodule update --init --recursive

poetry install --with dev
```

Node.js is also needed for the filesystem client demo (`npx` must be on
`PATH`); nothing else needs to be installed for it.

## Running the servers

Each server speaks MCP over stdio and is meant to be launched by an MCP
client (an MCP-compatible chat application, an agent framework, this repo's
own tests, etc.), not run standalone in a terminal:

```bash
poetry run python -m mcp_agent_platform.servers.document_qa_server
poetry run python -m mcp_agent_platform.servers.clauseguard_server
```

Run the filesystem client demo directly — it starts its own server
subprocess and exits when done:

```bash
poetry run python -m mcp_agent_platform.clients.filesystem_demo
```

## Docker

```bash
git submodule update --init --recursive
docker build -f docker/document-qa.Dockerfile -t mcp-agent-platform-document-qa .
docker build -f docker/clauseguard.Dockerfile -t mcp-agent-platform-clauseguard .

# or, with docker compose:
docker compose build
docker compose run --rm document-qa
docker compose run --rm clauseguard
```

Both images are also built in CI on every push, so a broken Dockerfile
fails the build.

## Testing

```bash
poetry run pytest -v
```

The `document_qa_server` tests substitute a small deterministic hashing
"embedding model" for the real `sentence-transformers` model, so the whole
suite runs fast with **zero network access** and zero cost — useful in
network-locked-down environments. The real embedding model path is the same
production code (`document-qa`'s own `rag.py`, unmodified) and is exercised
normally the first time either server actually runs with real documents,
which needs one-time internet access to download the model (free, no
account) — this happens automatically in CI, in the Docker build, and on a
normal developer machine; it just isn't exercised by the unit tests
themselves.

The filesystem client demo test (`tests/test_filesystem_demo.py`) *does* run
against the real official filesystem MCP server via `npx`, and is skipped
automatically if Node.js isn't available.

## Roadmap

Phase 1 (this repo) is the foundation: two MCP servers, one MCP client, CI,
Docker. Later phases (not yet built) extend this into a full multi-agent
platform: a LangGraph supervisor orchestrating these tools, guardrails,
CI-gated evals (Ragas/DeepEval), and observability (OpenTelemetry GenAI
conventions / Langfuse).
