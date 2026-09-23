"""
document_qa_server.py — exposes the `document-qa` project's hybrid RAG
retrieval pipeline (FAISS + BM25Plus, semantic chunking) as an MCP server.

Phase 1 scope: retrieval only. `document-qa`'s `generate_answer()` needs a
paid LLM API key and is deliberately NOT wrapped here, so this server has
zero running cost — embeddings run locally via sentence-transformers.

Tools exposed:
  - list_sample_documents : which bundled documents can be searched
  - search_documents       : hybrid semantic + keyword search over a document

Run directly for a local stdio smoke test:
    poetry run python -m mcp_agent_platform.servers.document_qa_server
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

# --- make the vendored document-qa package importable without modifying it ---
_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_DOC_QA = _REPO_ROOT / "vendor" / "document-qa"
if str(_VENDOR_DOC_QA) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DOC_QA))

from rag import parse_document, semantic_chunks, SearchIndex  # noqa: E402  (vendored module)

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

mcp = MCPServer(
    "document-qa",
    instructions=(
        "Hybrid RAG search (FAISS dense + BM25Plus sparse, reciprocal-rank "
        "fusion) over bundled sample documents. Fully local and free — no "
        "API key required. Retrieval only; this server does not call any "
        "LLM to generate answers."
    ),
)

_SAMPLES_DIR = _REPO_ROOT / "samples"
_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# module-level caches so the embedding model loads once and each document's
# index is built once per process, not once per tool call
_model = None
_indexes: dict[str, SearchIndex] = {}


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    return _model


def _get_index(document_name: str) -> SearchIndex:
    if document_name in _indexes:
        return _indexes[document_name]
    path = _SAMPLES_DIR / document_name
    if not path.is_file():
        available = ", ".join(p.name for p in _SAMPLES_DIR.glob("*") if p.is_file())
        raise ToolError(f"Unknown document '{document_name}'. Available: {available}")
    data = path.read_bytes()
    pages = parse_document(document_name, data)
    chunks = semantic_chunks(pages, _get_model())
    index = SearchIndex(chunks, _get_model())
    _indexes[document_name] = index
    return index


@mcp.tool()
def list_sample_documents() -> list[dict]:
    """List the bundled sample documents available to search_documents."""
    return [
        {"name": p.name, "size_bytes": p.stat().st_size}
        for p in sorted(_SAMPLES_DIR.glob("*"))
        if p.is_file()
    ]


@mcp.tool()
def search_documents(
    question: str,
    document: str = "sample_handbook.txt",
    top_k: int = 4,
    mode: Literal["hybrid", "semantic", "keyword"] = "hybrid",
) -> list[dict]:
    """Search a bundled document with hybrid dense+keyword retrieval.

    Runs document-qa's chunking + FAISS/BM25Plus fusion locally (no LLM
    call, no API key, no cost). Returns the top matching chunks with their
    document name, page number, and text — use list_sample_documents to see
    what's available for the `document` argument.
    """
    index = _get_index(document)
    results = index.search(question, top_k=top_k, mode=mode)
    return [
        {"chunk_id": c.id, "document": c.document, "page": c.page, "text": c.text}
        for c in results
    ]


if __name__ == "__main__":
    mcp.run(transport="stdio")
