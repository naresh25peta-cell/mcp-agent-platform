"""
Tests for document_qa_server.py.

The real sentence-transformers model requires a one-time download from
huggingface.co on first use (free, but needs open network — see README's
"Testing note" for why CI/production environments handle this fine while a
locked-down sandbox may not). To keep these tests fast, deterministic, and
runnable with zero network access, a small hashing "fake" embedding model
is substituted for the real one; it exercises the exact same code path
(parse_document -> semantic_chunks -> SearchIndex -> search) as production.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcp_agent_platform.servers import document_qa_server as dq  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

_DIM = 32


class _FakeEmbeddingModel:
    """Deterministic bag-of-words hashing "embedding" — no download, no cost."""

    def encode(self, texts, normalize_embeddings=True):
        vectors = []
        for text in texts:
            vec = np.zeros(_DIM, dtype="float32")
            for word in text.lower().split():
                idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % _DIM
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if normalize_embeddings and norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.asarray(vectors, dtype="float32")


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    monkeypatch.setattr(dq, "_get_model", lambda: _FakeEmbeddingModel())
    dq._indexes.clear()
    yield
    dq._indexes.clear()


@pytest.mark.asyncio
async def test_tools_are_registered():
    tools = await dq.mcp.list_tools()
    assert {t.name for t in tools} == {"list_sample_documents", "search_documents"}


@pytest.mark.asyncio
async def test_list_sample_documents_finds_bundled_handbook():
    result = await dq.mcp.call_tool("list_sample_documents", {})
    names = [d["name"] for d in result.structured_content["result"]]
    assert "sample_handbook.txt" in names


@pytest.mark.asyncio
async def test_search_documents_returns_chunks_with_expected_shape():
    result = await dq.mcp.call_tool(
        "search_documents",
        {"question": "how many days of annual leave do employees get", "top_k": 2},
    )
    assert not result.is_error
    chunks = result.structured_content["result"]
    assert 0 < len(chunks) <= 2
    for chunk in chunks:
        assert chunk["document"] == "sample_handbook.txt"
        assert isinstance(chunk["page"], int)
        assert chunk["text"]


@pytest.mark.asyncio
async def test_search_documents_unknown_document_raises():
    with pytest.raises(ToolError, match="Unknown document"):
        await dq.mcp.call_tool(
            "search_documents",
            {"question": "policy", "document": "does_not_exist.txt"},
        )
