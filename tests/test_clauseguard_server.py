"""Tests for clauseguard_server.py — pure SQLite reads, no network, no cost."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcp_agent_platform.servers import clauseguard_server as cg  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Redirect the server at a throwaway SQLite file for each test."""
    db_path = tmp_path / "clauseguard_test.db"
    monkeypatch.setattr(cg, "_DB_PATH", db_path)

    import clauseguard.database.schema as schema
    monkeypatch.setattr(schema, "DB_PATH", str(db_path))
    yield db_path


@pytest.mark.asyncio
async def test_tool_is_registered():
    tools = await cg.mcp.list_tools()
    assert [t.name for t in tools] == ["list_nomination_rules"]


@pytest.mark.asyncio
async def test_list_all_rules_seeds_on_first_call():
    result = await cg.mcp.call_tool("list_nomination_rules", {})
    rows = result.structured_content["result"]
    assert len(rows) == 3
    assert not result.is_error


@pytest.mark.asyncio
async def test_filter_by_agreement_guid():
    result = await cg.mcp.call_tool(
        "list_nomination_rules", {"agreement_guid": "AGR-001"}
    )
    rows = result.structured_content["result"]
    assert len(rows) == 2
    assert all(r["AgreementGuid"] == "AGR-001" for r in rows)


@pytest.mark.asyncio
async def test_filter_by_trade_group_id():
    result = await cg.mcp.call_tool(
        "list_nomination_rules", {"trade_group_id": "TG-200"}
    )
    rows = result.structured_content["result"]
    assert len(rows) == 1
    assert rows[0]["NominationRuleId"] == "NR-003"


@pytest.mark.asyncio
async def test_unknown_filter_returns_empty():
    result = await cg.mcp.call_tool(
        "list_nomination_rules", {"agreement_guid": "NOPE"}
    )
    assert result.structured_content["result"] == []
