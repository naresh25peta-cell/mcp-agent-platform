"""
clauseguard_server.py — exposes ClauseGuard's NominationRules reference data
as an MCP server.

Phase 1 scope: pure database reads only. ClauseGuard's extraction/scoring
tools call OpenAI-compatible LLM endpoints and are deliberately NOT wrapped
here, so this server has zero running cost.

Tools exposed:
  - list_nomination_rules : query seeded NominationRules by agreement/trade group

Run directly for a local stdio smoke test:
    poetry run python -m mcp_agent_platform.servers.clauseguard_server
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# --- make the vendored clauseguard package importable without modifying it ---
_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_CLAUSEGUARD_SRC = _REPO_ROOT / "vendor" / "clauseguard" / "src"
if str(_VENDOR_CLAUSEGUARD_SRC) not in sys.path:
    sys.path.insert(0, str(_VENDOR_CLAUSEGUARD_SRC))

# clauseguard.config reads DB_PATH from the environment at import time, so
# this must be set before the first `import clauseguard...` — point it at
# our own project's data directory rather than the vendored repo's.
_DB_PATH = _REPO_ROOT / "data" / "clauseguard.db"
os.environ.setdefault("DB_PATH", str(_DB_PATH))

from clauseguard.database.schema import get_connection, create_tables  # noqa: E402
from clauseguard.database.seed import seed  # noqa: E402

from mcp.server.mcpserver import MCPServer

mcp = MCPServer(
    "clauseguard",
    instructions=(
        "Read-only access to ClauseGuard's seeded NominationRules reference "
        "data (SQLite). Fully local and free — no API key required. "
        "Extraction/scoring tools that call an LLM are not exposed here."
    ),
)


def _ensure_seeded() -> None:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _DB_PATH.exists():
        return
    create_tables()
    seed()


@mcp.tool()
def list_nomination_rules(
    agreement_guid: str | None = None,
    trade_group_id: str | None = None,
) -> list[dict]:
    """List NominationRules, optionally filtered by AgreementGuid and/or TradeGroupId.

    Backed by ClauseGuard's seeded sample data (a plain SQLite read, no LLM
    call, no cost). Try agreement_guid="AGR-001" or trade_group_id="TG-100"
    to see the bundled sample rules.
    """
    _ensure_seeded()
    conn = get_connection()
    try:
        clauses, params = [], []
        if agreement_guid:
            clauses.append("AgreementGuid = ?")
            params.append(agreement_guid)
        if trade_group_id:
            clauses.append("TradeGroupId = ?")
            params.append(trade_group_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = conn.execute(
            f"SELECT * FROM NominationRules {where} ORDER BY id", params
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


if __name__ == "__main__":
    mcp.run(transport="stdio")
