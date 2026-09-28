"""
End-to-end test of the Phase 2 supervisor against the REAL Groq LLM (free
tier). Skipped automatically unless GROQ_API_KEY is set — this is the one
test in the suite that needs a real (free) API key and live network, so it
never blocks CI or a fresh clone; run it locally once your .env has a key:

    poetry run pytest tests/test_supervisor_live.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
)


@pytest.mark.asyncio
async def test_supervisor_answers_a_rules_question_live():
    from mcp_agent_platform.supervisor.graph import ask

    answer = await ask("What nomination rules exist for AgreementGuid AGR-001?")
    assert answer
    # Models may typeset IDs with Unicode hyphens (e.g. U+2011 "NR‑001").
    answer = answer.translate({ord(c): "-" for c in "‐‑‒–"})
    assert "NR-001" in answer or "AGR-001" in answer


@pytest.mark.asyncio
async def test_supervisor_answers_a_handbook_question_live():
    from mcp_agent_platform.supervisor.graph import ask

    answer = await ask("How many days of annual leave do full-time employees get?")
    assert answer
    assert "25" in answer
