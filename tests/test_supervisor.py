"""
Tests for the Phase 2 supervisor (src/mcp_agent_platform/supervisor/).

These spin up the REAL Phase 1 MCP servers as subprocesses and call their
REAL tools over the real MCP protocol — nothing about tool execution is
mocked. Only the routing/reasoning LLM is substituted with a small
deterministic fake (no GROQ_API_KEY, no network, no cost), because that's
the one piece that would otherwise require a live API key.

The fake model's tool choice targets `list_nomination_rules` (ClauseGuard)
rather than `search_documents` (document-qa), because the former is a
pure SQLite read with no external dependency, while the latter needs a
one-time download of the embedding model — exercised for real in
tests/test_document_qa_server.py with its own mocked-model approach, and
in test_supervisor_live.py below against the real Groq model.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcp_agent_platform.supervisor.graph import ask, build_supervisor  # noqa: E402
from mcp_agent_platform.supervisor.mcp_tools import load_all_tools  # noqa: E402


class FakeToolCallingModel(BaseChatModel):
    """Replays a fixed sequence of AIMessages instead of calling a real LLM.

    Implements `bind_tools` as a no-op (returns self) since create_agent
    calls it during graph construction; the tool schema itself is ignored
    because this fake's responses are pre-scripted, not generated.
    """

    responses: list[AIMessage]
    idx: int = 0

    def _generate(self, messages: list, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        message = self.responses[self.idx]
        self.idx += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, **kwargs: Any):
        return self

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"


@pytest.mark.asyncio
async def test_load_all_tools_finds_both_servers_tools():
    tools = await load_all_tools()
    names = {t.name for t in tools}
    assert {"search_documents", "list_sample_documents", "list_nomination_rules"} <= names


@pytest.mark.asyncio
async def test_supervisor_routes_question_to_clauseguard_tool():
    fake_model = FakeToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_nomination_rules",
                        "args": {"agreement_guid": "AGR-001"},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(
                content="AGR-001 has two nomination rules: NR-001 and NR-002."
            ),
        ]
    )

    agent = await build_supervisor(fake_model)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What are the nomination rules for AGR-001?"}]}
    )

    messages = result["messages"]
    tool_messages = [m for m in messages if type(m).__name__ == "ToolMessage"]
    assert len(tool_messages) == 1
    assert "NR-001" in tool_messages[0].content
    assert "AGR-001" in tool_messages[0].content

    final = messages[-1]
    assert "NR-001" in final.content


@pytest.mark.asyncio
async def test_ask_returns_final_answer_text():
    fake_model = FakeToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_nomination_rules",
                        "args": {"trade_group_id": "TG-200"},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content="TG-200 has one nomination rule: NR-003."),
        ]
    )

    answer = await ask("Nomination rules for TG-200?", model=fake_model)
    assert answer == "TG-200 has one nomination rule: NR-003."
