"""
Shared test doubles for the Phase 2/3 supervisor tests (test_supervisor.py,
test_guardrails.py). No real LLM, no network, no cost — every test in the
suite other than test_supervisor_live.py (itself skipped without a real
GROQ_API_KEY) runs without ever reaching Groq.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


class FakeToolCallingModel(BaseChatModel):
    """Replays a fixed sequence of AIMessages (with tool_calls set) instead
    of calling a real LLM.

    Implements `bind_tools` as a no-op (returns self) since create_agent
    calls it during graph construction; the tool schema itself is ignored
    because this fake's responses are pre-scripted, not generated.
    """

    responses: list[AIMessage]
    idx: int = 0

    def _generate(
        self, messages: list, stop=None, run_manager=None, **kwargs: Any
    ) -> ChatResult:
        message = self.responses[self.idx]
        self.idx += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, **kwargs: Any):
        return self

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"


class FakeJudgeModel(BaseChatModel):
    """Replays a fixed sequence of judge verdicts ("ALLOW: ..." / "BLOCK:
    ...") instead of calling a real Groq judge call. Clamps to the last
    scripted response if called more times than scripted — `ask()` may
    call the judge twice per question (once for the input, once for the
    output), and most tests only care about scripting one verdict."""

    responses: list[str]
    idx: int = 0

    def _generate(
        self, messages: list, stop=None, run_manager=None, **kwargs: Any
    ) -> ChatResult:
        content = self.responses[min(self.idx, len(self.responses) - 1)]
        self.idx += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self) -> str:
        return "fake-judge"


@pytest.fixture
def make_fake_tool_model():
    """Factory fixture: make_fake_tool_model([AIMessage(...), ...])."""

    def _make(responses: list[AIMessage]) -> FakeToolCallingModel:
        return FakeToolCallingModel(responses=responses)

    return _make


@pytest.fixture
def make_fake_judge():
    """Factory fixture: make_fake_judge(["ALLOW: ..."]) / (["BLOCK: ..."])."""

    def _make(responses: list[str]) -> FakeJudgeModel:
        return FakeJudgeModel(responses=responses)

    return _make


@pytest.fixture
def allow_judge(make_fake_judge):
    """A judge that always allows — for tests exercising the agent/tool
    behaviour rather than the guardrail layer itself."""

    return make_fake_judge(["ALLOW: ok"])
