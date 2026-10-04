"""
graph.py — the Phase 2 supervisor: a LangChain/LangGraph agent that reads a
natural-language question, decides whether the document-qa search tool or
the clauseguard rules-lookup tool (both from Phase 1) can answer it, calls
that tool, and answers using only what the tool returned.

Phase 3 wraps `ask()` with a guardrail layer (supervisor/guardrails.py):
a rule-based check runs first on both the question and the final answer,
and — unless disabled — a second Groq call judges each side too. See that
module's docstring for why there are two layers rather than one.

The routing/reasoning LLM is Groq's free tier (no cost, requires a free
GROQ_API_KEY — see .env.example). Nothing else in this module calls a paid
API; the tools it drives are the same $0 Phase 1 servers.
"""
from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain.agents import create_agent

from mcp_agent_platform.supervisor.guardrails import (
    check_input_rules,
    check_output_rules,
    judge_enabled,
    judge_input,
    judge_output,
)
from mcp_agent_platform.supervisor.mcp_tools import load_all_tools

SYSTEM_PROMPT = (
    "You are a supervisor agent with two tools available: `search_documents` "
    "(hybrid search over a company handbook for policy questions — pay, "
    "leave, remote work, equipment, expenses) and `list_nomination_rules` "
    "(looks up contract nomination rules by AgreementGuid and/or "
    "TradeGroupId). Read the user's question, pick the one tool that can "
    "actually answer it, call it, and answer using only what the tool "
    "returned. If neither tool is relevant to the question, say so plainly "
    "instead of guessing an answer."
)


def default_model(model_name: str | None = None) -> BaseChatModel:
    """Groq's free tier — no cost, requires GROQ_API_KEY in the environment."""
    from langchain_groq import ChatGroq

    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "(no card required) and put it in a .env file — see .env.example."
        )
    return ChatGroq(
        model=model_name or os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0,
    )


def default_judge_model() -> BaseChatModel:
    """The LLM used for the Phase 3 guardrail judge calls. Defaults to the
    same Groq key/model as the main supervisor; set GROQ_JUDGE_MODEL to use
    a different (e.g. smaller/faster) model just for judging."""
    return default_model(os.environ.get("GROQ_JUDGE_MODEL"))


async def build_supervisor(model: BaseChatModel | None = None):
    """Build the compiled agent, wired to both Phase 1 MCP servers.

    Pass `model` to substitute a test double (see tests/test_supervisor.py)
    without needing a live GROQ_API_KEY; omitted, it uses Groq's free tier.
    """
    tools = await load_all_tools()
    return create_agent(model or default_model(), tools, system_prompt=SYSTEM_PROMPT)


async def ask(
    question: str,
    model: BaseChatModel | None = None,
    judge_model: BaseChatModel | None = None,
) -> str:
    """Run one question through the Phase 3 guardrails and, if it passes,
    the Phase 2 supervisor agent — returning either the agent's final
    answer or a guardrail's refusal message.

    Pass `model`/`judge_model` to substitute test doubles (see
    tests/test_supervisor.py, tests/test_guardrails.py) without needing a
    live GROQ_API_KEY; omitted, both default to Groq's free tier. A judge
    model is only ever built (and GROQ_API_KEY only ever required for it)
    if the LLM judge layer is actually enabled — see guardrails.judge_enabled.
    """
    input_rule_check = check_input_rules(question)
    if not input_rule_check.allowed:
        return f"I can't help with that request: {input_rule_check.reason}."

    judge = judge_model
    if judge is None and judge_enabled():
        judge = default_judge_model()

    input_judge_verdict = await judge_input(question, judge)
    if not input_judge_verdict.allowed:
        return f"I can't help with that request: {input_judge_verdict.reason}."

    agent = await build_supervisor(model)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
    answer = result["messages"][-1].content

    output_rule_check = check_output_rules(answer)
    if not output_rule_check.allowed:
        return f"I can't share that answer: {output_rule_check.reason}."

    output_judge_verdict = await judge_output(question, answer, judge)
    if not output_judge_verdict.allowed:
        return f"I can't share that answer: {output_judge_verdict.reason}."

    return answer
