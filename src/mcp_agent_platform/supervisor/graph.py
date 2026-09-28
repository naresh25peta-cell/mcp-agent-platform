"""
graph.py — the Phase 2 supervisor: a LangChain/LangGraph agent that reads a
natural-language question, decides whether the document-qa search tool or
the clauseguard rules-lookup tool (both from Phase 1) can answer it, calls
that tool, and answers using only what the tool returned.

The routing/reasoning LLM is Groq's free tier (no cost, requires a free
GROQ_API_KEY — see .env.example). Nothing else in this module calls a paid
API; the tools it drives are the same $0 Phase 1 servers.
"""
from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain.agents import create_agent

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


def default_model() -> BaseChatModel:
    """Groq's free tier — no cost, requires GROQ_API_KEY in the environment."""
    from langchain_groq import ChatGroq

    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "(no card required) and put it in a .env file — see .env.example."
        )
    return ChatGroq(
        model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0,
    )


async def build_supervisor(model: BaseChatModel | None = None):
    """Build the compiled agent, wired to both Phase 1 MCP servers.

    Pass `model` to substitute a test double (see tests/test_supervisor.py)
    without needing a live GROQ_API_KEY; omitted, it uses Groq's free tier.
    """
    tools = await load_all_tools()
    return create_agent(model or default_model(), tools, system_prompt=SYSTEM_PROMPT)


async def ask(question: str, model: BaseChatModel | None = None) -> str:
    """Run one question through the supervisor and return its final answer."""
    agent = await build_supervisor(model)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content
