"""
mcp_tools.py — connects to this project's own Phase 1 MCP servers
(document-qa, clauseguard) as subprocesses and exposes their tools as
LangChain tools, so the Phase 2 supervisor can call them like any other
tool.

Hand-rolled rather than via the `langchain-mcp-adapters` package: its
current release imports `mcp.shared.context.RequestContext`, a name that
no longer exists in mcp 2.x's stateless-architecture rewrite — the same
kind of breaking rename Phase 1 hit with `FastMCP` -> `MCPServer` — so it
can't be installed alongside the Phase 1 servers, which already target
mcp 2.x. This module talks to `mcp.Client` directly instead (which does
support 2.x) and wraps each discovered tool as a LangChain StructuredTool.

Each server is launched fresh per tool call (a new stdio session per
invocation) — simple and stateless, at the cost of a per-call subprocess
startup. Fine for a portfolio-scale demo; a persistent-session variant
would be a reasonable Phase 3+ optimization.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from mcp import Client, StdioServerParameters

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / "src"

_SERVER_MODULES = {
    "document_qa": "mcp_agent_platform.servers.document_qa_server",
    "clauseguard": "mcp_agent_platform.servers.clauseguard_server",
}


def _server_env() -> dict[str, str]:
    """Subprocess env: inherit ours, but guarantee `src/` is importable even
    without an editable install of this package."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    parts = [str(_SRC_DIR)] + ([existing] if existing else [])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _server_params(module: str) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", module],
        cwd=str(_REPO_ROOT),
        env=_server_env(),
    )


def _make_langchain_tool(
    module: str, tool_name: str, description: str | None, input_schema: dict
) -> StructuredTool:
    async def _call(**kwargs: Any) -> Any:
        async with Client(_server_params(module)) as client:
            result = await client.call_tool(tool_name, kwargs)
        if result.structured_content is not None:
            content = result.structured_content
            return content.get("result", content) if isinstance(content, dict) else content
        return "\n".join(block.text for block in result.content if hasattr(block, "text"))

    return StructuredTool(
        name=tool_name,
        description=description or tool_name,
        args_schema=input_schema or {"type": "object", "properties": {}},
        coroutine=_call,
    )


async def load_all_tools() -> list[BaseTool]:
    """Spin up both Phase 1 servers, discover their tools, and return them
    as LangChain BaseTool objects ready to hand to a LangGraph/LangChain
    agent."""
    tools: list[BaseTool] = []
    for module in _SERVER_MODULES.values():
        async with Client(_server_params(module)) as client:
            listed = await client.list_tools()
        for t in listed.tools:
            tools.append(_make_langchain_tool(module, t.name, t.description, t.input_schema))
    return tools
