"""
filesystem_demo.py — a client that connects to the OFFICIAL filesystem MCP
reference server (`@modelcontextprotocol/server-filesystem`, run on demand
via `npx`) and calls its tools.

This demonstrates mcp-agent-platform acting as an MCP *client* against a
third-party MCP *server*, separate from the two servers this project hosts
itself (document-qa, clauseguard). Chosen deliberately because it needs no
API key and no account — it just reads/writes files under one sandboxed
directory, so this demo has zero running cost.

Requires Node.js (for `npx`); nothing else to install.

Run:
    poetry run python -m mcp_agent_platform.clients.filesystem_demo
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

from mcp import Client, StdioServerParameters


async def run_demo(sandbox_dir: Path) -> None:
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(sandbox_dir)],
    )

    async with Client(server_params) as client:
        tools = await client.list_tools()
        print(f"Connected to filesystem MCP server. {len(tools.tools)} tools available:")
        for tool in tools.tools:
            print(f"  - {tool.name}")

        print(f"\nWriting demo_note.txt into {sandbox_dir} via the write_file tool...")
        await client.call_tool(
            "write_file",
            {
                "path": str(sandbox_dir / "demo_note.txt"),
                "content": "Written by mcp-agent-platform's filesystem MCP client demo.",
            },
        )

        print("Listing the sandbox directory via the list_directory tool...")
        listing = await client.call_tool(
            "list_directory", {"path": str(sandbox_dir)}
        )
        for block in listing.content:
            if hasattr(block, "text"):
                print(block.text)

        print("\nReading demo_note.txt back via the read_text_file tool...")
        contents = await client.call_tool(
            "read_text_file", {"path": str(sandbox_dir / "demo_note.txt")}
        )
        for block in contents.content:
            if hasattr(block, "text"):
                print(block.text)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mcp_agent_platform_fs_demo_") as tmp:
        sandbox = Path(tmp)
        (sandbox / "greeting.txt").write_text(
            "Hello from the mcp-agent-platform filesystem demo sandbox.\n"
        )
        asyncio.run(run_demo(sandbox))


if __name__ == "__main__":
    main()
    sys.exit(0)
