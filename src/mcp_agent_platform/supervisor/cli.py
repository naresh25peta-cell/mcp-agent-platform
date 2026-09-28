"""
cli.py — ask the Phase 2 supervisor a question from the command line.

    poetry run python -m mcp_agent_platform.supervisor.cli "How many days of annual leave do I get?"
    poetry run python -m mcp_agent_platform.supervisor.cli "List nomination rules for AGR-001"

Requires a free GROQ_API_KEY (see .env.example) — nothing else here costs
anything.
"""
from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

from mcp_agent_platform.supervisor.graph import ask


def main() -> None:
    load_dotenv()
    if len(sys.argv) < 2:
        print('Usage: python -m mcp_agent_platform.supervisor.cli "<question>"')
        raise SystemExit(1)
    question = " ".join(sys.argv[1:])
    answer = asyncio.run(ask(question))
    print(answer)


if __name__ == "__main__":
    main()
