"""
Integration test for filesystem_demo.py against the REAL official filesystem
MCP server (launched via npx). Free (no API key), but needs Node.js on PATH
and network access to the npm registry the first time npx fetches the
package — both are present in CI and on a normal developer machine.
"""
import shutil
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("npx") is None, reason="Node.js/npx not available"
)


@pytest.mark.asyncio
async def test_filesystem_demo_round_trip():
    from mcp_agent_platform.clients.filesystem_demo import run_demo

    with tempfile.TemporaryDirectory(prefix="mcp_agent_platform_test_fs_") as tmp:
        sandbox = Path(tmp)
        (sandbox / "greeting.txt").write_text("hello\n")
        # run_demo prints its own progress; a clean run with no exception is
        # the pass condition, since it exercises write_file, list_directory,
        # and read_text_file against a real, freshly-spawned MCP server.
        await run_demo(sandbox)
        assert (sandbox / "demo_note.txt").exists()
