"""The MCP client: discovers and invokes tools on the MCP server.

Nothing here names a specific tool -- `list_mcp_tools()` returns whatever
`backend/mcp/server.py` currently registers, and `call_mcp_tool()` invokes
by name. Add a tool to the server and it's usable here with no client
changes.

WHY A SEPARATE SUBPROCESS, WITH A DIFFERENT PYTHON INTERPRETER
--------------------------------------------------------------------
Normal for MCP's stdio transport, not a workaround. The MCP server needs
`mcp`'s server-side code (`mcp.server.fastmcp`), which this process
(running LangGraph/LangChain, in `backend/.venv`) doesn't have and
shouldn't need -- `backend/mcp/.venv` is a separate environment because
the MCP server's dependencies conflict with FastAPI's (see
`backend/mcp/README.md`). `StdioServerParameters` below points at that
other interpreter running `backend/mcp/server.py`; the two processes only
talk over that subprocess's stdin/stdout.

ONE SESSION PER CALL
-------------------------
Every function here starts the server subprocess, does the MCP
handshake, does one thing, and shuts it down -- costs ~1-2s of
subprocess-startup overhead per call, but is by far the simplest version
of this code to read and trust. A persistent session (kept alive across
a whole graph run) is a reasonable future optimization once call volume
makes the overhead matter; nothing here design-locks that out.

SYNC WRAPPERS
-----------------
The MCP SDK's client API is async (built on `anyio`); every LangGraph
node here is a plain sync function. `asyncio.run()` bridges the two --
safe because nothing in this codebase runs an event loop of its own
(no `.ainvoke()` anywhere). `list_mcp_tools()`/`call_mcp_tool()` are the
real async implementations; `_sync` wrappers are what nodes import.
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool

# backend/langgraph/mcp_client/client.py -> two parents up is backend/,
# where both mcp/.venv (the server's isolated environment) and
# mcp/server.py live.
_BACKEND_DIR = Path(__file__).resolve().parents[2]
_MCP_SERVER_SCRIPT = _BACKEND_DIR / "mcp" / "server.py"


def _server_params() -> StdioServerParameters:
    # Containers can keep the server environment outside a bind-mounted /app.
    python = os.environ.get("MCP_SERVER_PYTHON")
    if not python:
        executable = ("Scripts", "python.exe") if sys.platform == "win32" else ("bin", "python")
        python = str(_BACKEND_DIR.joinpath("mcp", ".venv", *executable))
    return StdioServerParameters(command=python, args=[str(_MCP_SERVER_SCRIPT)])


@asynccontextmanager
async def mcp_session():
    """Start the MCP server as a subprocess and yield a ready-to-use,
    already-initialized session; torn down when the context exits."""
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def list_mcp_tools() -> list[Tool]:
    """Ask the MCP server what tools it currently exposes."""
    async with mcp_session() as session:
        result = await session.list_tools()
        return result.tools


async def call_mcp_tool(name: str, arguments: dict) -> str:
    """Invoke one MCP tool by name and return its text result.

    An unknown name surfaces as whatever error the MCP SDK raises;
    callers should catch it. MCP results carry a list of content blocks
    (to support tools returning e.g. text plus an image); this project's
    tools return exactly one text block, so only `content[0]` is used.
    """
    async with mcp_session() as session:
        result = await session.call_tool(name, arguments)
        return result.content[0].text


def list_mcp_tools_sync() -> list[Tool]:
    return asyncio.run(list_mcp_tools())


def call_mcp_tool_sync(name: str, arguments: dict) -> str:
    return asyncio.run(call_mcp_tool(name, arguments))
