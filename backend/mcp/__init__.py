"""A local MCP server exposing the ML engine's capabilities as tools.

Its own top-level package under `backend/`, a sibling of
`backend/langgraph/` and `backend/app/`, not a subpackage of either:
this package's dependency (`mcp`, installed in `backend/mcp/.venv`)
conflicts with `backend/app/`'s pinned `starlette` version (see
README.md), so keeping it a separate package with its own environment
isn't just tidy, it's required.

See README.md for the architecture and tool contract.
"""
