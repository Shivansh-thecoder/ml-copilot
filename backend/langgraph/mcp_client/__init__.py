"""The MCP client -- how this codebase talks to backend/mcp/server.py.

Discovers and invokes tools over MCP; a node that needs a tool should
only ever import from here, never spawn the server or speak the MCP
protocol directly.
"""
