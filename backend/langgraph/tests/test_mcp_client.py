from pathlib import Path

import pytest

from mcp_client import client


@pytest.mark.parametrize(
    ("platform", "executable"),
    [("linux", ("bin", "python")), ("darwin", ("bin", "python")), ("win32", ("Scripts", "python.exe"))],
)
def test_server_interpreter_matches_platform(monkeypatch, platform, executable):
    monkeypatch.delenv("MCP_SERVER_PYTHON", raising=False)
    monkeypatch.setattr(client.sys, "platform", platform)
    params = client._server_params()
    assert Path(params.command) == client._BACKEND_DIR.joinpath("mcp", ".venv", *executable)
    assert params.args == [str(client._MCP_SERVER_SCRIPT)]


def test_server_interpreter_override(monkeypatch):
    monkeypatch.setenv("MCP_SERVER_PYTHON", "/opt/mcp-venv/bin/python")
    assert client._server_params().command == "/opt/mcp-venv/bin/python"
