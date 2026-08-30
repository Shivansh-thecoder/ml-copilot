"""Importing `backend/langgraph/` pieces from inside `backend/app/`.

WHY THIS EXISTS
---------------
Reaching `app.main`/`app.api...` at all requires `backend/` on
`sys.path` -- but `backend/` also contains directories named `langgraph`
and `mcp`, which SHADOW the installed `langgraph` and `mcp` SDKs of the
same name (the collision documented in backend/mcp/README.md's
"naming-collision gotcha"). Putting `backend/langgraph` on `sys.path` is
not enough on its own: `backend/` still sits ahead of site-packages in
the search order, so `import langgraph` / `import mcp` keep resolving to
the local packages.

The fix, used by every router that imports something out of
`backend/langgraph/`: temporarily remove every `sys.path` entry pointing
at `backend/` itself, do the imports, then restore it so later `app.*`
imports still resolve. Python caches the correctly-resolved SDK in
`sys.modules` the moment an import succeeds, so every later `import
langgraph`/`import mcp` anywhere in the process hits that cache instead
of re-resolving -- which is why doing this once per module, at import
time, is enough.
"""

import sys
from contextlib import contextmanager
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]  # backend/
_LANGGRAPH_DIR = _BACKEND_DIR / "langgraph"


@contextmanager
def langgraph_imports():
    """Import `backend/langgraph/` modules (and the SDKs they import) safely."""
    if str(_LANGGRAPH_DIR) not in sys.path:
        sys.path.insert(0, str(_LANGGRAPH_DIR))

    hidden = [p for p in sys.path if Path(p or ".").resolve() == _BACKEND_DIR]
    for p in hidden:
        sys.path.remove(p)
    try:
        yield
    finally:
        sys.path[:0] = hidden
