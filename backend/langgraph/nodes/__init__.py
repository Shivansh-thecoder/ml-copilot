"""Node functions, one concept per file.

A "node" in LangGraph is just a Python callable: `(state) -> partial_update_dict`.
There is nothing framework-specific about the functions in this package --
they could be unit-tested and called directly with a plain dict, with no
graph involved at all. The `graphs/` package is what wires them into a
StateGraph; this package only contains the work each step performs.
"""
