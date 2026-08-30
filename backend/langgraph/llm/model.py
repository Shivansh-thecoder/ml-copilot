"""Centralized chat-model construction.

One place responsible for building the LLM client every node uses --
provider choice, model name, and API-key handling live here instead of
being duplicated across nodes. A node should only ever need
`from llm.model import get_chat_model`; it should never construct a
provider client itself.

Provider is OpenAI via `langchain-openai`'s `ChatOpenAI`. LangGraph
itself is provider-agnostic -- nodes and graphs never import
`langchain_openai` directly, only this file does, so swapping providers
later only touches this module.

This module deliberately does NOT reuse `app.core.config.Settings` (the
FastAPI app's settings object) -- `backend/langgraph/` stays import-
independent of `backend/app/`, with its own minimal configuration
instead of reaching into the FastAPI app's.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# backend/langgraph/llm/model.py -> two parents up is backend/, where
# .env (git-ignored) and .env.example both live.
_BACKEND_DIR = Path(__file__).resolve().parents[2]
load_dotenv(_BACKEND_DIR / ".env")

# Small, cheap, fast -- a reasonable default; override with
# MLCOPILOT_OPENAI_MODEL for a different one.
DEFAULT_MODEL = os.getenv("MLCOPILOT_OPENAI_MODEL", "gpt-4o-mini")


def get_chat_model() -> ChatOpenAI:
    """Build a configured chat model. Cheap to construct (no network call
    until `.invoke()`), so a fresh instance per call is fine."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy backend/.env.example to "
            "backend/.env and fill in your key, then try again."
        )
    return ChatOpenAI(model=DEFAULT_MODEL, api_key=api_key)
