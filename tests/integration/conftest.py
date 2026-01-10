"""Conftest for integration tests.

This module provides pytest fixtures and configuration for integration tests
that make real API calls to external LLM providers.

These tests are slow and should not run in CI - use `pytest -m slow` locally.
"""

import os

import pytest


def has_api_key(env_var: str) -> bool:
    """Check if an API key environment variable is set."""
    key = os.getenv(env_var)
    return bool(key and key.strip())


def is_ollama_running() -> bool:
    """Check if Ollama server is running locally."""
    try:
        import httpx

        response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


# Skip conditions for each provider
skip_no_groq = pytest.mark.skipif(
    not has_api_key("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set",
)

skip_no_openai = pytest.mark.skipif(
    not has_api_key("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)

skip_no_anthropic = pytest.mark.skipif(
    not has_api_key("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)

skip_no_gemini = pytest.mark.skipif(
    not has_api_key("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set",
)

skip_no_deepseek = pytest.mark.skipif(
    not has_api_key("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY not set",
)

skip_no_mistral = pytest.mark.skipif(
    not has_api_key("MISTRAL_API_KEY"),
    reason="MISTRAL_API_KEY not set",
)

skip_no_ollama = pytest.mark.skipif(
    not is_ollama_running(),
    reason="Ollama server not running on localhost:11434",
)
