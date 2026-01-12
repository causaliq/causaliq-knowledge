"""
Core caching infrastructure for causaliq.

This module provides a generic caching system with:
- SQLite-backed storage with concurrency support
- Pluggable encoders for type-specific compression
- Shared token dictionary for cross-entry compression
- Import/export for human-readable formats

Note: This module is designed for future migration to causaliq-core.
LLM-specific caching code remains in causaliq_knowledge.llm.cache.
"""

from causaliq_knowledge.cache.token_cache import TokenCache

__all__ = [
    "TokenCache",
]
