"""
Pluggable encoders for type-specific cache entry compression.

Encoders transform data to/from compact binary representations,
using a shared token dictionary for cross-entry compression.

Note: This submodule is designed for future migration to causaliq-core.
"""

from causaliq_knowledge.cache.encoders.base import EntryEncoder

__all__ = ["EntryEncoder"]
