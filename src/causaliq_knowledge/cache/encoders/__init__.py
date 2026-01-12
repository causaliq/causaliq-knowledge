"""
Pluggable encoders for type-specific cache entry compression.

Encoders transform data to/from compact binary representations,
using a shared token dictionary for cross-entry compression.

Note: This submodule is designed for future migration to causaliq-core.
"""

# Exports will be added as encoders are implemented:
# - EntryEncoder: ABC defining encoder interface
# - JsonEncoder: Generic JSON tokenisation encoder

__all__: list[str] = []
