# Cache Module (Core)

SQLite-backed caching infrastructure with shared token dictionary for
efficient storage.

!!! info "Migrated to causaliq-core"
    The core cache classes (`TokenCache`, `EntryEncoder`, `JsonEncoder`) have 
    been migrated to `causaliq-core`. Import them from `causaliq_core.cache`.
    LLM-specific code ([LLMEntryEncoder](../llm/cache.md)) remains in 
    `causaliq-knowledge`.

## Overview

The cache module provides:

- **TokenCache** - SQLite-backed cache with connection management
- **EntryEncoder** - Abstract base class for pluggable type-specific encoders
- **JsonEncoder** - Tokenised encoder for JSON-serialisable data (50-70% compression)

| Component | Migration Target |
|-----------|------------------|
| `TokenCache` | `causaliq-core` |
| `EntryEncoder` | `causaliq-core` |
| `JsonEncoder` | `causaliq-core` |

For LLM-specific caching, see [LLM Cache](../llm/cache.md) which provides
`LLMEntryEncoder` with structured data types for requests and responses.

## Design Philosophy

The cache uses SQLite for storage, providing:

- Fast indexed key lookup
- Built-in concurrency via SQLite locking
- In-memory mode via `:memory:` for testing
- Incremental updates without rewriting

See [Caching Architecture](../../architecture/caching.md) for full design details.

## Usage

### Basic In-Memory Cache

```python
from causaliq_core.cache import TokenCache

# In-memory cache (fast, non-persistent)
with TokenCache(":memory:") as cache:
    assert cache.table_exists("tokens")
    assert cache.table_exists("cache_entries")
```

### File-Based Persistent Cache

```python
from causaliq_core.cache import TokenCache

# File-based cache (persistent)
with TokenCache("my_cache.db") as cache:
    # Data persists across sessions
    print(f"Entries: {cache.entry_count()}")
    print(f"Tokens: {cache.token_count()}")
```

### Transaction Support

```python
from causaliq_core.cache import TokenCache

with TokenCache(":memory:") as cache:
    # Transactions auto-commit on success, rollback on exception
    with cache.transaction() as cursor:
        cursor.execute("INSERT INTO tokens (token) VALUES (?)", ("example",))
```

### Token Dictionary

The cache maintains a shared token dictionary for cross-entry compression.
Encoders use this to convert strings to compact integer IDs:

```python
from causaliq_core.cache import TokenCache

with TokenCache(":memory:") as cache:
    # Get or create token IDs (used by encoders)
    id1 = cache.get_or_create_token("hello")  # Returns 1
    id2 = cache.get_or_create_token("world")  # Returns 2
    id1_again = cache.get_or_create_token("hello")  # Returns 1 (cached)

    # Look up token by ID (used by decoders)
    token = cache.get_token(1)  # Returns "hello"
```

### Storing and Retrieving Entries

Cache entries are stored as binary blobs with a hash key and entry type:

```python
from causaliq_core.cache import TokenCache

with TokenCache(":memory:") as cache:
    # Store an entry
    cache.put("abc123", "llm", b"response data")
    
    # Check if entry exists
    if cache.exists("abc123", "llm"):
        # Retrieve entry
        data = cache.get("abc123", "llm")  # Returns b"response data"
    
    # Store with metadata
    cache.put("def456", "llm", b"data", metadata=b"extra info")
    result = cache.get_with_metadata("def456", "llm")
    # result = (b"data", b"extra info")
    
    # Delete entry
    cache.delete("abc123", "llm")
```

### Auto-Encoding with Registered Encoders

Register an encoder to automatically encode/decode entries:

```python
from causaliq_core.cache import TokenCache
from causaliq_core.cache.encoders import JsonEncoder

with TokenCache(":memory:") as cache:
    # Register encoder for "json" entry type
    cache.register_encoder("json", JsonEncoder())
    
    # Store data (auto-encoded)
    cache.put_data("hash1", "json", {"role": "user", "content": "Hello"})
    
    # Retrieve data (auto-decoded)
    data = cache.get_data("hash1", "json")
    # data = {"role": "user", "content": "Hello"}
    
    # Store with metadata
    cache.put_data("hash2", "json", 
                   {"response": "Hi!"}, 
                   metadata={"latency_ms": 150})
    result = cache.get_data_with_metadata("hash2", "json")
    # result = ({"response": "Hi!"}, {"latency_ms": 150})
```

### Exporting and Importing Entries

Export cache entries to files for backup, migration, or sharing.
Import entries from files into a cache:

```python
from pathlib import Path
from causaliq_core.cache import TokenCache
from causaliq_core.cache.encoders import JsonEncoder

# Export entries to directory
with TokenCache("my_cache.db") as cache:
    cache.register_encoder("json", JsonEncoder())
    
    # Export all entries of type "json" to directory
    # Creates one file per entry: {hash}.json
    count = cache.export_entries(Path("./export"), "json")
    print(f"Exported {count} entries")

# Import entries from directory
with TokenCache("new_cache.db") as cache:
    cache.register_encoder("json", JsonEncoder())
    
    # Import all .json files from directory
    # Uses filename (without extension) as hash key
    count = cache.import_entries(Path("./export"), "json")
    print(f"Imported {count} entries")
```

**Export behaviour:**

- Creates output directory if it doesn't exist
- Writes each entry to `{hash}.{ext}` (e.g., `abc123.json`)
- Uses encoder's `export()` method for human-readable format
- Returns count of exported entries

**Import behaviour:**

- Reads all files in directory (skips subdirectories)
- Uses filename stem as hash key (e.g., `abc123.json` → key `abc123`)
- Uses encoder's `import_()` method to parse content
- Returns count of imported entries

## API Reference

::: causaliq_core.cache.TokenCache
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - open
        - close
        - is_open
        - is_memory
        - conn
        - transaction
        - table_exists
        - entry_count
        - token_count
        - get_or_create_token
        - get_token
        - register_encoder
        - get_encoder
        - has_encoder
        - put
        - get
        - get_with_metadata
        - exists
        - delete
        - put_data
        - get_data
        - get_data_with_metadata
        - export_entries
        - import_entries

## EntryEncoder

The `EntryEncoder` abstract base class defines the interface for pluggable
cache encoders. Each encoder handles a specific entry type (e.g., LLM
requests, embeddings, documents).

### Creating a Custom Encoder

```python
from causaliq_core.cache import TokenCache
from causaliq_core.cache.encoders import EntryEncoder


class MyEncoder(EntryEncoder):
    """Example encoder for custom data types."""

    def encode(self, data: dict, cache: TokenCache) -> bytes:
        """Convert data to bytes for storage."""
        # Use cache.get_or_create_token() for string compression
        return b"encoded"

    def decode(self, data: bytes, cache: TokenCache) -> dict:
        """Convert bytes back to original data."""
        # Use cache.get_token() to restore strings
        return {"decoded": True}

    def export(self, data: bytes, cache: TokenCache, fmt: str) -> str:
        """Export to human-readable format (json, yaml)."""
        return '{"decoded": true}'

    def import_(self, data: str, cache: TokenCache, fmt: str) -> bytes:
        """Import from human-readable format."""
        return b"encoded"
```

### Encoder Interface

::: causaliq_core.cache.encoders.EntryEncoder
    options:
      show_root_heading: true
      show_source: false
      members:
        - encode
        - decode
        - export
        - import_
        - default_export_format
