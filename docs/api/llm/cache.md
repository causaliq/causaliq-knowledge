# LLM Cache

LLM-specific cache encoder and data structures for storing and retrieving
LLM requests and responses with rich metadata.

!!! info "Package Separation"
    This module stays in `causaliq-knowledge` as it contains LLM-specific logic.
    The core cache infrastructure (`TokenCache`, `EntryEncoder`, `JsonEncoder`) 
    is in `causaliq-core`. Import from `causaliq_core.cache`.

## Overview

The LLM cache module provides:

- **LLMEntryEncoder** - Extends JsonEncoder with LLM-specific convenience methods
- **LLMCacheEntry** - Complete cache entry with request, response, and metadata
- **LLMResponse** - Response data (content, finish reason, model version)
- **LLMMetadata** - Rich metadata (provider, tokens, latency, cost)
- **LLMTokenUsage** - Token usage statistics

## Design Philosophy

The LLM cache separates concerns:

| Component | Package |
|-----------|---------|  
| `TokenCache` | `causaliq_core.cache` |
| `EntryEncoder` | `causaliq_core.cache.encoders` |
| `JsonEncoder` | `causaliq_core.cache.encoders` |
| `LLMEntryEncoder` | `causaliq_knowledge.llm.cache` |
| `LLMCacheEntry` | `causaliq_knowledge.llm.cache` |

This allows the base cache to be reused across projects while keeping
LLM-specific logic in the appropriate package.

## Usage

### Creating Cache Entries

Use the `LLMCacheEntry.create()` factory method for convenient entry creation:

```python
from causaliq_knowledge.llm.cache import LLMCacheEntry

entry = LLMCacheEntry.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ],
    content="Hi there! How can I help you today?",
    temperature=0.7,
    max_tokens=1000,
    provider="openai",
    latency_ms=850,
    input_tokens=25,
    output_tokens=15,
    cost_usd=0.002,
)
```

### Encoding and Storing Entries

```python
from causaliq_core.cache import TokenCache
from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMEntryEncoder

with TokenCache(":memory:") as cache:
    encoder = LLMEntryEncoder()
    
    # Create an entry
    entry = LLMCacheEntry.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "What is Python?"}],
        content="Python is a programming language.",
        provider="openai",
    )
    
    # Encode to bytes
    blob = encoder.encode_entry(entry, cache)
    
    # Store in cache
    cache.put("request-hash", "llm", blob)
```

### Retrieving and Decoding Entries

```python
from causaliq_core.cache import TokenCache
from causaliq_knowledge.llm.cache import LLMEntryEncoder

with TokenCache("cache.db") as cache:
    encoder = LLMEntryEncoder()
    
    # Retrieve from cache
    blob = cache.get("request-hash", "llm")
    if blob:
        # Decode to LLMCacheEntry
        entry = encoder.decode_entry(blob, cache)
        print(f"Response: {entry.response.content}")
        print(f"Latency: {entry.metadata.latency_ms}ms")
```

### Exporting and Importing Entries

Export entries to JSON for inspection or migration:

```python
from pathlib import Path
from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMEntryEncoder

encoder = LLMEntryEncoder()

# Create entry
entry = LLMCacheEntry.create(
    model="claude-3",
    messages=[{"role": "user", "content": "Hello"}],
    content="Hi!",
    provider="anthropic",
)

# Export to JSON file
encoder.export_entry(entry, Path("entry.json"))

# Import from JSON file
restored = encoder.import_entry(Path("entry.json"))
```

### Using with TokenCache Auto-Encoding

Register the encoder for automatic encoding/decoding:

```python
from causaliq_core.cache import TokenCache
from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMEntryEncoder

with TokenCache(":memory:") as cache:
    # Register encoder for "llm" entry type
    cache.register_encoder("llm", LLMEntryEncoder())
    
    entry = LLMCacheEntry.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        content="Hi!",
    )
    
    # Store with auto-encoding
    cache.put_data("hash123", "llm", entry.to_dict())
    
    # Retrieve with auto-decoding
    data = cache.get_data("hash123", "llm")
    restored = LLMCacheEntry.from_dict(data)
```

### Using cached_completion with BaseLLMClient

The recommended way to use caching is via `BaseLLMClient.cached_completion()`:

```python
from causaliq_core.cache import TokenCache
from causaliq_knowledge.llm import GroqClient, LLMConfig

with TokenCache("llm_cache.db") as cache:
    client = GroqClient(LLMConfig(model="llama-3.1-8b-instant"))
    client.set_cache(cache)
    
    # First call - makes API request, caches response with latency
    response = client.cached_completion(
        [{"role": "user", "content": "What is Python?"}]
    )
    
    # Second call - returns from cache, no API call
    response = client.cached_completion(
        [{"role": "user", "content": "What is Python?"}]
    )
```

This automatically:

- Generates a deterministic cache key (SHA-256 of model + messages + params)
- Checks cache before making API calls
- Captures latency with `time.perf_counter()`
- Stores response with full metadata

### Importing Pre-Cached Responses

Load cached responses from JSON files for testing or migration:

```python
from pathlib import Path
from causaliq_core.cache import TokenCache
from causaliq_knowledge.llm.cache import LLMEntryEncoder

with TokenCache("llm_cache.db") as cache:
    cache.register_encoder("llm", LLMEntryEncoder())
    
    # Import all LLM entries from directory
    count = cache.import_entries(Path("./cached_responses"), "llm")
    print(f"Imported {count} cached LLM responses")
```

## Data Structures

### LLMTokenUsage

Token usage statistics for billing and analysis:

```python
from causaliq_knowledge.llm.cache import LLMTokenUsage

usage = LLMTokenUsage(
    input=100,   # Prompt tokens
    output=50,   # Completion tokens  
    total=150,   # Total tokens
)
```

### LLMMetadata

Rich metadata for debugging and analytics:

```python
from causaliq_knowledge.llm.cache import LLMMetadata, LLMTokenUsage

metadata = LLMMetadata(
    provider="openai",
    timestamp="2024-01-15T10:30:00+00:00",
    latency_ms=850,
    tokens=LLMTokenUsage(input=100, output=50, total=150),
    cost_usd=0.005,
    cache_hit=False,
)

# Convert to/from dict
data = metadata.to_dict()
restored = LLMMetadata.from_dict(data)
```

### LLMResponse

Response content and generation info:

```python
from causaliq_knowledge.llm.cache import LLMResponse

response = LLMResponse(
    content="The answer is 42.",
    finish_reason="stop",
    model_version="gpt-4-0125-preview",
)

# Convert to/from dict
data = response.to_dict()
restored = LLMResponse.from_dict(data)
```

### LLMCacheEntry

Complete cache entry combining request and response:

```python
from causaliq_knowledge.llm.cache import (
    LLMCacheEntry, LLMResponse, LLMMetadata
)

entry = LLMCacheEntry(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=1000,
    response=LLMResponse(content="Hi!"),
    metadata=LLMMetadata(provider="openai"),
)

# Preferred: use factory method
entry = LLMCacheEntry.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    content="Hi!",
    provider="openai",
)

# Convert to/from dict
data = entry.to_dict()
restored = LLMCacheEntry.from_dict(data)
```

## API Reference

### LLMEntryEncoder

::: causaliq_knowledge.llm.cache.LLMEntryEncoder
    options:
      show_root_heading: true
      show_source: false
      members:
        - encode_entry
        - decode_entry
        - export_entry
        - import_entry

### LLMCacheEntry

::: causaliq_knowledge.llm.cache.LLMCacheEntry
    options:
      show_root_heading: true
      show_source: false
      members:
        - create
        - to_dict
        - from_dict

### LLMResponse

::: causaliq_knowledge.llm.cache.LLMResponse
    options:
      show_root_heading: true
      show_source: false
      members:
        - to_dict
        - from_dict

### LLMMetadata

::: causaliq_knowledge.llm.cache.LLMMetadata
    options:
      show_root_heading: true
      show_source: false
      members:
        - to_dict
        - from_dict

### LLMTokenUsage

::: causaliq_knowledge.llm.cache.LLMTokenUsage
    options:
      show_root_heading: true
      show_source: false
