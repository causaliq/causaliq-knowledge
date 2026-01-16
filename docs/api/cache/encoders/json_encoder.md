# JsonEncoder

The `JsonEncoder` provides tokenised encoding for JSON-serialisable data,
achieving 50-70% compression through shared token dictionary usage.

## Overview

JsonEncoder is a concrete implementation of `EntryEncoder` that handles
any JSON-serialisable Python data structure:

- Dictionaries
- Lists
- Strings
- Integers and floats
- Booleans
- None

## Usage

### Direct Encoding

```python
from causaliq_knowledge.cache import TokenCache
from causaliq_knowledge.cache.encoders import JsonEncoder

with TokenCache(":memory:") as cache:
    encoder = JsonEncoder()
    
    # Encode any JSON-serialisable data
    data = {
        "messages": [
            {"role": "user", "content": "What is BMI?"},
        ],
        "temperature": 0.7,
        "max_tokens": 100,
    }
    
    # Encode to compact binary format
    blob = encoder.encode(data, cache)
    
    # Decode back to original structure
    decoded = encoder.decode(blob, cache)
    assert decoded == data
```

### With TokenCache Auto-Encoding

```python
from causaliq_knowledge.cache import TokenCache
from causaliq_knowledge.cache.encoders import JsonEncoder

with TokenCache(":memory:") as cache:
    # Register for auto-encoding
    cache.register_encoder("json", JsonEncoder())
    
    # Store and retrieve with automatic encoding
    cache.put_data("hash1", "json", {"key": "value"})
    data = cache.get_data("hash1", "json")
```

### Export/Import for Human-Readable Files

```python
from pathlib import Path
from causaliq_knowledge.cache.encoders import JsonEncoder

encoder = JsonEncoder()

# Export to JSON file
data = {"messages": [{"role": "user", "content": "Hello"}]}
encoder.export(data, Path("data.json"))

# Import from JSON file
imported = encoder.import_(Path("data.json"))
```

## Encoding Format

The encoder uses three type markers for compact binary representation:

| Marker | Value | Description |
|--------|-------|-------------|
| TOKEN_REF | 0x00 | Token ID reference (uint16) |
| LITERAL_INT | 0x01 | 64-bit signed integer |
| LITERAL_FLOAT | 0x02 | 64-bit double float |

### What Gets Tokenised

| Element | Tokenised | Rationale |
|---------|-----------|-----------|
| JSON structural chars (`{`, `}`, `[`, `]`, `:`, `,`) | Yes | Very frequent |
| String quotes (`"`) | Yes | Frequent |
| String content (words) | Yes | High repetition across entries |
| `null`, `true`, `false` | Yes | Fixed vocabulary |
| Integers | No | Stored as 8-byte literals |
| Floats | No | Stored as 8-byte literals |

### Compression Example

```
Original: {"role": "user", "content": "Hello world"}
Tokens:   { " role " : " user " , " content " : " Hello   world " }
          ↓   ↓     ↓ ↓   ↓    ↓ ↓     ↓      ↓ ↓   ↓       ↓    ↓
          T1  T2   T3 T2 T4   T5 T2 T6  T7    T8 T2 T4     T9   T10 T2 T11

Each token ID: 3 bytes (0x00 marker + uint16)
Typical compression: 50-70% vs raw JSON
```

## Token Reuse

Tokens are shared across all entries in a cache, providing cumulative
compression benefits:

```python
with TokenCache(":memory:") as cache:
    cache.register_encoder("json", JsonEncoder())
    
    # Common terms like "role", "content", "user" are tokenised once
    cache.put_data("h1", "json", {"role": "user", "content": "Hello"})
    cache.put_data("h2", "json", {"role": "assistant", "content": "Hi"})
    cache.put_data("h3", "json", {"role": "user", "content": "Bye"})
    
    # "role", "content", "user" reuse same token IDs across all entries
    print(f"Total tokens: {cache.token_count()}")  # Much less than unique words
```

## API Reference

::: causaliq_knowledge.cache.encoders.JsonEncoder
    options:
      show_root_heading: true
      show_source: false
      members:
        - encode
        - decode
        - export
        - import_
        - default_export_format
