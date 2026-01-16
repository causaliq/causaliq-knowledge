# LLM (and other) Caching

## Rationale

LLM requests and responses will be cached for the following purposes:

- **reducing costs** - unnecessary LLM requests are avoided;
- **reducing latency** - requests from cache can be serviced much more quickly than querying an external LLM API;
- **functional testing** - cached responses provide a mechanism for deterministic and fast functional testing both locally and in GitHub CI testing;
- **reproducibility** - of published results avoiding the issues of the non-deterministic nature of LLMs and their rapid evolution;
- **LLM analysis** - these caches provide a rich resource characterising the comparative behaviour of different models, prompt strategies etc.

In fact, these are generic requirements that may apply to other caching requirements, including, perhaps:

- structure learning results such as learned graphs, debug traces and experiment metadata - the conservative principle in CausalIQ workflows means that experiments are generally only performed if the results for that experiment are not present on disk
- node scores during structure learning, although these are generally held in memory

This argues for a generic caching capability, that would ultimately be provided by causaliq-core (though we may develop it first in causaliq-knowledge, as this is where the first use case arises, and then migrate it).

## Design Principles

- **Opt-out caching** - caching is enabled by default; users explicitly disable it when not wanted - for instance, if non-cached experiment timings are required
- **Flexible organisation** - defined by the application/user
  - each application defines a key which is SHA hashed with the truncated is hash used to locate the resource within a cache
    - for LLM queries this may be the model, messages, temperature and max tokens
    - for structure learning results, this could be the sample size, randomisation seed and resource type
    - for node scores, this could be the node and parents
  - each application defines the scope of each physical cache to optimise resource re-use
    - for LLM queries this is likely defined by the dataset - in this case, the cache therefore represents the "LLM learning about a particular dataset"
    - for structure learning results, this could be the experiment series id and network
    - for node scores, this could be the dataset, sample size, score type and score parameters
  - each application defines the data held for the cached resource
    - for LLM queries this includes the complete response, and rich metadata such as token counts, latency, cost, and timestamps to support LLM analysis
    - for structure learning results this includes the learned graph, experiment metadata (execution time, graph scores, hyperparameters etc) and optionally, the structure learning iteration trace
    - for node scores this is simply the score float value
- **Simple cache semantics** - on request: check cache → if hit, return cached response → if miss, make request, cache result, return response
- **Efficient storage** - much of this cached data has lots of repeated "tokens" - JSON and XML tags, variable names, LLM query and response words - storing it verbatim would be very inefficient
  - resources should be stored internally in a compressed format, with a "dictionary" used to compress and expand numeric tokens to tags, words etc (i.e. like ZIP?)
- **Fast access** - to resources
  - resource existence checking and retrieval should be very fast.
  - resource addition and update, and sequential scanning of all resources (for analysis) should be reasonably fast
- **Import/Export** - functions should be provided to import/export all or selected parts of a cache keys and resources to standards-based human-readable files:
  - this is useful for functional testing, for example, where test files are human readable, and then a fixture imports the test files to the cache bfore testing starts
  - for LLM queries the queries, response and metadata would be exported as JSON files
  - for experiment results the learned graphs would be exported as GraphML files, the metadata as JSON and the debug trace in CSV format
- **Centralised usage** - access to cache data should be centralised in the application code
  - for LLM queries caching logic implemented once in `BaseLLMClient`, available to all providers
  - for experiment results centralised in core CausalIQ WOrkflow code?
  - for node scores this might be called from the node_score function
- **Persistent and in-memory** - the application can opt to use the cache in-memory or from disk
- **Concurrency** - the cache manages multiple processes or threads retrievng and updating it

### Resolving the Format Tension

There is a tension between some of these principles, in particular, space/performance/concurrency and standards-based open formats. This tension is mitigated because these requirements arise at **different phases of the project lifecycle**:

| Phase | Primary Need | Format |
|-------|--------------|--------|
| **Test setup** | Human-readable, editable fixtures | JSON, GraphML |
| **Production runs** | Speed, concurrency, disk efficiency | SQLite + encoded blobs |
| **Result archival** (Zenodo) | Standards-based, long-term accessible | JSON, GraphML, CSV |

This lifecycle perspective leads to a clear resolution:

- **Compact, efficient formats are the norm** during active experimentation
- **Import** converts open formats → internal cache (test fixture setup)
- **Export** converts internal cache → open formats (archival, sharing)

The import/export capability is therefore not just a convenience feature - it is **essential architecture** that bridges incompatible requirements at different lifecycle stages.

## Existing Packages

Evaluation of existing packages against our requirements:

### Comparison Summary

| Requirement | zip | SQLite | diskcache | pickle |
|-------------|-----|--------|-----------|--------|
| Fast key lookup | ⚠️ Need index | ✅ Indexed | ✅ Indexed | ❌ Load all |
| Concurrency | ❌ Poor | ✅ Built-in | ✅ Built-in | ❌ None |
| Efficient storage | ✅ Compressed | ⚠️ Per-blob only | ❌ None | ⚠️ Reasonable |
| Cross-entry compression | ✅ Shared dict | ❌ No | ❌ No | ❌ No |
| In-memory mode | ❌ File-based | ✅ `:memory:` | ❌ Disk-based | ✅ Native |
| Incremental updates | ❌ Rewrite | ✅ Per-entry | ✅ Per-entry | ❌ Rewrite |
| Import/Export | ✅ File extract | ⚠️ Via SQL | ❌ Python-only | ❌ Python-only |
| Standards-based | ✅ Universal | ✅ Universal | ❌ Python-only | ❌ Python-only |
| Security | ✅ Safe | ✅ Safe | ✅ Safe | ❌ Code execution risk |

### zip

**Pros:**
- Excellent compression with shared dictionary across all files
- Universal standard format
- Built-in Python support (`zipfile` module)
- Could use hash as filename within archive

**Cons:**
- No true in-memory mode (must write to file or BytesIO)
- Poor incremental update support - must rewrite archive to add/modify entries
- No built-in concurrency protection
- Random access requires scanning central directory

**Verdict:** Good for archival/export, poor for active cache with frequent updates.

### SQLite

**Pros:**
- Excellent indexed key lookup - O(1) access
- Built-in concurrency with locking
- In-memory mode via `:memory:`
- Per-entry insert/update without rewriting
- Universal format, excellent tooling
- SQL enables rich querying for analysis

**Cons:**
- No cross-entry compression (can compress individual blobs, but no shared dictionary)
- Compression must be handled at application level

**Verdict:** Excellent for cache infrastructure. Compression concern addressed by our pluggable encoder + token dictionary approach.

### diskcache

**Pros:**
- Simple key-value API
- Built-in concurrency (uses SQLite underneath)
- Automatic eviction policies (LRU, etc.)
- Good performance for typical caching scenarios

**Cons:**
- No compression
- Python-specific - not portable to other languages/tools
- No cross-entry compression or shared dictionaries
- Limited querying capability - just key-value lookup
- Export would require custom code

**Verdict:** Convenient for simple caching but lacks compression and portability requirements.

### pickle

**Pros:**
- Native Python serialisation - handles arbitrary objects
- Simple API
- In-memory use is natural (just Python dicts)

**Cons:**
- **Security risk** - pickle can execute arbitrary code on load
- No concurrency protection - corruption risk with multiple writers
- Single-file approach requires loading entire cache into memory
- Directory-of-files approach has same scaling issues as JSON
- No cross-entry compression
- Python-specific - cannot share caches with other tools
- Not human-readable

**Verdict:** Unsuitable for production caches. Security risk alone disqualifies it. Only appropriate for small, single-process, trusted, ephemeral use cases.

### Recommendation

**SQLite + pluggable encoders with shared token dictionary** combines:
- SQLite's strengths: concurrency, fast lookup, in-memory mode, incremental updates
- Custom compression: pluggable encoders achieve domain-specific compression, shared token dictionary provides cross-entry compression benefits
- Standards compliance: SQLite is universal, export functions produce JSON/GraphML

## LLM Query/Response Caching

This section details the specific caching requirements for LLM queries and responses.

### Cache Key Construction

The cache key is a SHA-256 hash (truncated to 16 hex characters) of the following request parameters that affect the response:

| Field | Type | Rationale |
|-------|------|-----------|
| `model` | string | Different models produce different responses |
| `messages` | list[dict] | The full conversation including system prompt and user query |
| `temperature` | float | Affects response variability (0.0 recommended for reproducibility) |
| `max_tokens` | int | May truncate response |

**Not included in key:**
- `api_key` - authentication, doesn't affect response content
- `timeout` - client-side concern
- `stream` - delivery method, same content

**Example cache key input:**
```json
{
  "model": "gpt-4o",
  "messages": [
    {"role": "system", "content": "You are a causal inference expert..."},
    {"role": "user", "content": "What does variable 'BMI' typically represent in health studies?"}
  ],
  "temperature": 0.0,
  "max_tokens": 1000
}
```

**Resulting hash:** `a3f7b2c1e9d4f8a2`

### Cached Data Structure

Each cache entry stores both the response and rich metadata for analysis:

```json
{
  "cache_key": {
    "model": "gpt-4o",
    "messages": [...],
    "temperature": 0.0,
    "max_tokens": 1000
  },
  "response": {
    "content": "BMI (Body Mass Index) is a measure of body fat based on height and weight...",
    "finish_reason": "stop",
    "model_version": "gpt-4o-2024-08-06"
  },
  "metadata": {
    "provider": "openai",
    "timestamp": "2026-01-11T10:15:32.456Z",
    "latency_ms": 1847,
    "tokens": {
      "input": 142,
      "output": 203,
      "total": 345
    },
    "cost_usd": 0.00518,
    "cache_hit": false
  }
}
```

### Field Descriptions

**Response fields:**

| Field | Description |
|-------|-------------|
| `content` | The full text response from the LLM |
| `finish_reason` | Why generation stopped: `stop`, `length`, `content_filter` |
| `model_version` | Actual model version used (may differ from requested) |

**Metadata fields:**

| Field | Description | Use Case |
|-------|-------------|----------|
| `provider` | LLM provider name (openai, anthropic, etc.) | Analysis across providers |
| `timestamp` | When the original request was made | Tracking model evolution |
| `latency_ms` | Response time in milliseconds | Performance analysis |
| `tokens.input` | Prompt token count | Cost tracking |
| `tokens.output` | Completion token count | Cost tracking |
| `tokens.total` | Total tokens | Cost tracking |
| `cost_usd` | Estimated cost of the request | Budget monitoring |
| `cache_hit` | Whether this was served from cache | Always `false` when first stored |

### Cache Scope

LLM caches are typically **dataset-scoped**:

```
datasets/
  health_study/
    llm_cache.db          # All LLM queries about this dataset
  economic_data/
    llm_cache.db          # Separate cache for different dataset
```

This organisation means:

- Queries like "What does BMI mean?" are shared across all experiments on that dataset
- Different experiment series (algorithms, parameters) reuse the same cached LLM knowledge
- The cache represents "what the LLM has learned about this dataset"

### Integration with BaseLLMClient

```python
class BaseLLMClient:
    def __init__(self, cache: TokenCache | None = None, use_cache: bool = True):
        self.cache = cache
        self.use_cache = use_cache
    
    def query(self, messages: list[dict], **kwargs) -> LLMResponse:
        if self.use_cache and self.cache:
            cache_key = self._build_cache_key(messages, kwargs)
            cached = self.cache.get(cache_key, entry_type='llm')
            if cached:
                return LLMResponse.from_cache(cached)
        
        # Make actual API call
        start = time.perf_counter()
        response = self._call_api(messages, **kwargs)
        latency_ms = int((time.perf_counter() - start) * 1000)
        
        # Cache the result
        if self.use_cache and self.cache:
            entry = self._build_cache_entry(messages, kwargs, response, latency_ms)
            self.cache.put(cache_key, entry_type='llm', data=entry)
        
        return response
    
    def _build_cache_key(self, messages: list[dict], kwargs: dict) -> str:
        key_data = {
            'model': self.model,
            'messages': messages,
            'temperature': kwargs.get('temperature', 0.0),
            'max_tokens': kwargs.get('max_tokens', 1000),
        }
        key_json = json.dumps(key_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(key_json.encode()).hexdigest()[:16]
```

## Proposed Solution: SQLite + Pluggable Encoders

Given the requirements, we propose SQLite for storage/concurrency combined with **pluggable type-specific encoders** for compression. Generic tokenisation provides a good default, but domain-specific encoders achieve far better compression for well-defined structures.

### Why Pluggable Encoders?

Generic tokenisation is a lowest common denominator - it works for everything but optimises nothing. Consider learned graphs:

| Approach | Per Edge | 10,000 edges |
|----------|----------|--------------|
| GraphML (verbatim) | ~80-150 bytes | 800KB - 1.5MB |
| GraphML (tokenised) | ~30-50 bytes | 300-500KB |
| Domain-specific binary | 4 bytes | **40KB** |

Domain-specific encoding achieves **10-30x better compression** for structured data.

### SQLite Schema

```sql
-- Token dictionary (grows dynamically, shared across encoders)
CREATE TABLE tokens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- uint16 range (max 65,535)
    token TEXT UNIQUE NOT NULL,
    frequency INTEGER DEFAULT 1
);

-- Generic cache entries
CREATE TABLE cache_entries (
    hash TEXT PRIMARY KEY,
    entry_type TEXT NOT NULL,              -- 'llm', 'graph', 'score', etc.
    data BLOB NOT NULL,                    -- encoded by type-specific encoder
    created_at TEXT NOT NULL,
    metadata BLOB                          -- tokenised metadata (optional)
);

CREATE INDEX idx_entry_type ON cache_entries(entry_type);
CREATE INDEX idx_created_at ON cache_entries(created_at);
```

### Encoder Architecture

```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

class EntryEncoder(ABC):
    """Type-specific encoder/decoder for cache entries."""
    
    @abstractmethod
    def encode(self, data: Any, token_cache: 'TokenCache') -> bytes:
        """Encode data to bytes, optionally using shared token dictionary."""
        ...
    
    @abstractmethod
    def decode(self, blob: bytes, token_cache: 'TokenCache') -> Any:
        """Decode bytes back to original data structure."""
        ...
    
    @abstractmethod
    def export(self, data: Any, path: Path) -> None:
        """Export to human-readable format (GraphML, JSON, etc.)."""
        ...
    
    @abstractmethod
    def import_(self, path: Path) -> Any:
        """Import from human-readable format."""
        ...
```

### Concrete Encoders

#### GraphEncoder - Domain-Specific Binary

Achieves ~4 bytes per edge through bit-packing:

```python
class GraphEncoder(EntryEncoder):
    """Compact binary encoding for learned graphs.
    
    Format:
    - Header: node_count (uint16)
    - Node table: [node_name_token_id (uint16), ...] - maps index → name
    - Edge list: [packed_edge (uint32), ...] where:
        - bits 0-13:  source node index (up to 16,384 nodes)
        - bits 14-27: target node index
        - bits 28-31: edge type (4 bits for endpoint marks: -, >, o)
    """
    
    def encode(self, graph: nx.DiGraph, token_cache: 'TokenCache') -> bytes:
        nodes = list(graph.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        
        # Header
        data = struct.pack('H', len(nodes))
        
        # Node table - uses shared token dictionary for node names
        for node in nodes:
            token_id = token_cache.get_or_create_token(str(node))
            data += struct.pack('H', token_id)
        
        # Edge list - packed binary
        for src, tgt, attrs in graph.edges(data=True):
            edge_type = self._encode_edge_type(attrs)
            packed = (node_to_idx[src] | 
                     (node_to_idx[tgt] << 14) | 
                     (edge_type << 28))
            data += struct.pack('I', packed)
        
        return data
    
    def export(self, graph: nx.DiGraph, path: Path) -> None:
        """Export to standard GraphML for human readability."""
        nx.write_graphml(graph, path)
    
    def import_(self, path: Path) -> nx.DiGraph:
        """Import from GraphML."""
        return nx.read_graphml(path)
```

#### LLMEntryEncoder - Generic Tokenisation

Good default for text-heavy, variably-structured data:

```python
class LLMEntryEncoder(EntryEncoder):
    """Tokenised encoding for LLM cache entries.
    
    Uses shared token dictionary for JSON structure and text content.
    Numbers stored as literals. Achieves 50-70% compression on typical entries.
    """
    
    TOKEN_REF = 0x00
    LITERAL_INT = 0x01
    LITERAL_FLOAT = 0x02
    
    def encode(self, entry: dict, token_cache: 'TokenCache') -> bytes:
        # Tokenise JSON keys, structural chars, string words
        # Store integers and floats as literals
        ...
    
    def decode(self, blob: bytes, token_cache: 'TokenCache') -> dict:
        # Reconstruct JSON from token IDs and literals
        ...
    
    def export(self, entry: dict, path: Path) -> None:
        path.write_text(json.dumps(entry, indent=2))
    
    def import_(self, path: Path) -> dict:
        return json.loads(path.read_text())
```

#### ScoreEncoder - Minimal Encoding

For high-volume, simple data:

```python
class ScoreEncoder(EntryEncoder):
    """Compact encoding for node scores - just 8 bytes per entry."""
    
    def encode(self, score: float, token_cache: 'TokenCache') -> bytes:
        return struct.pack('d', score)  # 8-byte double
    
    def decode(self, blob: bytes, token_cache: 'TokenCache') -> float:
        return struct.unpack('d', blob)[0]
    
    def export(self, score: float, path: Path) -> None:
        path.write_text(str(score))
    
    def import_(self, path: Path) -> float:
        return float(path.read_text())
```

### Compression Comparison

| Entry Type | Encoder | Typical Size | vs Verbatim |
|------------|---------|--------------|-------------|
| LLM response | LLMEntryEncoder | 500 bytes | 50-70% smaller |
| Learned graph (1000 edges) | GraphEncoder | 4KB | **95% smaller** |
| Node score | ScoreEncoder | 8 bytes | 50% smaller |
| Experiment metadata | LLMEntryEncoder | 200 bytes | 60% smaller |

### TokenCache with Pluggable Encoders

```python
class TokenCache:
    def __init__(self, db_path: str, encoders: dict[str, EntryEncoder] = None):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()
        self._load_token_dict()
        
        # Register encoders - defaults provided, can be overridden
        self.encoders = encoders or {
            'llm': LLMEntryEncoder(),
            'graph': GraphEncoder(),
            'score': ScoreEncoder(),
        }
    
    def _load_token_dict(self):
        """Load token dictionary into memory for fast lookup."""
        cursor = self.conn.execute("SELECT token, id FROM tokens")
        self.token_to_id = {row[0]: row[1] for row in cursor}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
    
    def get_or_create_token(self, token: str) -> int:
        """Get token ID, creating new entry if needed. Used by encoders."""
        if token in self.token_to_id:
            return self.token_to_id[token]
        cursor = self.conn.execute(
            "INSERT INTO tokens (token) VALUES (?) RETURNING id", 
            (token,)
        )
        token_id = cursor.fetchone()[0]
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        return token_id
    
    def put(self, hash: str, entry_type: str, data: Any, metadata: dict = None):
        """Store entry using appropriate encoder."""
        encoder = self.encoders[entry_type]
        blob = encoder.encode(data, self)
        meta_blob = self.encoders['llm'].encode(metadata, self) if metadata else None
        self.conn.execute(
            "INSERT OR REPLACE INTO cache_entries VALUES (?, ?, ?, ?, ?)",
            (hash, entry_type, blob, datetime.utcnow().isoformat(), meta_blob)
        )
        self.conn.commit()
    
    def get(self, hash: str, entry_type: str) -> Any | None:
        """Retrieve and decode entry."""
        cursor = self.conn.execute(
            "SELECT data FROM cache_entries WHERE hash = ? AND entry_type = ?",
            (hash, entry_type)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        encoder = self.encoders[entry_type]
        return encoder.decode(row[0], self)
    
    def register_encoder(self, entry_type: str, encoder: EntryEncoder):
        """Register custom encoder for new entry types."""
        self.encoders[entry_type] = encoder
```

### Tokenisation Details (LLMEntryEncoder)

| Element | Tokenise? | Rationale |
|---------|-----------|-----------|
| JSON structural chars (`{`, `}`, `[`, `]`, `:`, `,`) | Yes | Very frequent, 1 char → 2 bytes is fine given volume |
| JSON keys (`"role"`, `"content"`) | Yes | Highly repetitive |
| String values (words) | Yes | Variable names, common words repeat often |
| Integers | No | Store as literal bytes (type marker + value) |
| Floats | No | Store as literal bytes (type marker + 8-byte double) |
| Booleans | Yes | Just two tokens: `true`, `false` |
| Null | Yes | Single token |

#### Token Boundary Rules

Tokenisation splits on:

1. **JSON structural characters** - each is its own token
2. **Whitespace** - discarded (JSON can be reconstructed without it)
3. **Within strings** - split on whitespace and punctuation boundaries

Example:
```
"BMI represents Body Mass Index" 
→ [", BMI, represents, Body, Mass, Index, "]
→ 7 tokens
```

#### Encoding Format

The data blob contains a sequence of **typed values**:

| Type Marker (uint8) | Followed By | Description |
|---------------------|-------------|-------------|
| 0x00 | uint16 | Token ID reference |
| 0x01 | int64 (8 bytes) | Literal integer |
| 0x02 | float64 (8 bytes) | Literal float |

Example encoding:
```
{"score": 3.14159, "node": "BMI"}

Tokens: { " score " : " node " : " BMI " }
         ↓   ↓     ↓ ↓   ↓    ↓ ↓   ↓   ↓
        T1  T2    T3 T4  T2   T5 T4  T2  T6  T2  T1→}

Encoded: [0x00,T1, 0x00,T2, 0x00,T3, 0x00,T2, 0x00,T4, 
          0x02,<8-byte float>,
          0x00,T4, 0x00,T2, 0x00,T5, 0x00,T2, 0x00,T4, 
          0x00,T2, 0x00,T6, 0x00,T2, 0x00,T7]
```

### In-Memory Mode

For node scores during structure learning (high-frequency, session-scoped):

```python
# In-memory SQLite - fast, no disk I/O
cache = TokenCache(":memory:")

# Can optionally persist at end of session
cache.save_to_disk("node_scores.db")
```

### Import/Export

Import/export delegates to the appropriate encoder:

```python
class TokenCache:
    def export_entries(self, output_dir: Path, entry_type: str, format: str = None):
        """Export cache entries to human-readable files."""
        encoder = self.encoders[entry_type]
        query = "SELECT hash, data FROM cache_entries WHERE entry_type = ?"
        for hash, blob in self.conn.execute(query, (entry_type,)):
            data = encoder.decode(blob, self)
            ext = format or encoder.default_export_format
            encoder.export(data, output_dir / f"{hash}.{ext}")
    
    def import_entries(self, input_dir: Path, entry_type: str):
        """Import human-readable files into cache."""
        encoder = self.encoders[entry_type]
        for file in input_dir.iterdir():
            if file.is_file():
                data = encoder.import_(file)
                hash = file.stem
                self.put(hash, entry_type, data)
```

### Cache Organisation

The cache is scoped by its physical location (database file):

| Context | Cache Location | Typical Size |
|---------|----------------|--------------|
| Dataset LLM queries | `{dataset_dir}/llm_cache.db` | 10K-100K entries |
| Experiment series | `{series_dir}/results.db` | 100K-10M entries |
| Functional tests | `tests/data/cache.db` | 100-1K entries |
| Node scores (session) | `:memory:` | 1M+ entries |

### Collision Handling

With truncated SHA-256 hashes, collisions are possible. On cache read:

1. Find entry by hash
2. Decode and verify original key matches request exactly
3. If mismatch, treat as cache miss

### CLI Tooling

```bash
# Export to human-readable format (uses encoder's export method)
causaliq cache export llm_cache.db --type llm --output cache_dump/
causaliq cache export results.db --type graph --format graphml --output graphs/

# Import from human-readable format  
causaliq cache import cache_dump/ --into llm_cache.db --type llm

# Statistics
causaliq cache stats results.db
# Output: 
#   Entries: 52,341 (llm: 12,341, graph: 40,000)
#   Tokens: 4,892
#   Size: 45.2 MB (estimated uncompressed: 142.8 MB)

# Query/inspect
causaliq cache query results.db --type llm --where "model='gpt-4o'" --limit 10
```

## Implementation Plan

The caching system will be built incrementally across the following commits. Feature branch: `feature/llm-caching`

### Module Structure

The implementation separates **core infrastructure** (future migration to `causaliq-core`) from **LLM-specific code** (remains in `causaliq-knowledge`):

```
src/causaliq_knowledge/
  cache/                          # CORE - will migrate to causaliq-core
    __init__.py
    token_cache.py                # TokenCache class, SQLite schema
    encoders/
      __init__.py
      base.py                     # EntryEncoder ABC
      json_encoder.py             # Generic JSON tokenisation encoder
  llm/
    cache.py                      # LLM-SPECIFIC - stays in causaliq-knowledge
                                  # LLMEntryEncoder, cache key building, metadata

tests/
  unit/
    cache/                        # CORE tests - migrate with core
      test_token_cache.py
      test_json_encoder.py
    llm/
      test_llm_cache.py           # LLM-SPECIFIC tests - stay here
  integration/
    test_llm_caching.py           # LLM-SPECIFIC integration tests
```

This separation ensures:
- Core cache infrastructure can be extracted to `causaliq-core` with its tests
- LLM-specific encoder and integration remain in `causaliq-knowledge`
- Clean dependency: `causaliq-knowledge` depends on `causaliq-core`, not vice versa

### Phase 1: Core Infrastructure *(migrates to causaliq-core)*

| # | Commit | Description |
|---|--------|-------------|
| 1 | Add cache module skeleton | Create `src/causaliq_knowledge/cache/` with `__init__.py`, empty module structure |
| 2 | Implement SQLite schema and TokenCache base | `TokenCache` class with `__init__`, `_init_schema()`, connection management, in-memory support |
| 3 | Add token dictionary management | `get_or_create_token()`, `_load_token_dict()`, in-memory token lookup |
| 4 | Add basic get/put operations | `put()`, `get()`, `exists()` methods (without encoding - raw blob storage) |
| 5 | Add unit tests for core TokenCache | `tests/unit/cache/test_token_cache.py` - schema, tokens, CRUD, in-memory |

### Phase 2: Encoder Architecture *(core migrates, LLM-specific stays)*

| # | Commit | Description |
|---|--------|-------------|
| 6 | Define EntryEncoder ABC | `cache/encoders/base.py` - ABC with `encode()`, `decode()`, `export()`, `import_()` *(core)* |
| 7 | Implement generic JSON encoder | `cache/encoders/json_encoder.py` - tokenisation, literals *(core)* |
| 8 | Integrate encoders with TokenCache | `register_encoder()`, encoder dispatch in `get()`/`put()` *(core)* |
| 9 | Add unit tests for encoders | `tests/unit/cache/test_json_encoder.py` *(core)* |
| 10 | Implement LLMEntryEncoder | `llm/cache.py` - extends JSON encoder with LLM-specific structure *(LLM-specific)* |
| 11 | Add unit tests for LLMEntryEncoder | `tests/unit/llm/test_llm_cache.py` *(LLM-specific)* |

### Phase 3: Import/Export *(core)*

| # | Commit | Description |
|---|--------|-------------|
| 12 | Add export functionality | `export_entries()` method in TokenCache |
| 13 | Add import functionality | `import_entries()` method in TokenCache |
| 14 | Add import/export tests | Round-trip tests in `tests/unit/cache/` |

### Phase 4: LLM Client Integration *(LLM-specific)*

| # | Commit | Description |
|---|--------|-------------|
| 15 | Add cache_key building to BaseLLMClient | `_build_cache_key()` method with SHA-256 hashing |
| 16 | Integrate caching into query flow | Cache lookup before API call, cache storage after |
| 17 | Add metadata capture | Latency, token counts, cost estimation, timestamps |
| 18 | Add LLM caching integration tests | `tests/integration/test_llm_caching.py` - mock API, cache hit/miss |
| 19 | Update functional tests to use cache import | Fixture imports test data, tests run from cache |

### Phase 5: CLI Tooling *(core commands, LLM-specific options)*

| # | Commit | Description |
|---|--------|-------------|
| 20 | Add cache CLI subcommand | `causaliq cache` command group *(core)* |
| 21 | Add cache export command | `causaliq cache export` with type/format options |
| 22 | Add cache import command | `causaliq cache import` |
| 23 | Add cache stats command | Entry counts, token count, size estimation |

### Phase 6: Documentation & Polish

| # | Commit | Description |
|---|--------|-------------|
| 24 | Add cache API documentation | Docstrings, mkdocs pages for cache module |
| 25 | Update architecture docs | Mark caching design as implemented, add usage examples |

### Dependency Graph

```
Phase 1 (core):     1 → 2 → 3 → 4 → 5
                                  ↓
Phase 2 (core):               6 → 7 → 8 → 9
                                        ↓
Phase 2 (llm):                    10 → 11
                                        ↓
Phase 3 (core):               12 → 13 → 14
                                        ↓
Phase 4 (llm):          15 → 16 → 17 → 18 → 19
                                            ↓
Phase 5:                      20 → 21 → 22 → 23
                                            ↓
Phase 6:                                24 → 25
```

### Milestones

- **After Phase 1** (commits 1-5): Working cache with raw blob storage, testable standalone
- **After Phase 2** (commits 6-11): Compression working, LLM encoder ready
- **After Phase 4** (commits 15-19): Cache fully usable for LLM queries
- **After Phase 5** (commits 20-23): CLI tools available for cache management

### Future: Migration to causaliq-core

When ready to migrate core infrastructure:

1. Move `cache/` directory to `causaliq-core`
2. Move `tests/unit/cache/` to `causaliq-core`
3. Update imports in `causaliq-knowledge` to use `causaliq_core.cache`
4. `LLMEntryEncoder` and LLM integration remain in `causaliq-knowledge`