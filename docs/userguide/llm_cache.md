# LLM Cache Management

CausalIQ Knowledge includes a caching system that stores LLM API responses to
avoid redundant API calls and reduce costs. These are **CLI-only** utility
commands and are not available as workflow actions.

## Why Use Caching?

LLM API calls are:

- **Expensive** — Token costs add up quickly during experiments
- **Slow** — Network latency and model inference take time
- **Rate-limited** — Providers impose request limits

The LLM cache stores request/response pairs keyed by model, prompt, and
parameters. Re-running experiments with the same parameters returns cached
responses instantly at zero cost.

---

## Cache Statistics

View statistics about a cache database including entry counts, token usage,
costs, and breakdown by model.

### Usage

```bash
cqknow cache-stats -i <cache.db>
```

### Parameters

| Parameter | CLI | Description |
|-----------|-----|-------------|
| `input` | `-i` | Path to the LLM cache database |

### Example

```bash
cqknow cache-stats -i ./llm_cache.db
```

**Output:**

```
Cache: ./llm_cache.db
============================================================
Entries:          42
Token dictionary: 1,523
Total cache hits: 128

Total cost:       $0.0234
Est. savings:     $0.0714
Total tokens:     45,231 in / 12,456 out

Model                             Entries      Hits  Hit Rate    Tokens In   Tokens Out        Cost     Latency
------------------------------------------------------------------------------------------------------------------
groq/llama-3.1-8b-instant              28        95    77.2%        32,145        8,234     $0.0123      245 ms
gemini/gemini-2.5-flash                14        33    70.2%        13,086        4,222     $0.0111      512 ms
```

---

## Exporting Cache Entries

Export cached LLM responses to human-readable JSON files for backup, sharing,
or inspection.

### Usage

```bash
cqknow export-cache -i <cache.db> -o <output>
```

### Parameters

| Parameter | CLI | Required | Description |
|-----------|-----|----------|-------------|
| `input` | `-i` | Yes | Path to the LLM cache database |
| `output` | `-o` | Yes | Output directory or `.zip` file path |
| `json` | `--json` | No | Output result as JSON |

### Export to Directory

```bash
cqknow export-cache -i ./llm_cache.db -o ./export_dir
```

Creates individual JSON files for each cache entry:

```
export_dir/
├── cli_2026-01-29-143052_groq.json
├── cli_2026-01-29-143055_groq.json
├── workflow_2026-01-30-091234_gemini.json
└── ...
```

### Export to ZIP Archive

```bash
cqknow export-cache -i ./llm_cache.db -o ./export.zip
```

Creates a compressed archive for easy sharing or backup.

### File Naming Format

Exported files use the naming format:

```
{id}_{timestamp}_{provider}.json
```

- **id** — Identifier from the generation request (default: "cli")
- **timestamp** — When the request was made
- **provider** — LLM provider (groq, gemini, etc.)

### JSON Output

```bash
cqknow export-cache -i ./llm_cache.db -o ./export_dir --json
```

```json
{
  "input_path": "./llm_cache.db",
  "output_path": "./export_dir",
  "format": "directory",
  "exported": 42
}
```

---

## Importing Cache Entries

Import previously exported LLM responses back into a cache database. Useful for
restoring backups or sharing cached responses between machines.

### Usage

```bash
cqknow import-cache -i <input> -o <cache.db>
```

### Parameters

| Parameter | CLI | Required | Description |
|-----------|-----|----------|-------------|
| `input` | `-i` | Yes | Directory or `.zip` file containing JSON files |
| `output` | `-o` | Yes | Path to cache database (created if needed) |
| `json` | `--json` | No | Output result as JSON |

### Import from Directory

```bash
cqknow import-cache -i ./export_dir -o ./new_cache.db
```

### Import from ZIP Archive

```bash
cqknow import-cache -i ./export.zip -o ./new_cache.db
```

### Entry Type Filtering

Only LLM cache entries (with `cache_key.model`, `cache_key.messages`, and
`response` fields) are imported. Other entry types are automatically skipped.

### JSON Output

```bash
cqknow import-cache -i ./export_dir -o ./new_cache.db --json
```

```json
{
  "input_path": "./export_dir",
  "output_path": "./new_cache.db",
  "format": "directory",
  "imported": 42,
  "skipped": 3
}
```

---

## Typical Workflows

### Backup and Restore

```bash
# Before major changes, backup the cache
cqknow export-cache -i ./llm_cache.db -o ./backup.zip

# If needed, restore from backup
cqknow import-cache -i ./backup.zip -o ./llm_cache.db
```

### Share Cache Between Machines

```bash
# On machine A: export the cache
cqknow export-cache -i ./llm_cache.db -o ./shared.zip

# Transfer shared.zip to machine B

# On machine B: import the cache
cqknow import-cache -i ./shared.zip -o ./llm_cache.db
```

### Audit API Usage

```bash
# Check total costs and usage
cqknow cache-stats -i ./llm_cache.db

# Export for detailed inspection
cqknow export-cache -i ./llm_cache.db -o ./audit/
```

---

## Cache Storage Details

The LLM cache uses SQLite with token-based compression:

- **Efficient storage** — Common tokens are stored once and referenced
- **Fast lookups** — Hash-based key indexing
- **Hit tracking** — Tracks how many times each entry is reused
- **Metadata rich** — Preserves tokens, cost, latency, and model information

Cache files are portable and can be copied between machines.
