# CausalIQ Knowledge CLI

The command-line interface provides a quick way to test LLM queries about
causal relationships.

## Installation

The CLI is automatically installed when you install the package:

```bash
pip install causaliq-knowledge
```

## Usage

### Basic Query

```bash
# Query using default model (Groq)
cqknow query smoking lung_cancer

# With domain context
cqknow query smoking lung_cancer --domain medicine
```

### Multiple Models

```bash
# Query multiple models for consensus
cqknow query X Y --model groq/llama-3.1-8b-instant --model gemini/gemini-2.5-flash
```

### JSON Output

```bash
# Get structured JSON output
cqknow query smoking lung_cancer --json
```

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--model` | `-m` | LLM model to query (can be repeated) |
| `--domain` | `-d` | Domain context (e.g., "medicine") |
| `--strategy` | `-s` | Consensus strategy: weighted_vote or highest_confidence |
| `--json` | | Output as JSON |
| `--temperature` | `-t` | LLM temperature (0.0-1.0) |

## Cache Management

The `cache` command group provides tools for inspecting and managing the LLM response cache.

### Cache Stats

View statistics about a cache database:

```bash
# Show cache statistics
cqknow cache stats ./llm_cache.db

# Output:
# Cache: ./llm_cache.db
# ========================================
# Entries:  42
# Tokens:   15,230
```

### Cache Export

Export cache entries to human-readable JSON files:

```bash
# Export all entries to a directory
cqknow cache export ./llm_cache.db ./export_dir

# Export to a zip archive (auto-detected from .zip extension)
cqknow cache export ./llm_cache.db ./export.zip
```

Files are named using a human-readable format:
```
gpt4_smoking_lung_edge_a1b2.json
gemini25_exercise_health_edge_c3d4.json
```

### JSON Output

```bash
# Get stats as JSON for scripting
cqknow cache stats ./llm_cache.db --json

# Get export result as JSON
cqknow cache export ./llm_cache.db ./export_dir --json
```

### Cache Command Options

| Command | Description |
|---------|-------------|
| `cache stats <path>` | Show entry and token counts |
| `cache stats <path> --json` | Output stats as JSON |
| `cache export <path> <dir>` | Export entries to directory |
| `cache export <path> <file.zip>` | Export entries to zip archive |
| `cache export <path> <output> --json` | Output export result as JSON |
| `cache import <cache> <input>` | Import entries from directory or zip |
| `cache import <cache> <input> --json` | Output import result as JSON |

### Cache Import

Import cache entries from JSON files:

```bash
# Import from a directory
cqknow cache import ./llm_cache.db ./import_dir

# Import from a zip archive (auto-detected from .zip extension)
cqknow cache import ./llm_cache.db ./export.zip

# Get import result as JSON
cqknow cache import ./llm_cache.db ./import_dir --json
```

Entry types are auto-detected from JSON structure:
- **LLM entries**: JSON containing `cache_key.model`, `cache_key.messages`, and `response`
- **Generic JSON**: Any other valid JSON file

This enables round-trip operations: export from one cache, import into another.

## CLI Entry Point

::: causaliq_knowledge.cli
    options:
        show_root_heading: true
        show_source: false
        heading_level: 3