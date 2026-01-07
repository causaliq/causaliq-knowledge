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

## CLI Entry Point

::: causaliq_knowledge.cli
    options:
        show_root_heading: true
        show_source: false
        heading_level: 3