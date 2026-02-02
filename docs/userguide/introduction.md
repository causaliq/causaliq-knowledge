# CausalIQ Knowledge User Guide

## What is CausalIQ Knowledge?

CausalIQ Knowledge is a Python package that provides **knowledge services** for causal discovery workflows. It enables you to query Large Language Models (LLMs) about potential causal relationships between variables, helping to resolve uncertainty in learned causal graphs.

## Primary Use Case

When averaging multiple causal graphs learned from data subsamples, some edges may be **uncertain** - appearing in some graphs but not others, or with inconsistent directions. CausalIQ Knowledge helps resolve this uncertainty by querying LLMs about whether:

1. A causal relationship exists between two variables
2. What the direction of causation is (A→B or B→A)

## Quick Start

### Installation

```bash
pip install causaliq-knowledge
```

### Basic Usage

```python
from causaliq_knowledge.llm import LLMKnowledge

# Initialize with Groq (default, free tier)
knowledge = LLMKnowledge(models=["groq/llama-3.1-8b-instant"])

# Query about a potential edge
result = knowledge.query_edge(
    node_a="smoking",
    node_b="lung_cancer",
    context={"domain": "epidemiology"}
)

print(f"Exists: {result.exists}")
print(f"Direction: {result.direction}")
print(f"Confidence: {result.confidence}")
print(f"Reasoning: {result.reasoning}")
```

### Using Gemini (Free Tier)

```python
# Use Google Gemini for inference
knowledge = LLMKnowledge(models=["gemini/gemini-2.5-flash"])
```

### Multi-Model Consensus

```python
# Query multiple models for more robust answers
knowledge = LLMKnowledge(
    models=["groq/llama-3.1-8b-instant", "gemini/gemini-2.5-flash"],
    consensus_strategy="weighted_vote"
)
```

## LLM Provider Setup

CausalIQ Knowledge uses **direct vendor-specific API clients** (not wrapper
libraries) to communicate with LLM providers. This approach provides
reliability and minimal dependencies. Currently supported:

- **Groq**: Free tier with fast inference
- **Google Gemini**: Generous free tier
- **OpenAI**: GPT-4o and other models
- **Anthropic**: Claude models
- **DeepSeek**: DeepSeek-V3 and R1 models
- **Mistral**: Mistral AI models
- **Ollama**: Local LLMs (free, runs locally)

### Free Options

#### Groq (Free Tier - Very Fast)

Groq offers a generous free tier with extremely fast inference:

1. Sign up at [console.groq.com](https://console.groq.com)
2. Create an API key
3. Set the environment variable (see [Storing API Keys](#storing-api-keys))
4. Use in code:

```python
from causaliq_knowledge.llm import LLMKnowledge

knowledge = LLMKnowledge(models=["groq/llama-3.1-8b-instant"])
result = knowledge.query_edge("smoking", "lung_cancer")
```

Available Groq models: `groq/llama-3.1-8b-instant`, `groq/llama-3.1-70b-versatile`, `groq/mixtral-8x7b-32768`

#### Google Gemini (Free Tier)

Google offers free access to Gemini models:

1. Sign up at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Create an API key
3. Set `GEMINI_API_KEY` environment variable
4. Use in code:

```python
knowledge = LLMKnowledge(models=["gemini/gemini-2.5-flash"])
```

#### OpenAI

OpenAI provides GPT-4o and other models:

1. Sign up at [platform.openai.com](https://platform.openai.com)
2. Create an API key
3. Set `OPENAI_API_KEY` environment variable

```python
knowledge = LLMKnowledge(models=["openai/gpt-4o-mini"])
```

#### Anthropic

Anthropic provides Claude models:

1. Sign up at [console.anthropic.com](https://console.anthropic.com)
2. Create an API key
3. Set `ANTHROPIC_API_KEY` environment variable

```python
knowledge = LLMKnowledge(models=["anthropic/claude-sonnet-4-20250514"])
```

#### DeepSeek

DeepSeek offers high-quality models at competitive prices:

1. Sign up at [platform.deepseek.com](https://platform.deepseek.com)
2. Create an API key
3. Set `DEEPSEEK_API_KEY` environment variable

```python
knowledge = LLMKnowledge(models=["deepseek/deepseek-chat"])
```

#### Mistral

Mistral AI provides models with EU data sovereignty:

1. Sign up at [console.mistral.ai](https://console.mistral.ai)
2. Create an API key
3. Set `MISTRAL_API_KEY` environment variable

```python
knowledge = LLMKnowledge(models=["mistral/mistral-small-latest"])
```

#### Ollama (Local)

Run models locally with Ollama (no API key needed):

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull a model: `ollama pull llama3`
3. Use in code:

```python
knowledge = LLMKnowledge(models=["ollama/llama3"])
```

### Storing API Keys

#### Option 1: User Environment Variables (Recommended)

Set permanently for your user account on Windows:

```powershell
[Environment]::SetEnvironmentVariable("GROQ_API_KEY", "your-key", "User")
[Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "your-key", "User")
```

On Linux/macOS, add to your `~/.bashrc` or `~/.zshrc`:

```bash
export GROQ_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
```

Restart your terminal for changes to take effect.

#### Option 2: Project `.env` File

Create a `.env` file in your project root:

```
GROQ_API_KEY=your-key-here
GEMINI_API_KEY=your-key-here
```

**Important:** Add `.env` to your `.gitignore` so keys aren't committed to version control!

#### Option 3: Password Manager

Store API keys in a password manager (LastPass, 1Password, Bitwarden, etc.) as a Secure Note. This provides:

- Encrypted backup of your keys
- Access from any machine
- Secure sharing with team members if needed

Copy keys from your password manager when setting environment variables.

### Verifying Your Setup

Test your configuration with the CLI:

```bash
# Using the installed CLI
cqknow query smoking lung_cancer --model groq/llama-3.1-8b-instant

# Or with Python module
python -m causaliq_knowledge.cli query smoking lung_cancer --model ollama/llama3
```

## Graph Generation

CausalIQ Knowledge can generate complete causal graphs from model
specifications using LLMs. This is useful for creating prior knowledge
structures or comparing LLM-generated graphs against ground truth.

### Command Overview

```bash
cqknow generate graph -s <model_spec.json> [options]
```

### Basic Examples

```bash
# Generate a graph using default settings (Groq, standard view)
cqknow generate graph -s research/models/cancer/cancer.json

# Use a specific LLM model
cqknow generate graph -s model.json -m gemini/gemini-2.5-flash

# Use rich context level for more detailed prompts
cqknow generate graph -s model.json --prompt-detail rich

# Save output to a JSON file
cqknow generate graph -s model.json -o graph.json
```

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--model-spec` | `-s` | Path to model specification JSON file (required) |
| `--prompt-detail` | `-p` | Context level: `minimal`, `standard`, or `rich` (default: standard) |
| `--llm` | `-m` | LLM model to use (default: groq/llama-3.1-8b-instant) |
| `--output` | `-o` | Output file path (JSON). Prints to stdout if not specified |
| `--format` | `-f` | Output format: `edge_list` or `adjacency_matrix` |
| `--json` | | Output result as JSON to stdout |
| `--id` | | Request identifier for export filenames (default: cli) |
| `--disguise` | `-D` | Enable variable name disguising to reduce LLM memorisation |
| `--seed` | | Random seed for reproducible disguising (requires --disguise) |
| `--use-benchmark-names` | | Use benchmark names instead of LLM names (test memorisation) |
| `--cache/--no-cache` | | Enable/disable LLM response caching (default: enabled) |
| `--cache-path` | `-c` | Path to cache database (default: `<model_spec>_llm.db`) |
| `--temperature` | `-t` | LLM temperature 0.0-1.0 (default: 0.1) |

### Prompt Detail Levels

The `--prompt-detail` option controls how much context is provided to the LLM:

- **minimal**: Variable names only - tests LLM's general knowledge
- **standard**: Names with types, states, and short descriptions
- **rich**: Full context including extended descriptions and sensitivity hints

### Variable Disguising

To reduce LLM memorisation of well-known benchmark networks, you can disguise
variable names:

```bash
# Disguise variable names with random seed for reproducibility
cqknow generate graph -s model.json --disguise --seed 42
```

The disguised names are automatically reversed in the output.

### Output Format

By default, the command prints human-readable output:

```
======================================================================
Generated Causal Graph: cancer
======================================================================
Domain:     epidemiology
Variables:  5
Model:      groq/llama-3.1-8b-instant
======================================================================

Proposed Edges (4):

   1. smoking → lung_cancer  [██████████] 100.0%
   2. pollution → lung_cancer  [████████░░]  80.0%
   ...
```

Use `--json` or `--output` for machine-readable JSON output.

### Caching

Graph generation requests are cached by default. The cache is stored alongside
your model specification file (e.g., `cancer.json` → `cancer_llm.db`).

```bash
# Disable caching
cqknow generate graph -s model.json --no-cache

# Use a custom cache path
cqknow generate graph -s model.json --cache-path /path/to/cache.db
```

For more information on model specification format, see
[Model Specification Format](model_specification.md).

## Cache Management

CausalIQ Knowledge includes a caching system that stores LLM responses to avoid redundant API calls. You can inspect, export, and import your cache using the CLI:

```bash
# View cache statistics
cqknow cache stats ./llm_cache.db

# Export cache entries to human-readable JSON files
cqknow cache export ./llm_cache.db ./export_dir

# Export to a zip archive for sharing
cqknow cache export ./llm_cache.db ./export.zip

# Import cache entries (auto-detects entry types)
cqknow cache import ./new_cache.db ./export.zip
```

Exported files use the naming format `{id}_{timestamp}_{provider}.json`, for
example `cli_2026-01-29-143052_groq.json`. The `id` comes from the `--id`
option when generating graphs (default: "cli").

The cache stores:

- **Entry count**: Number of cached LLM responses
- **Token count**: Total tokens across all cached entries

For programmatic cache usage, see [Caching Architecture](../architecture/caching.md).

## What's Next?

- [Architecture Overview](../architecture/overview.md) - Understand how the package works
- [LLM Integration Design](../architecture/llm_integration.md) - Detailed design documentation
- [API Reference](../api/overview.md) - Full API documentation