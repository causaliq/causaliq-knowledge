# CausalIQ Knowledge User Guide

## What is CausalIQ Knowledge?

CausalIQ Knowledge is a Python package that provides **LLM-based graph
generation** for causal discovery workflows. It enables you to generate
complete causal graphs from variable specifications using Large Language
Models (LLMs), providing prior knowledge structures that can be compared
against data-driven discoveries.

## Primary Use Case

CausalIQ Knowledge generates causal graphs from model specifications that
describe variables, their types, and domain context. This is useful for:

1. Creating prior knowledge graphs for causal discovery algorithms
2. Comparing LLM-generated graphs against ground truth benchmarks
3. Exploring how different LLMs reason about causal relationships

## Quick Start

### Installation

```bash
pip install causaliq-knowledge
```

### Command Line Usage

Generate a causal graph from a model specification:

```bash
# Generate graph with caching
cqknow generate_graph -s model.json -o results/ -c cache.db

# Use a specific LLM model
cqknow generate_graph -s model.json -o results/ -c cache.db -m gemini/gemini-2.5-flash

# Rich context for detailed prompts
cqknow generate_graph -s model.json -o results/ -c cache.db -p rich
```

### Python API Usage

```python
from causaliq_knowledge.graph import GraphGenerator, GraphGeneratorConfig
from causaliq_knowledge.graph import ModelLoader

# Load model specification
loader = ModelLoader()
model_spec = loader.load("model.json")

# Configure generator
config = GraphGeneratorConfig(
    llm_model="groq/llama-3.1-8b-instant",
    prompt_detail="standard",
    temperature=0.1,
)

# Generate graph
generator = GraphGenerator(config)
result = generator.generate(model_spec)

print(f"Generated {len(result.edges)} edges")
for edge in result.edges:
    print(f"  {edge.source} -> {edge.target} ({edge.confidence})")
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

```bash
cqknow generate_graph -s model.json -o results/ -c cache.db -m groq/llama-3.1-8b-instant
```

Available Groq models: `groq/llama-3.1-8b-instant`, `groq/llama-3.1-70b-versatile`, `groq/mixtral-8x7b-32768`

#### Google Gemini (Free Tier)

Google offers free access to Gemini models:

1. Sign up at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Create an API key
3. Set `GEMINI_API_KEY` environment variable
4. Use in code:

```bash
cqknow generate_graph -s model.json -o results/ -c cache.db -m gemini/gemini-2.5-flash
```

#### OpenAI

OpenAI provides GPT-4o and other models:

1. Sign up at [platform.openai.com](https://platform.openai.com)
2. Create an API key
3. Set `OPENAI_API_KEY` environment variable

```bash
cqknow generate_graph -s model.json -o results/ -c cache.db -m openai/gpt-4o-mini
```

#### Anthropic

Anthropic provides Claude models:

1. Sign up at [console.anthropic.com](https://console.anthropic.com)
2. Create an API key
3. Set `ANTHROPIC_API_KEY` environment variable

```bash
cqknow generate_graph -s model.json -o results/ -c cache.db -m anthropic/claude-sonnet-4-20250514
```

#### DeepSeek

DeepSeek offers high-quality models at competitive prices:

1. Sign up at [platform.deepseek.com](https://platform.deepseek.com)
2. Create an API key
3. Set `DEEPSEEK_API_KEY` environment variable

```bash
cqknow generate_graph -s model.json -o results/ -c cache.db -m deepseek/deepseek-chat
```

#### Mistral

Mistral AI provides models with EU data sovereignty:

1. Sign up at [console.mistral.ai](https://console.mistral.ai)
2. Create an API key
3. Set `MISTRAL_API_KEY` environment variable

```bash
cqknow generate_graph -s model.json -o results/ -c cache.db -m mistral/mistral-small-latest
```

#### Ollama (Local)

Run models locally with Ollama (no API key needed):

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull a model: `ollama pull llama3`
3. Use in code:

```bash
cqknow generate_graph -s model.json -o results/ -c cache.db -m ollama/llama3
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
# Generate a test graph (requires a model spec file)
cqknow generate_graph -s model.json -o none -c none -m groq/llama-3.1-8b-instant

# Or view cache statistics
cqknow cache stats ./cache.db
```

## Graph Generation

CausalIQ Knowledge can generate complete causal graphs from model
specifications using LLMs. This is useful for creating prior knowledge
structures or comparing LLM-generated graphs against ground truth.

### Command Line Interface

```bash
cqknow generate_graph -s <model_spec.json> -o <output> -c <cache> [options]
```

### Basic Examples

```bash
# Generate a graph, save to Workflow Cache with LLM caching
cqknow generate_graph -s model.json -o workflow.db -c cache.db

# Generate to a directory (GraphML + JSON files)
cqknow generate_graph -s model.json -o results/ -c cache.db

# Generate without caching, print adjacency matrix to stdout
cqknow generate_graph -s model.json -o none -c none

# Use a specific LLM model
cqknow generate_graph -s model.json -o workflow.db -c cache.db -m gemini/gemini-2.5-flash

# Use rich context level for more detailed prompts
cqknow generate_graph -s model.json -o workflow.db -c cache.db -p rich

# Test benchmark memorisation with original variable names
cqknow generate_graph -s model.json -o workflow.db -c cache.db --use-benchmark-names
```

### CLI Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model-spec` | `-s` | (required) | Path to model specification JSON file |
| `--output` | `-o` | (required) | Output: Workflow Cache `.db`, directory, or `none` |
| `--llm-cache` | `-c` | (required) | Cache: `.db` file path or `none` to disable |
| `--prompt-detail` | `-p` | `standard` | Detail level: `minimal`, `standard`, or `rich` |
| `--llm-model` | `-m` | `groq/llama-3.1-8b-instant` | LLM model with provider prefix |
| `--llm-temperature` | `-t` | `0.1` | Temperature (0.0-2.0), lower = deterministic |
| `--use-benchmark-names` | | `False` | Use benchmark names (test memorisation) |

### Prompt Detail Levels

The `--prompt-detail` option controls how much context is provided to the LLM:

- **minimal**: Variable names only - tests LLM's general knowledge
- **standard**: Names with types, states, and short descriptions
- **rich**: Full context including extended descriptions and sensitivity hints

### Output Behaviour

- **With `-o workflow.db`**: Writes graph to Workflow Cache database
- **With `-o results/`**: Writes `graph.graphml`, `metadata.json`, and
  `confidences.json` to the directory
- **With `-o none`**: Prints adjacency matrix to stdout

### Caching

LLM responses are cached to avoid redundant API calls. The cache stores
request/response pairs keyed by model, prompt, and parameters.

```bash
# Enable caching (recommended for development)
cqknow generate_graph -s model.json -o graph.json -c cache.db

# Disable caching (for fresh responses)
cqknow generate_graph -s model.json -o graph.json -c none
```

---

## Using with CausalIQ Workflows

CausalIQ Knowledge integrates with [causaliq-workflow](https://github.com/causaliq/causaliq-workflow)
for reproducible, automated causal discovery experiments.

### Installation

Ensure both packages are installed:

```bash
pip install causaliq-knowledge causaliq-workflow
```

### Basic Workflow Example

Create a workflow file `generate_graph.yaml`:

```yaml
description: "Generate causal graph using LLM"
id: "llm-graph-generation"

steps:
  - name: "Generate Graph"
    uses: "causaliq-knowledge"
    with:
      action: "generate_graph"
      model_spec: "models/cancer.json"
      output: "results/cancer_graph.json"
      llm_cache: "cache/cancer.db"
      llm_model: "groq/llama-3.1-8b-instant"
      prompt_detail: "standard"
```

Run the workflow:

```bash
# Dry-run (validate without executing)
causaliq-workflow generate_graph.yaml --mode dry-run

# Execute the workflow
causaliq-workflow generate_graph.yaml --mode run
```

### Workflow Action Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `action` | Yes | - | Must be `generate_graph` |
| `model_spec` | Yes | - | Path to model specification JSON |
| `output` | Yes | - | Output `.json` file path or `none` |
| `llm_cache` | Yes | - | Cache `.db` file path or `none` |
| `llm_model` | No | `groq/llama-3.1-8b-instant` | LLM model identifier |
| `prompt_detail` | No | `standard` | `minimal`, `standard`, or `rich` |
| `use_benchmark_names` | No | `false` | Use original benchmark names |
| `llm_temperature` | No | `0.1` | LLM temperature (0.0-2.0) |

### Matrix Expansion Example

Run the same analysis across multiple models and prompt detail levels:

```yaml
description: \"Compare LLM graph generation across models\"
id: \"llm-comparison\"
workflow_cache: \"results/{{id}}_cache.db\"

matrix:
  model:
    - \"groq/llama-3.1-8b-instant\"
    - \"gemini/gemini-2.5-flash\"
  detail:
    - \"minimal\"
    - \"standard\"
    - \"rich\"

steps:
  - name: \"Generate Graph\"
    uses: "causaliq-knowledge"
    with:
      action: "generate_graph"
      model_spec: "models/cancer.json"
      llm_cache: "cache/llm_cache.db"
      llm_model: "{{model}}"
      prompt_detail: "{{detail}}"
```

This generates 6 graphs (2 models × 3 detail levels), all stored in the
Workflow Cache with matrix values as keys.

### Multi-Network Comparison

Compare graph generation across different causal networks:

```yaml
description: "Generate graphs for multiple networks"
id: "multi-network"
workflow_cache: "results/{{id}}_cache.db"

matrix:
  network:
    - "asia"
    - "cancer"
    - "earthquake"

steps:
  - name: "Generate Graph"
    uses: "causaliq-knowledge"
    with:
      action: "generate_graph"
      model_spec: "models/{{network}}/{{network}}.json"
      llm_cache: "cache/{{network}}_llm.db"
      llm_model: "groq/llama-3.1-8b-instant"
```

### Workflow Output

When a workflow step completes, the action returns:

```json
{
  "status": "success",
  "edges_count": 5,
  "variables_count": 8,
  "output_file": "results/cancer_graph.json",
  "cache_stats": {
    "cache_hits": 2,
    "cache_misses": 6
  }
}
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

## What's Next?

- [Architecture Overview](../architecture/overview.md) - Understand how the package works
- [LLM Integration Design](../architecture/llm_integration.md) - Detailed design documentation
- [API Reference](../api/overview.md) - Full API documentation