# Generate Graph

The `generate_graph` action asks LLMs to propose causal edges together with
probabilities of the edge existing, and having a specific orientation. It
returns a Probability Dependency Graph (PDG) encapsulating this information.
These can be used to support, or be compared with, statistical structure
learning algorithms which will be provided in the `causaliq-discovery` package.

This is a `create` action (see
[workflow action patterns](https://workflow.causaliq.org/userguide/action_patterns/)) meaning it
creates new output entries from each matched input entry.

## Parameters

| Parameter | CLI | Default | Description |
|-----------|-----|---------|-------------|
| `network_context` | `-n` | None | Path to network context JSON file |
| `output` | `-o` | None | Output directory (CLI) or workflow cache (`.db`) |
| `llm_cache` | `-c` | None | LLM cache: `.db` file path or `none` to disable |
| `llm_model` | `-m` | `groq/llama-3.1-8b-instant` | LLM model identifier |
| `llm_temperature` | `-t` | 0.1 | LLM temperature (0.0-2.0) |
| `llm_max_tokens` | | 4000 | Maximum tokens in LLM response (100-100000) |
| `llm_timeout` | | 120.0 | LLM request timeout in seconds (10-600) |
| `llm_seed` | | None | Seed index for multi-sampling (busts cache) |
| `prompt_detail` | `-p` | `standard` | Detail level: `minimal`, `standard`, `rich` |
| `use_benchmark_names` | `-b` | `false` | Use benchmark names instead of LLM names |

**Notes:**

- Values must be supplied for all parameters without a default
- In CLI, parameter names use hyphens (e.g., `--network-context`, `--llm-cache`)
- The `llm_cache` parameter is **required** for both CLI and workflow usage.
  Use `none` to disable caching (not recommended for production)

---

## How It Works

### Step 1: Load Network Context

The network context JSON file defines the variables and domain context for
the graph generation. See [Network Context Format](model_specification.md) for
the complete specification.

### Step 2: Generate Edge Queries

For each pair of variables, the LLM is asked to estimate:

- Probability of a directed edge in each direction
- Probability of an undirected edge
- Probability of no edge

The prompt detail level controls how much context is provided:

| Level | Includes |
|-------|----------|
| `minimal` | Variable names only |
| `standard` | Names, types, states, short descriptions |
| `rich` | Full context including extended descriptions |

### Step 3: Aggregate to PDG

Responses are aggregated into a Probability Dependency Graph (PDG) where each
edge has four probability values:

$$P(forward) + P(backward) + P(undirected) + P(none) = 1.0$$

### Step 4: Return Results

The PDG is saved along with generation metadata including:

| Metadata | Description |
|----------|-------------|
| `model` | LLM model used |
| `provider` | LLM provider (groq, gemini, etc.) |
| `prompt_detail` | Detail level used |
| `tokens_input` | Total input tokens |
| `tokens_output` | Total output tokens |
| `cost_usd` | Estimated API cost |
| `latency_ms` | Total generation time |

---

## CLI Usage

### Basic Usage

Generate a causal graph from a network context file:

```bash
cqknow generate-graph -n asia.json -c cache.db -o results/
```

This creates:

- `results/graph.graphml` — The generated PDG
- `results/metadata.json` — Generation metadata

### With Specific Model

Use a different LLM model:

```bash
cqknow generate-graph -n asia.json -c cache.db -o results/ \
    -m gemini/gemini-2.5-flash
```

### Rich Prompt Context

Provide more context to the LLM for better results:

```bash
cqknow generate-graph -n asia.json -c cache.db -o results/ \
    -p rich
```

### Test Benchmark Memorisation

Use original benchmark names to test if the LLM has memorised the structure:

```bash
cqknow generate-graph -n asia.json -c cache.db -o results/ \
    --use-benchmark-names
```

### Without Output Files

Print results to stderr without writing files:

```bash
cqknow generate-graph -n asia.json -c none -o none
```

---

## Workflow Usage

In a CausalIQ workflow, `generate_graph` operates as a CREATE action that
generates a new PDG entry in the workflow cache:

```yaml
steps:
  - name: "Generate LLM graph"
    uses: "causaliq-knowledge"
    with:
      action: "generate_graph"
      network_context: "models/asia.json"
      output: "results/asia.db"
      llm_cache: "cache/llm_cache.db"
      llm_model: "groq/llama-3.1-8b-instant"
```

### Comparing Multiple Models

```yaml
description: "Compare graph generation across LLM providers"
id: "model-comparison"
workflow_cache: "results/{{id}}_cache.db"

matrix:
  model:
    - "groq/llama-3.1-8b-instant"
    - "gemini/gemini-2.5-flash"
    - "deepseek/deepseek-chat"

steps:
  - name: "Generate Graph"
    uses: "causaliq-knowledge"
    with:
      action: "generate_graph"
      network_context: "models/cancer.json"
      output: "results/cancer.db"
      llm_cache: "cache/llm_cache.db"
      llm_model: "{{model}}"
```

### Comparing Prompt Detail Levels

```yaml
description: "Compare prompt detail levels"
id: "detail-comparison"
workflow_cache: "results/{{id}}_cache.db"

matrix:
  detail:
    - "minimal"
    - "standard"
    - "rich"

steps:
  - name: "Generate Graph"
    uses: "causaliq-knowledge"
    with:
      action: "generate_graph"
      network_context: "models/asia.json"
      output: "results/asia.db"
      llm_cache: "cache/asia_llm.db"
      llm_model: "groq/llama-3.1-8b-instant"
      prompt_detail: "{{detail}}"
```

### Multi-Network Analysis

```yaml
description: "Generate graphs for benchmark networks"
id: "benchmark-analysis"
workflow_cache: "results/{{id}}_cache.db"

matrix:
  network:
    - "asia"
    - "cancer"
    - "earthquake"
    - "survey"

steps:
  - name: "Generate Graph"
    uses: "causaliq-knowledge"
    with:
      action: "generate_graph"
      network_context: "models/{{network}}/{{network}}.json"
      output: "results/{{network}}.db"
      llm_cache: "cache/{{network}}_llm.db"
```

---

## LLM Model Identifiers

Models must include a provider prefix. Use
[`cqknow list-models`](list_models.md) to see available models.

| Provider | Example Models |
|----------|----------------|
| Groq | `groq/llama-3.1-8b-instant`, `groq/llama-3.1-70b-versatile` |
| Gemini | `gemini/gemini-2.5-flash`, `gemini/gemini-2.0-flash` |
| OpenAI | `openai/gpt-4o-mini`, `openai/gpt-4o` |
| Anthropic | `anthropic/claude-sonnet-4-20250514` |
| DeepSeek | `deepseek/deepseek-chat`, `deepseek/deepseek-reasoner` |
| Mistral | `mistral/mistral-small-latest` |
| Ollama | `ollama/llama3`, `ollama/mistral` |

---

## LLM Caching

The `llm_cache` parameter specifies a SQLite database for caching LLM API
responses. This:

- **Reduces costs** by avoiding redundant API calls
- **Speeds up re-runs** of experiments
- **Enables reproducibility** by storing responses

Use [LLM Cache Management](llm_cache.md) commands to inspect, export, and
import cache contents.
```

### Full Comparison Matrix

```yaml
description: "Full model × detail × network comparison"
id: "full-comparison"
workflow_cache: "results/{{id}}_cache.db"

matrix:
  network:
    - "asia"
    - "cancer"
  model:
    - "groq/llama-3.1-8b-instant"
    - "gemini/gemini-2.5-flash"
  detail:
    - "minimal"
    - "standard"

steps:
  - name: "Generate Graph"
    uses: "causaliq-knowledge"
    with:
      action: "generate_graph"
      network_context: "models/{{network}}.json"
      llm_cache: "cache/llm_cache.db"
      llm_model: "{{model}}"
      prompt_detail: "{{detail}}"
```

This generates 8 graphs (2 networks × 2 models × 2 detail levels),
all stored in a single Workflow Cache with matrix values as keys.

## Action Output

When a workflow step completes successfully, it returns:

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

In dry-run mode, it returns validation results without executing:

```json
{
    "status": "skipped",
    "reason": "dry-run mode",
    "validated_params": {
        "network_context": "models/cancer.json",
        "output": "results/graph.json",
        "llm_cache": "cache/cancer.db"
    }
}
```

## Output File Format

The generated graph is saved as a PDG (Probabilistic Dependency Graph) in
GraphML format. Each edge carries separate existence and orientation
probabilities:

```xml
<edge source="smoking" target="lung_cancer">
  <data key="existence">0.95</data>
  <data key="orientation">0.85</data>
</edge>
```

Where:

- **existence**: Probability that a causal relationship exists (0.0-1.0)
- **orientation**: Confidence that the direction is source→target vs
  reverse (0.5 = uncertain, >0.5 = forward, <0.5 = reverse)

## Environment Setup

### API Keys

Set environment variables for your chosen LLM providers:

```bash
# Groq (free tier)
export GROQ_API_KEY="your-groq-key"

# Google Gemini (free tier)
export GEMINI_API_KEY="your-gemini-key"

# OpenAI
export OPENAI_API_KEY="your-openai-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-key"
```

See [LLM Provider Setup](introduction.md#llm-provider-setup) for details.

### Directory Structure

A typical project structure:

```
my-project/
├── models/
│   ├── asia.json
│   ├── cancer.json
│   └── smoking.json
├── workflows/
│   ├── generate_graphs.yaml
│   └── compare_models.yaml
├── results/           # Generated output files
├── cache/             # LLM response cache
└── .env               # API keys (add to .gitignore)
```

## Troubleshooting

### Action Not Found

```
ActionRegistryError: Action 'causaliq-knowledge' not found
```

Ensure both packages are installed in the same environment:

```bash
pip install causaliq-knowledge causaliq-workflow
```

### Schema Validation Error

```
WorkflowExecutionError: Schema file not found
```

Upgrade to the latest `causaliq-workflow`:

```bash
pip install --upgrade causaliq-workflow
```

### Invalid LLM Model

```
ValueError: LLM model must start with provider prefix
```

Include the provider prefix in `llm_model`:

```yaml
# Wrong
llm_model: "llama-3.1-8b-instant"

# Correct
llm_model: "groq/llama-3.1-8b-instant"
```

### Cache Path Error

```
ValueError: llm_cache must be 'none' or a path ending with .db
```

Use `.db` extension or `none`:

```yaml
# Wrong
llm_cache: "cache/data"

# Correct
llm_cache: "cache/data.db"
# Or
llm_cache: "none"
```

## Next Steps

- [Network Context Format](model_specification.md) - Define variables
- [User Guide](introduction.md) - Getting started and CLI usage
- [API Reference](../api/overview.md) - Programmatic access
