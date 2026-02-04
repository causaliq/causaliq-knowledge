# Workflow Integration

This guide explains how to use CausalIQ Knowledge as part of automated
CausalIQ Workflows for reproducible causal discovery experiments.

## Overview

CausalIQ Knowledge integrates with
[causaliq-workflow](https://github.com/causaliq/causaliq-workflow) through
Python entry points. When both packages are installed, the `causaliq-knowledge`
action becomes automatically available in workflow files.

## Installation

```bash
pip install causaliq-knowledge causaliq-workflow
```

Or install from Test PyPI for development versions:

```bash
pip install --extra-index-url https://test.pypi.org/simple/ \
    causaliq-knowledge causaliq-workflow
```

## Quick Start

### 1. Create a Model Specification

Create `models/smoking.json`:

```json
{
    "schema_version": "2.0",
    "dataset_id": "smoking",
    "domain": "epidemiology",
    "variables": [
        {
            "name": "smoking",
            "type": "binary",
            "short_description": "Patient smoking status"
        },
        {
            "name": "lung_cancer",
            "type": "binary",
            "short_description": "Lung cancer diagnosis"
        },
        {
            "name": "genetics",
            "type": "categorical",
            "states": ["low", "medium", "high"],
            "short_description": "Genetic risk factors"
        }
    ]
}
```

### 2. Create a Workflow File

Create `workflow.yaml`:

```yaml
description: "Generate causal graph for smoking study"
id: "smoking-graph"

steps:
  - name: "Generate Graph"
    uses: "causaliq-knowledge"
    with:
      action: "generate_graph"
      model_spec: "models/smoking.json"
      output: "results/smoking_graph.json"
      llm_cache: "cache/smoking.db"
      llm_model: "groq/llama-3.1-8b-instant"
```

### 3. Run the Workflow

```bash
# Validate without executing
causaliq-workflow workflow.yaml --mode dry-run

# Execute the workflow
causaliq-workflow workflow.yaml --mode run
```

## Action Parameters

The `causaliq-knowledge` action supports the `generate_graph` operation:

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `action` | Yes | - | Must be `generate_graph` |
| `model_spec` | Yes | - | Path to model specification JSON file |
| `output` | Yes | - | Output: `.json` file path or `none` for stdout |
| `llm_cache` | Yes | - | Cache: `.db` file path or `none` to disable |
| `llm_model` | No | `groq/llama-3.1-8b-instant` | LLM model identifier |
| `prompt_detail` | No | `standard` | Detail level: `minimal`, `standard`, `rich` |
| `use_benchmark_names` | No | `false` | Use original benchmark variable names |
| `llm_temperature` | No | `0.1` | LLM temperature (0.0-2.0) |

### LLM Model Identifiers

Models must include a provider prefix:

- **Groq**: `groq/llama-3.1-8b-instant`, `groq/llama-3.1-70b-versatile`
- **Gemini**: `gemini/gemini-2.5-flash`, `gemini/gemini-2.0-flash`
- **OpenAI**: `openai/gpt-4o-mini`, `openai/gpt-4o`
- **Anthropic**: `anthropic/claude-sonnet-4-20250514`
- **DeepSeek**: `deepseek/deepseek-chat`, `deepseek/deepseek-reasoner`
- **Mistral**: `mistral/mistral-small-latest`
- **Ollama**: `ollama/llama3`, `ollama/mistral`

### Prompt Detail Levels

- **minimal**: Variable names only (tests general LLM knowledge)
- **standard**: Names, types, states, and short descriptions
- **rich**: Full context including extended descriptions

## Workflow Examples

### Comparing Multiple LLM Models

```yaml
description: "Compare graph generation across LLM providers"
id: "model-comparison"

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
      model_spec: "models/cancer.json"
      output: "results/{{model}}/graph.json"
      llm_cache: "cache/{{model}}.db"
      llm_model: "{{model}}"
```

This generates 3 graphs, one for each model, in separate directories.

### Comparing Prompt Detail Levels

```yaml
description: "Compare prompt detail levels"
id: "detail-comparison"

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
      model_spec: "models/asia.json"
      output: "results/{{detail}}/graph.json"
      llm_cache: "cache/asia.db"
      llm_model: "groq/llama-3.1-8b-instant"
      prompt_detail: "{{detail}}"
```

### Multi-Network Analysis

```yaml
description: "Generate graphs for benchmark networks"
id: "benchmark-analysis"

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
      model_spec: "models/{{network}}/{{network}}.json"
      output: "results/{{network}}/graph.json"
      llm_cache: "cache/{{network}}.db"
```

### Full Comparison Matrix

```yaml
description: "Full model × detail × network comparison"
id: "full-comparison"

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
      model_spec: "models/{{network}}.json"
      output: "results/{{network}}/{{model}}/{{detail}}/graph.json"
      llm_cache: "cache/{{network}}_{{model}}.db"
      llm_model: "{{model}}"
      prompt_detail: "{{detail}}"
```

This generates 8 graphs (2 networks × 2 models × 2 detail levels).

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
        "model_spec": "models/cancer.json",
        "output": "results/graph.json",
        "llm_cache": "cache/cancer.db"
    }
}
```

## Output File Format

The generated graph JSON file contains:

```json
{
    "edges": [
        {"source": "smoking", "target": "lung_cancer", "confidence": 0.95},
        {"source": "genetics", "target": "lung_cancer", "confidence": 0.8}
    ],
    "variables": ["smoking", "lung_cancer", "genetics"],
    "reasoning": "Based on epidemiological evidence...",
    "metadata": {
        "model": "groq/llama-3.1-8b-instant",
        "prompt_detail": "standard",
        "timestamp": "2026-02-04T10:30:00Z"
    }
}
```

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

- [Model Specification Format](model_specification.md) - Define variables
- [CLI Reference](introduction.md#graph-generation) - Command-line usage
- [API Reference](../api/overview.md) - Programmatic access
