# CausalIQ Knowledge User Guide

## Overview

The causaliq-knowledge package is part of the
[CausalIQ ecosystem](https://causaliq.org/) for intelligent causal discovery
and inference. CausalIQ Knowledge provides the following capabilities for
integrating LLM knowledge into causal discovery workflows:

| Capability | Pattern | Description |
|------------|---------|-------------|
| **[`generate-graph`](generate_graph.md)** | Create | Generate causal graphs from network context using LLMs |

### CLI-Only Utilities

The following commands are available only through the CLI and are not
accessible as workflow actions:

| Command | Description |
|---------|-------------|
| **[`list-models`](list_models.md)** | List available LLM models from configured providers |
| **[`cache-stats`](llm_cache.md#cache-statistics)** | View LLM cache statistics and costs |
| **[`export-cache`](llm_cache.md#exporting-cache-entries)** | Export LLM cache entries to files |
| **[`import-cache`](llm_cache.md#importing-cache-entries)** | Import LLM cache entries from files |

## Command Line, Workflow or Programmatic Access

As with all CausalIQ packages, users may access the capabilities of CausalIQ
Knowledge in three ways:

- **Command Line Interface (CLI)** provides an easy introduction to generating
  graphs from the command line. This is primarily orientated to generating a
  single graph and thus gaining an initial understanding of how to use the
  capability.

- **CausalIQ Workflows** allows users to include CausalIQ Knowledge steps
  within workflows which can combine learning graphs from data or LLMs,
  performing inference, analysing results, through to generating publication
  tables and charts.

- **Programmatic Access** using the [Python API](../api/overview.md) for
  complete flexibility over the processing logic.

The CLI and workflow routes use the same command or action name respectively,
which in turn matches the capability name in the table above. Parameters are
named identically, and as far as practical, the capability behaviour is the
same in the CLI and workflow interfaces.

## Workflow Concepts

For common workflow concepts that apply across all CausalIQ packages, see the
[CausalIQ Workflow User Guide](https://workflow.causaliq.org/userguide/):

- [**Action patterns**](https://workflow.causaliq.org/userguide/action_patterns/)
  — Create, update, and aggregate patterns
- [**Common parameters**](https://workflow.causaliq.org/userguide/common_parameters/)
  — `input`, `output`, `filter` parameters
- [**Workflow caching**](https://workflow.causaliq.org/userguide/caching/)
  — Result storage and conservative execution
- [**CLI usage**](https://workflow.causaliq.org/userguide/cli/)
  — Execution modes (`run`, `dry-run`, `force`)

The individual action guides in this section document **knowledge-specific
parameters and behaviour** for each capability.

## LLM Provider Setup

CausalIQ Knowledge uses **direct vendor-specific API clients** (not wrapper
libraries) to communicate with LLM providers. This approach provides
reliability and minimal dependencies. Currently supported:

| Provider | Environment Variable | Free Tier | Console URL |
|----------|---------------------|-----------|-------------|
| Groq | `GROQ_API_KEY` | Yes (fast) | [console.groq.com](https://console.groq.com) |
| Google Gemini | `GEMINI_API_KEY` | Yes | [aistudio.google.com](https://aistudio.google.com) |
| OpenAI | `OPENAI_API_KEY` | No | [platform.openai.com](https://platform.openai.com) |
| Anthropic | `ANTHROPIC_API_KEY` | No | [console.anthropic.com](https://console.anthropic.com) |
| DeepSeek | `DEEPSEEK_API_KEY` | No | [platform.deepseek.com](https://platform.deepseek.com) |
| Mistral | `MISTRAL_API_KEY` | No | [console.mistral.ai](https://console.mistral.ai) |
| Ollama | (local) | Yes | [ollama.ai](https://ollama.ai) |

Use [`cqknow list-models`](list_models.md) to see which providers are
configured and what models are available.

### Storing API Keys

Set environment variables for your chosen providers:

```powershell
# PowerShell
$env:GROQ_API_KEY = "your-api-key"
$env:GEMINI_API_KEY = "your-api-key"
```

```bash
# Bash
export GROQ_API_KEY="your-api-key"
export GEMINI_API_KEY="your-api-key"
```

### Free Options

#### Groq (Recommended for Testing)

Groq offers a generous free tier with extremely fast inference:

1. Sign up at [console.groq.com](https://console.groq.com)
2. Create an API key
3. Set the environment variable
4. Use in code:

```bash
cqknow generate-graph -n context.json -o results/ -c cache.db \
    -m groq/llama-3.1-8b-instant
```

#### Google Gemini (Free Tier)

Google offers free access to Gemini models:

1. Sign up at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Create an API key
3. Set `GEMINI_API_KEY` environment variable
4. Use in code:

```bash
cqknow generate-graph -n context.json -o results/ -c cache.db \
    -m gemini/gemini-2.5-flash
```

#### Ollama (Local)

Run models locally with Ollama (no API key needed):

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull a model: `ollama pull llama3`
3. Use in commands with `-m ollama/llama3`

## What's Next?

- [**Generate Graph**](generate_graph.md) — Generate causal graphs from
  network context using LLMs
- [**Network Context Format**](model_specification.md) — Detailed specification
  for network context JSON files
- [**List Models**](list_models.md) — View available LLM models from configured
  providers
- [**LLM Cache Management**](llm_cache.md) — View, export, and import cached
  LLM responses