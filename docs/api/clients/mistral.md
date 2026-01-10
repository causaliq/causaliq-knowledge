# Mistral Client

Direct Mistral AI API client for Mistral models.

## Overview

Mistral AI is a French AI company known for high-quality open-weight and proprietary models.
Their API is **OpenAI-compatible**, making integration straightforward.

**Key features:**

- **Mistral Small**: Fast, cost-effective for simple tasks
- **Mistral Large**: Most capable, best for complex reasoning
- **Codestral**: Optimized for code generation
- Strong EU-based option for data sovereignty
- OpenAI-compatible API

## Configuration

The client requires a `MISTRAL_API_KEY` environment variable:

```bash
# Linux/macOS
export MISTRAL_API_KEY="your-api-key"

# Windows PowerShell
$env:MISTRAL_API_KEY="your-api-key"

# Windows cmd
set MISTRAL_API_KEY=your-api-key
```

Get your API key from: [https://console.mistral.ai](https://console.mistral.ai)

## Usage

### Basic Usage

```python
from causaliq_knowledge.llm import MistralClient, MistralConfig

# Default config (uses MISTRAL_API_KEY env var)
client = MistralClient()

# Or with custom config
config = MistralConfig(
    model="mistral-small-latest",
    temperature=0.1,
    max_tokens=500,
    timeout=30.0,
)
client = MistralClient(config)

# Make a completion request
messages = [{"role": "user", "content": "What is 2 + 2?"}]
response = client.completion(messages)
print(response.content)
```

### Using with CLI

```bash
# Query with Mistral
cqknow query smoking lung_cancer --model mistral/mistral-small-latest

# Use large model for complex queries
cqknow query income education --model mistral/mistral-large-latest --domain economics

# List available Mistral models
cqknow models mistral
```

### Using with LLMKnowledge Provider

```python
from causaliq_knowledge.llm import LLMKnowledge

# Single model
provider = LLMKnowledge(models=["mistral/mistral-small-latest"])
result = provider.query_edge("smoking", "lung_cancer")

# Multi-model consensus
provider = LLMKnowledge(
    models=[
        "mistral/mistral-large-latest",
        "groq/llama-3.1-8b-instant",
    ],
    consensus_strategy="weighted_vote",
)
```

## Available Models

| Model | Description | Best For |
|-------|-------------|----------|
| `mistral-small-latest` | Fast, cost-effective | Simple tasks |
| `mistral-medium-latest` | Balanced performance | General use |
| `mistral-large-latest` | Most capable | Complex reasoning |
| `codestral-latest` | Code-optimized | Programming tasks |
| `open-mistral-nemo` | 12B open model | Budget-friendly |
| `open-mixtral-8x7b` | MoE open model | Balanced open model |
| `ministral-3b-latest` | Ultra-small | Edge deployment |
| `ministral-8b-latest` | Small | Resource-constrained |

## Pricing

Mistral AI offers competitive pricing (as of Jan 2025):

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| mistral-small | $0.20 | $0.60 |
| mistral-medium | $2.70 | $8.10 |
| mistral-large | $2.00 | $6.00 |
| codestral | $0.20 | $0.60 |
| open-mistral-nemo | $0.15 | $0.15 |
| ministral-3b | $0.04 | $0.04 |
| ministral-8b | $0.10 | $0.10 |

See [Mistral pricing](https://mistral.ai/technology/#pricing) for details.

## API Reference

::: causaliq_knowledge.llm.mistral_client.MistralConfig
    options:
      show_source: false

::: causaliq_knowledge.llm.mistral_client.MistralClient
    options:
      show_source: false
