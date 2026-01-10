# DeepSeek Client

Direct DeepSeek API client for DeepSeek-V3 and DeepSeek-R1 models.

## Overview

DeepSeek is a Chinese AI company known for highly capable models at competitive prices.
Their API is **OpenAI-compatible**, making integration straightforward.

**Key features:**

- **DeepSeek-V3**: General purpose chat model, excellent performance
- **DeepSeek-R1**: Advanced reasoning model, rivals OpenAI o1 at much lower cost
- Very competitive pricing (~$0.14/1M input for chat)
- OpenAI-compatible API

## Configuration

The client requires a `DEEPSEEK_API_KEY` environment variable:

```bash
# Linux/macOS
export DEEPSEEK_API_KEY="your-api-key"

# Windows PowerShell
$env:DEEPSEEK_API_KEY="your-api-key"

# Windows cmd
set DEEPSEEK_API_KEY=your-api-key
```

Get your API key from: [https://platform.deepseek.com](https://platform.deepseek.com)

## Usage

### Basic Usage

```python
from causaliq_knowledge.llm import DeepSeekClient, DeepSeekConfig

# Default config (uses DEEPSEEK_API_KEY env var)
client = DeepSeekClient()

# Or with custom config
config = DeepSeekConfig(
    model="deepseek-chat",
    temperature=0.1,
    max_tokens=500,
    timeout=30.0,
)
client = DeepSeekClient(config)

# Make a completion request
messages = [{"role": "user", "content": "What is 2 + 2?"}]
response = client.completion(messages)
print(response.content)
```

### Using with CLI

```bash
# Query with DeepSeek
cqknow query smoking lung_cancer --model deepseek/deepseek-chat

# Use reasoning model for complex queries
cqknow query income education --model deepseek/deepseek-reasoner --domain economics

# List available DeepSeek models
cqknow models deepseek
```

### Using with LLMKnowledge Provider

```python
from causaliq_knowledge.llm import LLMKnowledge

# Single model
provider = LLMKnowledge(models=["deepseek/deepseek-chat"])
result = provider.query_edge("smoking", "lung_cancer")

# Multi-model consensus
provider = LLMKnowledge(
    models=[
        "deepseek/deepseek-chat",
        "groq/llama-3.1-8b-instant",
    ],
    consensus_strategy="weighted_vote",
)
```

## Available Models

| Model | Description | Best For |
|-------|-------------|----------|
| `deepseek-chat` | DeepSeek-V3 general purpose | Fast, general queries |
| `deepseek-reasoner` | DeepSeek-R1 reasoning model | Complex reasoning tasks |

## Pricing

DeepSeek offers very competitive pricing (as of Jan 2025):

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| deepseek-chat | $0.14 | $0.28 |
| deepseek-reasoner | $0.55 | $2.19 |

**Note:** Cache hits are even cheaper. See [DeepSeek pricing](https://platform.deepseek.com/pricing) for details.

## API Reference

::: causaliq_knowledge.llm.deepseek_client.DeepSeekConfig
    options:
      show_source: false

::: causaliq_knowledge.llm.deepseek_client.DeepSeekClient
    options:
      show_source: false
