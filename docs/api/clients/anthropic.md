# Anthropic Client API Reference

Direct Anthropic API client for Claude models. This client implements the
[BaseLLMClient](../base_client.md) interface using httpx to communicate
directly with the Anthropic API.

## Overview

The Anthropic client provides:

- Direct HTTP communication with Anthropic's API
- Implements the `BaseLLMClient` abstract interface
- JSON response parsing with error handling
- Call counting for usage tracking
- Configurable timeout and retry settings
- Proper handling of Anthropic's system prompt format

## Usage

```python
from causaliq_knowledge.llm import AnthropicClient, AnthropicConfig

# Create client with custom config
config = AnthropicConfig(
    model="claude-sonnet-4-20250514",
    temperature=0.1,
    max_tokens=500,
)
client = AnthropicClient(config=config)

# Make a completion request
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
]
response = client.completion(messages)
print(response.content)

# Parse JSON response
json_data = response.parse_json()
```

## Environment Variables

The Anthropic client requires the `ANTHROPIC_API_KEY` environment variable to be set:

```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

## AnthropicConfig

::: causaliq_knowledge.llm.anthropic_client.AnthropicConfig
    options:
        show_root_heading: true
        show_source: false
        heading_level: 3
        members: true

## AnthropicClient

::: causaliq_knowledge.llm.anthropic_client.AnthropicClient
    options:
        show_root_heading: true
        show_source: false
        heading_level: 3
        members: true

## Supported Models

Anthropic provides the Claude family of models:

| Model | Description | Free Tier |
|-------|-------------|-----------|
| `claude-sonnet-4-20250514` | Claude Sonnet 4 - balanced performance | ❌ No |
| `claude-opus-4-20250514` | Claude Opus 4 - highest capability | ❌ No |
| `claude-3-5-haiku-latest` | Claude 3.5 Haiku - fast and efficient | ❌ No |

See [Anthropic documentation](https://docs.anthropic.com/en/docs/about-claude/models) for the full list of available models.
