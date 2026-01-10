# OpenAI Client API Reference

Direct OpenAI API client for GPT models. This client implements the
[BaseLLMClient](../base_client.md) interface using httpx to communicate
directly with the OpenAI API.

## Overview

The OpenAI client provides:

- Direct HTTP communication with OpenAI's API
- Implements the `BaseLLMClient` abstract interface
- JSON response parsing with error handling
- Call counting for usage tracking
- Cost estimation for API calls
- Configurable timeout and retry settings

## Usage

```python
from causaliq_knowledge.llm import OpenAIClient, OpenAIConfig

# Create client with custom config
config = OpenAIConfig(
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=500,
)
client = OpenAIClient(config=config)

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

The OpenAI client requires the `OPENAI_API_KEY` environment variable to be set:

```bash
export OPENAI_API_KEY=your_api_key_here
```

## OpenAIConfig

::: causaliq_knowledge.llm.openai_client.OpenAIConfig
    options:
        show_root_heading: true
        show_source: false
        heading_level: 3
        members: true

## OpenAIClient

::: causaliq_knowledge.llm.openai_client.OpenAIClient
    options:
        show_root_heading: true
        show_source: false
        heading_level: 3
        members: true

## Supported Models

OpenAI provides the GPT family of models:

| Model | Description | Free Tier |
|-------|-------------|-----------|
| `gpt-4o` | GPT-4o - flagship multimodal model | ❌ No |
| `gpt-4o-mini` | GPT-4o Mini - affordable and fast | ❌ No |
| `gpt-4-turbo` | GPT-4 Turbo - high capability | ❌ No |
| `gpt-3.5-turbo` | GPT-3.5 Turbo - fast and economical | ❌ No |
| `o1` | o1 - reasoning model | ❌ No |
| `o1-mini` | o1 Mini - efficient reasoning | ❌ No |

See [OpenAI documentation](https://platform.openai.com/docs/models) for the full list of available models and pricing.
