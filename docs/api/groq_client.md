# Groq Client API Reference

Direct Groq API client for fast LLM inference. This client uses httpx to
communicate directly with the Groq API, providing reliable and predictable
behavior without wrapper library dependencies.

## Overview

The Groq client provides:

- Direct HTTP communication with Groq's API
- JSON response parsing with error handling
- Call counting for usage tracking
- Configurable timeout and retry settings

## Usage

```python
from causaliq_knowledge.llm import GroqClient, GroqConfig

# Create client with custom config
config = GroqConfig(
    model="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=500,
)
client = GroqClient(config=config)

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

The Groq client requires the `GROQ_API_KEY` environment variable to be set:

```bash
export GROQ_API_KEY=your_api_key_here
```

## GroqConfig

::: causaliq_knowledge.llm.groq_client.GroqConfig
    options:
        show_root_heading: true
        show_source: false
        heading_level: 3
        members: true

## GroqResponse

::: causaliq_knowledge.llm.groq_client.GroqResponse
    options:
        show_root_heading: true
        show_source: false
        heading_level: 3
        members: true

## GroqClient

::: causaliq_knowledge.llm.groq_client.GroqClient
    options:
        show_root_heading: true
        show_source: false
        heading_level: 3
        members: true

## Supported Models

Groq provides fast inference for open-source models:

| Model | Description | Free Tier |
|-------|-------------|-----------|
| `llama-3.1-8b-instant` | Fast Llama 3.1 8B model | ✅ Yes |
| `llama-3.1-70b-versatile` | Larger Llama 3.1 model | ✅ Yes |
| `mixtral-8x7b-32768` | Mixtral MoE model | ✅ Yes |

See [Groq documentation](https://console.groq.com/docs/models) for the full list of available models.
