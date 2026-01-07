# Gemini Client API Reference

Direct Google Gemini API client. This client uses httpx to communicate
directly with the Gemini API, providing reliable and predictable behavior
without wrapper library dependencies.

## Overview

The Gemini client provides:

- Direct HTTP communication with Google's Generative Language API
- Automatic conversion from OpenAI-style messages to Gemini format
- JSON response parsing with error handling
- Call counting for usage tracking
- Configurable timeout settings

## Usage

```python
from causaliq_knowledge.llm import GeminiClient, GeminiConfig

# Create client with custom config
config = GeminiConfig(
    model="gemini-2.5-flash",
    temperature=0.1,
    max_tokens=500,
)
client = GeminiClient(config=config)

# Make a completion request (OpenAI-style messages)
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

The Gemini client requires the `GEMINI_API_KEY` environment variable to be set:

```bash
export GEMINI_API_KEY=your_api_key_here
```

## GeminiConfig

::: causaliq_knowledge.llm.gemini_client.GeminiConfig
    options:
        show_root_heading: true
        show_source: false
        heading_level: 3
        members: true

## GeminiResponse

::: causaliq_knowledge.llm.gemini_client.GeminiResponse
    options:
        show_root_heading: true
        show_source: false
        heading_level: 3
        members: true

## GeminiClient

::: causaliq_knowledge.llm.gemini_client.GeminiClient
    options:
        show_root_heading: true
        show_source: false
        heading_level: 3
        members: true

## Message Format Conversion

The client automatically converts OpenAI-style messages to Gemini's format:

| OpenAI Role | Gemini Role |
|-------------|-------------|
| `system` | System instruction (separate field) |
| `user` | `user` |
| `assistant` | `model` |

## Supported Models

Google Gemini provides a generous free tier:

| Model | Description | Free Tier |
|-------|-------------|-----------|
| `gemini-2.5-flash` | Fast and efficient | ✅ Yes |
| `gemini-2.5-pro` | Most capable | ✅ Limited |
| `gemini-1.5-flash` | Previous generation | ✅ Yes |

See [Google AI documentation](https://ai.google.dev/models/gemini) for the full list of available models.
