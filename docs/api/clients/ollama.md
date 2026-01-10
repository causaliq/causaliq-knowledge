# Ollama Client API Reference

Local Ollama API client for running Llama and other open-source models locally.
This client implements the [BaseLLMClient](../base_client.md) interface using
httpx to communicate with a locally running Ollama server.

## Overview

The Ollama client provides:

- Local LLM inference without API keys or internet access
- Implements the `BaseLLMClient` abstract interface
- Support for Llama 3.2, Llama 3.1, Mistral, and other models
- JSON response parsing with error handling
- Call counting for usage tracking
- Availability checking via `is_available()` method

## Prerequisites

1. Install Ollama from [ollama.com/download](https://ollama.com/download)
2. Pull a model:
   ```bash
   ollama pull llama3.2:1b    # Small, fast (~1.3GB)
   ollama pull llama3.2       # Medium (~2GB)
   ollama pull llama3.1:8b    # Larger, better quality (~4.7GB)
   ```
3. Ensure Ollama is running (it usually auto-starts after installation)

## Usage

```python
from causaliq_knowledge.llm import OllamaClient, OllamaConfig

# Create client with default config (llama3.2:1b on localhost:11434)
client = OllamaClient()

# Or with custom config
config = OllamaConfig(
    model="llama3.1:8b",
    temperature=0.1,
    max_tokens=500,
    timeout=120.0,  # Local inference can be slow
)
client = OllamaClient(config=config)

# Check if Ollama is available
if client.is_available():
    # Make a completion request
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    response = client.completion(messages)
    print(response.content)
else:
    print("Ollama not running or model not installed")
```

## Using with LLMKnowledge Provider

```python
from causaliq_knowledge.llm import LLMKnowledge

# Use local Ollama for causal queries
provider = LLMKnowledge(models=["ollama/llama3.2:1b"])
result = provider.query_edge("smoking", "lung_cancer")
print(f"Exists: {result.exists}, Confidence: {result.confidence}")

# Mix local and cloud models for consensus
provider = LLMKnowledge(
    models=[
        "ollama/llama3.2:1b",
        "groq/llama-3.1-8b-instant",
    ],
    consensus_strategy="weighted_vote"
)
```

## OllamaConfig

::: causaliq_knowledge.llm.ollama_client.OllamaConfig
    options:
        show_root_heading: true
        show_source: false
        heading_level: 3
        members: true

## OllamaClient

::: causaliq_knowledge.llm.ollama_client.OllamaClient
    options:
        show_root_heading: true
        show_source: false
        heading_level: 3
        members: true

## Supported Models

Ollama supports many open-source models. Recommended for causal queries:

| Model | Size | RAM Needed | Quality |
|-------|------|------------|---------|
| `llama3.2:1b` | ~1.3GB | 4GB+ | Good for simple queries |
| `llama3.2` | ~2GB | 6GB+ | Better reasoning |
| `llama3.1:8b` | ~4.7GB | 10GB+ | Best quality |
| `mistral` | ~4GB | 8GB+ | Good alternative |

See [Ollama Library](https://ollama.com/library) for all available models.

## Troubleshooting

**"Could not connect to Ollama"**
- Ensure Ollama is installed and running
- Run `ollama serve` in a terminal, or start the Ollama app
- Check that nothing else is using port 11434

**"Model not found"**
- Run `ollama pull <model-name>` to download the model
- Run `ollama list` to see installed models

**Slow responses**
- Local inference is CPU/GPU bound
- Use smaller models like `llama3.2:1b`
- Increase the timeout in `OllamaConfig`
- Consider using GPU acceleration if available
