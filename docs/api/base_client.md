# LLM Client Base Interface

Abstract base class and common types for all LLM vendor clients. This module
defines the interface that all vendor-specific clients must implement,
ensuring consistent behavior across different LLM providers.

## Overview

The base client module provides:

- **BaseLLMClient** - Abstract base class defining the client interface
- **LLMConfig** - Base configuration dataclass for all clients
- **LLMResponse** - Unified response format from any LLM provider

### Caching Support

BaseLLMClient includes built-in caching integration:

- **set_cache()** - Configure a TokenCache for response caching
- **cached_completion()** - Make completion requests with automatic caching
- **_build_cache_key()** - Generate deterministic cache keys (SHA-256)

## Design Philosophy

We use vendor-specific API clients rather than wrapper libraries like
LiteLLM or LangChain. This provides:

- Minimal dependencies (httpx only for HTTP)
- Reliable and predictable behavior
- Easy debugging without abstraction layers
- Full control over API interactions

The abstract interface ensures that all vendor clients behave consistently,
making it easy to swap providers or add new ones.

## Usage

Vendor-specific clients inherit from `BaseLLMClient`:

```python
from causaliq_knowledge.llm import (
    BaseLLMClient,
    LLMConfig,
    LLMResponse,
    GroqClient,
    GeminiClient,
)

# All clients share the same interface
def query_llm(client: BaseLLMClient, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    response = client.completion(messages)
    return response.content

# Works with any client
groq = GroqClient()
gemini = GeminiClient()

result1 = query_llm(groq, "What is 2+2?")
result2 = query_llm(gemini, "What is 2+2?")
```

### Caching LLM Responses

Enable caching to avoid redundant API calls:

```python
from causaliq_knowledge.cache import TokenCache
from causaliq_knowledge.llm import GroqClient, LLMConfig

# Create a persistent cache
with TokenCache("llm_cache.db") as cache:
    client = GroqClient(LLMConfig(model="llama-3.1-8b-instant"))
    client.set_cache(cache)
    
    messages = [{"role": "user", "content": "What is Python?"}]
    
    # First call - hits API, stores in cache
    response1 = client.cached_completion(messages)
    
    # Second call - returns from cache, no API call
    response2 = client.cached_completion(messages)
    
    assert response1.content == response2.content
    assert client.call_count == 1  # Only one API call made
```

The cache uses the LLMEntryEncoder automatically, storing:

- Request details (model, messages, temperature, max_tokens)
- Response content
- Metadata (provider, token counts, cost, latency)

Each cached entry captures latency timing automatically using `time.perf_counter()`,
enabling performance analysis across providers and models.

See [LLM Cache](llm/cache.md) for details on the cache entry structure.
```

## Creating a Custom Client

To add support for a new LLM provider, implement the `BaseLLMClient` interface:

```python
from causaliq_knowledge.llm import BaseLLMClient, LLMConfig, LLMResponse

class MyCustomClient(BaseLLMClient):
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._total_calls = 0

    @property
    def provider_name(self) -> str:
        return "my_provider"

    def completion(self, messages, **kwargs) -> LLMResponse:
        # Implement API call here
        ...
        return LLMResponse(
            content="response text",
            model=self.config.model,
            input_tokens=10,
            output_tokens=20,
        )

    @property
    def call_count(self) -> int:
        return self._total_calls
```

## LLMConfig

::: causaliq_knowledge.llm.base_client.LLMConfig

## LLMResponse

::: causaliq_knowledge.llm.base_client.LLMResponse

## BaseLLMClient

::: causaliq_knowledge.llm.base_client.BaseLLMClient
