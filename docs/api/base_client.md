# LLM Client Base Interface

Abstract base class and common types for all LLM vendor clients. This module
defines the interface that all vendor-specific clients must implement,
ensuring consistent behavior across different LLM providers.

## Overview

The base client module provides:

- **BaseLLMClient** - Abstract base class defining the client interface
- **LLMConfig** - Base configuration dataclass for all clients
- **LLMResponse** - Unified response format from any LLM provider

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
