# CausalIQ Knowledge API Reference

API documentation for causaliq-knowledge, organized by module.

## Import Patterns

Core models are available from the top-level package:

```python
from causaliq_knowledge import EdgeKnowledge, EdgeDirection, KnowledgeProvider
```

Cache infrastructure is available from the `cache` submodule:

```python
from causaliq_knowledge.cache import TokenCache
```

LLM-specific classes should be imported from the `llm` submodule:

```python
from causaliq_knowledge.llm import (
    # Abstract base interface
    BaseLLMClient,
    LLMConfig,
    LLMResponse,
    # Main provider
    LLMKnowledge,
    # Vendor clients
    GroqClient,
    GroqConfig,
    GeminiClient,
    GeminiConfig,
    OpenAIClient,
    OpenAIConfig,
    AnthropicClient,
    AnthropicConfig,
    DeepSeekClient,
    DeepSeekConfig,
    MistralClient,
    MistralConfig,
    OllamaClient,
    OllamaConfig,
    # Prompts
    EdgeQueryPrompt,
    parse_edge_response,
)
```

## Modules

### [Models](models.md)

Core Pydantic models for representing causal knowledge:

- **EdgeDirection** - Enum for causal edge direction (a_to_b, b_to_a, undirected)
- **EdgeKnowledge** - Structured knowledge about a potential causal edge

### [Cache](cache/overview.md)

SQLite-backed caching infrastructure:

- **TokenCache** - Cache with connection management and transaction support

### [Base](base.md)

Abstract interfaces for knowledge providers:

- **KnowledgeProvider** - Abstract base class that all knowledge sources implement

### [LLM Provider](provider.md)

Main entry point for LLM-based knowledge queries:

- **LLMKnowledge** - KnowledgeProvider implementation using vendor-specific API clients
- **weighted_vote** - Multi-model consensus by weighted voting
- **highest_confidence** - Select response with highest confidence

### [LLM Client Interface](base_client.md)

Abstract base class and common types for LLM vendor clients:

- **BaseLLMClient** - Abstract interface all vendor clients implement
- **LLMConfig** - Base configuration dataclass
- **LLMResponse** - Unified response format

### Vendor API Clients

Direct API clients for specific LLM providers. All implement the `BaseLLMClient` interface.

- **[Groq Client](clients/groq.md)** - Fast inference via Groq API
- **[Gemini Client](clients/gemini.md)** - Google Gemini API
- **[OpenAI Client](clients/openai.md)** - OpenAI GPT models
- **[Anthropic Client](clients/anthropic.md)** - Anthropic Claude models
- **[DeepSeek Client](clients/deepseek.md)** - DeepSeek models
- **[Mistral Client](clients/mistral.md)** - Mistral AI models
- **[Ollama Client](clients/ollama.md)** - Local LLMs via Ollama

### [Prompts](prompts.md)

Prompt templates for LLM edge queries:

- **EdgeQueryPrompt** - Builder for edge existence/orientation prompts
- **parse_edge_response** - Parse LLM JSON responses to EdgeKnowledge

### [CLI](cli.md)

Command-line interface for testing and querying.