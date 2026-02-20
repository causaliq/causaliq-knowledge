# CausalIQ Knowledge API Reference

API documentation for causaliq-knowledge, organised by module.

## Import Patterns

Graph generation classes are available from the `graph` submodule:

```python
from causaliq_knowledge.graph import (
    # Network context (main model)
    NetworkContext,
    NetworkLoadError,
    # Variable specification
    VariableSpec,
    VariableType,
    VariableRole,
    # Supporting models
    ViewDefinition,
    Provenance,
    LLMGuidance,
    Constraints,
    CausalPrinciple,
    GroundTruth,
    PromptDetails,
    # Filtering
    ViewFilter,
    PromptDetail,
    # Generation
    GraphGenerator,
    GraphGeneratorConfig,
    GeneratedGraph,
    ProposedEdge,
    GenerationMetadata,
    # Parameters
    GenerateGraphParams,
    # Prompts
    GraphQueryPrompt,
    OutputFormat,
    # Cache
    GraphCompressor,
)
```

Cache infrastructure is available from causaliq-core:

```python
from causaliq_core.cache import TokenCache
```

LLM clients should be imported from the `llm` submodule:

```python
from causaliq_knowledge.llm import (
    # Abstract base interface
    BaseLLMClient,
    LLMConfig,
    LLMResponse,
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
)
```

## Modules

### [Graph Module](graph/overview.md)

LLM-based causal graph generation from network context specifications:

- **[Graph Generator](graph/generator.md)** - Generate complete causal graphs
  - GraphGenerator, GraphGeneratorConfig
  - GeneratedGraph, ProposedEdge, GenerationMetadata
- **[Network Context](graph/models.md)** - Pydantic models for network context
  - NetworkContext, NetworkLoadError
  - VariableSpec, VariableType, VariableRole
  - PromptDetails, ViewDefinition, Provenance, Constraints
- **[View Filter](graph/view_filter.md)** - Extract context levels
  - ViewFilter, PromptDetail (MINIMAL, STANDARD, RICH)
- **[Graph Prompts](graph/prompts.md)** - Prompt builders
  - GraphQueryPrompt, OutputFormat
- **[Response Models](graph/response.md)** - Response parsing
  - ProposedEdge, GeneratedGraph, GenerationMetadata

### [LLM Client Interface](base_client.md)

Abstract base class and common types for LLM vendor clients:

- **BaseLLMClient** - Abstract interface all vendor clients implement
- **LLMConfig** - Base configuration dataclass
- **LLMResponse** - Unified response format

### Vendor API Clients

Direct API clients for specific LLM providers. All implement the
`BaseLLMClient` interface.

- **[Groq Client](clients/groq.md)** - Fast inference via Groq API
- **[Gemini Client](clients/gemini.md)** - Google Gemini API
- **[OpenAI Client](clients/openai.md)** - OpenAI GPT models
- **[Anthropic Client](clients/anthropic.md)** - Anthropic Claude models
- **[DeepSeek Client](clients/deepseek.md)** - DeepSeek models
- **[Mistral Client](clients/mistral.md)** - Mistral AI models
- **[Ollama Client](clients/ollama.md)** - Local LLMs via Ollama

### [CLI](cli.md)

Command-line interface for graph generation and cache management.