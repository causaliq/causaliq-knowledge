# CausalIQ Knowledge API Reference

API documentation for causaliq-knowledge, organised by module.

## Import Patterns

Graph generation classes are available from the `graph` submodule:

```python
from causaliq_knowledge.graph import (
    # Model specification
    ModelSpec,
    VariableSpec,
    VariableType,
    VariableRole,
    ViewDefinition,
    Provenance,
    LLMGuidance,
    Constraints,
    GroundTruth,
    # Loading
    ModelLoader,
    ModelLoadError,
    # Filtering
    ViewFilter,
    PromptDetail,
    # Disguising
    VariableDisguiser,
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

LLM-based causal graph generation from variable specifications:

- **[Graph Generator](graph/generator.md)** - Generate complete causal graphs
  - GraphGenerator, GraphGeneratorConfig
  - GeneratedGraph, ProposedEdge, GenerationMetadata
- **[Model Specification](graph/models.md)** - Pydantic models for model specs
  - ModelSpec, VariableSpec, VariableType, VariableRole
  - PromptDetails, ViewDefinition, Provenance, Constraints
- **[Model Loader](graph/loader.md)** - Load and validate JSON model files
  - ModelLoader, ModelLoadError
- **[View Filter](graph/view_filter.md)** - Extract context levels
  - ViewFilter, PromptDetail (MINIMAL, STANDARD, RICH)
- **[Variable Disguiser](graph/disguiser.md)** - Name obfuscation
  - VariableDisguiser with reproducible seed-based mapping

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