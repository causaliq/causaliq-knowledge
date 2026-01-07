# CausalIQ Knowledge API Reference

API documentation for causaliq-knowledge, organized by module.

## Import Patterns

Core models are available from the top-level package:

```python
from causaliq_knowledge import EdgeKnowledge, EdgeDirection, KnowledgeProvider
```

LLM-specific classes should be imported from the `llm` submodule:

```python
from causaliq_knowledge.llm import LLMKnowledge
from causaliq_knowledge.llm import GroqClient, GroqConfig, GroqResponse
from causaliq_knowledge.llm import GeminiClient, GeminiConfig, GeminiResponse
from causaliq_knowledge.llm import EdgeQueryPrompt, parse_edge_response
```

## Modules

### [Models](models.md)

Core Pydantic models for representing causal knowledge:

- **EdgeDirection** - Enum for causal edge direction (a_to_b, b_to_a, undirected)
- **EdgeKnowledge** - Structured knowledge about a potential causal edge

### [Base](base.md)

Abstract interfaces for knowledge providers:

- **KnowledgeProvider** - Abstract base class that all knowledge sources implement

### [LLM Provider](provider.md)

Main entry point for LLM-based knowledge queries:

- **LLMKnowledge** - KnowledgeProvider implementation using vendor-specific API clients
- **weighted_vote** - Multi-model consensus by weighted voting
- **highest_confidence** - Select response with highest confidence

### [Groq Client](groq_client.md)

Direct Groq API client for fast LLM inference:

- **GroqConfig** - Configuration for Groq API client
- **GroqResponse** - Response from Groq API
- **GroqClient** - Client for making Groq API calls

### [Gemini Client](gemini_client.md)

Direct Google Gemini API client:

- **GeminiConfig** - Configuration for Gemini API client
- **GeminiResponse** - Response from Gemini API
- **GeminiClient** - Client for making Gemini API calls

### [Prompts](prompts.md)

Prompt templates for LLM edge queries:

- **EdgeQueryPrompt** - Builder for edge existence/orientation prompts
- **parse_edge_response** - Parse LLM JSON responses to EdgeKnowledge

### [CLI](cli.md)

Command-line interface for testing and querying.