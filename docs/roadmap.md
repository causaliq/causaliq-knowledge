# CausalIQ Knowledge - Development Roadmap

**Last updated**: January 07, 2026  

This project roadmap fits into the [overall ecosystem roadmap](https://causaliq.org/projects/ecosystem_roadmap/)

## 🎯 Current Release

### Release v0.1.0 - Foundation LLM [January 2026]

Simple LLM queries to 1 or 2 LLMs about edge existence and orientation to support graph averaging.

**Scope:**

- Abstract `KnowledgeProvider` interface
- `EdgeKnowledge` Pydantic model for structured responses
- `LLMKnowledge` implementation using vendor-specific API clients
- Direct API clients for Groq and Google Gemini
- Single-model and multi-model consensus queries
- Basic prompt templates for edge existence/orientation
- CLI for testing queries

**Architecture Decision:**

We use **vendor-specific API clients** (GroqClient, GeminiClient) rather than wrapper libraries like LiteLLM or LangChain. This provides:

- Minimal dependencies (httpx only for HTTP)
- Reliable and predictable behavior
- Easy debugging without abstraction layers
- Full control over API interactions

**Out of scope for v0.1.0:**

- Response caching (v0.3.0)
- Rich context/RAG (v0.4.0)
- Direct algorithm integration (v0.5.0)

**Implementation milestones:**

- ~~v0.1.0a: Core models and abstract interface~~ ✅
- ~~v0.1.0b: Direct vendor API clients (Groq, Gemini)~~ ✅
- ~~v0.1.0c: Edge existence/orientation prompts~~ ✅
- ~~v0.1.0d: Multi-LLM consensus logic~~ ✅
- v0.1.0e: CLI and documentation

---

## ✅ Previous Releases

*See Git commit history for detailed implementation progress*

- None


## 🛣️ Upcoming Releases

### Release v0.2.0 - Additional LLM Providers

- OpenAI client for GPT-4 models
- Anthropic client for Claude models
- Provider-specific prompt optimizations
- Cost tracking and reporting utilities

### Release v0.3.0 - LLM Caching

- Disk-based response caching (diskcache)
- Cache key: (node_a, node_b, context_hash, model)
- Cache invalidation strategies
- Semantic similarity caching (optional)

### Release v0.4.0 - LLM Context

- Variable descriptions and roles
- Domain context specification
- Literature retrieval (RAG) - evaluate lightweight alternatives to LangChain
- Vector store integration for document search

### Release v0.5.0 - Algorithm Integration

- Integration hooks for structure learning algorithms
- Knowledge-guided constraint generation
- Integration with causaliq-analysis `average()` function
- Entropy-based automatic query triggering

### Release v0.6.0 - Legacy Reference

- Support for deriving knowledge from reference networks
- Migration of functionality from legacy discovery repo

## 📦 Dependencies Evolution

```toml
# v0.1.0 (current)
dependencies = [
    "click>=8.0.0",
    "httpx>=0.24.0",
    "pydantic>=2.0.0",
]

# v0.3.0 (add)
"diskcache>=5.0.0"

# v0.4.0 (evaluate - prefer lightweight solutions)
# Consider: llama-index, simple RAG, or custom implementation
# Avoid: langchain (heavy dependencies)
``` 
