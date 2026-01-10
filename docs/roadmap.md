# CausalIQ Knowledge - Development Roadmap

**Last updated**: January 10, 2026  

This project roadmap fits into the [overall ecosystem roadmap](https://causaliq.org/projects/ecosystem_roadmap/)

## 🎯 Current Release

### Release v0.2.0 - Additional LLMs [January 2026]

Expanded LLM provider support from 2 to 7 providers.

**Scope:**

- OpenAI client for GPT-4o and GPT-4o-mini models
- Anthropic client for Claude models
- DeepSeek client for DeepSeek-V3 and R1 models
- Mistral client for Mistral AI models
- Ollama client for local LLM inference
- OpenAI-compatible base client for API-compatible providers
- Integration tests for all providers
- Cost estimation utilities for each provider


---

## ✅ Previous Releases

### Release v0.1.0 - Foundation LLM [January 2026]

Simple LLM queries to 1 or 2 LLMs about edge existence and orientation to support graph averaging.

**Delivered:**

- Abstract `KnowledgeProvider` interface
- `EdgeKnowledge` Pydantic model for structured responses
- `LLMKnowledge` implementation using vendor-specific API clients
- Direct API clients for Groq and Google Gemini
- Single-model and multi-model consensus queries
- Basic prompt templates for edge existence/orientation
- CLI for testing queries
- 100% test coverage
- Comprehensive documentation


## 🛣️ Upcoming Releases

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
