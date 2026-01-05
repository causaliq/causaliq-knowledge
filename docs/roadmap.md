# CausalIQ Knowledge - Development Roadmap

**Last updated**: January 05, 2026  

This project roadmap fits into the [overall ecosystem roadmap](https://causaliq.org/projects/ecosystem_roadmap/)

## 🎯 Current Release

### Release v0.1.0 - Foundation LLM [January 2026]

Simple LLM queries to 1 or 2 LLMs about edge existence and orientation to support graph averaging.

**Scope:**

- Abstract `KnowledgeProvider` interface
- `EdgeKnowledge` Pydantic model for structured responses
- `LLMKnowledge` implementation using LiteLLM
- Support for OpenAI, Anthropic, Google, Groq, and Ollama (local)
- Single-model and multi-model consensus queries
- Basic prompt templates for edge existence/orientation
- CLI for testing queries

**Out of scope for v0.1.0:**

- Response caching (v0.3.0)
- Rich context/RAG (v0.4.0)
- Direct algorithm integration (v0.5.0)

**Implementation milestones:**

- v0.1.0a: Core models and abstract interface
- v0.1.0b: LiteLLM client wrapper
- v0.1.0c: Edge existence/orientation prompts
- v0.1.0d: Multi-LLM consensus logic
- v0.1.0e: CLI and documentation

---

## ✅ Previous Releases

*See Git commit history for detailed implementation progress*

- none


## 🛣️ Upcoming Releases

### Release v0.2.0 - Additional LLMs
- Expanded provider configurations
- Provider-specific prompt optimizations
- Cost tracking and reporting utilities
- Budget management features

### Release v0.3.0 - LLM Caching
- Disk-based response caching (diskcache)
- Cache key: (node_a, node_b, context_hash, model)
- Cache invalidation strategies
- Semantic similarity caching (optional)

### Release v0.4.0 - LLM Context
- Variable descriptions and roles
- Domain context specification
- Literature retrieval (RAG) using LangChain components
- ChromaDB or FAISS vector store integration

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
# v0.1.0
dependencies = [
    "click>=8.0.0",
    "litellm>=1.0.0",
    "pydantic>=2.0.0",
]

# v0.3.0 (add)
"diskcache>=5.0.0"

# v0.4.0 (add)
"langchain-core>=0.2.0"
"langchain-community>=0.2.0"
"chromadb>=0.4.0"
``` 
