# CausalIQ Knowledge - Development Roadmap

**Last updated**: February 4, 2026  

This project roadmap fits into the [overall ecosystem roadmap](https://causaliq.org/projects/ecosystem_roadmap/)

## 🚧  Under development

*No releases currently under active development.*

---

## ✅ Previous Releases

### Release v0.4.0 - Graph Generation [February 2026]

CLI tools and CausalIQ workflows for LLM-generated causal graphs.

**Scope:**

- `GraphGenerator` class for generating complete causal graphs from variable specifications
- `ModelSpec` and `VariableSpec` models for JSON-based model definitions
- `ModelLoader` for loading and validating specification files
- `ViewFilter` for extracting minimal/standard/rich context levels
- `VariableDisguiser` for name obfuscation to counteract LLM memorisation
- `GraphQueryPrompt` builder with configurable context levels and output formats
- `ProposedEdge`, `GeneratedGraph`, `GenerationMetadata` response models
- CLI `generate_graph` command with comprehensive options
- Request ID tracking for improved export file naming
- CausalIQ Workflow integration via entry points
- Comprehensive API documentation with Python usage examples

### Release v0.3.0 - LLM Caching [January 2026]

SQLite-based response caching with CLI tools for cache management.

**Scope:**

- `TokenCache` - SQLite storage with WAL mode for concurrent access
- `CacheEncoder` system - Extensible serialization (JsonEncoder, LLMEntryEncoder)
- `LLMCacheEntry` - Structured model for cached responses
- `BaseLLMClient` caching integration - Automatic cache-first lookup
- CLI commands: `cache stats`, `cache export`, `cache import`
- Human-readable export filenames with edge query detection
- Zip archive support for export/import

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


### Release v0.1.0 - Foundation LLM [January 2026]

Foundation release establishing LLM client infrastructure for causal graph
generation.

**Scope:**

- Direct API clients for multiple LLM providers (Groq, Gemini, OpenAI, etc.)
- Unified LLM client interface with consistent response format
- LLM response caching infrastructure
- CLI for basic operations
- 100% test coverage
- Comprehensive documentation


## 🛣️ Upcoming Releases (speculative)

### Release v0.5.0 - Workflow Cache Integration

Update `generate_graph` to write results into Workflow Caches rather than
individual files. Migrate common caching infrastructure to causaliq-core.

**Background**

This release is part of the coordinated Workflow Cache feature spanning
causaliq-core (v0.4.0), causaliq-knowledge (v0.5.0), and causaliq-workflow
(v0.2.0). See causaliq-core documentation for cache architecture details.

**Scope**

*Commit 1: Create GraphEntryEncoder* ✅

- `TokenCache`, `EntryEncoder`, `JsonEncoder` now from causaliq-core
- New `graph/cache.py` module
- `GraphEntryEncoder` extends `EntryEncoder`
- Uses `SDG.compress()`/`SDG.decompress()` from causaliq-core
- Metadata stored as tokenised JSON (provenance, edge confidences)
- Unit tests for encode/decode round-trip

*Commit 2: Add cache parameter to generate_graph action* ✅

- New `workflow_cache` parameter in `GenerateGraphParams`
- Path to Workflow Cache `.db` file
- Deprecate `output` parameter (file-based output)

*Commit 3: Update GraphGenerator to write to Workflow Cache* ✅

- Write graph via `GraphEntryEncoder` to cache
- Include metadata: provenance, edge confidences, generation params
- Cache key from workflow context (matrix values)
- Unit and functional tests

*Commit 4: Documentation*

- Update generate_graph CLI and API docs
- Document Workflow Cache integration
- Migration guide from file-based output

**Dependencies**: Requires causaliq-core v0.4.0.dev2 or later

**Test PyPI**: Publish `0.5.0.dev1` after commit 4

### Release v0.6.0 - LLM Provider Cost Tracking

Query LLM provider APIs for usage and cost statistics.

**Scope:**

*Commit 1: Provider usage API clients*

- Create `UsageClient` base class and provider-specific implementations
- Support OpenAI, Anthropic, Google (Gemini), Groq usage APIs
- Handle authentication and API differences between providers
- Unit tests with mocked API responses

*Commit 2: CLI `llm costs` command*

- Add `cqknow llm costs` command
- Aggregate costs by time period (daily, monthly)
- JSON and human-readable output formats
- Functional tests for CLI

*Commit 3: Cache savings analysis*

- Comparison between cached savings and actual API costs
- Integration with existing cache statistics
- Summary reports showing cost efficiency

### Release v0.7.0 - LLM Context

- Literature retrieval (RAG) - evaluate lightweight alternatives to LangChain
- Vector store integration for document search

### Release v0.8.0 - Algorithm Integration

- Integration hooks for structure learning algorithms
- Knowledge-guided constraint generation
- Integration with causaliq-analysis `average()` function
- Entropy-based automatic query triggering

### Release v0.8.0 - Legacy Reference

- Support for deriving knowledge from reference networks
- Migration of functionality from legacy discovery repo

## 📦 Dependencies Evolution

```toml
# v0.1.0 - v0.3.0
dependencies = [
    "click>=8.0.0",
    "httpx>=0.24.0",
    "pydantic>=2.0.0",
]

# v0.4.0 (current)
dependencies = [
    "causaliq-workflow>=0.1.1.dev3",  # CausalIQ Action integration
    "click>=8.0.0",
    "httpx>=0.24.0",
    "pydantic>=2.0.0",
]

# Future (evaluate - prefer lightweight solutions)
# Consider: llama-index, simple RAG, or custom implementation
# Avoid: langchain (heavy dependencies)
``` 
