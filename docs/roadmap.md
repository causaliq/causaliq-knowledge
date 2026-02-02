# CausalIQ Knowledge - Development Roadmap

**Last updated**: February 2, 2026  

This project roadmap fits into the [overall ecosystem roadmap](https://causaliq.org/projects/ecosystem_roadmap/)

## 🚧  Under development

### Release v0.4.0 - Graph Generation

CLI tools and CausalIQ workflows for LLM-generated causal graphs

**Scope**

- request LLMs to generate a causal graph from variables specification
- complements existing edge-by-edge query capability (v0.1.0) with full graph generation
- leverages LLM response caching (v0.3.0) for efficient repeated queries
- domain and variables will be defined in a `.json` file (see `research/models/cancer/cancer.json`, for example) 
- users will be able to specify amount of context information provided to the LLMs: minimal, standard or rich
- users can opt to disguise variable names to try and counteract LLM memorisation
- ideally, response will be of directed edges or adjacency matrix with some
indication of edge confidence
- requests can be made through the CLI, or as CausalIQ workflow steps (so integration with the CausalIQ Action interface will be needed )

**Implementation Plan**

*Commit 1: Model specification schema and loader* ✅

- Create `VariableSpec` Pydantic model for variable definitions (name, type, states, role, descriptions)
- Create `ModelSpec` Pydantic model for full model specification (domain, variables, views, provenance)
- Implement `ModelLoader` class to load and validate `.json` model files
- Add schema validation with helpful error messages
- Unit tests for models and loader

*Commit 2: View filtering and variable disguising* ✅

- Implement `ViewFilter` to extract minimal/standard/rich views from `ModelSpec`
- Implement `VariableDisguiser` for name obfuscation (random mapping with seed for reproducibility)
- Add reverse mapping to translate LLM responses back to original names
- Unit tests for filtering and disguising

*Commit 3: Graph generation prompts* ✅

- Create `GraphQueryPrompt` builder for full graph generation (distinct from edge queries)
- Implement prompt templates for minimal/standard/rich context levels
- Define expected JSON response schema (edges list with confidence scores)
- Add prompt for adjacency matrix format as alternative output
- Unit tests for prompt generation

*Commit 4: Graph response models and parsing* ✅

- Create `ProposedEdge` model (source, target, confidence, reasoning)
- Create `GeneratedGraph` model (list of edges, metadata, model used)
- Implement robust JSON parsing for LLM graph responses
- Handle both edge list and adjacency matrix formats
- Unit tests for response parsing

*Commit 5: GraphGenerator provider class* ✅

- Create `GraphGenerator` class extending knowledge provider pattern
- Implement `generate_graph()` method using LLM clients
- Support all existing LLM providers (Groq, Gemini, OpenAI, etc.)
- Integrate with `TokenCache` for caching graph generation requests
- Ensure cache keys distinguish graph queries from edge queries
- Unit tests and integration tests

*Commit 6: CLI `generate` command* ✅

- Add `cqknow generate` command group
- Implement `generate graph` subcommand with options:
  - `--model-spec` / `-s`: Path to model specification JSON
  - `--prompt-detail` / `-p`: Context level (minimal/standard/rich)
  - `--disguise` / `-D`: Enable variable name disguising
  - `--llm` / `-m`: LLM model(s) to use
  - `--output` / `-o`: Output file path (JSON/CSV)
  - `--format` / `-f`: Output format (edges/adjacency)
- Human-readable output and JSON export
- Functional tests for CLI

*Commit 7: Improved export request file naming* ✅

- All requests (graph generate or edge existence etc) made to LLM have a string id parameter which is stored as part of the request metadata (i.e. is not used in the request hash)
- Export file names now have the format `{id}_{yyyy-mm-dd-hhmmss}_{provider}.json` e.g., `expt23_2026-01-27-201346_groq.json`
- An optional `--id` parameter is supported by the CLI, with a default value of "cli"
- Default cache filename changed from `_cache.db` to `_llm.db`

*Commit 8: CausalIQ Action integration* ✅

- Define `GenerateGraphParams` model for shared parameter validation
- Implement `GenerateGraphAction` compatible with CausalIQ workflows
- Add workflow step registration via `CausalIQAction` export
- Integration tests with mock workflow context
- Added `py.typed` marker for PEP 561 compliance

*Commit 9: Documentation and examples*

- API documentation for new modules
- User guide for graph generation workflow
- Example model specification files
- Update roadmap to mark v0.4.0 complete, scope v0.5.0

---

## ✅ Previous Releases

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

Simple LLM queries to 1 or 2 LLMs about edge existence and orientation to support graph averaging.

**Scope:**

- Abstract `KnowledgeProvider` interface
- `EdgeKnowledge` Pydantic model for structured responses
- `LLMKnowledge` implementation using vendor-specific API clients
- Direct API clients for Groq and Google Gemini
- Single-model and multi-model consensus queries
- Basic prompt templates for edge existence/orientation
- CLI for testing queries
- 100% test coverage
- Comprehensive documentation


## 🛣️ Upcoming Releases (speculative)

### Release v0.5.0 - LLM Provider Cost Tracking

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

### Release v0.6.0 - LLM Context

- Literature retrieval (RAG) - evaluate lightweight alternatives to LangChain
- Vector store integration for document search

### Release v0.7.0 - Algorithm Integration

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
    "causaliq-workflow>=0.1.1.dev1",  # CausalIQ Action integration
    "click>=8.0.0",
    "httpx>=0.24.0",
    "pydantic>=2.0.0",
]

# Future (evaluate - prefer lightweight solutions)
# Consider: llama-index, simple RAG, or custom implementation
# Avoid: langchain (heavy dependencies)
``` 
