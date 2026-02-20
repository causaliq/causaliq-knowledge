# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Nothing yet

### Changed

- Nothing yet

### Deprecated

- Nothing yet

### Removed

- Nothing yet

### Fixed

- Nothing yet

### Security

- Nothing yet

## [0.5.0] - Workflow Integration - 2026-02-20

Integration into CausalIQ Workflows including writing results to cache.

- **GraphCompressor**: Workflow Cache integration for generated graphs
  - Compression/decompression of `GeneratedGraph` objects
  - Integration with causaliq-workflow cache system
- **CausalIQ Workflow Integration**: Action-based workflow support
  - `ActionProvider` class for workflow step registration
  - `network_context` parameter for specifying network context file
  - Entry point discovery via `causaliq.actions` namespace
  - `py.typed` marker for PEP 561 compliance
- **Comprehensive API Documentation**: Updated docs for Python usage
  - Complete `GraphGenerator` API reference with examples
  - Quick start guides for graph generation
  - Workflow integration documentation

### Changed

- **Dependencies**: Updated to `causaliq-workflow>=0.2.0` and `causaliq-core>=0.4.0`


## [0.4.0] - Graph Generation - 2026-02-04

LLM-based causal graph generation from network context specifications.

### Added

- **Graph Generation Module**: Complete LLM-based causal graph generation
  - `GraphGenerator` class for generating complete causal graphs
  - `GraphGeneratorConfig` for configuration (temperature, max_tokens, timeout,
    output format, prompt detail level)
  - Support for all 7 LLM providers (Anthropic, DeepSeek, Gemini, Groq,
    Mistral, Ollama, OpenAI)
  - Integration with `TokenCache` for response caching
- **Network Context System**: JSON-based network definitions
  - `NetworkContext` dataclass with `load()`, `from_dict()`, `load_and_validate()`
  - `VariableSpec` with name, `llm_name`, type, role, states, and descriptions
  - `VariableType` and `VariableRole` enums for structured variable metadata
  - Support for domain, provenance, and constraint definitions
- **View Filtering**: Context level extraction for LLM prompts
  - `ViewFilter` to extract minimal/standard/rich views from network context
  - `PromptDetail` enum (MINIMAL, STANDARD, RICH)
  - `use_llm_names` parameter for semantic name disguising
- **Semantic Name Disguising**: Built-in name obfuscation via `llm_name` field
  - Each variable can specify an `llm_name` for LLM prompts
  - Automatic reverse translation of LLM responses to original names
  - No separate disguiser class needed - integrated into ViewFilter
- **Graph Prompts**: Structured prompt building for graph generation
  - `GraphQueryPrompt` builder with configurable context levels
  - `OutputFormat` enum (EDGE_LIST, ADJACENCY_MATRIX)
  - JSON response schemas for validation
- **Response Models**: Structured output from graph generation
  - `ProposedEdge` model with source, target, confidence, rationale
  - `GeneratedGraph` model with edges and metadata
  - `GenerationMetadata` with model, latency, tokens, cost, cache status
- **CLI Graph Generation**: Command-line interface for graph generation
  - `cqknow generate_graph` command with comprehensive options
  - `--network-context/-n` option for specifying network context file
  - Support for model selection, prompt detail, output format
  - JSON export with request ID tracking
- **Request ID Tracking**: Improved export file naming
  - Optional `--id` parameter for CLI commands
  - Export format: `{id}_{timestamp}_{provider}.json`
  - Default cache filename changed to `_llm.db`


### Changed

- **Dependencies**: Added `causaliq-workflow>=0.1.1.dev3` for workflow integration
- **Black Version**: Pinned to `>=25.0.0,<26.0.0` for Python 3.9 compatibility


## [0.3.0] - LLM Caching - 2026-01-26

### Added

- **TokenCache**: SQLite-based caching system for LLM responses
  - Disk-based persistence with WAL mode for concurrent access
  - Token counting for cost tracking
  - Entry count and type listing methods
- **Compressor System**: Extensible serialisation framework
  - `Compressor` abstract base class in `causaliq-core`
  - `JsonCompressor` for generic JSON serialisation with compression
  - `LLMCompressor` for LLM-specific entry handling
- **LLMCacheEntry**: Structured data model for cached LLM responses
  - Dataclass with cache_key, response, and metadata
  - `LLMResponse`, `LLMMetadata`, `LLMTokenUsage` data structures
  - Factory method `create()` for easy creation from API responses
- **BaseLLMClient Caching**: Automatic response caching in all LLM clients
  - Cache-first lookup with fallback to API
  - Configurable cache path
- **CLI Cache Commands**:
  - `cqknow cache stats` - View entry and token counts
  - `cqknow cache export` - Export to directory or zip archive
  - `cqknow cache import` - Import from directory or zip with auto-detection
- **Human-Readable Export Filenames**: Edge query detection for meaningful names
  - Pattern: `{id}_{timestamp}_{provider}.json`
- **Zip Archive Support**: Export/import with automatic .zip detection


## [0.2.0] - Additional LLMs - 2026-01-10

Expanded LLM provider support from 2 to 7 providers, covering major commercial and open-source options.

### Added

- **OpenAI Client**: Direct API client for GPT-4o, GPT-4o-mini and other OpenAI models
- **Anthropic Client**: Direct API client for Claude models (claude-sonnet-4-20250514, etc.)
- **DeepSeek Client**: Direct API client for DeepSeek-V3 and DeepSeek-R1 models (OpenAI-compatible)
- **Mistral Client**: Direct API client for Mistral AI models (OpenAI-compatible)
- **Ollama Client**: Local LLM support via Ollama (llama3, mistral, phi, etc.)
- **OpenAI-Compatible Base**: Shared client implementation for OpenAI-compatible APIs
- **Integration Tests**: Real API integration tests for all 7 providers (marked slow, skipped on CI)
- **Provider Documentation**: Individual API reference pages for each client

### Changed

- Updated documentation to reflect all supported providers
- Enhanced provider detection in `LLMKnowledge` to route to correct client
- Updated roadmap and user guide with new provider setup instructions


## [0.1.0] - Foundation LLM - 2026-01-07

First release of causaliq-knowledge providing LLM-based knowledge services for causal discovery.

### Added

- **Core Models**: `EdgeKnowledge` Pydantic model for structured causal edge knowledge with existence, direction, confidence, and reasoning fields
- **EdgeDirection Enum**: Type-safe direction representation (a_to_b, b_to_a, undirected)
- **KnowledgeProvider Interface**: Abstract base class defining the knowledge provider contract
- **LLMKnowledge Provider**: Main entry point for querying LLMs about causal relationships
  - Support for Groq API (llama-3.1-8b-instant, etc.)
  - Support for Google Gemini API (gemini-2.5-flash, etc.)
  - Multi-model consensus with weighted voting and highest confidence strategies
  - Configurable temperature, max tokens, and timeout settings
- **Direct API Clients**: Vendor-specific API clients using httpx for reliability
  - `GroqClient`: Direct Groq API integration
  - `GeminiClient`: Direct Google Gemini API integration
- **Prompt Templates**: Structured prompts for edge existence and orientation queries
  - `EdgeQueryPrompt`: Builder for constructing LLM prompts with domain context
  - `parse_edge_response`: JSON response parsing with validation
- **CLI Interface**: Command-line tool for testing queries
  - `cqknow query` command with model, domain, and output format options
  - JSON and human-readable output formats
- **Comprehensive Documentation**: MkDocs-based documentation site
  - Architecture overview and design notes
  - API reference with one page per module
  - User guide with quickstart examples
  - Development roadmap
- **100% Test Coverage**: Complete unit test suite covering all modules

### Architecture Decisions

- **Vendor-Specific APIs**: Use direct API clients (httpx) rather than wrapper libraries like LiteLLM or LangChain for reliability, minimal dependencies, and predictable behavior
- **Pydantic Models**: Structured response validation ensuring type safety and clear interfaces
- **Abstract Provider Interface**: Extensible design allowing future knowledge sources (rule-based, human-input, etc.)
