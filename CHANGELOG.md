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
