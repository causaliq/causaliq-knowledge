# Graph Module API Reference

The `graph` module provides functionality for LLM-based causal graph generation
from network context specifications.

## Quick Start

Generate a causal graph in Python:

```python
from causaliq_knowledge.graph import GraphGenerator, NetworkContext

# Create a generator with your chosen model
generator = GraphGenerator(model="groq/llama-3.1-8b-instant")

# Load a network context and generate
context = NetworkContext.load("research/models/asia/asia.json")
graph = generator.generate_from_context(context)

# Access the results
for edge in graph.edges:
    print(f"{edge.source} -> {edge.target}")
```

For complete examples and configuration options, see
[Graph Generator](generator.md).

## Import Patterns

All graph module classes are available from `causaliq_knowledge.graph`:

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
    PromptDetails,
    ViewDefinition,
    Provenance,
    LLMGuidance,
    Constraints,
    CausalPrinciple,
    GroundTruth,
    # Filtering
    ViewFilter,
    PromptDetail,
    # Prompts
    GraphQueryPrompt,
    OutputFormat,
    EDGE_LIST_RESPONSE_SCHEMA,
    ADJACENCY_MATRIX_RESPONSE_SCHEMA,
    # Response models
    ProposedEdge,
    GeneratedGraph,
    GenerationMetadata,
    parse_graph_response,
    # Graph generation
    GraphGenerator,
    GraphGeneratorConfig,
    # Parameters
    GenerateGraphParams,
    # Cache integration
    GraphCompressor,
)
```

## Modules

### [Network Context](models.md)

Pydantic models for defining network context specifications:

- **NetworkContext** - Complete network context with variables and metadata
- **NetworkLoadError** - Exception for context loading failures
- **VariableSpec** - Single variable definition with type, role, descriptions
- **VariableType** - Enum for variable types (binary, categorical, ordinal, continuous)
- **VariableRole** - Enum for causal roles (exogenous, endogenous, latent)
- **PromptDetails** - Prompt detail definitions for minimal/standard/rich context levels
- **ViewDefinition** - Single view configuration with included fields

### [View Filter](view_filter.md)

Filtering network context to extract specific context levels:

- **ViewFilter** - Extract minimal/standard/rich views from NetworkContext
- **PromptDetail** - Enum for context levels (MINIMAL, STANDARD, RICH)

### [Graph Prompts](prompts.md)

Prompt builders for LLM graph generation queries:

- **GraphQueryPrompt** - Builder for system and user prompts
- **OutputFormat** - Enum for response formats (edge list, adjacency matrix)
- Response schemas for validation

### [Response Models](response.md)

Data models and parsing for LLM graph generation responses:

- **ProposedEdge** - Single proposed causal edge with confidence
- **GeneratedGraph** - Complete generated graph with edges and metadata
- **GenerationMetadata** - Metadata about the generation process
- **parse_graph_response** - Parse LLM responses into structured objects

### [Graph Generator](generator.md)

High-level graph generation orchestration:

- **GraphGenerator** - Main class for generating causal graphs via LLMs
- **GraphGeneratorConfig** - Configuration for generation parameters
- Support for all LLM providers with caching integration
