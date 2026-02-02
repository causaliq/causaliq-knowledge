# Graph Module API Reference

The `graph` module provides functionality for LLM-based causal graph generation
from variable specifications.

## Import Patterns

All graph module classes are available from `causaliq_knowledge.graph`:

```python
from causaliq_knowledge.graph import (
    # Model specification
    ModelSpec,
    VariableSpec,
    VariableType,
    VariableRole,
    Views,
    ViewDefinition,
    Provenance,
    LLMGuidance,
    Constraints,
    CausalPrinciple,
    GroundTruth,
    # Loading
    ModelLoader,
    ModelLoadError,
    # Filtering
    ViewFilter,
    PromptDetail,
    # Disguising
    VariableDisguiser,
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
)
```

## Modules

### [Model Specification](models.md)

Pydantic models for defining causal model specifications:

- **ModelSpec** - Complete model specification with variables and metadata
- **VariableSpec** - Single variable definition with type, role, descriptions
- **VariableType** - Enum for variable types (binary, categorical, ordinal, continuous)
- **VariableRole** - Enum for causal roles (exogenous, endogenous, latent)
- **Views** - View definitions for minimal/standard/rich context levels
- **ViewDefinition** - Single view configuration with included fields

### [Model Loader](loader.md)

Loading and validation of model specification JSON files:

- **ModelLoader** - Static methods for loading and validating model specs
- **ModelLoadError** - Exception for model loading failures

### [View Filter](view_filter.md)

Filtering model specifications to extract specific context levels:

- **ViewFilter** - Extract minimal/standard/rich views from ModelSpec
- **PromptDetail** - Enum for context levels (MINIMAL, STANDARD, RICH)

### [Variable Disguiser](disguiser.md)

Variable name obfuscation for LLM queries:

- **VariableDisguiser** - Reproducible name mapping with reverse translation

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
