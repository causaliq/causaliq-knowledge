# Graph Generator

The `generator` module provides the `GraphGenerator` class for generating
complete causal graphs from variable specifications using LLMs.

## Import Pattern

```python
from causaliq_knowledge.graph import (
    GraphGenerator,
    GraphGeneratorConfig,
    ModelLoader,
    GeneratedGraph,
    PromptDetail,
    OutputFormat,
)
from causaliq_knowledge.cache import TokenCache
```

## Overview

`GraphGenerator` orchestrates the full graph generation workflow:

1. Create a generator with model and configuration
2. Optionally set up caching with `TokenCache`
3. Generate graphs from variable dictionaries or `ModelSpec` files
4. Receive structured `GeneratedGraph` objects with edges and metadata

## Quick Start

Here's a complete working example:

```python
from causaliq_knowledge.graph import (
    GraphGenerator,
    GraphGeneratorConfig,
    ModelLoader,
    PromptDetail,
    OutputFormat,
)

# Create generator with model identifier
# Format: "provider/model_name"
generator = GraphGenerator(model="groq/llama-3.1-8b-instant")

# Option 1: Generate from a list of variables
graph = generator.generate_graph(
    variables=[
        {"name": "smoking"},
        {"name": "lung_cancer"},
        {"name": "age"},
    ],
    domain="oncology",
)

# Option 2: Generate from a model specification file
spec = ModelLoader.load("research/models/my_model.json")
graph = generator.generate_from_spec(spec)

# Access the results
print(f"Generated {len(graph.edges)} edges")
for edge in graph.edges:
    print(f"  {edge.source} -> {edge.target} ({edge.confidence:.2f})")

# Access metadata
print(f"Model: {graph.metadata.model}")
print(f"Latency: {graph.metadata.latency_ms}ms")
print(f"Cost: ${graph.metadata.cost_usd:.4f}")
```

## Configuration

### GraphGeneratorConfig

Configuration dataclass for graph generation parameters.

```python
from causaliq_knowledge.graph import GraphGeneratorConfig, PromptDetail, OutputFormat

config = GraphGeneratorConfig(
    temperature=0.1,              # LLM sampling temperature
    max_tokens=2000,              # Maximum response tokens
    timeout=60.0,                 # Request timeout in seconds
    output_format=OutputFormat.EDGE_LIST,  # or ADJACENCY_MATRIX
    prompt_detail=PromptDetail.STANDARD,   # MINIMAL, STANDARD, or RICH
    use_llm_names=True,           # Use llm_name field from specs
    request_id="",                # Optional request identifier
)
```

**Attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 0.1 | LLM sampling temperature (lower = more deterministic) |
| `max_tokens` | int | 2000 | Maximum tokens in LLM response |
| `timeout` | float | 60.0 | Request timeout in seconds |
| `output_format` | OutputFormat | EDGE_LIST | Response format (EDGE_LIST or ADJACENCY_MATRIX) |
| `prompt_detail` | PromptDetail | STANDARD | Detail level (MINIMAL, STANDARD, or RICH) |
| `use_llm_names` | bool | True | Use llm_name instead of benchmark name |
| `request_id` | str | "" | Optional identifier for requests |

## Creating a Generator

```python
from causaliq_knowledge.graph import GraphGenerator, GraphGeneratorConfig
from causaliq_knowledge.cache import TokenCache

# Basic creation with just a model
generator = GraphGenerator(model="groq/llama-3.1-8b-instant")

# With custom configuration
config = GraphGeneratorConfig(
    temperature=0.2,
    prompt_detail=PromptDetail.RICH,
)
generator = GraphGenerator(model="gemini/gemini-2.0-flash", config=config)

# With caching enabled
cache = TokenCache(db_path="graph_cache.db")
generator = GraphGenerator(
    model="openai/gpt-4o",
    config=config,
    cache=cache,
)

# Or set cache after creation
generator = GraphGenerator(model="anthropic/claude-sonnet-4-20250514")
generator.set_cache(cache, use_cache=True)
```

## Generating from Variables

Use `generate_graph()` when you have a list of variable dictionaries:

```python
from causaliq_knowledge.graph import GraphGenerator, PromptDetail

generator = GraphGenerator(model="groq/llama-3.1-8b-instant")

# Minimal - just variable names
graph = generator.generate_graph(
    variables=[
        {"name": "smoking"},
        {"name": "lung_cancer"},
        {"name": "age"},
        {"name": "genetics"},
    ],
    domain="oncology",
)

# With more context
graph = generator.generate_graph(
    variables=[
        {
            "name": "smoking",
            "type": "binary",
            "description": "Whether the patient smokes",
        },
        {
            "name": "lung_cancer",
            "type": "binary",
            "description": "Diagnosis of lung cancer",
        },
    ],
    domain="oncology",
    level=PromptDetail.RICH,  # Override config's prompt_detail
)

# Access results
for edge in graph.edges:
    print(f"{edge.source} -> {edge.target}")
    print(f"  Confidence: {edge.confidence}")
    print(f"  Rationale: {edge.rationale}")
```

## Generating from Model Specifications

Use `generate_from_spec()` when you have a JSON model specification file:

```python
from causaliq_knowledge.graph import GraphGenerator, ModelLoader, PromptDetail

generator = GraphGenerator(model="gemini/gemini-2.0-flash")

# Load the specification
spec = ModelLoader.load("research/models/asia/asia.json")

# Generate with default settings from config
graph = generator.generate_from_spec(spec)

# Override settings for this specific call
graph = generator.generate_from_spec(
    spec=spec,
    level=PromptDetail.MINIMAL,
    use_llm_names=False,  # Use benchmark names instead
)
```

## Supported LLM Providers

GraphGenerator supports all providers via the `provider/model` format:

| Provider | Example Model String |
|----------|---------------------|
| Anthropic | `anthropic/claude-sonnet-4-20250514` |
| DeepSeek | `deepseek/deepseek-chat` |
| Gemini | `gemini/gemini-2.0-flash` |
| Groq | `groq/llama-3.1-8b-instant` |
| Mistral | `mistral/mistral-large-latest` |
| Ollama | `ollama/llama3.2` |
| OpenAI | `openai/gpt-4o` |

```python
# Using different providers
gen_groq = GraphGenerator(model="groq/llama-3.1-8b-instant")
gen_gemini = GraphGenerator(model="gemini/gemini-2.0-flash")
gen_openai = GraphGenerator(model="openai/gpt-4o")
gen_anthropic = GraphGenerator(model="anthropic/claude-sonnet-4-20250514")
```

## Caching

GraphGenerator integrates with `TokenCache` for response caching:

```python
from causaliq_knowledge.graph import GraphGenerator
from causaliq_knowledge.cache import TokenCache

# Create cache and generator
cache = TokenCache(db_path="graph_cache.db")
generator = GraphGenerator(
    model="gemini/gemini-2.0-flash",
    cache=cache,
)

# First call - hits the LLM
graph1 = generator.generate_graph(
    variables=[{"name": "A"}, {"name": "B"}],
    domain="test",
)
print(f"From cache: {graph1.metadata.from_cache}")  # False

# Second call with same inputs - uses cache
graph2 = generator.generate_graph(
    variables=[{"name": "A"}, {"name": "B"}],
    domain="test",
)
print(f"From cache: {graph2.metadata.from_cache}")  # True

# Disable caching for specific generator
generator.set_cache(cache, use_cache=False)
```

## Prompt Detail Levels

Control the amount of context provided to the LLM:

| Level | Description |
|-------|-------------|
| `PromptDetail.MINIMAL` | Variable names only |
| `PromptDetail.STANDARD` | Names, types, and brief descriptions |
| `PromptDetail.RICH` | Full descriptions, roles, states, and constraints |

```python
from causaliq_knowledge.graph import GraphGenerator, GraphGeneratorConfig, PromptDetail

# Set at config level (default for all calls)
config = GraphGeneratorConfig(prompt_detail=PromptDetail.MINIMAL)
generator = GraphGenerator(model="groq/llama-3.1-8b-instant", config=config)

# Override per call
graph = generator.generate_graph(
    variables=[{"name": "A"}, {"name": "B"}],
    domain="test",
    level=PromptDetail.RICH,  # Use rich for this call only
)
```

## Output Formats

Choose between edge list and adjacency matrix output:

```python
from causaliq_knowledge.graph import GraphGenerator, GraphGeneratorConfig, OutputFormat

# Edge list format (default)
config = GraphGeneratorConfig(output_format=OutputFormat.EDGE_LIST)
generator = GraphGenerator(model="groq/llama-3.1-8b-instant", config=config)

# Adjacency matrix format
config = GraphGeneratorConfig(output_format=OutputFormat.ADJACENCY_MATRIX)
generator = GraphGenerator(model="groq/llama-3.1-8b-instant", config=config)
```

## Working with Results

### GeneratedGraph

The result of generation is a `GeneratedGraph` object:

```python
graph = generator.generate_graph(...)

# Access edges
for edge in graph.edges:
    print(f"Source: {edge.source}")
    print(f"Target: {edge.target}")
    print(f"Confidence: {edge.confidence}")
    print(f"Rationale: {edge.rationale}")

# Access metadata
meta = graph.metadata
print(f"Model: {meta.model}")
print(f"Provider: {meta.provider}")
print(f"Timestamp: {meta.timestamp}")
print(f"Latency: {meta.latency_ms}ms")
print(f"Input tokens: {meta.input_tokens}")
print(f"Output tokens: {meta.output_tokens}")
print(f"Cost: ${meta.cost_usd:.6f}")
print(f"From cache: {meta.from_cache}")
```

### Generator Statistics

```python
generator = GraphGenerator(model="groq/llama-3.1-8b-instant")

# After some generations...
stats = generator.get_stats()
print(f"Model: {stats['model']}")
print(f"Call count: {stats['call_count']}")
print(f"Client call count: {stats['client_call_count']}")
```

## Complete Example

Here's a full example showing a typical workflow:

```python
"""Generate a causal graph from a model specification."""

from pathlib import Path

from causaliq_knowledge.graph import (
    GraphGenerator,
    GraphGeneratorConfig,
    ModelLoader,
    PromptDetail,
    OutputFormat,
)
from causaliq_knowledge.cache import TokenCache


def main():
    # Set up caching
    cache = TokenCache(db_path=Path("cache/graph_cache.db"))

    # Configure the generator
    config = GraphGeneratorConfig(
        temperature=0.1,
        max_tokens=2000,
        prompt_detail=PromptDetail.STANDARD,
        output_format=OutputFormat.EDGE_LIST,
    )

    # Create generator
    generator = GraphGenerator(
        model="groq/llama-3.1-8b-instant",
        config=config,
        cache=cache,
    )

    # Load model specification
    spec = ModelLoader.load("research/models/asia/asia.json")
    print(f"Loaded spec: {spec.name}")
    print(f"Variables: {len(spec.variables)}")

    # Generate graph
    graph = generator.generate_from_spec(spec)

    # Display results
    print(f"\nGenerated {len(graph.edges)} edges:")
    for edge in graph.edges:
        print(f"  {edge.source} -> {edge.target} ({edge.confidence:.2f})")

    # Show metadata
    print(f"\nMetadata:")
    print(f"  Model: {graph.metadata.provider}/{graph.metadata.model}")
    print(f"  Latency: {graph.metadata.latency_ms}ms")
    print(f"  Tokens: {graph.metadata.input_tokens} in, "
          f"{graph.metadata.output_tokens} out")
    print(f"  Cost: ${graph.metadata.cost_usd:.6f}")
    print(f"  Cached: {graph.metadata.from_cache}")


if __name__ == "__main__":
    main()
```

## API Reference

::: causaliq_knowledge.graph.generator
    options:
      show_root_heading: false
      members:
        - GraphGeneratorConfig
        - GraphGenerator
