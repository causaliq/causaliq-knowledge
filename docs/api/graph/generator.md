# Graph Generator

The `generator` module provides the `GraphGenerator` class for generating
complete causal graphs from variable specifications using LLMs.

## Import Pattern

```python
from causaliq_knowledge.graph import (
    GraphGenerator,
    GraphGeneratorConfig,
)
```

## Overview

`GraphGenerator` orchestrates the full graph generation workflow:

1. Load and validate model specifications
2. Apply view filtering (minimal/standard/rich context)
3. Optionally disguise variable names
4. Build prompts for LLM queries
5. Execute queries with caching support
6. Parse and validate responses
7. Return structured `GeneratedGraph` objects

## Configuration

### GraphGeneratorConfig

Configuration dataclass for graph generation parameters.

```python
from causaliq_knowledge.graph import GraphGeneratorConfig

config = GraphGeneratorConfig(
    view_level="standard",
    output_format="edge_list",
    disguise=False,
    disguise_seed=None,
    cache_enabled=True,
    confidence_threshold=0.0
)
```

**Attributes:**

- `view_level` (str): Context level - "minimal", "standard", or "rich"
- `output_format` (str): Response format - "edge_list" or "adjacency_matrix"
- `disguise` (bool): Whether to disguise variable names
- `disguise_seed` (int, optional): Seed for reproducible disguising
- `cache_enabled` (bool): Whether to use response caching
- `confidence_threshold` (float): Minimum confidence for including edges

## Basic Usage

### Creating a Generator

```python
from causaliq_knowledge.graph import GraphGenerator, GraphGeneratorConfig
from causaliq_knowledge.llm import LLMClient

# Create with default configuration
generator = GraphGenerator()

# Create with custom configuration
config = GraphGeneratorConfig(
    view_level="rich",
    output_format="edge_list",
    disguise=True,
    disguise_seed=42
)
generator = GraphGenerator(config=config)

# Create with a specific LLM client
client = LLMClient.create("gemini", model="gemini-2.0-flash")
generator = GraphGenerator(client=client)
```

### Generating from Variables

```python
from causaliq_knowledge.graph import GraphGenerator

generator = GraphGenerator()

# Generate a graph from variable names
graph = generator.generate_graph(
    variables=["smoking", "lung_cancer", "age", "genetics"],
    domain="oncology",
    model="gemini-2.0-flash"
)

# Access results
for edge in graph.edges:
    print(f"{edge.source} -> {edge.target} ({edge.confidence:.2f})")
```

### Generating from Model Specification

```python
from causaliq_knowledge.graph import GraphGenerator, ModelLoader

generator = GraphGenerator()

# Load a model specification
spec = ModelLoader.load("research/models/cancer/cancer.json")

# Generate using the specification
graph = generator.generate_from_spec(
    spec=spec,
    model="gemini-2.0-flash"
)
```

## LLM Provider Support

GraphGenerator supports all LLM providers available in causaliq-knowledge:

```python
from causaliq_knowledge.graph import GraphGenerator

generator = GraphGenerator()

# Use different providers
graph_gemini = generator.generate_graph(
    variables=["A", "B", "C"],
    domain="example",
    model="gemini-2.0-flash"
)

graph_openai = generator.generate_graph(
    variables=["A", "B", "C"],
    domain="example",
    model="gpt-4o"
)

graph_anthropic = generator.generate_graph(
    variables=["A", "B", "C"],
    domain="example",
    model="claude-sonnet-4-20250514"
)
```

## Caching

GraphGenerator integrates with `TokenCache` for response caching:

```python
from causaliq_knowledge.graph import GraphGenerator, GraphGeneratorConfig
from causaliq_knowledge.cache import TokenCache

# Enable caching (default)
config = GraphGeneratorConfig(cache_enabled=True)
generator = GraphGenerator(config=config)

# Provide a custom cache
cache = TokenCache(db_path="my_cache.db")
generator = GraphGenerator(config=config, cache=cache)

# Disable caching
config = GraphGeneratorConfig(cache_enabled=False)
generator = GraphGenerator(config=config)
```

Cache keys are generated to distinguish graph generation queries from
edge-by-edge queries, ensuring proper cache isolation.

## Variable Disguising

To counteract potential LLM memorisation of known causal relationships:

```python
from causaliq_knowledge.graph import GraphGenerator, GraphGeneratorConfig

config = GraphGeneratorConfig(
    disguise=True,
    disguise_seed=42  # For reproducibility
)
generator = GraphGenerator(config=config)

# Variable names are disguised before sending to LLM
# and automatically mapped back in the response
graph = generator.generate_graph(
    variables=["smoking", "lung_cancer"],
    domain="oncology",
    model="gemini-2.0-flash"
)

# Results use original variable names
print(graph.edges[0].source)  # "smoking", not "VAR_A"
```

## View Levels

Control the amount of context provided to the LLM:

| Level | Description |
|-------|-------------|
| `minimal` | Variable names only |
| `standard` | Names, types, and brief descriptions |
| `rich` | Full descriptions, roles, states, and constraints |

```python
from causaliq_knowledge.graph import GraphGenerator, GraphGeneratorConfig

# Minimal context - tests LLM's inherent knowledge
config = GraphGeneratorConfig(view_level="minimal")

# Rich context - provides maximum guidance
config = GraphGeneratorConfig(view_level="rich")
```

## API Reference

::: causaliq_knowledge.graph.generator
    options:
      show_root_heading: false
      members:
        - GraphGeneratorConfig
        - GraphGenerator
