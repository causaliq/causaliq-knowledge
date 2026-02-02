# Graph Prompts API Reference

Prompt templates for LLM graph generation queries.

## Overview

This module provides prompt builders for generating complete causal
graphs from variable specifications. These are distinct from the
edge-by-edge queries in the `llm.prompts` module.

```python
from causaliq_knowledge.graph.prompts import (
    GraphQueryPrompt,
    OutputFormat,
)
```

## OutputFormat

Enumeration of output formats for graph generation responses.

```python
class OutputFormat(str, Enum):
    EDGE_LIST = "edge_list"
    ADJACENCY_MATRIX = "adjacency_matrix"
```

**Values:**

| Value | Description |
|-------|-------------|
| `EDGE_LIST` | Graph represented as a list of edges with source, target, and confidence |
| `ADJACENCY_MATRIX` | Graph represented as a matrix where entry (i,j) is the confidence that variable i causes variable j |

**Example:**

```python
from causaliq_knowledge.graph.prompts import OutputFormat

fmt = OutputFormat.EDGE_LIST
print(fmt.value)  # "edge_list"
```

## GraphQueryPrompt

Builder for graph generation query prompts. Constructs system and user
prompts for querying an LLM to generate a complete causal graph from
variable specifications.

### Constructor

```python
GraphQueryPrompt(
    variables: list[dict[str, Any]],
    level: PromptDetail = PromptDetail.STANDARD,
    domain: Optional[str] = None,
    output_format: OutputFormat = OutputFormat.EDGE_LIST,
    system_prompt: Optional[str] = None,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `variables` | `list[dict[str, Any]]` | Required | List of filtered variable dictionaries |
| `level` | `PromptDetail` | `STANDARD` | The prompt detail level (minimal, standard, rich) |
| `domain` | `Optional[str]` | `None` | Optional domain context for the query |
| `output_format` | `OutputFormat` | `EDGE_LIST` | Desired output format |
| `system_prompt` | `Optional[str]` | `None` | Custom system prompt (uses default if None) |

**Example:**

```python
from causaliq_knowledge.graph import ModelLoader, ViewFilter, PromptDetail
from causaliq_knowledge.graph.prompts import GraphQueryPrompt

spec = ModelLoader.load("model.json")
view_filter = ViewFilter(spec)
variables = view_filter.filter_variables(PromptDetail.STANDARD)

prompt = GraphQueryPrompt(
    variables=variables,
    level=PromptDetail.STANDARD,
    domain=spec.domain,
)
```

### Methods

#### build

Build the system and user prompts for the LLM query.

```python
def build(self) -> tuple[str, str]
```

**Returns:**

A tuple of `(system_prompt, user_prompt)` strings ready for use with an
LLM client.

**Example:**

```python
prompt = GraphQueryPrompt(
    variables=variables,
    level=PromptDetail.STANDARD,
)
system, user = prompt.build()

# Use with an LLM client
response = client.query(system_prompt=system, user_prompt=user)
```

#### get_variable_names

Get the list of variable names from the filtered variables.

```python
def get_variable_names(self) -> list[str]
```

**Returns:**

List of variable names extracted from the variables dictionaries.

**Example:**

```python
prompt = GraphQueryPrompt(variables=variables, level=PromptDetail.MINIMAL)
names = prompt.get_variable_names()
# ["age", "income", "education", ...]
```

#### from_model_spec (class method)

Create a `GraphQueryPrompt` directly from a `ModelSpec`. This is a
convenience method that handles view filtering automatically.

```python
@classmethod
def from_model_spec(
    cls,
    spec: ModelSpec,
    level: PromptDetail = PromptDetail.STANDARD,
    output_format: OutputFormat = OutputFormat.EDGE_LIST,
    system_prompt: Optional[str] = None,
) -> GraphQueryPrompt
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spec` | `ModelSpec` | Required | The model specification |
| `level` | `PromptDetail` | `STANDARD` | The prompt detail level |
| `output_format` | `OutputFormat` | `EDGE_LIST` | Desired output format |
| `system_prompt` | `Optional[str]` | `None` | Custom system prompt |

**Returns:**

A `GraphQueryPrompt` instance configured from the model specification.

**Example:**

```python
from causaliq_knowledge.graph import ModelLoader, PromptDetail
from causaliq_knowledge.graph.prompts import GraphQueryPrompt

spec = ModelLoader.load("model.json")
prompt = GraphQueryPrompt.from_model_spec(
    spec,
    level=PromptDetail.RICH,
)
system, user = prompt.build()
```

## Response Schemas

The module provides JSON schemas for validating LLM responses.

### Edge List Response Schema

```python
EDGE_LIST_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["edges"],
    "properties": {
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["source", "target"],
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
            },
        },
        "reasoning": {"type": "string"},
    },
}
```

**Example Response:**

```json
{
  "edges": [
    {"source": "age", "target": "income", "confidence": 0.8},
    {"source": "education", "target": "income", "confidence": 0.9}
  ],
  "reasoning": "Age and education both influence earning potential."
}
```

### Adjacency Matrix Response Schema

```python
ADJACENCY_MATRIX_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["variables", "adjacency_matrix"],
    "properties": {
        "variables": {
            "type": "array",
            "items": {"type": "string"},
        },
        "adjacency_matrix": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
        "reasoning": {"type": "string"},
    },
}
```

**Example Response:**

```json
{
  "variables": ["age", "education", "income"],
  "adjacency_matrix": [
    [0.0, 0.0, 0.8],
    [0.0, 0.0, 0.9],
    [0.0, 0.0, 0.0]
  ],
  "reasoning": "Age and education both influence income directly."
}
```

## System Prompts

The module provides default system prompts for different output formats:

- `GRAPH_SYSTEM_PROMPT_EDGE_LIST`: Instructions for edge list format
- `GRAPH_SYSTEM_PROMPT_ADJACENCY`: Instructions for adjacency matrix format

These prompts instruct the LLM to:

- Respond with valid JSON only
- Include only direct causal relationships
- Provide confidence scores from 0.0 to 1.0
- Consider domain knowledge and temporal ordering
- Avoid self-loops

## User Prompt Templates

User prompts are selected based on the `PromptDetail`:

| Level | Without Domain | With Domain |
|-------|---------------|-------------|
| `MINIMAL` | Variable names only | Variable names with domain context |
| `STANDARD` | Names, types, descriptions | Same with domain context |
| `RICH` | Full metadata including roles, categories, hints | Same with domain context |

## Complete Example

```python
from causaliq_knowledge.graph import ModelLoader, PromptDetail
from causaliq_knowledge.graph.prompts import GraphQueryPrompt, OutputFormat

# Load model specification
spec = ModelLoader.load("research/models/health_model.json")

# Create prompt with rich context
prompt = GraphQueryPrompt.from_model_spec(
    spec,
    level=PromptDetail.RICH,
    output_format=OutputFormat.EDGE_LIST,
)

# Build prompts
system_prompt, user_prompt = prompt.build()

# Get variable names for result validation
variable_names = prompt.get_variable_names()

# Use with your LLM client
# response = llm_client.query(system=system_prompt, user=user_prompt)
```
