# Response Models

The `response` module provides data models and parsing functions for LLM graph
generation responses. It handles both edge list and adjacency matrix formats
with robust JSON extraction.

## Import Pattern

```python
from causaliq_knowledge.graph import (
    ProposedEdge,
    GeneratedGraph,
    GenerationMetadata,
    parse_graph_response,
)
```

## Data Models

### ProposedEdge

A Pydantic model representing a single proposed causal edge from an LLM.

```python
from causaliq_knowledge.graph import ProposedEdge

edge = ProposedEdge(
    source="smoking",
    target="lung_cancer",
    confidence=0.95,
    reasoning="Well-established causal relationship from epidemiological studies"
)
```

**Attributes:**

- `source` (str): Name of the source (cause) variable
- `target` (str): Name of the target (effect) variable
- `confidence` (float): Confidence score between 0.0 and 1.0
- `reasoning` (str, optional): LLM's reasoning for proposing this edge

### GenerationMetadata

A dataclass containing metadata about the graph generation process.

```python
from causaliq_knowledge.graph import GenerationMetadata

metadata = GenerationMetadata(
    model_name="gemini-2.0-flash",
    prompt_tokens=450,
    completion_tokens=320,
    view_level="standard",
    disguised=False,
    output_format="edge_list"
)
```

**Attributes:**

- `model_name` (str): Name of the LLM model used
- `prompt_tokens` (int, optional): Number of tokens in the prompt
- `completion_tokens` (int, optional): Number of tokens in the response
- `view_level` (str, optional): Context level used (minimal/standard/rich)
- `disguised` (bool): Whether variable names were disguised
- `output_format` (str): Response format (edge_list/adjacency_matrix)

### GeneratedGraph

A dataclass representing a complete generated causal graph.

```python
from causaliq_knowledge.graph import GeneratedGraph, ProposedEdge

graph = GeneratedGraph(
    edges=[
        ProposedEdge(source="A", target="B", confidence=0.9),
        ProposedEdge(source="B", target="C", confidence=0.85),
    ],
    variables=["A", "B", "C"],
    reasoning="Based on temporal ordering and domain knowledge...",
    metadata=None
)
```

**Attributes:**

- `edges` (list[ProposedEdge]): List of proposed causal edges
- `variables` (list[str]): List of variable names in the graph
- `reasoning` (str, optional): Overall reasoning for the graph structure
- `metadata` (GenerationMetadata, optional): Generation metadata

## Parsing Functions

### parse_graph_response

Parse an LLM response string into a `GeneratedGraph` object.

```python
from causaliq_knowledge.graph import parse_graph_response

response_text = '''```json
{
    "edges": [
        {"source": "A", "target": "B", "confidence": 0.9},
        {"source": "B", "target": "C", "confidence": 0.85}
    ],
    "reasoning": "Based on causal principles..."
}
```'''

graph = parse_graph_response(
    response_text=response_text,
    variables=["A", "B", "C"],
    output_format="edge_list"
)
```

**Parameters:**

- `response_text` (str): Raw LLM response text (may include markdown)
- `variables` (list[str]): Expected variable names for validation
- `output_format` (str): Expected format ("edge_list" or "adjacency_matrix")

**Returns:** `GeneratedGraph` object

**Raises:** `ValueError` if JSON parsing fails or format is invalid

### Response Formats

The module supports two response formats:

#### Edge List Format

```json
{
    "edges": [
        {
            "source": "variable_a",
            "target": "variable_b",
            "confidence": 0.9,
            "reasoning": "Optional per-edge reasoning"
        }
    ],
    "reasoning": "Overall graph reasoning"
}
```

#### Adjacency Matrix Format

```json
{
    "variables": ["A", "B", "C"],
    "adjacency_matrix": [
        [0.0, 0.9, 0.0],
        [0.0, 0.0, 0.85],
        [0.0, 0.0, 0.0]
    ],
    "reasoning": "Overall graph reasoning"
}
```

Values in the adjacency matrix represent confidence scores. A value at
position `[i][j]` indicates an edge from `variables[i]` to `variables[j]`.

## API Reference

::: causaliq_knowledge.graph.response
    options:
      show_root_heading: false
      members:
        - ProposedEdge
        - GenerationMetadata
        - GeneratedGraph
        - parse_graph_response
