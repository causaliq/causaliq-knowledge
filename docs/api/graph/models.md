# Model Specification API Reference

Pydantic models for defining causal model specifications in JSON format.

## Overview

Model specifications define the variables and metadata for a causal model,
enabling LLMs to generate causal graphs with appropriate domain context.

```python
from causaliq_knowledge.graph import (
    ModelSpec,
    VariableSpec,
    VariableType,
    VariableRole,
    PromptDetails,
    ViewDefinition,
)
```

## VariableType

Enumeration of supported variable types.

```python
class VariableType(str, Enum):
    BINARY = "binary"           # Two states (e.g., yes/no)
    CATEGORICAL = "categorical" # Multiple unordered states
    ORDINAL = "ordinal"         # Multiple ordered states
    CONTINUOUS = "continuous"   # Numeric values
```

**Example:**

```python
from causaliq_knowledge.graph import VariableType

var_type = VariableType.BINARY
print(var_type.value)  # "binary"
```

## VariableRole

Enumeration of causal roles in the graph structure.

```python
class VariableRole(str, Enum):
    EXOGENOUS = "exogenous"   # No parents (root cause)
    ENDOGENOUS = "endogenous" # Has parents (caused by other variables)
    LATENT = "latent"         # Unobserved variable
```

**Example:**

```python
from causaliq_knowledge.graph import VariableRole

role = VariableRole.EXOGENOUS
print(role.value)  # "exogenous"
```

## VariableSpec

Specification for a single variable in the causal model.

### Attributes

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | Yes | Primary identifier used in experiments |
| `type` | `VariableType` | Yes | Variable type (binary, categorical, etc.) |
| `canonical_name` | `str` | No | Original benchmark name (for evaluation) |
| `display_name` | `str` | No | Human-readable display name |
| `aliases` | `list[str]` | No | Alternative names for the variable |
| `states` | `list[str]` | No | Possible values for discrete variables |
| `role` | `VariableRole` | No | Causal role (exogenous, endogenous, latent) |
| `category` | `str` | No | Domain-specific category |
| `short_description` | `str` | No | Brief description of the variable |
| `extended_description` | `str` | No | Detailed description with domain context |
| `base_rate` | `dict[str, float]` | No | Prior probabilities for each state |
| `conditional_rates` | `dict` | No | Conditional probabilities |
| `sensitivity_hints` | `str` | No | Hints about causal relationships |
| `related_domain_knowledge` | `list[str]` | No | Domain knowledge statements |
| `references` | `list[str]` | No | Literature references |

### Example

```python
from causaliq_knowledge.graph import VariableSpec, VariableType, VariableRole

smoking = VariableSpec(
    name="smoking_status",
    type=VariableType.BINARY,
    states=["never", "ever"],
    role=VariableRole.EXOGENOUS,
    short_description="Patient has history of tobacco smoking.",
    extended_description="Self-reported smoking history, known risk factor.",
    base_rate={"never": 0.7, "ever": 0.3},
)
```

## ViewDefinition

Configuration for a single context view level.

### Attributes

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `description` | `str` | No | Human-readable description of the view |
| `include_fields` | `list[str]` | Yes | Variable fields to include in this view |

### Example

```python
from causaliq_knowledge.graph import ViewDefinition

minimal_view = ViewDefinition(
    description="Variable names only",
    include_fields=["name"]
)

standard_view = ViewDefinition(
    description="Names with basic metadata",
    include_fields=["name", "type", "short_description", "states"]
)
```

## PromptDetails

Container for the three standard prompt detail levels.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `minimal` | `ViewDefinition` | Minimal context (names only) |
| `standard` | `ViewDefinition` | Standard context (names + descriptions) |
| `rich` | `ViewDefinition` | Rich context (full metadata) |

### Default Prompt Details

If not specified, the following defaults are used:

```python
PromptDetails(
    minimal=ViewDefinition(include_fields=["name"]),
    standard=ViewDefinition(
        include_fields=["name", "type", "short_description", "states"]
    ),
    rich=ViewDefinition(
        include_fields=[
            "name", "type", "role", "short_description",
            "extended_description", "states", "sensitivity_hints"
        ]
    ),
)
```

## Provenance

Provenance information for the model specification.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `source_network` | `str` | Name of the source benchmark network |
| `source_reference` | `str` | Citation for the original source |
| `source_url` | `str` | URL to the source data |
| `disguise_strategy` | `str` | Strategy used for variable name disguising |
| `memorization_risk` | `str` | Risk level for LLM memorisation |
| `notes` | `str` | Additional notes about the source |

## LLMGuidance

Guidance for LLM interactions with the model.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `usage_notes` | `list[str]` | Notes about using this model with LLMs |
| `do_not_provide` | `list[str]` | Information to withhold from LLMs |
| `expected_difficulty` | `str` | Expected difficulty level |

## Constraints

Structural constraints on the causal graph.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `forbidden_edges` | `list[list[str]]` | Edges that must not exist |
| `required_edges` | `list[list[str]]` | Edges that must exist |
| `partial_order` | `list[list[str]]` | Temporal ordering constraints |
| `causal_principles` | `list[CausalPrinciple]` | Domain causal principles |

## GroundTruth

Ground truth edges for evaluation.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `edges_expert` | `list[list[str]]` | Expert-defined edges |
| `edges_experiment` | `list[list[str]]` | Experimentally-derived edges |
| `edges_observational` | `list[list[str]]` | Observationally-derived edges |

## ModelSpec

Complete model specification combining all components.

### Attributes

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `schema_version` | `str` | No | Schema version (default: "2.0") |
| `dataset_id` | `str` | Yes | Unique identifier for the dataset |
| `domain` | `str` | Yes | Domain of the model (e.g., "epidemiology") |
| `purpose` | `str` | No | Purpose of the model |
| `variables` | `list[VariableSpec]` | Yes | List of variable specifications |
| `provenance` | `Provenance` | No | Source and provenance information |
| `llm_guidance` | `LLMGuidance` | No | Guidance for LLM interactions |
| `prompt_details` | `PromptDetails` | No | Prompt detail definitions (uses defaults if omitted) |
| `constraints` | `Constraints` | No | Structural constraints |
| `ground_truth` | `GroundTruth` | No | Ground truth for evaluation |

### Methods

#### `get_variable_names() -> list[str]`

Return list of all variable names.

```python
spec = ModelLoader.load("model.json")
names = spec.get_variable_names()
# ["smoking", "cancer", "age"]
```

#### `get_variable(name: str) -> VariableSpec | None`

Get a variable specification by name.

```python
spec = ModelLoader.load("model.json")
smoking = spec.get_variable("smoking")
```

#### `get_exogenous_variables() -> list[VariableSpec]`

Get all variables with exogenous role.

```python
spec = ModelLoader.load("model.json")
root_causes = spec.get_exogenous_variables()
```

### Example

```python
from causaliq_knowledge.graph import (
    ModelSpec,
    VariableSpec,
    VariableType,
    VariableRole,
)

spec = ModelSpec(
    dataset_id="smoking_cancer",
    domain="epidemiology",
    purpose="Causal model for smoking and cancer",
    variables=[
        VariableSpec(
            name="smoking",
            type=VariableType.BINARY,
            role=VariableRole.EXOGENOUS,
            short_description="Smoking status",
        ),
        VariableSpec(
            name="cancer",
            type=VariableType.BINARY,
            role=VariableRole.ENDOGENOUS,
            short_description="Cancer diagnosis",
        ),
    ],
)
```

## JSON Schema

Model specifications are typically stored as JSON files. See
[Model Specification Format](../../userguide/model_specification.md) for
the complete JSON schema and examples.
