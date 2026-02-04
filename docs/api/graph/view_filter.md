# View Filter API Reference

Filtering model specifications to extract specific context levels for LLM queries.

## Overview

The `ViewFilter` class extracts variable information according to view
definitions (minimal, standard, rich) from a model specification. This
allows controlling how much context is provided to LLMs.

```python
from causaliq_knowledge.graph import ViewFilter, PromptDetail
```

## PromptDetail

Enumeration of context levels for filtering.

```python
class PromptDetail(str, Enum):
    MINIMAL = "minimal"   # Variable names only
    STANDARD = "standard" # Names with basic descriptions
    RICH = "rich"         # Full metadata and context
```

**Example:**

```python
from causaliq_knowledge.graph import PromptDetail

level = PromptDetail.STANDARD
print(level.value)  # "standard"
```

## ViewFilter

Filter model specifications to extract specific view levels.

### Constructor

```python
ViewFilter(spec: ModelSpec)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `spec` | `ModelSpec` | The model specification to filter |

**Example:**

```python
from causaliq_knowledge.graph import ModelLoader, ViewFilter

spec = ModelLoader.load("model.json")
view_filter = ViewFilter(spec)
```

### Properties

#### `spec -> ModelSpec`

Return the model specification.

```python
filter = ViewFilter(spec)
print(filter.spec.domain)
```

### Methods

#### `get_include_fields(level: PromptDetail) -> list[str]`

Get the fields to include for a given view level.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `level` | `PromptDetail` | The prompt detail level |

**Returns:** `list[str]` - Field names to include.

**Example:**

```python
from causaliq_knowledge.graph import ViewFilter, PromptDetail

filter = ViewFilter(spec)

# Get fields for minimal view
fields = filter.get_include_fields(PromptDetail.MINIMAL)
# ["name"]

# Get fields for standard view
fields = filter.get_include_fields(PromptDetail.STANDARD)
# ["name", "type", "short_description", "states"]
```

#### `filter_variable(variable: VariableSpec, level: PromptDetail) -> dict`

Filter a single variable to include only specified fields.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `variable` | `VariableSpec` | The variable to filter |
| `level` | `PromptDetail` | The prompt detail level |

**Returns:** `dict` - Dictionary with only the included fields.

**Example:**

```python
filter = ViewFilter(spec)
var = spec.variables[0]

minimal = filter.filter_variable(var, PromptDetail.MINIMAL)
# {"name": "smoking"}

standard = filter.filter_variable(var, PromptDetail.STANDARD)
# {"name": "smoking", "type": "binary", "short_description": "..."}
```

#### `filter_variables(level: PromptDetail) -> list[dict]`

Filter all variables to the specified view level.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `level` | `PromptDetail` | The prompt detail level |

**Returns:** `list[dict]` - List of filtered variable dictionaries.

**Example:**

```python
filter = ViewFilter(spec)

# Get minimal view of all variables
minimal_vars = filter.filter_variables(PromptDetail.MINIMAL)
# [{"name": "smoking"}, {"name": "cancer"}, ...]

# Get rich view with full context
rich_vars = filter.filter_variables(PromptDetail.RICH)
```

#### `get_variable_names() -> list[str]`

Get all variable names from the specification.

**Returns:** `list[str]` - List of variable names.

```python
filter = ViewFilter(spec)
names = filter.get_variable_names()
# ["smoking", "cancer", "age"]
```

#### `get_domain() -> str`

Get the domain from the specification.

**Returns:** `str` - The domain string.

```python
filter = ViewFilter(spec)
domain = filter.get_domain()
# "epidemiology"
```

#### `get_context_summary(level: PromptDetail) -> dict`

Get a complete context summary for LLM prompts.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `level` | `PromptDetail` | The prompt detail level for variable filtering |

**Returns:** `dict` - Dictionary with domain, dataset_id, and filtered variables.

**Example:**

```python
filter = ViewFilter(spec)
summary = filter.get_context_summary(PromptDetail.STANDARD)

# {
#     "domain": "epidemiology",
#     "dataset_id": "cancer",
#     "variables": [
#         {"name": "smoking", "type": "binary", ...},
#         {"name": "cancer", "type": "binary", ...},
#     ]
# }
```

## Usage Patterns

### Generating LLM Context

```python
from causaliq_knowledge.graph import ModelLoader, ViewFilter, PromptDetail
import json

# Load model and create filter
spec = ModelLoader.load("model.json")
filter = ViewFilter(spec)

# Get context for LLM prompt
context = filter.get_context_summary(PromptDetail.STANDARD)

# Format for prompt
prompt = f"""
Domain: {context['domain']}

Variables:
{json.dumps(context['variables'], indent=2)}

Please generate a causal graph for these variables.
"""
```

### Comparing Prompt Detail Levels

```python
from causaliq_knowledge.graph import ViewFilter, PromptDetail

filter = ViewFilter(spec)

# Compare information at different levels
for level in PromptDetail:
    vars = filter.filter_variables(level)
    fields = set()
    for v in vars:
        fields.update(v.keys())
    print(f"{level.value}: {sorted(fields)}")

# minimal: ['name']
# standard: ['name', 'short_description', 'states', 'type']
# rich: ['extended_description', 'name', 'role', 'short_description', ...]
```

### Custom Prompt Details

Models can define custom prompt detail configurations:

```json
{
    "prompt_details": {
        "minimal": {
            "include_fields": ["name"]
        },
        "standard": {
            "include_fields": ["name", "type", "short_description"]
        },
        "rich": {
            "include_fields": [
                "name", "type", "role", "short_description",
                "extended_description", "sensitivity_hints"
            ]
        }
    }
}
```
