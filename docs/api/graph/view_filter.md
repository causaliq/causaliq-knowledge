# View Filter API Reference

Filtering network context to extract specific context levels for LLM queries.

## Overview

The `ViewFilter` class extracts variable information according to view
definitions (minimal, standard, rich) from a network context. This
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

Filter network context to extract specific view levels.

### Constructor

```python
ViewFilter(context: NetworkContext, *, use_llm_names: bool = True)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `context` | `NetworkContext` | - | The network context to filter |
| `use_llm_names` | `bool` | `True` | Output llm_name as 'name' field |

**Example:**

```python
from causaliq_knowledge.graph import NetworkContext, ViewFilter

context = NetworkContext.load("model.json")
view_filter = ViewFilter(context)

# Use benchmark names instead of LLM names
view_filter = ViewFilter(context, use_llm_names=False)
```

### Properties

#### `context -> NetworkContext`

Return the network context.

```python
filter = ViewFilter(context)
print(filter.context.domain)
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

filter = ViewFilter(context)

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
filter = ViewFilter(context)
var = context.variables[0]

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
filter = ViewFilter(context)

# Get minimal view of all variables
minimal_vars = filter.filter_variables(PromptDetail.MINIMAL)
# [{"name": "smoking"}, {"name": "cancer"}, ...]

# Get rich view with full context
rich_vars = filter.filter_variables(PromptDetail.RICH)
```

#### `get_variable_names() -> list[str]`

Get all variable names from the context.

**Returns:** `list[str]` - List of variable names.

```python
filter = ViewFilter(context)
names = filter.get_variable_names()
# ["smoking", "cancer", "age"]
```

#### `get_domain() -> str`

Get the domain from the context.

**Returns:** `str` - The domain string.

```python
filter = ViewFilter(context)
domain = filter.get_domain()
# "epidemiology"
```

#### `get_context_summary(level: PromptDetail) -> dict`

Get a complete context summary for LLM prompts.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `level` | `PromptDetail` | The prompt detail level for variable filtering |

**Returns:** `dict` - Dictionary with domain, network, and filtered variables.

**Example:**

```python
filter = ViewFilter(context)
summary = filter.get_context_summary(PromptDetail.STANDARD)

# {
#     "domain": "epidemiology",
#     "network": "cancer",
#     "variables": [
#         {"name": "smoking", "type": "binary", ...},
#         {"name": "cancer", "type": "binary", ...},
#     ]
# }
```

## Usage Patterns

### Generating LLM Context

```python
from causaliq_knowledge.graph import NetworkContext, ViewFilter, PromptDetail
import json

# Load network context and create filter
context = NetworkContext.load("model.json")
filter = ViewFilter(context)

# Get context for LLM prompt
summary = filter.get_context_summary(PromptDetail.STANDARD)

# Format for prompt
prompt = f"""
Domain: {summary['domain']}

Variables:
{json.dumps(summary['variables'], indent=2)}

Please generate a causal graph for these variables.
"""
```

### Comparing Prompt Detail Levels

```python
from causaliq_knowledge.graph import ViewFilter, PromptDetail

filter = ViewFilter(context)

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

Network context can define custom prompt detail configurations:

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
