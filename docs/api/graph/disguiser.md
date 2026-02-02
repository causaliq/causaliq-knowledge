# Variable Disguiser API Reference

Variable name obfuscation for LLM queries to counteract memorisation.

## Overview

The `VariableDisguiser` class creates reproducible mappings from original
variable names to disguised names (V1, V2, etc.) and provides methods to
translate between representations. This helps prevent LLMs from using
memorised knowledge about well-known benchmark datasets.

```python
from causaliq_knowledge.graph import VariableDisguiser
```

## VariableDisguiser

Obfuscate variable names for LLM queries.

### Constructor

```python
VariableDisguiser(
    spec: ModelSpec,
    seed: int | None = None,
    prefix: str = "V"
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spec` | `ModelSpec` | - | Model specification containing variables |
| `seed` | `int \| None` | `None` | Random seed for reproducible disguising |
| `prefix` | `str` | `"V"` | Prefix for disguised names |

**Example:**

```python
from causaliq_knowledge.graph import ModelLoader, VariableDisguiser

spec = ModelLoader.load("model.json")

# Without seed - random mapping each time
disguiser = VariableDisguiser(spec)

# With seed - reproducible mapping
disguiser = VariableDisguiser(spec, seed=42)

# Custom prefix
disguiser = VariableDisguiser(spec, seed=42, prefix="VAR")
```

### Properties

#### `seed -> int | None`

Return the random seed used for mapping.

```python
disguiser = VariableDisguiser(spec, seed=42)
print(disguiser.seed)  # 42
```

#### `prefix -> str`

Return the prefix used for disguised names.

```python
disguiser = VariableDisguiser(spec, prefix="VAR")
print(disguiser.prefix)  # "VAR"
```

#### `original_to_disguised -> dict[str, str]`

Return a copy of the original-to-disguised mapping.

```python
disguiser = VariableDisguiser(spec, seed=42)
mapping = disguiser.original_to_disguised
# {"smoking": "V2", "cancer": "V1", "age": "V3"}
```

#### `disguised_to_original -> dict[str, str]`

Return a copy of the disguised-to-original mapping.

```python
disguiser = VariableDisguiser(spec, seed=42)
mapping = disguiser.disguised_to_original
# {"V2": "smoking", "V1": "cancer", "V3": "age"}
```

### Methods

#### `disguise_name(original: str) -> str`

Convert an original variable name to its disguised form.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `original` | `str` | The original variable name |

**Returns:** `str` - The disguised variable name.

**Raises:** `KeyError` - If the original name is not in the mapping.

**Example:**

```python
disguiser = VariableDisguiser(spec, seed=42)
disguised = disguiser.disguise_name("smoking")
# "V2"
```

#### `reveal_name(disguised: str) -> str`

Convert a disguised variable name back to its original form.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `disguised` | `str` | The disguised variable name |

**Returns:** `str` - The original variable name.

**Raises:** `KeyError` - If the disguised name is not in the mapping.

**Example:**

```python
disguiser = VariableDisguiser(spec, seed=42)
original = disguiser.reveal_name("V2")
# "smoking"
```

#### `disguise_text(text: str) -> str`

Replace all original variable names in text with disguised names.

Replacement is case-insensitive. Longer names are replaced first to
prevent partial replacements.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Text containing original variable names |

**Returns:** `str` - Text with original names replaced by disguised names.

**Example:**

```python
disguiser = VariableDisguiser(spec, seed=42)

text = "Does smoking cause cancer?"
disguised = disguiser.disguise_text(text)
# "Does V2 cause V1?"
```

#### `reveal_text(text: str) -> str`

Replace all disguised variable names in text with original names.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Text containing disguised variable names |

**Returns:** `str` - Text with disguised names replaced by original names.

**Example:**

```python
disguiser = VariableDisguiser(spec, seed=42)

text = "V2 causes V1"
revealed = disguiser.reveal_text(text)
# "smoking causes cancer"
```

#### `disguise_names_list(names: list[str]) -> list[str]`

Convert a list of original names to disguised names.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `names` | `list[str]` | List of original variable names |

**Returns:** `list[str]` - List of disguised variable names.

**Example:**

```python
disguiser = VariableDisguiser(spec, seed=42)
disguised = disguiser.disguise_names_list(["smoking", "cancer"])
# ["V2", "V1"]
```

#### `reveal_names_list(names: list[str]) -> list[str]`

Convert a list of disguised names to original names.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `names` | `list[str]` | List of disguised variable names |

**Returns:** `list[str]` - List of original variable names.

**Example:**

```python
disguiser = VariableDisguiser(spec, seed=42)
original = disguiser.reveal_names_list(["V2", "V1"])
# ["smoking", "cancer"]
```

## Usage Patterns

### Disguising LLM Queries

```python
from causaliq_knowledge.graph import (
    ModelLoader,
    ViewFilter,
    PromptDetail,
    VariableDisguiser,
)

# Load and prepare
spec = ModelLoader.load("cancer.json")
view_filter = ViewFilter(spec)
disguiser = VariableDisguiser(spec, seed=42)

# Get filtered context
context = view_filter.get_context_summary(PromptDetail.STANDARD)

# Disguise variable names in context
for var in context["variables"]:
    var["name"] = disguiser.disguise_name(var["name"])

# Build prompt with disguised names
prompt = f"Generate a causal graph for: {context['variables']}"

# Send to LLM...
llm_response = "V1 -> V2, V2 -> V3"

# Translate response back to original names
original_response = disguiser.reveal_text(llm_response)
# "smoking -> cancer, cancer -> death"
```

### Reproducible Experiments

Using a seed ensures the same disguising across runs:

```python
# Run 1
disguiser1 = VariableDisguiser(spec, seed=42)
mapping1 = disguiser1.original_to_disguised

# Run 2 (different session)
disguiser2 = VariableDisguiser(spec, seed=42)
mapping2 = disguiser2.original_to_disguised

assert mapping1 == mapping2  # Same mapping
```

### Translating LLM Edge Responses

```python
# LLM returns edges with disguised names
llm_edges = [("V1", "V2"), ("V2", "V3")]

# Translate back to original names
original_edges = [
    (disguiser.reveal_name(src), disguiser.reveal_name(tgt))
    for src, tgt in llm_edges
]
# [("smoking", "cancer"), ("cancer", "death")]
```

### Custom Prefixes for Clarity

```python
# Use meaningful prefix for debugging
disguiser = VariableDisguiser(spec, seed=42, prefix="VAR")
print(disguiser.disguise_name("smoking"))
# "VAR2"

# Use short prefix for token efficiency
disguiser = VariableDisguiser(spec, seed=42, prefix="X")
print(disguiser.disguise_name("smoking"))
# "X2"
```

## Why Disguise Variables?

LLMs may have memorised causal relationships from well-known benchmark
datasets (ASIA, ALARM, SACHS, etc.). Disguising variable names helps:

1. **Prevent memorisation bias** - LLM must reason from provided context
2. **Test genuine reasoning** - Evaluate LLM's causal reasoning ability
3. **Fair benchmarking** - Compare LLMs without memorisation advantages
4. **Reproducible experiments** - Seed ensures consistent disguising

## Disguising Strategies

There are two complementary approaches to preventing LLM memorisation:

### 1. Semantic Disguising (Built into Model Specs)

Model specifications support **semantic renaming** through the `name` vs
`canonical_name` fields. Variables use meaningful names that differ from
the well-known benchmark names:

```json
{
    "name": "endemic_travel",
    "canonical_name": "asia",
    "short_description": "Travel to TB-endemic region"
}
```

Here, `endemic_travel` is meaningful and aids LLM reasoning, but differs
from the ASIA benchmark's original `asia` variable name that LLMs may
have memorised.

**When to use:** This is the default approach. Model specs should always
use semantically meaningful `name` fields that differ from `canonical_name`.
The LLM receives helpful context while avoiding memorised benchmark names.

### 2. Abstract Disguising (VariableDisguiser)

The `VariableDisguiser` class provides **abstract renaming** where all
semantic information is removed:

```python
disguiser = VariableDisguiser(spec, seed=42)
# "endemic_travel" → "V1"
# "tuberculosis" → "V2"
```

**When to use:** For experiments testing pure structural reasoning without
any semantic hints. This is a more aggressive approach that forces the LLM
to rely entirely on the provided descriptions and domain context, with no
help from variable name semantics.

### Choosing a Strategy

| Strategy | Variable Names | Use Case |
|----------|---------------|----------|
| Semantic | `endemic_travel`, `airway_inflammation` | Normal use - meaningful but non-canonical |
| Abstract | `V1`, `V2`, `V3` | Testing structural reasoning without semantic cues |

For most experiments, **semantic disguising is sufficient** - just ensure
your model spec uses meaningful names in the `name` field that differ from
well-known benchmark names stored in `canonical_name`.

Use `VariableDisguiser` when you want to test whether LLMs can infer
structure purely from descriptions, without any help from variable name
semantics.
