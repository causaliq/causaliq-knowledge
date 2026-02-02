# Model Loader API Reference

Loading and validation of model specification JSON files.

## Overview

The `ModelLoader` class provides static methods for loading model specifications
from JSON files or dictionaries, with comprehensive validation.

```python
from causaliq_knowledge.graph import ModelLoader, ModelLoadError
```

## ModelLoadError

Exception raised when model loading fails.

```python
class ModelLoadError(Exception):
    """Raised when a model specification cannot be loaded or validated."""
    pass
```

**Example:**

```python
from causaliq_knowledge.graph import ModelLoader, ModelLoadError

try:
    spec = ModelLoader.load("nonexistent.json")
except ModelLoadError as e:
    print(f"Failed to load model: {e}")
```

## ModelLoader

Static class for loading and validating model specifications.

### Methods

#### `load(path: str | Path) -> ModelSpec`

Load a model specification from a JSON file.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str \| Path` | Path to the JSON file |

**Returns:** `ModelSpec` - The loaded and validated model specification.

**Raises:** `ModelLoadError` - If the file cannot be read or validation fails.

**Example:**

```python
from pathlib import Path
from causaliq_knowledge.graph import ModelLoader

# Load from string path
spec = ModelLoader.load("models/cancer.json")

# Load from Path object
spec = ModelLoader.load(Path("models") / "cancer.json")

# Access loaded data
print(f"Domain: {spec.domain}")
print(f"Variables: {spec.get_variable_names()}")
```

#### `from_dict(data: dict) -> ModelSpec`

Create a model specification from a dictionary.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `dict` | Dictionary containing model specification |

**Returns:** `ModelSpec` - The validated model specification.

**Raises:** `ModelLoadError` - If validation fails.

**Example:**

```python
from causaliq_knowledge.graph import ModelLoader

data = {
    "dataset_id": "test",
    "domain": "epidemiology",
    "variables": [
        {"name": "smoking", "type": "binary"},
        {"name": "cancer", "type": "binary"},
    ],
}

spec = ModelLoader.from_dict(data)
```

#### `validate_variables(spec: ModelSpec) -> None`

Validate variable references in the model specification.

Checks that:

- All variable names are unique
- Edge references in constraints use valid variable names
- Ground truth edges reference valid variables

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `spec` | `ModelSpec` | The model specification to validate |

**Raises:** `ModelLoadError` - If validation fails.

**Example:**

```python
from causaliq_knowledge.graph import ModelLoader, ModelLoadError

spec = ModelLoader.from_dict(data)

try:
    ModelLoader.validate_variables(spec)
except ModelLoadError as e:
    print(f"Invalid variable references: {e}")
```

#### `load_and_validate(path: str | Path) -> ModelSpec`

Load a model specification and perform full validation.

This is the recommended method for loading models, as it performs both
file loading and comprehensive validation.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str \| Path` | Path to the JSON file |

**Returns:** `ModelSpec` - The fully validated model specification.

**Raises:** `ModelLoadError` - If loading or validation fails.

**Example:**

```python
from causaliq_knowledge.graph import ModelLoader

# Recommended way to load models
spec = ModelLoader.load_and_validate("models/cancer.json")
```

## Error Handling

The loader provides helpful error messages for common issues:

```python
from causaliq_knowledge.graph import ModelLoader, ModelLoadError

try:
    spec = ModelLoader.load_and_validate("model.json")
except ModelLoadError as e:
    # Error messages include:
    # - File not found errors
    # - JSON parsing errors with line numbers
    # - Validation errors with field names
    # - Duplicate variable name errors
    # - Invalid edge reference errors
    print(f"Error: {e}")
```

## Usage Patterns

### Loading Research Models

```python
from pathlib import Path
from causaliq_knowledge.graph import ModelLoader

# Load from research directory
models_dir = Path("research/models")
cancer_spec = ModelLoader.load(models_dir / "cancer" / "cancer.json")
asia_spec = ModelLoader.load(models_dir / "asia" / "asia.json")
```

### Programmatic Model Creation

```python
from causaliq_knowledge.graph import (
    ModelLoader,
    ModelSpec,
    VariableSpec,
    VariableType,
)

# Create model programmatically
spec = ModelSpec(
    dataset_id="my_model",
    domain="test",
    variables=[
        VariableSpec(name="X", type=VariableType.BINARY),
        VariableSpec(name="Y", type=VariableType.BINARY),
    ],
)

# Validate the model
ModelLoader.validate_variables(spec)
```
