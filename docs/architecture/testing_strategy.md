# Testing Strategy Design Note

## Overview

Testing LLM-dependent code presents unique challenges: API calls cost money,
responses are non-deterministic, and external services may be unavailable.
This document describes the testing strategy for causaliq-knowledge.

## Testing Pyramid

```
                    ┌─────────────────┐
                    │   Functional    │  ← Cached responses
                    │     Tests       │     Real scenarios, reproducible
                    └────────┬────────┘
               ┌─────────────┴─────────────┐
               │     Integration Tests     │  ← Real API calls (optional)
               │   (with live LLM APIs)    │     Expensive, non-deterministic
               └─────────────┬─────────────┘
    ┌────────────────────────┴────────────────────────┐
    │                  Unit Tests                      │  ← Mocked LLM responses
    │           (mocked LLM responses)                 │     Fast, free, deterministic
    └──────────────────────────────────────────────────┘
```

## Test Categories

### 1. Unit Tests (Always Run in CI)

Unit tests mock all LLM calls, making them:

- **Fast**: No network latency
- **Free**: No API costs
- **Deterministic**: Same result every time
- **Isolated**: No external dependencies

```python
# tests/unit/graph/test_generator.py
import pytest
from unittest.mock import MagicMock


# Test graph generator creates edges from LLM response.
def test_generator_creates_edges_from_response(monkeypatch):
    """Test that valid LLM JSON is correctly parsed into edges."""
    from causaliq_knowledge.graph import GraphGenerator, GraphGeneratorConfig
    from causaliq_knowledge.graph import ModelSpec, VariableSpec

    # Create test model spec
    spec = ModelSpec(
        name="test",
        variables=[
            VariableSpec(id="A", name="smoking"),
            VariableSpec(id="B", name="cancer"),
        ],
    )

    config = GraphGeneratorConfig(
        llm_model="groq/llama-3.1-8b-instant",
        prompt_detail="standard",
    )

    # Mock the LLM client
    mock_response = {
        "edges": [
            {"source": "A", "target": "B", "confidence": 0.85}
        ]
    }

    generator = GraphGenerator(config)
    # Mock internal client call
    generator._client = MagicMock()
    generator._client.complete_json.return_value = (mock_response, None)

    result = generator.generate(spec)

    assert len(result.edges) == 1
    assert result.edges[0].source == "A"
    assert result.edges[0].target == "B"
```

### 2. Integration Tests (Optional, Manual or CI with Secrets)

Integration tests use real LLM APIs to validate actual behaviour:

- **Expensive**: May cost money per call (though free tiers available)
- **Non-deterministic**: LLM responses vary
- **Slow**: Network latency
- **Validates real integration**: Catches API changes

```python
# tests/integration/test_graph_generation_live.py
import pytest
import os

pytestmark = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set"
)


@pytest.mark.slow
@pytest.mark.integration
def test_groq_generates_valid_graph():
    """Validate real Groq API returns parseable graph."""
    from causaliq_knowledge.graph import GraphGenerator, GraphGeneratorConfig
    from causaliq_knowledge.graph import ModelLoader

    loader = ModelLoader()
    spec = loader.load("tests/data/simple_model.json")

    config = GraphGeneratorConfig(
        llm_model="groq/llama-3.1-8b-instant",
        prompt_detail="standard",
    )

    generator = GraphGenerator(config)
    result = generator.generate(spec)

    # Don't assert specific values - LLM may vary
    # Just validate structure and reasonable bounds
    assert len(result.edges) >= 0
    for edge in result.edges:
        assert 0.0 <= edge.confidence <= 1.0
```

### 3. Functional Tests with Cached Responses

Functional tests use cached LLM responses for reproducible testing:

- **Realistic**: Uses actual LLM responses (captured once)
- **Deterministic**: Same cached response every time
- **Free**: No API calls after initial capture
- **Fast**: Disk read instead of network call

#### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                     Test Fixture Generation                      │
│                      (run once, manually)                        │
│                                                                  │
│   1. Run graph generation against real LLMs                      │
│   2. Cache stores responses in tests/data/functional/cache/     │
│   3. Commit cache files to git                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Functional Tests (CI)                        │
│                                                                  │
│   1. Load cached responses from tests/data/functional/cache/    │
│   2. GraphGenerator configured to use cache-only mode           │
│   3. Tests run with real LLM responses, no API calls            │
└─────────────────────────────────────────────────────────────────┘
```

## CI Configuration

### pytest Markers

```toml
# pyproject.toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests requiring live external APIs",
    "functional: marks functional tests using cached responses",
]
addopts = "-ra -q --strict-markers -m 'not slow and not integration'"
```

### GitHub Actions Strategy

| Test Type | When | API Keys | Cost |
|-----------|------|----------|------|
| **Unit** | Every push/PR | No | Free |
| **Functional** | Every push/PR | Uses cache | Free |
| **Integration** | Main branch only, optional | GitHub Secrets | ~$0.01/run |

```yaml
# .github/workflows/ci.yml (conceptual)
jobs:
  unit-and-functional:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run unit and functional tests
        run: pytest tests/unit tests/functional

  integration:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'  # Only on main
    steps:
      - uses: actions/checkout@v4
      - name: Run integration tests
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
        run: pytest tests/integration -m integration
```

## Test Data Management

### Directory Structure

```
tests/
├── __init__.py
├── unit/
│   ├── __init__.py
│   ├── graph/
│   │   ├── test_generator.py   # Graph generation
│   │   ├── test_models.py      # Model specs
│   │   └── test_loader.py      # JSON loading
│   └── llm/
│       ├── test_clients.py     # LLM clients
│       └── test_config.py      # Configuration
├── integration/
│   ├── __init__.py
│   └── test_graph_live.py      # Real API calls
├── functional/
│   ├── __init__.py
│   └── test_graph_cached.py    # Using cached responses
└── data/
    ├── model_specs/            # Test model specifications
    │   ├── simple.json
    │   └── cancer.json
    └── functional/
        └── cache/              # Committed to git
            └── groq/
                └── simple_graph.json
```

## Benefits of This Strategy

| Benefit | How Achieved |
|---------|--------------|
| **Fast CI** | Unit tests are mocked, functional use cache |
| **Low cost** | Only integration tests (optional) call APIs |
| **Reproducible** | Cached responses are deterministic |
| **Realistic** | Functional tests use real LLM responses |
| **Stable experiments** | Same cache = same results across runs |
| **Version controlled** | Cache files in git track response changes |

## Future Considerations

### Cache Invalidation for Tests

When updating test fixtures:

1. Delete relevant cache files
2. Run fixture generation script
3. Review new responses
4. Commit updated cache files

### Model Version Tracking

Cache files should include model version to detect when responses might
change due to model updates.
