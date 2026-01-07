# Testing Strategy Design Note

## Overview

Testing LLM-dependent code presents unique challenges: API calls cost money, responses are non-deterministic, and external services may be unavailable. This document describes the testing strategy for causaliq-knowledge.

## Testing Pyramid

```
                    ┌─────────────────┐
                    │   Functional    │  ← Cached responses (v0.3.0+)
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
# tests/unit/test_llm_providers.py
import pytest
from unittest.mock import MagicMock


def test_query_edge_parses_valid_response(monkeypatch):
    """Test that valid LLM JSON is correctly parsed."""
    from causaliq_knowledge.llm import LLMKnowledge
    from causaliq_knowledge.llm.groq_client import GroqClient

    # Mock the Groq client's complete_json method
    mock_json = {
        "exists": True,
        "direction": "a_to_b",
        "confidence": 0.85,
        "reasoning": "Smoking causes lung cancer via carcinogens."
    }
    
    mock_client = MagicMock(spec=GroqClient)
    mock_client.complete_json.return_value = (mock_json, MagicMock())
    
    knowledge = LLMKnowledge(models=["groq/llama-3.1-8b-instant"])
    knowledge._clients["groq/llama-3.1-8b-instant"] = mock_client
    
    result = knowledge.query_edge("smoking", "lung_cancer")
    
    assert result.exists is True
    assert result.direction.value == "a_to_b"
    assert result.confidence == 0.85


def test_query_edge_handles_malformed_json(monkeypatch):
    """Test graceful handling of invalid LLM response."""
    from causaliq_knowledge.llm import LLMKnowledge
    from causaliq_knowledge.llm.groq_client import GroqClient

    # Mock returning None (failed parse)
    mock_client = MagicMock(spec=GroqClient)
    mock_client.complete_json.return_value = (None, MagicMock())
    
    knowledge = LLMKnowledge(models=["groq/llama-3.1-8b-instant"])
    knowledge._clients["groq/llama-3.1-8b-instant"] = mock_client
    
    result = knowledge.query_edge("A", "B")
    
    assert result.exists is None  # Uncertain
    assert result.confidence == 0.0
```

### 2. Integration Tests (Optional, Manual or CI with Secrets)

Integration tests use real LLM APIs to validate actual behavior:

- **Expensive**: May cost money per call (though free tiers available)
- **Non-deterministic**: LLM responses vary
- **Slow**: Network latency
- **Validates real integration**: Catches API changes

```python
# tests/integration/test_llm_live.py
import pytest
import os

pytestmark = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set"
)

@pytest.mark.slow
@pytest.mark.integration
def test_groq_returns_valid_response():
    """Validate real Groq API returns parseable response."""
    from causaliq_knowledge.llm import LLMKnowledge
    
    knowledge = LLMKnowledge(models=["groq/llama-3.1-8b-instant"])
    result = knowledge.query_edge("smoking", "lung_cancer")
    
    # Don't assert specific values - LLM may vary
    # Just validate structure and reasonable bounds
    assert result.exists in [True, False, None]
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.reasoning) > 0
```

### 3. Functional Tests with Cached Responses (v0.3.0+)

Once response caching is implemented, we can create **reproducible functional tests** using cached LLM responses. This is the best of both worlds:

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
│   1. Run queries against real LLMs                               │
│   2. Cache stores responses in tests/data/functional/cache/     │
│   3. Commit cache files to git                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Functional Tests (CI)                        │
│                                                                  │
│   1. Load cached responses from tests/data/functional/cache/    │
│   2. LLMKnowledge configured to use cache-only mode             │
│   3. Tests run with real LLM responses, no API calls            │
└─────────────────────────────────────────────────────────────────┘
```

#### Example Functional Test

```python
# tests/functional/test_edge_queries.py
import pytest
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "data" / "cache"

@pytest.fixture
def cached_knowledge():
    """LLMKnowledge using only cached responses."""
    return LLMKnowledge(
        models=["groq/llama-3.1-8b-instant"],
        cache_dir=str(CACHE_DIR),
        cache_only=True  # Fail if cache miss, don't call API
    )

def test_smoking_cancer_relationship(cached_knowledge):
    """Test with cached response for smoking->cancer query."""
    result = cached_knowledge.query_edge("smoking", "lung_cancer")
    
    # Can assert specific values since response is cached
    assert result.exists is True
    assert result.direction == "a_to_b"
    assert result.confidence > 0.8

def test_consensus_across_models(cached_knowledge):
    """Test multi-model consensus with cached responses."""
    knowledge = LLMKnowledge(
        models=["groq/llama-3.1-8b-instant", "gemini/gemini-2.5-flash"],
        cache_dir=str(CACHE_DIR),
        cache_only=True
    )
    result = knowledge.query_edge("exercise", "heart_health")
    
    assert result.exists is True
```

#### Generating Test Fixtures

```python
# scripts/generate_test_fixtures.py
"""
Run this script manually to generate/update cached responses for functional tests.
Requires API keys for all models being tested.
"""
from causaliq_knowledge.llm import LLMKnowledge
from pathlib import Path

CACHE_DIR = Path("tests/data/functional/cache")
TEST_EDGES = [
    ("smoking", "lung_cancer"),
    ("exercise", "heart_health"),
    ("education", "income"),
    ("rain", "wet_ground"),
]

def generate_fixtures():
    knowledge = LLMKnowledge(
        models=["groq/llama-3.1-8b-instant", "gemini/gemini-2.5-flash"],
        cache_dir=str(CACHE_DIR)
    )
    
    for node_a, node_b in TEST_EDGES:
        print(f"Caching: {node_a} -> {node_b}")
        knowledge.query_edge(node_a, node_b)
    
    print(f"Fixtures saved to {CACHE_DIR}")

if __name__ == "__main__":
    generate_fixtures()
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
| **Unit** | Every push/PR | ❌ Not needed | Free |
| **Functional** | Every push/PR | ❌ Uses cache | Free |
| **Integration** | Main branch only, optional | ✅ GitHub Secrets | ~$0.01/run |

```yaml
# .github/workflows/ci.yml (conceptual addition)
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
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: pytest tests/integration -m integration
```

## Test Data Management

### Directory Structure

```
tests/
├── __init__.py
├── unit/
│   ├── __init__.py
│   ├── test_models.py          # EdgeKnowledge, etc.
│   ├── test_prompts.py         # Prompt formatting
│   └── test_llm_providers.py   # Mocked LLM calls
├── integration/
│   ├── __init__.py
│   └── test_llm_live.py        # Real API calls
├── functional/
│   ├── __init__.py
│   └── test_edge_queries.py    # Using cached responses
└── data/
    └── functional/
        └── cache/              # Committed to git
            ├── groq/
            │   ├── smoking_lung_cancer.json
            │   └── exercise_heart_health.json
            └── gemini/
                └── ...
```

### Cache File Format

```json
{
  "query": {
    "node_a": "smoking",
    "node_b": "lung_cancer",
    "context": {"domain": "epidemiology"}
  },
  "model": "groq/llama-3.1-8b-instant",
  "timestamp": "2026-01-05T10:30:00Z",
  "response": {
    "exists": true,
    "direction": "a_to_b",
    "confidence": 0.92,
    "reasoning": "Smoking is an established cause of lung cancer..."
  }
}
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

Cache files should include model version to detect when responses might change due to model updates.

### Semantic Similarity Testing

For v0.4.0+, consider testing that semantically similar queries hit cache (e.g., "smoking" vs "tobacco use" → "cancer" vs "lung cancer").
