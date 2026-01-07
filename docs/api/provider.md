# LLM Knowledge Provider

The `LLMKnowledge` class is the main entry point for querying LLMs about
causal relationships. It implements the `KnowledgeProvider` interface and
supports multi-model consensus using vendor-specific API clients.

## Architecture

`LLMKnowledge` uses **direct vendor-specific API clients** rather than wrapper
libraries like LiteLLM or LangChain. Currently supported providers:

- **Groq**: Fast inference for open-source models (free tier)
- **Gemini**: Google's Gemini models (generous free tier)

## Usage

```python
from causaliq_knowledge.llm import LLMKnowledge

# Single model (default: Groq)
provider = LLMKnowledge()

# Query about a potential edge
result = provider.query_edge("smoking", "lung_cancer")
print(f"Exists: {result.exists}")
print(f"Direction: {result.direction}")
print(f"Confidence: {result.confidence}")

# Multi-model consensus
provider = LLMKnowledge(
    models=["groq/llama-3.1-8b-instant", "gemini/gemini-2.5-flash"],
    consensus_strategy="weighted_vote",
)
result = provider.query_edge(
    "exercise",
    "heart_health",
    context={"domain": "medicine"},
)
```

## Model Identifiers

Models are specified with a provider prefix:

| Provider | Format | Example |
|----------|--------|---------|
| Groq | `groq/<model>` | `groq/llama-3.1-8b-instant` |
| Gemini | `gemini/<model>` | `gemini/gemini-2.5-flash` |

## LLMKnowledge

::: causaliq_knowledge.llm.provider.LLMKnowledge
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - query_edge
        - get_stats
        - name
        - models
        - consensus_strategy

## Consensus Strategies

When using multiple models, responses are combined using a consensus
strategy.

### weighted_vote

The default strategy. Combines responses by:

1. **Existence**: Weighted vote by confidence (True, False, or None)
2. **Direction**: Weighted majority among agreeing models
3. **Confidence**: Average confidence of agreeing models
4. **Reasoning**: Combined from all models

::: causaliq_knowledge.llm.provider.weighted_vote
    options:
      show_root_heading: true
      show_source: false

### highest_confidence

Simply returns the response with the highest confidence score.

::: causaliq_knowledge.llm.provider.highest_confidence
    options:
      show_root_heading: true
      show_source: false

## Example: Multi-Model Comparison

```python
from causaliq_knowledge.llm import LLMKnowledge

# Query multiple models (Groq + Gemini)
provider = LLMKnowledge(
    models=["groq/llama-3.1-8b-instant", "gemini/gemini-2.5-flash"],
    consensus_strategy="weighted_vote",
    temperature=0.1,  # Low temperature for consistency
)

# Query with domain context
result = provider.query_edge(
    node_a="interest_rate",
    node_b="inflation",
    context={
        "domain": "macroeconomics",
        "descriptions": {
            "interest_rate": "Central bank policy rate",
            "inflation": "Year-over-year CPI change",
        },
    },
)

print(f"Combined result: {result.exists} ({result.direction})")
print(f"Confidence: {result.confidence:.2f}")
print(f"Reasoning: {result.reasoning}")

# Check usage stats
stats = provider.get_stats()
print(f"Total cost: ${stats['total_cost']:.4f}")
```

## Using Local Models (Free)

```python
# Use Ollama for free local inference
# First: install Ollama and run `ollama pull llama3`
provider = LLMKnowledge(models=["ollama/llama3"])

# Or mix local and cloud models
provider = LLMKnowledge(
    models=["ollama/llama3", "gpt-4o-mini"],
    consensus_strategy="weighted_vote",
)
```
