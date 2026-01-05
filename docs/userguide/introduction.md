# CausalIQ Knowledge User Guide

## What is CausalIQ Knowledge?

CausalIQ Knowledge is a Python package that provides **knowledge services** for causal discovery workflows. It enables you to query Large Language Models (LLMs) about potential causal relationships between variables, helping to resolve uncertainty in learned causal graphs.

## Primary Use Case

When averaging multiple causal graphs learned from data subsamples, some edges may be **uncertain** - appearing in some graphs but not others, or with inconsistent directions. CausalIQ Knowledge helps resolve this uncertainty by querying LLMs about whether:

1. A causal relationship exists between two variables
2. What the direction of causation is (A→B or B→A)

## Quick Start

### Installation

```bash
pip install causaliq-knowledge
```

### Basic Usage

```python
from causaliq_knowledge import LLMKnowledge, EdgeKnowledge

# Initialize with your preferred model
knowledge = LLMKnowledge(models=["gpt-4o-mini"])

# Query about a potential edge
result: EdgeKnowledge = knowledge.query_edge(
    node_a="smoking",
    node_b="lung_cancer",
    context={"domain": "epidemiology"}
)

print(f"Exists: {result.exists}")
print(f"Direction: {result.direction}")
print(f"Confidence: {result.confidence}")
print(f"Reasoning: {result.reasoning}")
```

### Using Local Models (Free)

```python
# Use Ollama for free local inference
# First: install Ollama and run `ollama pull llama3`
knowledge = LLMKnowledge(models=["ollama/llama3"])
```

### Multi-Model Consensus

```python
# Query multiple models for more robust answers
knowledge = LLMKnowledge(
    models=["gpt-4o-mini", "ollama/llama3"],
    consensus_strategy="weighted_vote"
)
```

## LLM Provider Setup

CausalIQ Knowledge uses [LiteLLM](https://github.com/BerriAI/litellm) to support 100+ LLM providers through a unified interface. You need API access to at least one provider:

### Free Options

| Provider | Setup |
|----------|-------|
| **Ollama** (local) | Install from [ollama.ai](https://ollama.ai), run `ollama pull llama3` |
| **Groq** | Sign up at [console.groq.com](https://console.groq.com), set `GROQ_API_KEY` |
| **Google Gemini** | Sign up at [makersuite.google.com](https://makersuite.google.com), set `GEMINI_API_KEY` |

### Paid Options (Pay-per-use)

| Provider | Setup | Cost |
|----------|-------|------|
| **OpenAI** | Sign up at [platform.openai.com](https://platform.openai.com), set `OPENAI_API_KEY` | ~$0.15/1M tokens (mini) |
| **Anthropic** | Sign up at [console.anthropic.com](https://console.anthropic.com), set `ANTHROPIC_API_KEY` | ~$0.25/1M tokens (haiku) |

## What's Next?

- [Architecture Overview](../architecture/overview.md) - Understand how the package works
- [LLM Integration Design](../architecture/llm_integration.md) - Detailed design documentation
- [API Reference](../api/overview.md) - Full API documentation