# causaliq-knowledge

![Python Versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)

The CausalIQ Knowledge project represents a novel approach to causal discovery by combining the traditional statistical structure learning algorithms with the contextual understanding and reasoning capabilities of Large Language Models. This integration enables more interpretable, domain-aware, and human-friendly causal discovery workflows. It is part of the [CausalIQ ecosystem](https://causaliq.org/) for intelligent causal discovery.

## Status

🚧 **Active Development** - this repository is currently in active development, which involves:

- Adding new knowledge features, in particular knowledge from LLMs
- Migrating functionality which provides knowledge based on standard reference networks from the legacy monolithic discovery repo
- Ensuring CausalIQ development standards are met


## Quick Start

```python
from causaliq_knowledge.llm import LLMKnowledge

# Query an LLM about a potential causal relationship
knowledge = LLMKnowledge(models=["groq/llama-3.1-8b-instant"])
result = knowledge.query_edge("smoking", "lung_cancer")

print(f"Exists: {result.exists}, Direction: {result.direction}")
print(f"Confidence: {result.confidence}")
print(f"Reasoning: {result.reasoning}")
```

## Features

Currently implemented releases:

- **Release v0.1.0 - Foundation LLM**: Simple LLM queries to 1 or 2 LLMs about edge existence and orientation to support graph averaging
- **Release v0.2.0 - Additional LLMs**: Support for 7 LLM providers (Groq, Gemini, OpenAI, Anthropic, DeepSeek, Mistral, Ollama)
- **Release v0.3.0 - LLM Caching**: SQLite-based response caching with CLI tools for cache management
- **Release v0.4.0 - Graph Generation**: CLI and CausalIQ workflow action for LLM-generated causal graphs

Planned:

- **Release v0.5.0 - Graph Caching**: save generated graphs to Workflow caches
- **Release v0.6.0 - LLM Cost Tracking**: Query LLM provider APIs for usage and cost statistics
- **Release v0.7.0 - LLM Context**: Variable/role/literature etc context
- **Release v0.8.0 - Algorithm integration**: Integration into structure learning algorithms

## Implementation Approach

### Technology Stack

- **Vendor-Specific API Clients**: Direct integration with LLM providers using httpx
- **[Pydantic](https://docs.pydantic.dev/)**: Structured response validation
- **[Click](https://click.palletsprojects.com/)**: Command-line interface

### Why Vendor-Specific APIs (not LiteLLM/LangChain)?

We use **direct vendor-specific API clients** rather than wrapper libraries:

| Aspect | Direct APIs | Wrapper Libraries |
|--------|-------------|-------------------|
| Reliability | ✅ Full control | ❌ Wrapper bugs |
| Dependencies | ✅ Minimal (httpx) | ❌ Heavy (~50-100MB) |
| Debugging | ✅ Clear traces | ❌ Abstraction layers |
| Maintenance | ✅ We control | ❌ Wait for updates |

This approach keeps the package lightweight, reliable, and easy to debug.

### Supported LLM Providers

| Provider | Client | Models | Free Tier |
|----------|--------|--------|-----------|
| **Groq** | `GroqClient` | llama-3.1-8b-instant | ✅ Generous |
| **Google Gemini** | `GeminiClient` | gemini-2.5-flash | ✅ Generous |
| **OpenAI** | `OpenAIClient` | gpt-4o-mini | ❌ Paid |
| **Anthropic** | `AnthropicClient` | claude-sonnet-4-20250514 | ❌ Paid |
| **DeepSeek** | `DeepSeekClient` | deepseek-chat | ✅ Low cost |
| **Mistral** | `MistralClient` | mistral-small-latest | ❌ Paid |
| **Ollama** | `OllamaClient` | llama3 | ✅ Free (local) |

## Upcoming Key Innovations

### 🧠 LLMs support Causal Discovery and Inference

- Initially LLM will work with **graph averaging** to resolve uncertain edges (use entropy to decide edges with uncertain existence or direction)
- Integration into **structure learning** algorithms to provide knowledge for "uncertain" areas of the graph
- LLMs analyse learning process and errors to **suggest improved algorithms**
- LLMs used to preprocess **text and visual data** so they can be used as inputs to structure learning

### 🤝 Human Engagement

- **Natural language constraints**: Specify domain knowledge in plain English
- **Expert knowledge incorporation** by converting expert understanding into algorithmic constraints
- LLMs convert **natural language questions** to causal queries
- **Interactive causal discovery** where structure learning or LLMs identify areas of causal uncertainty and can test causal hypotheses through dialogue

### 🪟 Transparency and interpretability

- LLMs **interpret structure learning process** and outputs, including their uncertainties
- LLMs **interpret causal inference** results including uncertainties
- **Contextual graph interpretation** to explain variable meanings and relationships
- **Uncertainty communication** with clear explanation of confidence levels and limitations
- **Report generation** including automated research summaries and methodology descriptions

### 🔒 Stability and reproducibility

- **Cache queries and responses** so that experiments are stable and repeatable even if LLMs themselves are not
- **Stable randomisation** of e.g. data sub-sampling

### 💰 Efficient use of LLM resources (important as an independent researcher)

- **Cache queries and results** so that knowledge can be re-used
- Evaluation and development of **simple context-adapted LLMs**


## Upcoming Integration with CausalIQ Ecosystem

- 🔍 CausalIQ Discovery makes use of this package to learn more accurate graphs.
- 🧪 CausalIQ Analysis uses this package to explain the learning process, intelligently combine and explain results.
- 🔮 CausalIQ Predict uses this package to explain predictions made by learnt models.

## Documentation

- [User Guide](docs/userguide/introduction.md) - Getting started
- [Architecture Overview](docs/architecture/overview.md) - Design and components
- [LLM Integration Design](docs/architecture/llm_integration.md) - Detailed LLM design
- [Roadmap](docs/roadmap.md) - Release planning

---

**Supported Python Versions**: 3.9, 3.10, 3.11, 3.12, 3.13  
**Default Python Version**: 3.11
