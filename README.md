# causaliq-knowledge

[![Python Support](https://img.shields.io/pypi/pyversions/zenodo-sync.svg)](https://pypi.org/project/zenodo-sync/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The CausalIQ Knowledge project represents a novel approach to causal discovery by combining the traditional statistical structure learning algorithms with the contextual understanding and reasoning capabilities of Large Language Models. This integration enables more interpretable, domain-aware, and human-friendly causal discovery workflows. It is part of the [CausalIQ ecosystem](https://causaliq.org/) for intelligent causal discovery.

## Status

🚧 **Active Development** - this repository is currently in active development, which involves:

- adding new knowledge features, in particular knowledge from LLMs
- migrating functionality which provides knowledge based on standard reference networks from the legacy monolithic discovery repo
- ensure CausalIQ development standards are met


## Quick Start

```python
from causaliq_knowledge import LLMKnowledge

# Query an LLM about a potential causal relationship
knowledge = LLMKnowledge(models=["gpt-4o-mini"])
result = knowledge.query_edge("smoking", "lung_cancer")

print(f"Exists: {result.exists}, Direction: {result.direction}")
print(f"Confidence: {result.confidence}")
print(f"Reasoning: {result.reasoning}")
```

## Features

Under development:

- **Release v0.1.0 - Foundation LLM**: simple LLM queries to 1 or 2 LLMs about edge existence and orientation to support graph averaging

Currently implemented releases:

- none

Planned 

- **Release v0.2.0 - Additional LLMs**: support for more LLMs
- **Release v0.3.0 - LLM Caching**: caching of LLM query and responses
- **Release v0.4.0 - LLM Context**: variable/role/literature etc context
- **Release v0.5.0 - Algorithm integration**: integration into structure learning algorithms
- **Release v0.6.0 - Legacy Reference**: support for legacy approaches of deriving knowledge from reference networks

## Implementation Approach

### Technology Stack

- **[LiteLLM](https://github.com/BerriAI/litellm)**: Unified API for 100+ LLM providers with built-in cost tracking
- **[Pydantic](https://docs.pydantic.dev/)**: Structured response validation
- **LangChain** (v0.4.0+): RAG capabilities for literature context

### Why LiteLLM?

| Requirement | LiteLLM | LangChain |
|-------------|---------|-----------|
| Multi-provider unified API | ✅ | ✅ |
| Built-in cost tracking | ✅ | ❌ |
| Lightweight | ✅ (~5MB) | ❌ (~100MB+) |
| Complexity | Low | High |

For v0.1.0-v0.3.0 (simple edge queries), LiteLLM provides everything needed. LangChain will be added in v0.4.0 for RAG/context features.

### Supported LLM Providers

| Provider | Models | Free Tier |
|----------|--------|-----------|
| **Ollama** (local) | llama3, mistral, etc. | ✅ Free |
| **Groq** | llama3-70b, mixtral | ✅ Limited |
| **Google** | gemini-pro, gemini-flash | ✅ Generous |
| **OpenAI** | gpt-4o, gpt-4o-mini | ❌ Pay-per-use |
| **Anthropic** | claude-3-sonnet, haiku | ❌ Pay-per-use |

## Upcoming Key Innovations

### 🧠 LLMs support Causal Discovery and Inference
- initially LLM will work with **graph averaging** to resolve uncertain edges (use entropy to decide edges with uncertain existence or direction)
- integration into **structure learning** algorithms to provide knowledge for "uncertain" areas of the graph
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
- **cache queries and responses** so that experiments are stable and repeatable even if LLMs themselves are not
- **stable randomisation** of e.g. data sub-sampling

### 💰 Efficient use of LLM resources (important as an independent researcher)
- **cache queries and results** so that knowledge can be re-used
- evaluation and development of **simple context-adapted LLMs**


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

**Supported Python Versions**: 3.9, 3.10, 3.11, 3.12  
**Default Python Version**: 3.11
