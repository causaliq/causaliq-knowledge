# Architecture Vision for causaliq-knowledge

## CausalIQ Ecosystem

causaliq-knowledge is a component of the overall [CausalIQ ecosystem architecture](https://causaliq.org/projects/ecosystem_architecture/).

This package provides **knowledge services** to other CausalIQ packages, enabling them to incorporate LLM-derived and human-specified knowledge into causal discovery and inference workflows.

## Architectural Principles 

### Simplicity First
- Use lightweight libraries over heavy frameworks
- Start with minimal viable features, extend incrementally
- Prefer explicit code over framework "magic"

### Cost Efficiency
- Built-in cost tracking and budget management (critical for independent research)
- Caching of LLM queries and responses to avoid redundant API calls
- Support for cheap/free local models alongside cloud providers

### Transparency and Reproducibility
- Cache all LLM interactions for experiment reproducibility
- Provide reasoning/explanations with all knowledge outputs
- Log confidence levels to enable uncertainty-aware decisions

### Clean Interfaces
- Abstract `KnowledgeProvider` interface allows multiple implementations
- LLM-based, rule-based, and human-input knowledge sources use same interface
- Easy integration with causaliq-analysis and causaliq-discovery

## Architecture Components

### Core Components (v0.1.0)

```
causaliq_knowledge/
├── __init__.py              # Package exports
├── cli.py                   # Command-line interface
├── base.py                  # Abstract KnowledgeProvider interface
├── models.py                # Pydantic models (EdgeKnowledge, etc.)
├── llm/
│   ├── __init__.py
│   ├── client.py            # LiteLLM wrapper with configuration
│   ├── prompts.py           # Prompt templates for edge queries
│   └── providers.py         # LLMKnowledge implementation
└── utils/
    └── __init__.py
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                 Consuming Package (e.g., causaliq-analysis)     │
│                                                                 │
│   uncertain_edges = df[df["entropy"] > threshold]               │
│                          │                                      │
│                          ▼                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              causaliq-knowledge                          │   │
│   │                                                          │   │
│   │   knowledge.query_edge("smoking", "cancer")              │   │
│   │       │                                                  │   │
│   │       ▼                                                  │   │
│   │   ┌───────────┐    ┌───────────┐    ┌───────────┐       │   │
│   │   │  LLM 1    │    │  LLM 2    │    │  Cache    │       │   │
│   │   │ (GPT-4o)  │    │ (Llama3)  │    │ (disk)    │       │   │
│   │   └───────────┘    └───────────┘    └───────────┘       │   │
│   │       │                 │                │               │   │
│   │       └─────────────────┴────────────────┘               │   │
│   │                         │                                │   │
│   │                         ▼                                │   │
│   │               EdgeKnowledge(                             │   │
│   │                   exists=True,                           │   │
│   │                   direction="a_to_b",                    │   │
│   │                   confidence=0.85,                       │   │
│   │                   reasoning="Established medical..."     │   │
│   │               )                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                          │                                      │
│                          ▼                                      │
│   Combine with statistical probabilities                        │
└─────────────────────────────────────────────────────────────────┘
```

## Technology Choices

### LiteLLM over LangChain (for v0.1.0 - v0.3.0)

| Requirement | LiteLLM | LangChain |
|-------------|---------|-----------|
| Multi-provider unified API | ✅ 100+ providers | ✅ Many providers |
| Built-in cost tracking | ✅ Yes | ❌ OpenAI only |
| Built-in caching | ✅ disk/redis/semantic | ✅ Various |
| Complexity | Low | High |
| Package size | ~5MB | ~100MB+ |

**Rationale**: For simple, structured queries about edge existence/orientation, LiteLLM provides everything needed with less complexity. LangChain may be added in v0.4.0+ when RAG capabilities are needed for literature context.

### Key Dependencies

- **litellm**: Unified LLM API with cost tracking
- **pydantic**: Structured response validation
- **diskcache** (v0.3.0): Persistent query caching

## Integration Points

### With causaliq-analysis

The primary integration point is the `average()` function which produces edge probability tables. Future versions will accept a `knowledge` parameter:

```python
# Future usage (v0.5.0 of causaliq-analysis)
from causaliq_knowledge import LLMKnowledge

knowledge = LLMKnowledge(models=["gpt-4o-mini"])
df = average(traces, sample_size=1000, knowledge=knowledge)
```

### With causaliq-discovery

Structure learning algorithms will use knowledge to guide search in uncertain areas of the graph space.

## See Also

- [LLM Integration Design Note](llm_integration.md) - Detailed design for LLM queries
- [Roadmap](../roadmap.md) - Release planning