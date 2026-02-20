# Architecture Vision for causaliq-knowledge

## CausalIQ Ecosystem

causaliq-knowledge is a component of the overall [CausalIQ ecosystem architecture](https://causaliq.org/projects/ecosystem_architecture/).

This package provides **knowledge services** to other CausalIQ packages, enabling them to incorporate LLM-derived and human-specified knowledge into causal discovery and inference workflows.

## Architectural Principles 

### Simplicity First

- Use lightweight libraries over heavy frameworks
- Start with minimal viable features, extend incrementally
- Prefer explicit code over framework "magic"
- Use vendor-specific APIs rather than abstraction wrappers

### Cost Efficiency

- Built-in cost tracking and budget management (critical for independent research)
- Caching of LLM queries and responses to avoid redundant API calls
- Support for cheap/free providers (Groq, Gemini free tiers)

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
└── llm/
    ├── __init__.py          # LLM module exports
    ├── groq_client.py       # Direct Groq API client
    ├── gemini_client.py     # Direct Google Gemini API client
    ├── prompts.py           # Prompt templates for edge queries
    └── provider.py          # LLMKnowledge implementation
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

### Graph Generation Components (v0.4.0)

```
causaliq_knowledge/
├── action.py                # CausalIQ workflow action integration
├── cli/                     # Command-line interface
│   ├── main.py              # Core CLI entry point
│   ├── cache.py             # Cache management commands
│   ├── generate.py          # Graph generation commands
│   └── models.py            # Model listing command
└── graph/
    ├── __init__.py          # Module exports
    ├── models.py            # NetworkContext, VariableSpec, etc.
    ├── generator.py         # GraphGenerator class
    ├── view_filter.py       # ViewFilter for context levels
    ├── prompts.py           # GraphQueryPrompt builder
    ├── response.py          # GeneratedGraph, ProposedEdge
    ├── params.py            # GenerateGraphParams
    └── cache.py             # GraphCompressor for workflow cache
```

### Graph Generation Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Graph Generation Flow                         │
│                                                                 │
│   network_context.json                                          │
│       │                                                         │
│       ▼                                                         │
│   ┌───────────────┐                                             │
│   │NetworkContext │  Load and validate JSON context             │
│   │   .load()     │                                             │
│   └───────────────┘                                             │
│       │                                                         │
│       ▼                                                         │
│   ┌───────────────┐    ┌─────────────────────┐                  │
│   │  ViewFilter   │───▶│ PromptDetail.MINIMAL │  Names only     │
│   │               │    │ PromptDetail.STANDARD│  + descriptions │
│   │               │    │ PromptDetail.RICH    │  Full metadata  │
│   └───────────────┘    └─────────────────────┘                  │
│       │                                                         │
│       ▼                                                         │
│   name/llm_name mapping:  smoke → tobacco_history               │
│   (via VariableSpec.llm_name field)                             │
│       │                                                         │
│       ▼                                                         │
│   ┌───────────────┐                                             │
│   │GraphGenerator │  Generate causal graph                      │
│   │  + LLM Client │  (multiple provider support)                │
│   │  + TokenCache │  (cached responses)                         │
│   └───────────────┘                                             │
│       │                                                         │
│       ▼                                                         │
│   llm_name → name mapping:  tobacco_history → smoke             │
│   (automatic translation in response)                           │
│       │                                                         │
│       ▼                                                         │
│   GeneratedGraph(edges=[("smoke", "cancer"), ...])              │
│       │                                                         │
│       ▼                                                         │
│   ┌───────────────┐                                             │
│   │GraphCompressor│  Compress for Workflow Cache                │
│   └───────────────┘                                             │
└─────────────────────────────────────────────────────────────────┘
```

## Technology Choices

### Vendor-Specific APIs over Wrapper Libraries

We use **direct vendor-specific API clients** rather than wrapper libraries like LiteLLM or LangChain. This architectural decision provides:

| Aspect | Direct APIs | Wrapper Libraries |
|--------|-------------|-------------------|
| Reliability | ✅ Full control, predictable | ❌ Wrapper bugs, version drift |
| Debugging | ✅ Clear stack traces | ❌ Abstraction layers |
| Dependencies | ✅ Minimal (httpx only) | ❌ Heavy transitive deps |
| API Coverage | ✅ Full vendor features | ❌ Lowest common denominator |
| Maintenance | ✅ We control updates | ❌ Wait for wrapper updates |

**Why Not LiteLLM?**

- Adds 50+ transitive dependencies
- Version conflicts with other packages
- Wrapper bugs mask vendor API issues
- We only need 2-3 providers, not 100+

**Why Not LangChain?**

- Massive dependency footprint (~100MB+)
- Over-engineered for simple structured queries  
- Rapid breaking changes between versions
- May reconsider for v0.4.0+ RAG features only

### Current Provider Clients

- **GroqClient**: Direct Groq API via httpx (free tier, fast inference)
- **GeminiClient**: Direct Google Gemini API via httpx (generous free tier)

### Key Dependencies

- **httpx**: HTTP client for API calls
- **pydantic**: Structured response validation
- **click**: Command-line interface
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