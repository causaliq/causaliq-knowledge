# LLM Integration Design Note

## Overview

This document describes how causaliq-knowledge integrates with Large Language Models (LLMs) to provide knowledge about causal relationships. The primary use case for v0.1.0 is answering queries about **edge existence** and **edge orientation** to support graph averaging in causaliq-analysis.

## How it works

### Query Flow

1. **Consumer requests knowledge** about a potential edge (e.g., "Does smoking cause cancer?")
2. **KnowledgeProvider** receives the query with optional context
3. **LLM client** formats the query using structured prompts
4. **One or more LLMs** are queried (configurable)
5. **Responses are parsed** into structured `EdgeKnowledge` objects
6. **Multi-LLM consensus** combines responses (if multiple models used)
7. **Result returned** with confidence score and reasoning

### Core Interface

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel

class EdgeKnowledge(BaseModel):
    """Structured knowledge about a potential causal edge."""
    exists: bool | None           # True, False, or None (uncertain)
    direction: str | None         # "a_to_b", "b_to_a", "undirected", None
    confidence: float             # 0.0 to 1.0
    reasoning: str                # Human-readable explanation
    model: str | None = None      # Which LLM provided this (for logging)

class KnowledgeProvider(ABC):
    """Abstract interface for all knowledge sources."""
    
    @abstractmethod
    def query_edge(
        self,
        node_a: str,
        node_b: str,
        context: dict | None = None
    ) -> EdgeKnowledge:
        """
        Query whether a causal edge exists between two nodes.
        
        Args:
            node_a: Name of first variable
            node_b: Name of second variable  
            context: Optional context (domain, variable descriptions, etc.)
            
        Returns:
            EdgeKnowledge with existence, direction, confidence, reasoning
        """
        pass
```

### LLM Implementation

```python
class LLMKnowledge(KnowledgeProvider):
    """LLM-based knowledge provider using LiteLLM."""
    
    def __init__(
        self,
        models: list[str] = ["gpt-4o-mini"],
        consensus_strategy: str = "weighted_vote",
        cache_dir: str | None = None,
    ):
        """
        Initialize LLM knowledge provider.
        
        Args:
            models: List of LiteLLM model identifiers
                   e.g., ["gpt-4o-mini", "ollama/llama3", "claude-3-haiku"]
            consensus_strategy: How to combine multi-model responses
                               "weighted_vote" or "highest_confidence"
            cache_dir: Directory for caching responses (None = no cache)
        """
        ...
```

## LLM Provider Configuration

### Supported Providers (via LiteLLM)

| Provider | Model Examples | API Key Environment Variable |
|----------|---------------|------------------------------|
| OpenAI | `gpt-4o`, `gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `claude-3-sonnet`, `claude-3-haiku` | `ANTHROPIC_API_KEY` |
| Google | `gemini-pro`, `gemini-flash` | `GEMINI_API_KEY` |
| Groq | `groq/llama3-70b` | `GROQ_API_KEY` |
| Ollama (local) | `ollama/llama3`, `ollama/mistral` | None (local) |
| Together AI | `together_ai/mistral-7b` | `TOGETHER_API_KEY` |

### Cost Considerations

For edge queries (~500 tokens each):

| Model | Cost per 1000 queries | Quality | Speed |
|-------|----------------------|---------|-------|
| GPT-4o-mini | ~$0.08 | Good | Fast |
| Claude 3 Haiku | ~$0.13 | Good | Fast |
| Gemini Flash | Free tier | Good | Fast |
| Groq Llama3 | Free tier | Good | Very fast |
| Ollama (local) | $0 | Varies | Depends on hardware |

**Recommendation**: Start with Groq free tier or local Ollama for development, use GPT-4o-mini for production quality.

## Prompt Design

### Edge Existence Query

```
System: You are an expert in causal reasoning and domain knowledge. 
Your task is to assess whether a causal relationship exists between two variables.
Respond in JSON format with: exists (true/false/null), direction (a_to_b/b_to_a/undirected/null), confidence (0-1), reasoning (string).

User: In the domain of {domain}, does a causal relationship exist between "{node_a}" and "{node_b}"?
Consider:
- Direct causation (A causes B)
- Reverse causation (B causes A)  
- Bidirectional/feedback relationships
- No causal relationship (correlation only or independence)

Variable context:
{variable_descriptions}
```

### Response Format

```json
{
  "exists": true,
  "direction": "a_to_b",
  "confidence": 0.85,
  "reasoning": "Smoking is an established cause of lung cancer through well-documented biological mechanisms including DNA damage from carcinogens in tobacco smoke."
}
```

## Multi-LLM Consensus

When multiple models are configured, responses are combined:

### Weighted Vote Strategy (default)

```python
def weighted_vote(responses: list[EdgeKnowledge]) -> EdgeKnowledge:
    """Combine responses weighted by confidence."""
    # For existence: weighted majority vote
    # For direction: weighted majority among those agreeing on existence
    # Final confidence: average confidence of agreeing models
    # Reasoning: concatenate key points from each model
```

### Highest Confidence Strategy

```python
def highest_confidence(responses: list[EdgeKnowledge]) -> EdgeKnowledge:
    """Return response with highest confidence."""
    return max(responses, key=lambda r: r.confidence)
```

## Integration with Graph Averaging

The primary consumer is `causaliq_analysis.graph.average()`:

```python
# Current output from average()
df = average(traces, sample_size=1000)
# Returns: node_a, node_b, p_a_to_b, p_b_to_a, p_undirected, p_no_edge

# Entropy calculation identifies uncertain edges
def edge_entropy(row):
    probs = [row.p_a_to_b, row.p_b_to_a, row.p_undirected, row.p_no_edge]
    probs = [p for p in probs if p > 0]
    return -sum(p * math.log2(p) for p in probs)

df["entropy"] = df.apply(edge_entropy, axis=1)
uncertain_edges = df[df["entropy"] > 1.5]  # High uncertainty

# Query LLM for uncertain edges
knowledge = LLMKnowledge(models=["gpt-4o-mini"])
for _, row in uncertain_edges.iterrows():
    result = knowledge.query_edge(row.node_a, row.node_b)
    # Combine statistical and LLM probabilities...
```

## Design Rationale

### Why LiteLLM?

1. **Unified API**: Single interface for 100+ providers
2. **Built-in cost tracking**: Essential for budget-conscious research
3. **Lightweight**: ~5MB vs ~100MB+ for LangChain
4. **Simple**: No complex abstractions for straightforward queries

### Why structured JSON responses?

1. **Reliable parsing**: Avoids regex/heuristic extraction
2. **Validation**: Pydantic ensures response integrity
3. **Consistency**: Same structure regardless of model

### Why multi-model consensus?

1. **Reduced hallucination**: Multiple models catch individual errors
2. **Confidence calibration**: Agreement increases confidence
3. **Robustness**: Not dependent on single provider availability

## Error Handling and Resilience

### API Failures

- Automatic retry with exponential backoff (LiteLLM built-in)
- Fallback to next model in list if primary fails
- Return `EdgeKnowledge(exists=None, confidence=0.0)` if all fail

### Invalid Responses

- Pydantic validation catches malformed JSON
- Default to `exists=None` if parsing fails
- Log warnings for debugging

### Rate Limiting

- LiteLLM handles rate limit errors with automatic retry
- Configure `max_retries` and `timeout` per model

## Performance

### Latency

- Single query: 0.5-2s depending on model/provider
- Batch queries: Can parallelize across edges (async)
- Cached queries: <10ms

### Throughput (v0.3.0 with caching)

- First query to new edge: 1-2s
- Cached query: <10ms
- 1000 unique edges: ~20-30 minutes (sequential), ~5 min (parallel)

## Future Extensions

### v0.3.0: Caching
- Disk-based cache keyed by (node_a, node_b, context_hash)
- Semantic similarity cache for similar variable names

### v0.4.0: Rich Context
- Variable descriptions and roles
- Domain-specific literature retrieval (RAG)
- Conversation history for follow-up queries

### v0.5.0: Algorithm Integration
- Direct integration with structure learning search
- Knowledge-guided constraint generation
