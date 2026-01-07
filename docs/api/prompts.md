# Prompts Module

The `prompts` module provides prompt templates and utilities for querying
LLMs about causal relationships between variables.

## Overview

This module contains:

- **EdgeQueryPrompt**: A dataclass for building prompts to query edge existence and orientation
- **parse_edge_response**: A function to parse LLM JSON responses into `EdgeKnowledge` objects
- **Template constants**: Pre-defined prompt templates for system and user messages

## EdgeQueryPrompt

::: causaliq_knowledge.llm.prompts.EdgeQueryPrompt
    options:
      show_root_heading: true
      show_source: true
      members:
        - build
        - from_context

### Usage Example

```python
from causaliq_knowledge.llm import EdgeQueryPrompt
from causaliq_knowledge.llm import GroqClient, GroqConfig

# Create a prompt for querying the relationship between two variables
prompt = EdgeQueryPrompt(
    node_a="smoking",
    node_b="lung_cancer",
    domain="medicine",
    descriptions={
        "smoking": "Tobacco consumption frequency",
        "lung_cancer": "Diagnosis of lung cancer",
    },
)

# Build the system and user prompts
system_prompt, user_prompt = prompt.build()

# Use with GroqClient
config = GroqConfig(model="llama-3.1-8b-instant")
client = GroqClient(config=config)
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]
json_data, response = client.complete_json(messages)
```

### Using from_context

The `from_context` class method provides a convenient way to create
prompts from a context dictionary, which is the format used by
`KnowledgeProvider.query_edge()`:

```python
context = {
    "domain": "economics",
    "descriptions": {
        "interest_rate": "Central bank interest rate",
        "inflation": "Consumer price index change",
    },
}

prompt = EdgeQueryPrompt.from_context(
    node_a="interest_rate",
    node_b="inflation",
    context=context,
)
```

## parse_edge_response

::: causaliq_knowledge.llm.prompts.parse_edge_response
    options:
      show_root_heading: true
      show_source: true

### Usage Example

```python
from causaliq_knowledge.llm import (
    EdgeQueryPrompt,
    GroqClient,
    GroqConfig,
    parse_edge_response,
)

# Create client and prompt
config = GroqConfig(model="llama-3.1-8b-instant")
client = GroqClient(config=config)
prompt = EdgeQueryPrompt("X", "Y", domain="statistics")
system, user = prompt.build()

# Query the LLM
messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": user},
]
json_data, response = client.complete_json(messages)

# Parse the response into EdgeKnowledge
knowledge = parse_edge_response(json_data, model="groq/llama-3.1-8b-instant")

print(f"Edge exists: {knowledge.exists}")
print(f"Direction: {knowledge.direction}")
print(f"Confidence: {knowledge.confidence}")
print(f"Reasoning: {knowledge.reasoning}")
```

## Prompt Templates

The module exports several template constants that can be customized:

### DEFAULT_SYSTEM_PROMPT

The default system prompt instructs the LLM to act as a causal reasoning
expert and respond with structured JSON.

### USER_PROMPT_TEMPLATE

Basic user prompt template for querying edge relationships without
domain context.

### USER_PROMPT_WITH_DOMAIN_TEMPLATE

User prompt template that includes domain context for more accurate
responses.

### VARIABLE_DESCRIPTIONS_TEMPLATE

Template addition for including variable descriptions in the prompt.

## Custom System Prompts

You can provide a custom system prompt to `EdgeQueryPrompt`:

```python
custom_system = """You are a biomedical expert.
Assess causal relationships based on established medical literature.
Respond with JSON: {"exists": bool, "direction": str, "confidence": float, "reasoning": str}
"""

prompt = EdgeQueryPrompt(
    node_a="gene_X",
    node_b="protein_Y",
    domain="molecular_biology",
    system_prompt=custom_system,
)
```
