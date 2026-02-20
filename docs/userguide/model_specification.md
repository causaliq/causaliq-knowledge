# Network Context Format

This guide describes the JSON format for network context files used
in causaliq-knowledge graph generation.

## Overview

Network context files define the variables, metadata, and constraints for
a causal network. They enable LLMs to generate causal graphs with appropriate
domain context while allowing control over how much information is provided.

## Minimal Example

The simplest valid network context:

```json
{
    "network": "simple",
    "domain": "test",
    "variables": [
        {"name": "X", "type": "binary"},
        {"name": "Y", "type": "binary"}
    ]
}
```

## Complete Example

A comprehensive network context with all optional fields:

```json
{
    "schema_version": "2.0",
    "network": "smoking_cancer",
    "domain": "epidemiology",
    "purpose": "Causal model for smoking and lung cancer relationship",
    
    "provenance": {
        "source_network": "LUNG",
        "source_reference": "Lauritzen & Spiegelhalter (1988)",
        "source_url": "https://example.com/lung-network",
        "memorization_risk": "high",
        "notes": "Well-known benchmark, LLMs may have memorised structure"
    },
    
    "llm_guidance": {
        "usage_notes": [
            "Focus on biological plausibility",
            "Consider temporal ordering of variables"
        ],
        "do_not_provide": [
            "Ground truth edges",
            "Canonical variable names"
        ],
        "expected_difficulty": "medium"
    },
    
    "prompt_details": {
        "minimal": {
            "description": "Variable names only",
            "include_fields": ["name"]
        },
        "standard": {
            "description": "Names with basic metadata",
            "include_fields": ["name", "type", "short_description", "states"]
        },
        "rich": {
            "description": "Full context for complex reasoning",
            "include_fields": [
                "name", "type", "role", "short_description",
                "extended_description", "states", "sensitivity_hints"
            ]
        }
    },
    
    "variables": [
        {
            "name": "smoke",
            "llm_name": "tobacco_history",
            "display_name": "Smoking Status",
            "type": "binary",
            "states": ["never", "ever"],
            "role": "exogenous",
            "category": "exposure",
            "short_description": "Patient tobacco smoking history.",
            "extended_description": "Self-reported lifetime smoking history. Known major risk factor for lung cancer with dose-response relationship.",
            "base_rate": {"never": 0.65, "ever": 0.35},
            "sensitivity_hints": "Strong causal effect on respiratory outcomes.",
            "related_domain_knowledge": [
                "Smoking contains carcinogens that damage lung tissue",
                "Risk increases with duration and intensity"
            ],
            "references": ["Doll & Hill (1950)", "IARC Monograph"]
        },
        {
            "name": "cancer",
            "llm_name": "lung_malignancy",
            "display_name": "Lung Cancer Status",
            "type": "binary",
            "states": ["negative", "positive"],
            "role": "endogenous",
            "category": "outcome",
            "short_description": "Lung cancer diagnosis.",
            "extended_description": "Confirmed lung cancer diagnosis. Primary outcome variable in smoking studies.",
            "sensitivity_hints": "Caused by multiple factors including smoking and genetics."
        },
        {
            "name": "genetics",
            "llm_name": "genetic_predisposition",
            "type": "categorical",
            "states": ["low", "medium", "high"],
            "role": "exogenous",
            "category": "genetic",
            "short_description": "Genetic predisposition to lung cancer."
        }
    ],
    
    "constraints": {
        "forbidden_edges": [
            ["cancer", "smoke"],
            ["cancer", "genetics"]
        ],
        "required_edges": [],
        "partial_order": [
            ["smoke", "cancer"],
            ["genetics", "cancer"]
        ]
    },
    
    "ground_truth": {
        "edges_expert": [
            ["smoke", "cancer"],
            ["genetics", "cancer"]
        ]
    }
}
```

## Field Reference

### Root Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | string | No | Schema version (default: "2.0") |
| `network` | string | Yes | Identifier for the network (e.g., "asia") |
| `domain` | string | Yes | Domain (e.g., "epidemiology", "genetics") |
| `purpose` | string | No | Purpose or description of the context |
| `provenance` | object | No | Source and provenance information |
| `llm_guidance` | object | No | Guidance for LLM interactions |
| `prompt_details` | object | No | Custom prompt detail definitions |
| `variables` | array | Yes | List of variable specifications |
| `constraints` | object | No | Structural constraints |
| `causal_principles` | array | No | Domain causal principles |
| `ground_truth` | object | No | Ground truth for evaluation |

### Variable Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Benchmark/literature name for ground truth |
| `llm_name` | string | No | Name used for LLM queries (defaults to name) |
| `type` | string | Yes | One of: binary, categorical, ordinal, continuous |
| `display_name` | string | No | Human-readable name for reports |
| `aliases` | array | No | Alternative names |

#### Semantic Disguising with name vs llm_name

The `name` and `llm_name` fields enable **semantic disguising** - using
meaningful but non-canonical names to reduce LLM memorisation whilst still
allowing evaluation against benchmarks.

**Example**: For the ASIA network's "Tuberculosis" variable:

```json
{
  "name": "tub",
  "llm_name": "HasTB",
  "display_name": "Tuberculosis Status",
  "type": "binary"
}
```

- **`name`**: "tub" - the original ASIA benchmark identifier for evaluation
- **`llm_name`**: "HasTB" - sent to the LLM (meaningful but not the benchmark name)
- **`display_name`**: "Tuberculosis Status" - for human-readable reports

This approach:

1. **Reduces memorisation** - LLM sees "HasTB" not "tub" from the benchmark
2. **Preserves semantics** - The llm_name still conveys clinical meaning
3. **Enables evaluation** - Results mapped back via `name` field

#### Additional Variable Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `states` | array | No | Possible values for discrete variables |
| `role` | string | No | One of: exogenous, endogenous, latent |
| `category` | string | No | Domain-specific category |
| `short_description` | string | No | Brief description |
| `extended_description` | string | No | Detailed description with context |
| `base_rate` | object | No | Prior probabilities {"state": probability} |
| `conditional_rates` | object | No | Conditional probabilities |
| `sensitivity_hints` | string | No | Hints about causal relationships |
| `related_domain_knowledge` | array | No | Domain knowledge statements |
| `references` | array | No | Literature references |

### Variable Types

- **binary** - Two states (yes/no, present/absent)
- **categorical** - Multiple unordered states (colors, categories)
- **ordinal** - Multiple ordered states (low/medium/high, stages)
- **continuous** - Numeric values (age, temperature, concentrations)

### Variable Roles

- **exogenous** - No parents in the causal graph (root causes)
- **endogenous** - Has parents (caused by other variables)
- **latent** - Unobserved/hidden variable

## Prompt Details

Prompt details control how much variable information is provided to LLMs:

### Minimal View

Only variable names - tests pure structural reasoning:

```json
{"include_fields": ["name"]}
```

Output: `[{"name": "smoking"}, {"name": "cancer"}]`

### Standard View

Names with basic metadata - balanced context:

```json
{"include_fields": ["name", "type", "short_description", "states"]}
```

### Rich View

Full context for complex reasoning:

```json
{
    "include_fields": [
        "name", "type", "role", "short_description",
        "extended_description", "states", "sensitivity_hints"
    ]
}
```

## Constraints

Structural constraints guide graph generation:

### Forbidden Edges

Edges that must not exist:

```json
{
    "forbidden_edges": [
        ["effect", "cause"],
        ["outcome", "exposure"]
    ]
}
```

### Required Edges

Edges that must exist:

```json
{
    "required_edges": [
        ["treatment", "outcome"]
    ]
}
```

### Partial Order

Temporal ordering constraints (A must precede B):

```json
{
    "partial_order": [
        ["birth_year", "diagnosis_age"],
        ["exposure", "disease"]
    ]
}
```

## Ground Truth

For evaluation, ground truth edges can be specified:

```json
{
    "ground_truth": {
        "edges_expert": [["A", "B"], ["B", "C"]],
        "edges_experiment": [["A", "B"]],
        "edges_observational": [["A", "B"], ["B", "C"], ["A", "C"]]
    }
}
```

- **edges_expert** - Edges from domain expert consensus
- **edges_experiment** - Edges confirmed by experiments
- **edges_observational** - Edges from observational studies

## Loading Network Context

```python
from causaliq_knowledge.graph import NetworkContext

# Load from file
context = NetworkContext.load("models/cancer.json")

# Load with full validation
context, warnings = NetworkContext.load_and_validate("models/cancer.json")

# Access data
print(f"Network: {context.network}")
print(f"Domain: {context.domain}")
print(f"Variables: {context.get_variable_names()}")
print(f"LLM Names: {context.get_llm_names()}")
```

## Example Network Contexts

Example network context files are in the `research/models/` directory:

- `asia/` - ASIA network (pulmonary disease)
- `cancer/` - Lung cancer model
- `sachs/` - SACHS protein signalling network
- `diabetes/` - Diabetes risk factors
- `sepsis/` - Sepsis clinical model

## Best Practices

1. **Use meaningful llm_names** - Aids LLM reasoning whilst avoiding memorisation
2. **Provide short descriptions** - Essential context for LLMs
3. **Define custom prompt details** - Control information disclosure
4. **Set provenance** - Document data sources
5. **Include ground truth** - Enable evaluation
6. **Add constraints** - Encode domain knowledge
