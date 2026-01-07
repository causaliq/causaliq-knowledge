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
from causaliq_knowledge.llm import LLMKnowledge

# Initialize with Groq (default, free tier)
knowledge = LLMKnowledge(models=["groq/llama-3.1-8b-instant"])

# Query about a potential edge
result = knowledge.query_edge(
    node_a="smoking",
    node_b="lung_cancer",
    context={"domain": "epidemiology"}
)

print(f"Exists: {result.exists}")
print(f"Direction: {result.direction}")
print(f"Confidence: {result.confidence}")
print(f"Reasoning: {result.reasoning}")
```

### Using Gemini (Free Tier)

```python
# Use Google Gemini for inference
knowledge = LLMKnowledge(models=["gemini/gemini-2.5-flash"])
```

### Multi-Model Consensus

```python
# Query multiple models for more robust answers
knowledge = LLMKnowledge(
    models=["groq/llama-3.1-8b-instant", "gemini/gemini-2.5-flash"],
    consensus_strategy="weighted_vote"
)
```

## LLM Provider Setup

CausalIQ Knowledge uses **direct vendor-specific API clients** (not wrapper libraries) to communicate with LLM providers. This approach provides reliability and minimal dependencies. Currently supported:

- **Groq**: Free tier with fast inference
- **Google Gemini**: Generous free tier

### Free Options

#### Groq (Free Tier - Very Fast)

Groq offers a generous free tier with extremely fast inference:

1. Sign up at [console.groq.com](https://console.groq.com)
2. Create an API key
3. Set the environment variable (see [Storing API Keys](#storing-api-keys))
4. Use in code:

```python
from causaliq_knowledge.llm import LLMKnowledge

knowledge = LLMKnowledge(models=["groq/llama-3.1-8b-instant"])
result = knowledge.query_edge("smoking", "lung_cancer")
```

Available Groq models: `groq/llama-3.1-8b-instant`, `groq/llama-3.1-70b-versatile`, `groq/mixtral-8x7b-32768`

#### Google Gemini (Free Tier)

Google offers free access to Gemini models:

1. Sign up at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Create an API key
3. Set `GEMINI_API_KEY` environment variable
4. Use in code:

```python
knowledge = LLMKnowledge(models=["gemini/gemini-2.5-flash"])
```

### Coming Soon (v0.2.0)

Additional providers will be added in future releases:

| Provider | Status | Notes |
|----------|--------|-------|
| **OpenAI** | Planned v0.2.0 | GPT-4 models |
| **Anthropic** | Planned v0.2.0 | Claude models |

### Storing API Keys

#### Option 1: User Environment Variables (Recommended)

Set permanently for your user account on Windows:

```powershell
[Environment]::SetEnvironmentVariable("GROQ_API_KEY", "your-key", "User")
[Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "your-key", "User")
```

On Linux/macOS, add to your `~/.bashrc` or `~/.zshrc`:

```bash
export GROQ_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
```

Restart your terminal for changes to take effect.

#### Option 2: Project `.env` File

Create a `.env` file in your project root:

```
GROQ_API_KEY=your-key-here
GEMINI_API_KEY=your-key-here
```

**Important:** Add `.env` to your `.gitignore` so keys aren't committed to version control!

#### Option 3: Password Manager

Store API keys in a password manager (LastPass, 1Password, Bitwarden, etc.) as a Secure Note. This provides:

- Encrypted backup of your keys
- Access from any machine
- Secure sharing with team members if needed

Copy keys from your password manager when setting environment variables.

### Verifying Your Setup

Test your configuration with the CLI:

```bash
# Using the installed CLI
cqknow query smoking lung_cancer --model groq/llama-3.1-8b-instant

# Or with Python module
python -m causaliq_knowledge.cli query smoking lung_cancer --model ollama/llama3
```

## What's Next?

- [Architecture Overview](../architecture/overview.md) - Understand how the package works
- [LLM Integration Design](../architecture/llm_integration.md) - Detailed design documentation
- [API Reference](../api/overview.md) - Full API documentation