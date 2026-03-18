# List Models

The `list-models` command displays available LLM models from each configured
provider. This is a **CLI-only** utility command and is not available as a
workflow action.

## Usage

```bash
cqknow list-models
```

## Parameters

| Parameter | CLI | Default | Description |
|-----------|-----|---------|-------------|
| `provider` | `-p` | no filter | Filter to a specific provider |

## Output

The command queries each provider's API and displays:

- **Available providers** with their models (green `[OK]`)
- **Unconfigured providers** with setup instructions (red `[X]`)
- **Providers with issues** such as no models installed (yellow `[!]`)

### Example Output

```
Available LLM Models:

  [OK] Groq (3 models):
      groq/llama-3.1-8b-instant
      groq/llama-3.1-70b-versatile
      groq/mixtral-8x7b-32768

  [X] Anthropic:
      ANTHROPIC_API_KEY not set

  [OK] Gemini (2 models):
      gemini/gemini-2.5-flash
      gemini/gemini-2.0-flash

  [!] Ollama (Local):
      No models installed. Run: ollama pull <model>

  [X] OpenAI:
      OPENAI_API_KEY not set

  [X] DeepSeek:
      DEEPSEEK_API_KEY not set

  [X] Mistral:
      MISTRAL_API_KEY not set

Provider Setup:
  GROQ_API_KEY: configured - https://console.groq.com
  ANTHROPIC_API_KEY: not set - https://console.anthropic.com
  GEMINI_API_KEY: configured - https://aistudio.google.com
  Ollama server: running - https://ollama.ai
  OPENAI_API_KEY: not set - https://platform.openai.com
  DEEPSEEK_API_KEY: not set - https://platform.deepseek.com
  MISTRAL_API_KEY: not set - https://console.mistral.ai

Note: Some models may require a paid plan. Free tier availability varies
by provider.

Default model: groq/llama-3.1-8b-instant
```

---

## Filtering by Provider

To show models from a specific provider only:

```bash
cqknow list-models -p groq
```

Valid provider names:

- `groq`
- `anthropic`
- `gemini`
- `ollama`
- `openai`
- `deepseek`
- `mistral`

---

## Provider Setup

### Groq

1. Sign up at [console.groq.com](https://console.groq.com)
2. Create an API key
3. Set the environment variable:

```powershell
$env:GROQ_API_KEY = "your-api-key"
```

### Google Gemini

1. Sign up at [aistudio.google.com](https://aistudio.google.com)
2. Create an API key
3. Set the environment variable:

```powershell
$env:GEMINI_API_KEY = "your-api-key"
```

### Ollama (Local)

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull a model: `ollama pull llama3`
3. Ensure the Ollama server is running

### OpenAI

1. Sign up at [platform.openai.com](https://platform.openai.com)
2. Create an API key
3. Set `OPENAI_API_KEY` environment variable

### Anthropic

1. Sign up at [console.anthropic.com](https://console.anthropic.com)
2. Create an API key
3. Set `ANTHROPIC_API_KEY` environment variable

### DeepSeek

1. Sign up at [platform.deepseek.com](https://platform.deepseek.com)
2. Create an API key
3. Set `DEEPSEEK_API_KEY` environment variable

### Mistral

1. Sign up at [console.mistral.ai](https://console.mistral.ai)
2. Create an API key
3. Set `MISTRAL_API_KEY` environment variable

---

## Free Tier Availability

| Provider | Free Tier | Notes |
|----------|-----------|-------|
| Groq | Yes | Generous free tier, very fast inference |
| Gemini | Yes | Free tier available |
| Ollama | Yes | Runs locally, no API costs |
| OpenAI | No | Paid only |
| Anthropic | No | Paid only |
| DeepSeek | No | Paid only |
| Mistral | No | Paid only |
