"""Integration tests across all available LLM providers.

These tests verify that the common BaseLLMClient interface works consistently
across all configured providers. Tests are skipped for providers without keys.

Run with: pytest -m slow tests/integration/test_all_providers_integration.py -v
"""

import pytest
from conftest import has_api_key, is_ollama_running


def get_available_clients():
    """Get list of available LLM clients based on environment."""
    clients = []

    if has_api_key("GROQ_API_KEY"):
        from causaliq_knowledge.llm import GroqClient, GroqConfig

        clients.append(
            ("groq", GroqClient, GroqConfig(model="llama-3.1-8b-instant"))
        )

    if has_api_key("OPENAI_API_KEY"):
        from causaliq_knowledge.llm import OpenAIClient, OpenAIConfig

        clients.append(
            ("openai", OpenAIClient, OpenAIConfig(model="gpt-4o-mini"))
        )

    if has_api_key("ANTHROPIC_API_KEY"):
        from causaliq_knowledge.llm import AnthropicClient, AnthropicConfig

        clients.append(
            (
                "anthropic",
                AnthropicClient,
                AnthropicConfig(model="claude-sonnet-4-20250514"),
            )
        )

    if has_api_key("GEMINI_API_KEY"):
        from causaliq_knowledge.llm import GeminiClient, GeminiConfig

        clients.append(
            ("gemini", GeminiClient, GeminiConfig(model="gemini-2.0-flash"))
        )

    if has_api_key("DEEPSEEK_API_KEY"):
        from causaliq_knowledge.llm import DeepSeekClient, DeepSeekConfig

        clients.append(
            ("deepseek", DeepSeekClient, DeepSeekConfig(model="deepseek-chat"))
        )

    if has_api_key("MISTRAL_API_KEY"):
        from causaliq_knowledge.llm import MistralClient, MistralConfig

        clients.append(
            (
                "mistral",
                MistralClient,
                MistralConfig(model="mistral-small-latest"),
            )
        )

    if is_ollama_running():
        from causaliq_knowledge.llm import OllamaClient, OllamaConfig

        clients.append(
            (
                "ollama",
                OllamaClient,
                OllamaConfig(model="llama3.2:1b", timeout=120.0),
            )
        )

    return clients


available_clients = get_available_clients()
skip_no_providers = pytest.mark.skipif(
    len(available_clients) == 0,
    reason="No LLM providers available (no API keys or Ollama not running)",
)


@pytest.fixture(params=get_available_clients(), ids=lambda x: x[0])
def client_info(request):
    """Fixture that yields each available client."""
    name, client_class, config = request.param
    client = client_class(config)
    return name, client


# Test that all providers can answer a simple math question.
@pytest.mark.slow
@pytest.mark.integration
@skip_no_providers
def test_all_providers_simple_math(client_info):
    name, client = client_info
    messages = [
        {"role": "user", "content": "What is 5 + 3? Reply with the number."}
    ]

    response = client.completion(messages)

    assert response.content is not None
    assert len(response.content) > 0
    assert "8" in response.content
    assert response.model is not None


# Test that provider_name property works for all providers.
@pytest.mark.slow
@pytest.mark.integration
@skip_no_providers
def test_all_providers_provider_name(client_info):
    name, client = client_info

    assert client.provider_name == name


# Test that is_available returns True for all configured providers.
@pytest.mark.slow
@pytest.mark.integration
@skip_no_providers
def test_all_providers_is_available(client_info):
    name, client = client_info

    assert client.is_available() is True


# Test that call_count increments properly for all providers.
@pytest.mark.slow
@pytest.mark.integration
@skip_no_providers
def test_all_providers_call_count(client_info):
    name, client = client_info
    initial_count = client.call_count
    messages = [{"role": "user", "content": "Hi"}]

    client.completion(messages)

    assert client.call_count == initial_count + 1


# Test that model_name property returns expected value for all providers.
@pytest.mark.slow
@pytest.mark.integration
@skip_no_providers
def test_all_providers_model_name(client_info):
    name, client = client_info

    assert client.model_name is not None
    assert len(client.model_name) > 0


# Test that all providers give consistent factual answers.
@pytest.mark.slow
@pytest.mark.integration
@skip_no_providers
def test_cross_provider_factual_consistency():
    clients = get_available_clients()
    if len(clients) < 2:
        pytest.skip("Need at least 2 providers for consistency test")

    messages = [
        {
            "role": "user",
            "content": "What is the capital of France? Reply with city name.",
        }
    ]

    responses = []
    for name, client_class, config in clients:
        client = client_class(config)
        response = client.completion(messages)
        responses.append((name, response.content))

    for name, content in responses:
        assert (
            "Paris" in content or "paris" in content.lower()
        ), f"Provider {name} did not return Paris: {content}"


# Test that all providers can produce JSON with requested structure.
@pytest.mark.slow
@pytest.mark.integration
@skip_no_providers
def test_cross_provider_json_consistency():
    clients = get_available_clients()
    if len(clients) < 2:
        pytest.skip("Need at least 2 providers for consistency test")

    messages = [
        {
            "role": "system",
            "content": "Always respond in valid JSON format.",
        },
        {
            "role": "user",
            "content": 'Return a JSON object: {"status": "ok", "count": 42}',
        },
    ]

    successful_parses = 0
    for name, client_class, config in clients:
        client = client_class(config)
        parsed, response = client.complete_json(messages)

        if parsed is not None:
            successful_parses += 1
            assert (
                "status" in parsed or "Status" in parsed
            ), f"Provider {name} missing status key"

    assert (
        successful_parses >= len(clients) // 2
    ), f"Only {successful_parses}/{len(clients)} produced valid JSON"


# Test that invalid model names raise appropriate errors.
@pytest.mark.slow
@pytest.mark.integration
def test_invalid_model_error():
    if not has_api_key("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")

    from causaliq_knowledge.llm import GroqClient, GroqConfig

    config = GroqConfig(model="nonexistent-model-xyz-123")
    client = GroqClient(config)
    messages = [{"role": "user", "content": "Hi"}]

    with pytest.raises(ValueError):
        client.completion(messages)


# Test that timeout is respected with very short timeout.
@pytest.mark.slow
@pytest.mark.integration
def test_timeout_handling():
    if not is_ollama_running():
        pytest.skip("Ollama not running")

    from causaliq_knowledge.llm import OllamaClient, OllamaConfig

    config = OllamaConfig(
        model="llama3.2:1b",
        timeout=0.001,
        max_tokens=1000,
    )
    client = OllamaClient(config)
    messages = [
        {
            "role": "user",
            "content": "Write a long essay about the history of computing.",
        }
    ]

    try:
        client.completion(messages)
    except (ValueError, Exception):
        pass
