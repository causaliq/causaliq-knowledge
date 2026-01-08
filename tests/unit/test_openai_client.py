"""Unit tests for OpenAI client."""

import httpx
import pytest

from causaliq_knowledge.llm.base_client import BaseLLMClient, LLMConfig
from causaliq_knowledge.llm.openai_client import (
    OpenAIClient,
    OpenAIConfig,
)

# --- OpenAIConfig Tests ---


# Test OpenAIConfig defaults
def test_openai_config_defaults(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    config = OpenAIConfig()

    assert config.model == "gpt-4o-mini"
    assert config.temperature == 0.1
    assert config.max_tokens == 500
    assert config.timeout == 30.0
    assert config.api_key == "test-key"


# Test OpenAIConfig custom values
def test_openai_config_custom(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    config = OpenAIConfig(
        model="gpt-4o",
        temperature=0.5,
        max_tokens=1000,
        timeout=60.0,
        api_key="custom-key",
    )

    assert config.model == "gpt-4o"
    assert config.temperature == 0.5
    assert config.max_tokens == 1000
    assert config.timeout == 60.0
    assert config.api_key == "custom-key"


# Test OpenAIConfig requires API key
def test_openai_config_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIConfig()


# --- OpenAIClient Tests ---


# Test OpenAIClient default config
def test_openai_client_default_config(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    client = OpenAIClient()

    assert client.config.model == "gpt-4o-mini"
    assert client.call_count == 0


# Test OpenAIClient custom config
def test_openai_client_custom_config():
    config = OpenAIConfig(
        model="gpt-4o",
        api_key="test-key",
    )
    client = OpenAIClient(config)

    assert client.config.model == "gpt-4o"


# Test OpenAIClient provider name
def test_openai_client_provider_name():
    config = OpenAIConfig(api_key="test-key")
    client = OpenAIClient(config)

    assert client.provider_name == "openai"


# Test OpenAIClient inherits from BaseLLMClient
def test_openai_client_inherits_from_base():
    assert issubclass(OpenAIClient, BaseLLMClient)


# Test OpenAIConfig inherits from LLMConfig
def test_openai_config_inherits_from_base():
    assert issubclass(OpenAIConfig, LLMConfig)


# Test _default_config returns OpenAIConfig
def test_openai_default_config(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = OpenAIClient()
    default = client._default_config()

    assert isinstance(default, OpenAIConfig)
    assert default.model == "gpt-4o-mini"
    assert default.api_key == "test-key"


# --- completion() Tests ---


# Test successful completion
def test_openai_completion_success(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [{"message": {"content": "Hello! How can I help?"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8},
    }
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    config = OpenAIConfig(api_key="test-key")
    client = OpenAIClient(config)
    messages = [{"role": "user", "content": "Hello"}]

    response = client.completion(messages)

    assert response.content == "Hello! How can I help?"
    assert response.model == "gpt-4o-mini"
    assert response.input_tokens == 10
    assert response.output_tokens == 8
    assert client.call_count == 1


# Test completion with system message
def test_openai_completion_with_system(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "chatcmpl-123",
        "model": "gpt-4o-mini",
        "choices": [{"message": {"content": "I am helpful."}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5},
    }
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    config = OpenAIConfig(api_key="test-key")
    client = OpenAIClient(config)
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Who are you?"},
    ]

    client.completion(messages)

    # Verify system message is passed in messages
    call_args = mock_client.post.call_args
    payload = call_args.kwargs["json"]
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"


# Test completion with kwargs override
def test_openai_completion_with_kwargs(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "chatcmpl-123",
        "model": "gpt-4o-mini",
        "choices": [{"message": {"content": "Response"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    config = OpenAIConfig(api_key="test-key", temperature=0.1, max_tokens=100)
    client = OpenAIClient(config)
    messages = [{"role": "user", "content": "Test"}]

    client.completion(messages, temperature=0.8, max_tokens=200)

    call_args = mock_client.post.call_args
    payload = call_args.kwargs["json"]
    assert payload["temperature"] == 0.8
    assert payload["max_tokens"] == 200


# Test completion HTTP error
def test_openai_completion_http_error(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"
    mock_response.json.return_value = {"error": {"message": "Invalid API key"}}

    def raise_for_status():
        raise httpx.HTTPStatusError(
            "Unauthorized",
            request=httpx.Request("POST", "http://test"),
            response=mock_response,
        )

    mock_response.raise_for_status = raise_for_status

    mock_client = mocker.Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    config = OpenAIConfig(api_key="bad-key")
    client = OpenAIClient(config)
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(ValueError, match="(?i)openai API error"):
        client.completion(messages)


# Test completion timeout
def test_openai_completion_timeout(mocker):
    mock_client = mocker.Mock()
    mock_client.post.side_effect = httpx.TimeoutException("Timeout")
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    config = OpenAIConfig(api_key="test-key")
    client = OpenAIClient(config)
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(ValueError, match="timed out"):
        client.completion(messages)


# Test completion generic error
def test_openai_completion_generic_error(mocker):
    mock_client = mocker.Mock()
    mock_client.post.side_effect = RuntimeError("Connection failed")
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    config = OpenAIConfig(api_key="test-key")
    client = OpenAIClient(config)
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(ValueError, match="(?i)openai API error"):
        client.completion(messages)


# --- complete_json() Tests ---


# Test complete_json success
def test_openai_complete_json_success(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "chatcmpl-123",
        "model": "gpt-4o-mini",
        "choices": [
            {"message": {"content": '{"exists": true, "confidence": 0.9}'}}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15},
    }
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    config = OpenAIConfig(api_key="test-key")
    client = OpenAIClient(config)
    messages = [{"role": "user", "content": "Return JSON"}]

    parsed, response = client.complete_json(messages)

    assert parsed is not None
    assert parsed["exists"] is True
    assert parsed["confidence"] == 0.9
    assert response.content == '{"exists": true, "confidence": 0.9}'


# Test complete_json with invalid JSON
def test_openai_complete_json_invalid(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "chatcmpl-123",
        "model": "gpt-4o-mini",
        "choices": [{"message": {"content": "Not valid JSON"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    config = OpenAIConfig(api_key="test-key")
    client = OpenAIClient(config)
    messages = [{"role": "user", "content": "Return JSON"}]

    parsed, response = client.complete_json(messages)

    assert parsed is None
    assert response.content == "Not valid JSON"


# --- is_available() Tests ---


# Test is_available returns True when key set
def test_openai_is_available_true():
    config = OpenAIConfig(api_key="test-key")
    client = OpenAIClient(config)

    assert client.is_available() is True


# Test is_available returns False when key empty
def test_openai_is_available_false(monkeypatch):
    # Create config with key, then clear it
    config = OpenAIConfig(api_key="test-key")
    config.api_key = ""
    client = OpenAIClient.__new__(OpenAIClient)
    client.config = config
    client._total_calls = 0

    assert client.is_available() is False


# --- list_models() Tests ---


# Test list_models returns models from API
def test_openai_list_models(mocker):
    config = OpenAIConfig(api_key="test-key")
    client = OpenAIClient(config)

    # Mock the HTTP client
    mock_response = mocker.Mock()
    mock_response.json.return_value = {
        "data": [
            {"id": "gpt-4o", "object": "model"},
            {"id": "gpt-4o-mini", "object": "model"},
            {"id": "gpt-3.5-turbo", "object": "model"},
            {"id": "whisper-1", "object": "model"},  # Should be filtered
            {"id": "dall-e-3", "object": "model"},  # Should be filtered
            {"id": "text-embedding-ada-002", "object": "model"},  # Filtered
            {"id": "gpt-4-vision-preview", "object": "model"},  # Filtered
        ],
    }
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.get.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    models = client.list_models()

    assert isinstance(models, list)
    assert "gpt-4o" in models
    assert "gpt-4o-mini" in models
    assert "gpt-3.5-turbo" in models
    # Should filter non-chat models
    assert "whisper-1" not in models
    assert "dall-e-3" not in models
    assert "text-embedding-ada-002" not in models
    # Should filter vision variants
    assert "gpt-4-vision-preview" not in models
    # Should be sorted
    assert models == sorted(models)


# Test list_models with o1 models
def test_openai_list_models_includes_o1(mocker):
    config = OpenAIConfig(api_key="test-key")
    client = OpenAIClient(config)

    mock_response = mocker.Mock()
    mock_response.json.return_value = {
        "data": [
            {"id": "o1", "object": "model"},
            {"id": "o1-mini", "object": "model"},
            {"id": "o1-preview", "object": "model"},
        ],
    }
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.get.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    models = client.list_models()

    assert "o1" in models
    assert "o1-mini" in models
    assert "o1-preview" in models


# Test list_models handles API error
def test_openai_list_models_api_error(mocker):
    config = OpenAIConfig(api_key="test-key")
    client = OpenAIClient(config)

    # Mock HTTP error
    mock_response = mocker.Mock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"

    mock_client = mocker.Mock()
    mock_client.get.side_effect = httpx.HTTPStatusError(
        "Error",
        request=httpx.Request("GET", "http://test"),
        response=mock_response,
    )
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    with pytest.raises(ValueError, match="(?i)openai API error"):
        client.list_models()


# Test list_models handles generic exception
def test_openai_list_models_generic_error(mocker):
    config = OpenAIConfig(api_key="test-key")
    client = OpenAIClient(config)

    # Mock generic exception
    mock_client = mocker.Mock()
    mock_client.get.side_effect = Exception("Connection error")
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    with pytest.raises(ValueError, match="(?i)failed to list openai models"):
        client.list_models()


# --- Cost Calculation Tests ---


# Test cost calculation for gpt-4o-mini
def test_openai_cost_calculation_gpt4o_mini():
    config = OpenAIConfig(api_key="test-key", model="gpt-4o-mini")
    client = OpenAIClient(config)

    # 1000 input tokens, 500 output tokens
    cost = client._calculate_cost("gpt-4o-mini", 1000, 500)

    # gpt-4o-mini: $0.15/1M input, $0.60/1M output
    expected = (1000 / 1_000_000 * 0.15) + (500 / 1_000_000 * 0.60)
    assert abs(cost - expected) < 0.0001


# Test cost calculation for gpt-4o
def test_openai_cost_calculation_gpt4o():
    config = OpenAIConfig(api_key="test-key", model="gpt-4o")
    client = OpenAIClient(config)

    cost = client._calculate_cost("gpt-4o", 1000, 500)

    # gpt-4o: $2.50/1M input, $10.00/1M output
    expected = (1000 / 1_000_000 * 2.50) + (500 / 1_000_000 * 10.00)
    assert abs(cost - expected) < 0.0001


# Test cost calculation for unknown model returns 0
def test_openai_cost_calculation_unknown_model():
    config = OpenAIConfig(api_key="test-key")
    client = OpenAIClient(config)

    cost = client._calculate_cost("unknown-model", 1000, 500)

    assert cost == 0.0


# Test cost calculation with model variant
def test_openai_cost_calculation_model_variant():
    config = OpenAIConfig(api_key="test-key")
    client = OpenAIClient(config)

    # Should match gpt-4o prefix
    cost = client._calculate_cost("gpt-4o-2024-08-06", 1000, 500)

    expected = (1000 / 1_000_000 * 2.50) + (500 / 1_000_000 * 10.00)
    assert abs(cost - expected) < 0.0001


# --- Response Tests ---


# Test response includes cost
def test_openai_response_includes_cost(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "chatcmpl-123",
        "model": "gpt-4o-mini",
        "choices": [{"message": {"content": "Test response"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    config = OpenAIConfig(api_key="test-key", model="gpt-4o-mini")
    client = OpenAIClient(config)
    messages = [{"role": "user", "content": "Test"}]

    response = client.completion(messages)

    # gpt-4o-mini pricing
    expected_cost = (100 / 1_000_000 * 0.15) + (50 / 1_000_000 * 0.60)
    assert abs(response.cost - expected_cost) < 0.0001


# Test response with empty content
def test_openai_completion_empty_content(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "chatcmpl-123",
        "model": "gpt-4o-mini",
        "choices": [{"message": {"content": None}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0},
    }
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    config = OpenAIConfig(api_key="test-key")
    client = OpenAIClient(config)
    messages = [{"role": "user", "content": "Test"}]

    response = client.completion(messages)

    assert response.content == ""


# Test base class _filter_models default implementation
def test_openai_compat_base_filter_models_default(monkeypatch):
    from causaliq_knowledge.llm.openai_compat_client import OpenAICompatClient

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = OpenAIClient()

    # Call the base class _filter_models directly (not overridden version)
    models = ["model-a", "model-b", "model-c"]
    result = OpenAICompatClient._filter_models(client, models)

    assert result == models
