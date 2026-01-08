"""Unit tests for Anthropic client."""

import httpx
import pytest

from causaliq_knowledge.llm.anthropic_client import (
    AnthropicClient,
    AnthropicConfig,
)
from causaliq_knowledge.llm.base_client import BaseLLMClient, LLMConfig


# Test AnthropicConfig defaults
def test_anthropic_config_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    config = AnthropicConfig()

    assert config.model == "claude-sonnet-4-20250514"
    assert config.temperature == 0.1
    assert config.max_tokens == 500
    assert config.timeout == 30.0
    assert config.api_key == "test-key"


# Test AnthropicConfig custom values
def test_anthropic_config_custom(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")

    config = AnthropicConfig(
        model="claude-3-haiku-20240307",
        temperature=0.5,
        max_tokens=1000,
        timeout=60.0,
        api_key="custom-key",
    )

    assert config.model == "claude-3-haiku-20240307"
    assert config.temperature == 0.5
    assert config.max_tokens == 1000
    assert config.timeout == 60.0
    assert config.api_key == "custom-key"


# Test AnthropicConfig requires API key
def test_anthropic_config_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicConfig()


# Test AnthropicClient default config
def test_anthropic_client_default_config(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    client = AnthropicClient()

    assert client.config.model == "claude-sonnet-4-20250514"
    assert client.call_count == 0


# Test AnthropicClient custom config
def test_anthropic_client_custom_config():
    config = AnthropicConfig(
        model="claude-3-haiku-20240307",
        api_key="test-key",
    )
    client = AnthropicClient(config)

    assert client.config.model == "claude-3-haiku-20240307"


# Test AnthropicClient provider name
def test_anthropic_client_provider_name():
    config = AnthropicConfig(api_key="test-key")
    client = AnthropicClient(config)

    assert client.provider_name == "anthropic"


# Test AnthropicClient inherits from BaseLLMClient
def test_anthropic_client_inherits_from_base():
    assert issubclass(AnthropicClient, BaseLLMClient)


# Test AnthropicConfig inherits from LLMConfig
def test_anthropic_config_inherits_from_base():
    assert issubclass(AnthropicConfig, LLMConfig)


# --- completion() Tests ---


# Test successful completion
def test_anthropic_completion_success(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "msg_123",
        "type": "message",
        "model": "claude-sonnet-4-20250514",
        "content": [{"type": "text", "text": "Hello! How can I help?"}],
        "usage": {"input_tokens": 10, "output_tokens": 8},
    }
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    config = AnthropicConfig(api_key="test-key")
    client = AnthropicClient(config)
    messages = [{"role": "user", "content": "Hello"}]

    response = client.completion(messages)

    assert response.content == "Hello! How can I help?"
    assert response.model == "claude-sonnet-4-20250514"
    assert response.input_tokens == 10
    assert response.output_tokens == 8
    assert client.call_count == 1


# Test completion with system message
def test_anthropic_completion_with_system(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "msg_123",
        "model": "claude-sonnet-4-20250514",
        "content": [{"type": "text", "text": "I am helpful."}],
        "usage": {"input_tokens": 20, "output_tokens": 5},
    }
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    config = AnthropicConfig(api_key="test-key")
    client = AnthropicClient(config)
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Who are you?"},
    ]

    client.completion(messages)

    # Verify system was extracted and passed correctly
    call_args = mock_client.post.call_args
    payload = call_args.kwargs["json"]
    assert payload["system"] == "You are helpful."
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["role"] == "user"


# Test completion with kwargs override
def test_anthropic_completion_with_kwargs(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "msg_123",
        "model": "claude-sonnet-4-20250514",
        "content": [{"type": "text", "text": "Response"}],
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    config = AnthropicConfig(
        api_key="test-key", temperature=0.1, max_tokens=100
    )
    client = AnthropicClient(config)
    messages = [{"role": "user", "content": "Test"}]

    client.completion(messages, temperature=0.8, max_tokens=200)

    call_args = mock_client.post.call_args
    payload = call_args.kwargs["json"]
    assert payload["temperature"] == 0.8
    assert payload["max_tokens"] == 200


# Test completion HTTP error
def test_anthropic_completion_http_error(mocker):
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

    config = AnthropicConfig(api_key="bad-key")
    client = AnthropicClient(config)
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(ValueError, match="Anthropic API error"):
        client.completion(messages)


# Test completion HTTP error with non-JSON response
def test_anthropic_completion_http_error_no_json(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    def raise_json():
        raise ValueError("No JSON")

    mock_response.json = raise_json

    def raise_for_status():
        raise httpx.HTTPStatusError(
            "Server Error",
            request=httpx.Request("POST", "http://test"),
            response=mock_response,
        )

    mock_response.raise_for_status = raise_for_status

    mock_client = mocker.Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    config = AnthropicConfig(api_key="test-key")
    client = AnthropicClient(config)
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(ValueError, match="Internal Server Error"):
        client.completion(messages)


# Test completion timeout
def test_anthropic_completion_timeout(mocker):
    mock_client = mocker.Mock()
    mock_client.post.side_effect = httpx.TimeoutException("Timeout")
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    config = AnthropicConfig(api_key="test-key")
    client = AnthropicClient(config)
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(ValueError, match="timed out"):
        client.completion(messages)


# Test completion generic error
def test_anthropic_completion_generic_error(mocker):
    mock_client = mocker.Mock()
    mock_client.post.side_effect = RuntimeError("Connection failed")
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    config = AnthropicConfig(api_key="test-key")
    client = AnthropicClient(config)
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(ValueError, match="Anthropic API error"):
        client.completion(messages)


# --- complete_json() Tests ---


# Test complete_json success
def test_anthropic_complete_json_success(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "msg_123",
        "model": "claude-sonnet-4-20250514",
        "content": [
            {"type": "text", "text": '{"exists": true, "confidence": 0.9}'}
        ],
        "usage": {"input_tokens": 10, "output_tokens": 15},
    }
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    config = AnthropicConfig(api_key="test-key")
    client = AnthropicClient(config)
    messages = [{"role": "user", "content": "Return JSON"}]

    parsed, response = client.complete_json(messages)

    assert parsed is not None
    assert parsed["exists"] is True
    assert parsed["confidence"] == 0.9
    assert response.content == '{"exists": true, "confidence": 0.9}'


# Test complete_json with invalid JSON
def test_anthropic_complete_json_invalid(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "msg_123",
        "model": "claude-sonnet-4-20250514",
        "content": [{"type": "text", "text": "Not valid JSON"}],
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    config = AnthropicConfig(api_key="test-key")
    client = AnthropicClient(config)
    messages = [{"role": "user", "content": "Return JSON"}]

    parsed, response = client.complete_json(messages)

    assert parsed is None
    assert response.content == "Not valid JSON"


# --- is_available() Tests ---


# Test is_available returns True when key set
def test_anthropic_is_available_true():
    config = AnthropicConfig(api_key="test-key")
    client = AnthropicClient(config)

    assert client.is_available() is True


# Test is_available returns False when key empty
def test_anthropic_is_available_false(monkeypatch):
    # Create config with key, then clear it
    config = AnthropicConfig(api_key="test-key")
    config.api_key = ""
    client = AnthropicClient.__new__(AnthropicClient)
    client.config = config
    client._total_calls = 0

    assert client.is_available() is False


# --- list_models() Tests ---


# Test list_models returns models from API
def test_anthropic_list_models(mocker):
    config = AnthropicConfig(api_key="test-key")
    client = AnthropicClient(config)

    # Mock the HTTP client
    mock_response = mocker.Mock()
    mock_response.json.return_value = {
        "data": [
            {"id": "claude-sonnet-4-20250514", "type": "model"},
            {"id": "claude-3-5-sonnet-20241022", "type": "model"},
            {"id": "claude-3-haiku-20240307", "type": "model"},
        ],
        "has_more": False,
    }
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.get.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    models = client.list_models()

    assert isinstance(models, list)
    assert len(models) == 3
    assert "claude-sonnet-4-20250514" in models
    assert "claude-3-5-sonnet-20241022" in models
    # Should be sorted
    assert models == sorted(models)


# Test list_models returns empty list on API error
def test_anthropic_list_models_api_error(mocker):
    config = AnthropicConfig(api_key="test-key")
    client = AnthropicClient(config)

    # Mock HTTP error
    mock_client = mocker.Mock()
    mock_client.get.side_effect = httpx.HTTPStatusError(
        "Error", request=mocker.Mock(), response=mocker.Mock()
    )
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    models = client.list_models()

    assert models == []


# Test list_models returns empty list when no API key
def test_anthropic_list_models_no_key():
    config = AnthropicConfig(api_key="test-key")
    client = AnthropicClient(config)
    client.config.api_key = None

    models = client.list_models()

    assert models == []


# Test list_models handles generic exception
def test_anthropic_list_models_generic_error(mocker):
    config = AnthropicConfig(api_key="test-key")
    client = AnthropicClient(config)

    # Mock generic exception
    mock_client = mocker.Mock()
    mock_client.get.side_effect = Exception("Connection error")
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    models = client.list_models()

    assert models == []
