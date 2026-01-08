"""Unit tests for DeepSeek client."""

import httpx
import pytest

from causaliq_knowledge.llm.deepseek_client import (
    DeepSeekClient,
    DeepSeekConfig,
)

# --- DeepSeekConfig Tests ---


# Test default configuration
def test_deepseek_config_defaults(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    config = DeepSeekConfig()

    assert config.model == "deepseek-chat"
    assert config.temperature == 0.1
    assert config.max_tokens == 500
    assert config.timeout == 30.0
    assert config.api_key == "test-key"


# Test custom configuration
def test_deepseek_config_custom(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "default-key")
    config = DeepSeekConfig(
        model="deepseek-reasoner",
        temperature=0.5,
        max_tokens=1000,
        timeout=60.0,
        api_key="custom-key",
    )

    assert config.model == "deepseek-reasoner"
    assert config.temperature == 0.5
    assert config.max_tokens == 1000
    assert config.timeout == 60.0
    assert config.api_key == "custom-key"


# Test config requires API key
def test_deepseek_config_requires_api_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        DeepSeekConfig()


# Test config reads from env var
def test_deepseek_config_env_var(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-api-key")
    config = DeepSeekConfig()

    assert config.api_key == "env-api-key"


# --- DeepSeekClient Tests ---


# Test client initialization
def test_deepseek_client_init(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient()

    assert client.config.model == "deepseek-chat"
    assert client.provider_name == "deepseek"
    assert client.call_count == 0


# Test client with custom config
def test_deepseek_client_custom_config(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    config = DeepSeekConfig(model="deepseek-reasoner", api_key="custom-key")
    client = DeepSeekClient(config)

    assert client.config.model == "deepseek-reasoner"
    assert client.config.api_key == "custom-key"


# Test is_available with key
def test_deepseek_is_available_with_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient()

    assert client.is_available() is True


# Test is_available without key
def test_deepseek_is_available_without_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    config = DeepSeekConfig()
    config.api_key = None  # Clear after init
    client = DeepSeekClient(config)

    assert client.is_available() is False


# Test _default_config returns DeepSeekConfig
def test_deepseek_default_config(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient()
    default = client._default_config()

    assert isinstance(default, DeepSeekConfig)
    assert default.model == "deepseek-chat"
    assert default.api_key == "test-key"


# --- Completion Tests ---


# Test successful completion
def test_deepseek_completion_success(mocker, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    mock_response = mocker.Mock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Hello, world!"}}],
        "model": "deepseek-chat",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = DeepSeekClient()
    messages = [{"role": "user", "content": "Hello"}]
    response = client.completion(messages)

    assert response.content == "Hello, world!"
    assert response.model == "deepseek-chat"
    assert response.input_tokens == 10
    assert response.output_tokens == 5
    assert client.call_count == 1


# Test completion with custom params
def test_deepseek_completion_custom_params(mocker, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    mock_response = mocker.Mock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Response"}}],
        "model": "deepseek-reasoner",
        "usage": {"prompt_tokens": 20, "completion_tokens": 10},
    }
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    config = DeepSeekConfig(model="deepseek-reasoner", api_key="test-key")
    client = DeepSeekClient(config)
    messages = [{"role": "user", "content": "Think step by step"}]
    response = client.completion(messages, temperature=0.7, max_tokens=2000)

    assert response.content == "Response"

    # Check that custom params were passed
    call_args = mock_client.post.call_args
    payload = call_args.kwargs["json"]
    assert payload["temperature"] == 0.7
    assert payload["max_tokens"] == 2000


# Test completion HTTP error
def test_deepseek_completion_http_error(mocker, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "bad-key")

    mock_response = mocker.Mock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"

    def raise_for_status():
        raise httpx.HTTPStatusError(
            "Error",
            request=httpx.Request("POST", "http://test"),
            response=mock_response,
        )

    mock_response.raise_for_status = raise_for_status

    mock_client = mocker.Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = DeepSeekClient()
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(ValueError, match="(?i)deepseek API error"):
        client.completion(messages)


# Test completion timeout
def test_deepseek_completion_timeout(mocker, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    mock_client = mocker.Mock()
    mock_client.post.side_effect = httpx.TimeoutException("Timeout")
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = DeepSeekClient()
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(ValueError, match="(?i)deepseek API request timed out"):
        client.completion(messages)


# --- List Models Tests ---


# Test list_models success
def test_deepseek_list_models_success(mocker, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    mock_response = mocker.Mock()
    mock_response.json.return_value = {
        "data": [
            {"id": "deepseek-chat"},
            {"id": "deepseek-reasoner"},
            {"id": "deepseek-coder"},
        ]
    }
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.get.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = DeepSeekClient()
    models = client.list_models()

    # Should filter to only deepseek- models
    assert "deepseek-chat" in models
    assert "deepseek-reasoner" in models
    assert "deepseek-coder" in models


# Test list_models API error
def test_deepseek_list_models_api_error(mocker, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    mock_response = mocker.Mock()
    mock_response.status_code = 500
    mock_response.text = "Server Error"

    mock_client = mocker.Mock()
    mock_client.get.side_effect = httpx.HTTPStatusError(
        "Error",
        request=httpx.Request("GET", "http://test"),
        response=mock_response,
    )
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = DeepSeekClient()

    with pytest.raises(ValueError, match="(?i)deepseek API error"):
        client.list_models()


# --- Cost Calculation Tests ---


# Test cost calculation for deepseek-chat
def test_deepseek_cost_calculation_chat(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    config = DeepSeekConfig(model="deepseek-chat")
    client = DeepSeekClient(config)

    # deepseek-chat: $0.14/1M input, $0.28/1M output
    cost = client._calculate_cost("deepseek-chat", 1_000_000, 1_000_000)

    assert cost == pytest.approx(0.42, rel=0.01)  # 0.14 + 0.28


# Test cost calculation for deepseek-reasoner
def test_deepseek_cost_calculation_reasoner(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    config = DeepSeekConfig(model="deepseek-reasoner")
    client = DeepSeekClient(config)

    # deepseek-reasoner: $0.55/1M input, $2.19/1M output
    cost = client._calculate_cost("deepseek-reasoner", 1_000_000, 1_000_000)

    assert cost == pytest.approx(2.74, rel=0.01)  # 0.55 + 2.19


# Test cost calculation for unknown model
def test_deepseek_cost_calculation_unknown(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    config = DeepSeekConfig(model="unknown-model", api_key="test-key")
    client = DeepSeekClient(config)

    cost = client._calculate_cost("unknown-model", 1000, 1000)

    assert cost == 0.0


# --- complete_json Tests ---


# Test complete_json success
def test_deepseek_complete_json_success(mocker, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    mock_response = mocker.Mock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '{"result": "success"}'}}],
        "model": "deepseek-chat",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)

    mocker.patch("httpx.Client", return_value=mock_client)

    client = DeepSeekClient()
    messages = [{"role": "user", "content": "Return JSON"}]
    parsed, response = client.complete_json(messages)

    assert parsed == {"result": "success"}
    assert response.content == '{"result": "success"}'
