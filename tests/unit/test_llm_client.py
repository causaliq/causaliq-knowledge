"""Unit tests for LLM client wrapper."""

import pytest
from pydantic import BaseModel

from causaliq_knowledge.llm.client import LLMClient, LLMConfig, LLMResponse


# Helper to create a mock LLM response object
def _mock_response(
    content="Response", model="gpt-4o-mini", prompt=10, compl=5
):
    class MockUsage:
        prompt_tokens = prompt
        completion_tokens = compl

    class MockMessage:
        def __init__(self):
            self.content = content

    class MockChoice:
        def __init__(self):
            self.message = MockMessage()

    class MockResponse:
        def __init__(self):
            self.choices = [MockChoice()]
            self.model = model
            self.usage = MockUsage() if prompt or compl else None

    return MockResponse()


# ============================================================================
# LLMConfig tests
# ============================================================================


# Check default configuration values
def test_config_default_values():
    config = LLMConfig()
    assert config.model == "gpt-4o-mini"
    assert config.temperature == 0.0
    assert config.max_tokens == 1024
    assert config.timeout == 30.0
    assert config.max_retries == 3
    assert config.api_base is None


# Check custom configuration values
def test_config_custom_values():
    config = LLMConfig(
        model="ollama/llama3",
        temperature=0.7,
        max_tokens=2048,
        timeout=60.0,
        max_retries=5,
        api_base="http://localhost:11434",
    )
    assert config.model == "ollama/llama3"
    assert config.temperature == 0.7
    assert config.max_tokens == 2048
    assert config.timeout == 60.0
    assert config.max_retries == 5
    assert config.api_base == "http://localhost:11434"


# ============================================================================
# LLMResponse tests
# ============================================================================


# Check basic response creation
def test_response_basic():
    response = LLMResponse(
        content="Hello, world!",
        model="gpt-4o-mini",
        input_tokens=10,
        output_tokens=5,
        cost=0.0001,
    )
    assert response.content == "Hello, world!"
    assert response.model == "gpt-4o-mini"
    assert response.input_tokens == 10
    assert response.output_tokens == 5
    assert response.cost == 0.0001


# Check parsing valid JSON content
def test_response_parse_json_valid():
    response = LLMResponse(
        content='{"exists": true, "confidence": 0.85}',
        model="gpt-4o-mini",
    )
    result = response.parse_json()
    assert result == {"exists": True, "confidence": 0.85}


# Check parsing JSON wrapped in markdown code fence
def test_response_parse_json_with_markdown_fence():
    response = LLMResponse(
        content='```json\n{"exists": true}\n```',
        model="gpt-4o-mini",
    )
    result = response.parse_json()
    assert result == {"exists": True}


# Check parsing JSON wrapped in plain code fence
def test_response_parse_json_with_plain_fence():
    response = LLMResponse(
        content='```\n{"exists": false}\n```',
        model="gpt-4o-mini",
    )
    result = response.parse_json()
    assert result == {"exists": False}


# Check parsing invalid JSON returns None
def test_response_parse_json_invalid():
    response = LLMResponse(
        content="This is not JSON",
        model="gpt-4o-mini",
    )
    result = response.parse_json()
    assert result is None


# Check default values for optional fields
def test_response_defaults():
    response = LLMResponse(content="test", model="test-model")
    assert response.input_tokens == 0
    assert response.output_tokens == 0
    assert response.cost == 0.0
    assert response.raw_response is None


# ============================================================================
# LLMClient tests
# ============================================================================


# Check client initialization with default config
def test_client_init_default_config():
    client = LLMClient()
    assert client.config.model == "gpt-4o-mini"
    assert client.total_cost == 0.0
    assert client.call_count == 0


# Check client initialization with custom config
def test_client_init_custom_config():
    config = LLMConfig(model="claude-3-haiku")
    client = LLMClient(config)
    assert client.config.model == "claude-3-haiku"


# Check statistics properties return correct initial values
def test_client_stats_properties():
    client = LLMClient()
    assert client.total_cost == 0.0
    assert client.total_input_tokens == 0
    assert client.total_output_tokens == 0
    assert client.call_count == 0


# Check resetting statistics clears all values
def test_client_reset_stats():
    client = LLMClient()
    client._total_cost = 1.0
    client._total_input_tokens = 100
    client._total_output_tokens = 50
    client._call_count = 5

    client.reset_stats()

    assert client.total_cost == 0.0
    assert client.total_input_tokens == 0
    assert client.total_output_tokens == 0
    assert client.call_count == 0


# Check get_stats returns correct dictionary
def test_client_get_stats():
    client = LLMClient()
    client._total_cost = 0.05
    client._total_input_tokens = 1000
    client._total_output_tokens = 500
    client._call_count = 10

    stats = client.get_stats()

    assert stats["total_cost"] == 0.05
    assert stats["total_input_tokens"] == 1000
    assert stats["total_output_tokens"] == 500
    assert stats["call_count"] == 10
    assert stats["model"] == "gpt-4o-mini"


# Check basic completion call works correctly
def test_client_complete_basic(monkeypatch):
    mock_resp = _mock_response("Test response", "gpt-4o-mini", 10, 5)

    import causaliq_knowledge.llm.client as client_module

    monkeypatch.setattr(
        client_module.litellm, "completion", lambda **kw: mock_resp
    )
    monkeypatch.setattr(
        client_module.litellm, "completion_cost", lambda **kw: 0.0001
    )

    client = LLMClient()
    response = client.complete(user="Hello")

    assert response.content == "Test response"
    assert response.model == "gpt-4o-mini"
    assert response.input_tokens == 10
    assert response.output_tokens == 5
    assert response.cost == 0.0001
    assert client.call_count == 1
    assert client.total_cost == 0.0001


# Check completion with system message constructs messages correctly
def test_client_complete_with_system(monkeypatch):
    mock_resp = _mock_response("Response", "gpt-4o-mini", 20, 10)
    captured_kwargs = {}

    def capture_completion(**kwargs):
        captured_kwargs.update(kwargs)
        return mock_resp

    import causaliq_knowledge.llm.client as client_module

    monkeypatch.setattr(
        client_module.litellm, "completion", capture_completion
    )
    monkeypatch.setattr(
        client_module.litellm, "completion_cost", lambda **kw: 0.0002
    )

    client = LLMClient()
    client.complete(
        system="You are a helpful assistant.",
        user="What is 2+2?",
    )

    messages = captured_kwargs["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "What is 2+2?"


# Check completion with custom API base passes it through
def test_client_complete_with_api_base(monkeypatch):
    mock_resp = _mock_response("Response", "ollama/llama3", 0, 0)
    mock_resp.usage = None
    captured_kwargs = {}

    def capture_completion(**kwargs):
        captured_kwargs.update(kwargs)
        return mock_resp

    import causaliq_knowledge.llm.client as client_module

    monkeypatch.setattr(
        client_module.litellm, "completion", capture_completion
    )
    monkeypatch.setattr(
        client_module.litellm, "completion_cost", lambda **kw: 0.0
    )

    config = LLMConfig(
        model="ollama/llama3",
        api_base="http://localhost:11434",
    )
    client = LLMClient(config)
    client.complete(user="Hello")

    assert captured_kwargs["api_base"] == "http://localhost:11434"


# Check multiple calls accumulate statistics
def test_client_complete_accumulates_stats(monkeypatch):
    mock_resp = _mock_response("Response", "gpt-4o-mini", 10, 5)

    import causaliq_knowledge.llm.client as client_module

    monkeypatch.setattr(
        client_module.litellm, "completion", lambda **kw: mock_resp
    )
    monkeypatch.setattr(
        client_module.litellm, "completion_cost", lambda **kw: 0.0001
    )

    client = LLMClient()
    client.complete(user="Call 1")
    client.complete(user="Call 2")
    client.complete(user="Call 3")

    assert client.call_count == 3
    assert client.total_input_tokens == 30
    assert client.total_output_tokens == 15
    assert client.total_cost == pytest.approx(0.0003)


# Check complete_json parses JSON response
def test_client_complete_json(monkeypatch):
    mock_resp = _mock_response(
        '{"exists": true, "confidence": 0.9}', "gpt-4o-mini", 10, 5
    )

    import causaliq_knowledge.llm.client as client_module

    monkeypatch.setattr(
        client_module.litellm, "completion", lambda **kw: mock_resp
    )
    monkeypatch.setattr(
        client_module.litellm, "completion_cost", lambda **kw: 0.0001
    )

    client = LLMClient()
    parsed, response = client.complete_json(user="Query")

    assert parsed == {"exists": True, "confidence": 0.9}
    assert response.content == '{"exists": true, "confidence": 0.9}'


# Check complete_json handles invalid JSON gracefully
def test_client_complete_json_invalid(monkeypatch):
    mock_resp = _mock_response("Not valid JSON", "gpt-4o-mini", 10, 5)

    import causaliq_knowledge.llm.client as client_module

    monkeypatch.setattr(
        client_module.litellm, "completion", lambda **kw: mock_resp
    )
    monkeypatch.setattr(
        client_module.litellm, "completion_cost", lambda **kw: 0.0001
    )

    client = LLMClient()
    parsed, response = client.complete_json(user="Query")

    assert parsed is None
    assert response.content == "Not valid JSON"


# Check completion handles None content gracefully
def test_client_complete_handles_none_content(monkeypatch):
    mock_resp = _mock_response(None, "gpt-4o-mini", 10, 0)

    import causaliq_knowledge.llm.client as client_module

    monkeypatch.setattr(
        client_module.litellm, "completion", lambda **kw: mock_resp
    )
    monkeypatch.setattr(
        client_module.litellm, "completion_cost", lambda **kw: 0.0
    )

    client = LLMClient()
    response = client.complete(user="Hello")

    assert response.content == ""


# Check completion handles missing usage info
def test_client_complete_handles_missing_usage(monkeypatch):
    mock_resp = _mock_response("Response", "gpt-4o-mini", 0, 0)
    mock_resp.usage = None

    import causaliq_knowledge.llm.client as client_module

    monkeypatch.setattr(
        client_module.litellm, "completion", lambda **kw: mock_resp
    )
    monkeypatch.setattr(
        client_module.litellm, "completion_cost", lambda **kw: 0.0
    )

    client = LLMClient()
    response = client.complete(user="Hello")

    assert response.input_tokens == 0
    assert response.output_tokens == 0


# Check completion with response_format passes it through
def test_client_complete_with_response_format(monkeypatch):
    class TestFormat(BaseModel):
        answer: str

    mock_resp = _mock_response('{"answer": "test"}', "gpt-4o-mini", 10, 5)
    captured_kwargs = {}

    def capture_completion(**kwargs):
        captured_kwargs.update(kwargs)
        return mock_resp

    import causaliq_knowledge.llm.client as client_module

    monkeypatch.setattr(
        client_module.litellm, "completion", capture_completion
    )
    monkeypatch.setattr(
        client_module.litellm, "completion_cost", lambda **kw: 0.0001
    )

    client = LLMClient()
    client.complete(user="Hello", response_format=TestFormat)

    assert "response_format" in captured_kwargs
    assert captured_kwargs["response_format"] == TestFormat
