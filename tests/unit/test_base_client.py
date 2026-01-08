"""Unit tests for BaseLLMClient abstract interface."""

from typing import Any, Dict, List

import pytest

from causaliq_knowledge.llm.base_client import (
    BaseLLMClient,
    LLMConfig,
    LLMResponse,
)

# --- LLMConfig Tests ---


# Test that LLMConfig has sensible default values
def test_llm_config_default_values():
    config = LLMConfig(model="test-model")
    assert config.model == "test-model"
    assert config.temperature == 0.1
    assert config.max_tokens == 500
    assert config.timeout == 30.0
    assert config.api_key is None


# Test that LLMConfig accepts custom values
def test_llm_config_custom_values():
    config = LLMConfig(
        model="custom-model",
        temperature=0.5,
        max_tokens=1000,
        timeout=60.0,
        api_key="test-key",
    )
    assert config.model == "custom-model"
    assert config.temperature == 0.5
    assert config.max_tokens == 1000
    assert config.timeout == 60.0
    assert config.api_key == "test-key"


# --- LLMResponse Tests ---


# Test basic LLMResponse creation with minimal fields
def test_llm_response_basic():
    response = LLMResponse(
        content="Hello, world!",
        model="test-model",
    )
    assert response.content == "Hello, world!"
    assert response.model == "test-model"
    assert response.input_tokens == 0
    assert response.output_tokens == 0
    assert response.cost == 0.0
    assert response.raw_response is None


# Test LLMResponse with all fields populated
def test_llm_response_full():
    raw = {"id": "test-id", "choices": []}
    response = LLMResponse(
        content="Test content",
        model="test-model",
        input_tokens=10,
        output_tokens=20,
        cost=0.001,
        raw_response=raw,
    )
    assert response.input_tokens == 10
    assert response.output_tokens == 20
    assert response.cost == 0.001
    assert response.raw_response == raw


# Test parsing valid JSON content
def test_llm_response_parse_json_valid():
    response = LLMResponse(
        content='{"key": "value", "number": 42}',
        model="test-model",
    )
    result = response.parse_json()
    assert result == {"key": "value", "number": 42}


# Test parsing JSON wrapped in markdown json code block
def test_llm_response_parse_json_with_markdown_code_block():
    response = LLMResponse(
        content='```json\n{"key": "value"}\n```',
        model="test-model",
    )
    result = response.parse_json()
    assert result == {"key": "value"}


# Test parsing JSON wrapped in generic code block
def test_llm_response_parse_json_with_generic_code_block():
    response = LLMResponse(
        content='```\n{"key": "value"}\n```',
        model="test-model",
    )
    result = response.parse_json()
    assert result == {"key": "value"}


# Test that invalid JSON returns None
def test_llm_response_parse_json_invalid():
    response = LLMResponse(
        content="This is not JSON",
        model="test-model",
    )
    result = response.parse_json()
    assert result is None


# Test that empty content returns None when parsing
def test_llm_response_parse_json_empty():
    response = LLMResponse(
        content="",
        model="test-model",
    )
    result = response.parse_json()
    assert result is None


# --- BaseLLMClient Tests ---


# Helper: Create a concrete mock implementation of BaseLLMClient
def _create_mock_client_class():
    class MockClient(BaseLLMClient):
        def __init__(self, config: LLMConfig) -> None:
            self.config = config
            self._calls = 0
            self._response_content = "Mock response"

        @property
        def provider_name(self) -> str:
            return "mock"

        def completion(
            self, messages: List[Dict[str, str]], **kwargs: Any
        ) -> LLMResponse:
            self._calls += 1
            return LLMResponse(
                content=self._response_content,
                model=self.config.model,
            )

        @property
        def call_count(self) -> int:
            return self._calls

    return MockClient


# Test that BaseLLMClient cannot be instantiated directly
def test_base_llm_client_cannot_instantiate():
    with pytest.raises(TypeError, match="abstract"):
        BaseLLMClient(LLMConfig(model="test"))  # type: ignore


# Test that a concrete implementation works correctly
def test_base_llm_client_concrete_implementation():
    MockClient = _create_mock_client_class()
    config = LLMConfig(model="mock-model")
    client = MockClient(config)

    assert client.provider_name == "mock"
    assert client.call_count == 0
    assert client.model_name == "mock-model"

    response = client.completion([{"role": "user", "content": "Hello"}])
    assert response.content == "Mock response"
    assert client.call_count == 1


# Test that complete_json calls completion and parses the response
def test_base_llm_client_complete_json():
    MockClient = _create_mock_client_class()
    config = LLMConfig(model="mock-model")
    client = MockClient(config)
    client._response_content = '{"exists": true, "confidence": 0.9}'

    parsed, response = client.complete_json(
        [{"role": "user", "content": "Test"}]
    )

    assert parsed == {"exists": True, "confidence": 0.9}
    assert response.content == '{"exists": true, "confidence": 0.9}'
    assert client.call_count == 1


# --- Inheritance Verification Tests ---


# Test that GroqClient properly inherits from BaseLLMClient
def test_groq_client_inherits_from_base():
    from causaliq_knowledge.llm.groq_client import GroqClient

    assert issubclass(GroqClient, BaseLLMClient)


# Test that GeminiClient properly inherits from BaseLLMClient
def test_gemini_client_inherits_from_base():
    from causaliq_knowledge.llm.gemini_client import GeminiClient

    assert issubclass(GeminiClient, BaseLLMClient)


# Test that GroqConfig properly inherits from LLMConfig
def test_groq_config_inherits_from_base():
    from causaliq_knowledge.llm.groq_client import GroqConfig

    assert issubclass(GroqConfig, LLMConfig)


# Test that GeminiConfig properly inherits from LLMConfig
def test_gemini_config_inherits_from_base():
    from causaliq_knowledge.llm.gemini_client import GeminiConfig

    assert issubclass(GeminiConfig, LLMConfig)
