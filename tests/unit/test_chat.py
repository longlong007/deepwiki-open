from types import SimpleNamespace

import pytest

from api.chat import ChatStreamer
from api.chat._stream import (
    text_from_openai_stream_event,
    AnthropicChatStreamer,
    AzureChatStreamer,
    BedrockChatStreamer,
    DashScopeChatStreamer,
    GoogleGenerativeChatStreamer,
    LiteLLMChatStreamer,
    OllamaChatStreamer,
    OpenAIChatStreamer,
    OpenRouterChatStreamer,
)


@pytest.mark.parametrize(
    "provider, expected",
    [
        ("ollama", OllamaChatStreamer),
        ("openrouter", OpenRouterChatStreamer),
        ("openai", OpenAIChatStreamer),
        ("azure", AzureChatStreamer),
        ("bedrock", BedrockChatStreamer),
        ("dashscope", DashScopeChatStreamer),
        ("google", GoogleGenerativeChatStreamer),
        ("litellm", LiteLLMChatStreamer),
        ("anthropic", AnthropicChatStreamer),
    ],
)
def test_every_provider_is_registered(provider, expected):
    assert ChatStreamer._registry[provider] is expected


@pytest.mark.parametrize(
    "provider, expected",
    [
        ("ollama", OllamaChatStreamer),
        ("openrouter", OpenRouterChatStreamer),
        ("openai", OpenAIChatStreamer),
        ("azure", AzureChatStreamer),
        ("bedrock", BedrockChatStreamer),
        ("dashscope", DashScopeChatStreamer),
        ("google", GoogleGenerativeChatStreamer),
        ("litellm", LiteLLMChatStreamer),
        ("anthropic", AnthropicChatStreamer),
    ],
)
def test_create_returns_correct_subclass(monkeypatch, provider, expected):
    monkeypatch.setattr(expected, "__init__", lambda self, **kw: None)
    s = ChatStreamer.create(provider=provider, model="m", model_config={"model": "m"})
    assert isinstance(s, expected)


def test_create_unknown_provider_raises():
    with pytest.raises(RuntimeError, match="not registered"):
        ChatStreamer.create(provider="nope", model=None, model_config={})


def test_response_created_event_is_ignored():
    event = SimpleNamespace(type="response.created")
    assert text_from_openai_stream_event(event) is None


def test_response_text_delta_is_returned():
    event = SimpleNamespace(type="response.output_text.delta", delta="<wiki_structure>")
    assert text_from_openai_stream_event(event) == "<wiki_structure>"


def test_chat_completion_chunk_still_returns_content():
    chunk = SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content="hello"))]
    )
    assert text_from_openai_stream_event(chunk) == "hello"


def test_empty_chat_completion_chunk_is_ignored():
    chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))])
    assert text_from_openai_stream_event(chunk) is None


def test_response_error_event_raises():
    event = SimpleNamespace(type="error", message="bad key")
    with pytest.raises(RuntimeError, match="bad key"):
        text_from_openai_stream_event(event)
