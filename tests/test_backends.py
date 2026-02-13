import asyncio
import sys
import types

import pytest

from tuichat.backends import (BackendError, EchoBackend, OllamaBackend, OpenAIBackend,
                              create_backend)


def collect(backend, messages):
    async def go():
        return [c async for c in backend.stream(messages)]
    return asyncio.run(go())


def test_echo_streams_a_reply_that_mentions_the_prompt():
    text = "".join(collect(EchoBackend(), [{"role": "user", "content": "ping"}]))
    assert "ping" in text


def test_create_backend_dispatch_and_defaults():
    assert isinstance(create_backend("ollama"), OllamaBackend)
    assert create_backend("ollama").model == "deepseek-r1:14b"
    assert create_backend("openai", model="m", base_url="http://x/v1").base_url == "http://x/v1"
    assert isinstance(create_backend("echo"), EchoBackend)
    with pytest.raises(ValueError):
        create_backend("nope")


def test_openai_without_key_warns_and_refuses(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = OpenAIBackend()
    assert "OPENAI_API_KEY" in backend.preflight()
    with pytest.raises(BackendError, match="OPENAI_API_KEY"):
        collect(backend, [])


def test_openai_with_key_has_no_warning():
    assert OpenAIBackend(api_key="k").preflight() is None


def test_openai_streams_deltas_and_skips_empty_ones(monkeypatch):
    def chunk(text):
        delta = types.SimpleNamespace(content=text)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)])

    class FakeStream:
        def __init__(self):
            self.items = [chunk("Hel"), chunk(None), types.SimpleNamespace(choices=[]), chunk("lo")]

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.items:
                raise StopAsyncIteration
            return self.items.pop(0)

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

        async def create(self, **kwargs):
            assert kwargs["stream"] is True and kwargs["model"] == "m"
            return FakeStream()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeClient))
    assert collect(OpenAIBackend(model="m", api_key="k"), [{"role": "user", "content": "hi"}]) == ["Hel", "lo"]


def test_openai_auth_error_is_translated(monkeypatch):
    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

        async def create(self, **kwargs):
            raise RuntimeError("Authentication failed")

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeClient))
    with pytest.raises(BackendError, match="Authentication failed: check OPENAI_API_KEY"):
        collect(OpenAIBackend(api_key="bad"), [])


def test_ollama_streams_content(monkeypatch):
    class FakeAsyncClient:
        def __init__(self, host=None):
            pass

        async def chat(self, model, messages, stream):
            async def gen():
                for part in ("a", "b"):
                    yield {"message": {"content": part}}
            return gen()

    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(AsyncClient=FakeAsyncClient))
    assert collect(OllamaBackend(), [{"role": "user", "content": "x"}]) == ["a", "b"]


def test_ollama_connection_error_is_actionable(monkeypatch):
    class FakeAsyncClient:
        def __init__(self, host=None):
            pass

        async def chat(self, **kwargs):
            raise ConnectionError("refused")

    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(AsyncClient=FakeAsyncClient))
    with pytest.raises(BackendError, match="ollama serve"):
        collect(OllamaBackend(), [])
