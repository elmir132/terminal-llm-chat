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
        def __init__(self, host=None, **kwargs):
            pass

        async def chat(self, model, messages, stream, **kwargs):
            async def gen():
                for part in ("a", "b"):
                    yield {"message": {"content": part}}
            return gen()

    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(AsyncClient=FakeAsyncClient))
    assert collect(OllamaBackend(), [{"role": "user", "content": "x"}]) == ["a", "b"]


def test_ollama_connection_error_is_actionable(monkeypatch):
    class FakeAsyncClient:
        def __init__(self, host=None, **kwargs):
            pass

        async def chat(self, **kwargs):
            raise ConnectionError("refused")

    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(AsyncClient=FakeAsyncClient))
    with pytest.raises(BackendError, match="ollama serve"):
        collect(OllamaBackend(), [])


# ---- timeout, sampling options, model listing -----------------------------------------------------

def test_ollama_gets_the_timeout_and_the_sampling_options(monkeypatch):
    seen = {}

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            seen["client"] = kwargs

        async def chat(self, **kwargs):
            seen["chat"] = kwargs

            async def gen():
                yield {"message": {"content": "x"}}
            return gen()

    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(AsyncClient=FakeAsyncClient))
    collect(OllamaBackend(timeout=30, temperature=0.2, num_ctx=8192), [])
    assert seen["client"]["timeout"] == 30
    assert seen["chat"]["options"] == {"temperature": 0.2, "num_ctx": 8192}
    seen.clear()
    collect(OllamaBackend(), [])
    assert "timeout" not in seen["client"] and "options" not in seen["chat"]        # nothing extra unless asked


def test_openai_gets_the_timeout_and_temperature_only_when_set(monkeypatch):
    seen = {}

    class FakeStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    class FakeClient:
        def __init__(self, **kwargs):
            seen["client"] = kwargs
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

        async def create(self, **kwargs):
            seen["create"] = kwargs
            return FakeStream()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeClient))
    collect(OpenAIBackend(api_key="k", timeout=45, temperature=0), [])
    assert seen["client"]["timeout"] == 45 and seen["create"]["temperature"] == 0         # 0 is a real value
    collect(OpenAIBackend(api_key="k"), [])
    assert "timeout" not in seen["client"] and "temperature" not in seen["create"]


@pytest.mark.parametrize("error", [TimeoutError("timed out"), type("ReadTimeout", (Exception,), {})("x")])
def test_a_silent_server_gives_a_clear_timeout_message(monkeypatch, error):
    class FakeAsyncClient:
        def __init__(self, **kwargs):
            pass

        async def chat(self, **kwargs):
            raise error

    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(AsyncClient=FakeAsyncClient))
    with pytest.raises(BackendError, match=r"No reply from ollama for 20 seconds.*--timeout"):
        collect(OllamaBackend(timeout=20), [])


def test_openai_timeout_is_translated_too(monkeypatch):
    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

        async def create(self, **kwargs):
            raise type("APITimeoutError", (Exception,), {})("Request timed out.")

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeClient))
    with pytest.raises(BackendError, match="No reply from openai for 10 seconds"):
        collect(OpenAIBackend(api_key="k", timeout=10), [])


def test_model_listing_for_each_backend(monkeypatch):
    def listing(backend):
        async def go():
            return await backend.list_models()
        return asyncio.run(go())

    class NewOllama:                      # newer client: objects with a .model attribute
        def __init__(self, **kwargs):
            pass

        async def list(self):
            return types.SimpleNamespace(models=[types.SimpleNamespace(model="b:1"), types.SimpleNamespace(model="a:2")])

    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(AsyncClient=NewOllama))
    assert listing(OllamaBackend()) == ["a:2", "b:1"]

    class OldOllama(NewOllama):           # older client: plain dicts with "name"
        async def list(self):
            return {"models": [{"name": "z"}, {"name": "y"}]}

    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(AsyncClient=OldOllama))
    assert listing(OllamaBackend()) == ["y", "z"]

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.models = types.SimpleNamespace(list=self.list)

        async def list(self):
            return types.SimpleNamespace(data=[types.SimpleNamespace(id="gpt-b"), types.SimpleNamespace(id="gpt-a")])

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeOpenAI))
    assert listing(OpenAIBackend(api_key="k")) == ["gpt-a", "gpt-b"]
    assert listing(EchoBackend("e")) == ["e"]


def test_listing_failures_become_readable_errors(monkeypatch):
    class Down:
        def __init__(self, **kwargs):
            pass

        async def list(self):
            raise ConnectionError("refused")

    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(AsyncClient=Down))
    with pytest.raises(BackendError, match="ollama serve"):
        asyncio.run(OllamaBackend().list_models())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(BackendError, match="OPENAI_API_KEY"):
        asyncio.run(OpenAIBackend().list_models())


def test_create_backend_passes_the_options_through():
    backend = create_backend("ollama", timeout=9, temperature=0.5, num_ctx=4096)
    assert (backend.timeout, backend.temperature, backend.num_ctx) == (9, 0.5, 4096)
    assert create_backend("openai", timeout=3).timeout == 3
