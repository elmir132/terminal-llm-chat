"""Model backends. Each one turns a message list into a stream of text chunks."""
import asyncio
import os
from typing import AsyncIterator, Dict, List

Messages = List[Dict[str, str]]


class BackendError(Exception):
    """A backend failed in a way worth showing to the user."""


class Backend:
    name = "base"
    icon = "•"
    tagline = ""

    def __init__(self, model: str):
        self.model = model

    @property
    def title(self) -> str:
        return f"{self.icon} {self.name} ({self.model})"

    def preflight(self):
        """A warning to show at startup, or None."""
        return None

    async def stream(self, messages: Messages) -> AsyncIterator[str]:
        raise NotImplementedError
        yield ""  # pragma: no cover  (marks this as an async generator)


class OllamaBackend(Backend):
    """A model running locally through Ollama. Needs the `ollama` extra."""
    name = "ollama"
    icon = "🦙"
    tagline = "Local • Private"
    default_model = "deepseek-r1:14b"

    def __init__(self, model=None, host=None):
        super().__init__(model or self.default_model)
        self.host = host

    async def stream(self, messages):
        try:
            import ollama
        except ImportError as exc:
            raise BackendError("Install the ollama extra: pip install 'tuichat[ollama]'") from exc
        client = ollama.AsyncClient(host=self.host)
        try:
            response = await client.chat(model=self.model, messages=messages, stream=True)
            async for chunk in response:
                yield chunk["message"]["content"]
        except Exception as exc:  # connection refused, unknown model, ...
            raise BackendError(f"Ollama error: {exc}. Is `ollama serve` running and is "
                               f"`{self.model}` pulled?") from exc


class OpenAIBackend(Backend):
    """OpenAI or any OpenAI-compatible server (set base_url). Needs the `openai` extra."""
    name = "openai"
    icon = "🤖"
    tagline = "Cloud API • /model to change"
    default_model = "gpt-4o"

    def __init__(self, model=None, base_url=None, api_key=None):
        super().__init__(model or self.default_model)
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def preflight(self):
        if not self.api_key:
            return "OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY='your-key'"
        return None

    async def stream(self, messages):
        if not self.api_key:
            raise BackendError("OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY='your-key'")
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise BackendError("Install the openai extra: pip install 'tuichat[openai]'") from exc
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        try:
            stream = await client.chat.completions.create(model=self.model, messages=messages, stream=True)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as exc:
            text = str(exc)
            if "api_key" in text.lower() or "authentication" in text.lower():
                raise BackendError("Authentication failed: check OPENAI_API_KEY") from exc
            raise BackendError(f"OpenAI error: {text}") from exc


class EchoBackend(Backend):
    """Offline demo backend: no model, no network. Used for tests and screenshots."""
    name = "echo"
    icon = "🔁"
    tagline = "Demo • no model"
    default_model = "echo-1"

    def __init__(self, model=None, delay=0.0, think=False):
        super().__init__(model or self.default_model)
        self.delay = delay
        self.think = think

    async def stream(self, messages):
        last = messages[-1]["content"] if messages else ""
        reply = f"You said: {last}\n\nThis is a **demo** reply from the echo backend."
        if self.think:
            yield "<think>Pretending to reason about the question…</think>"
        for word in reply.split(" "):
            if self.delay:
                await asyncio.sleep(self.delay)
            yield word + " "


def create_backend(kind: str, model=None, base_url=None) -> Backend:
    if kind == "ollama":
        return OllamaBackend(model)
    if kind == "openai":
        return OpenAIBackend(model, base_url=base_url)
    if kind == "echo":
        return EchoBackend(model, delay=0.03, think=True)
    raise ValueError(f"Unknown backend: {kind}")
