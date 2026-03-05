"""Model backends. Each one turns a message list into a stream of text chunks."""
import asyncio
import os
from typing import AsyncIterator, Dict, List

Messages = List[Dict[str, str]]


class BackendError(Exception):
    """A backend failed in a way worth showing to the user."""


def _is_timeout(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return "timeout" in text or "timed out" in text


class Backend:
    name = "base"
    icon = "•"
    tagline = ""

    def __init__(self, model: str, timeout=None, temperature=None, num_ctx=None):
        self.model = model
        self.timeout = timeout            # seconds without any data before giving up (None = wait for ever)
        self.temperature = temperature
        self.num_ctx = num_ctx

    def _timeout_error(self) -> "BackendError":
        return BackendError(f"No reply from {self.name} for {self.timeout:g} seconds. "
                            "The server may be busy or stuck; use --timeout to wait longer.")

    @property
    def title(self) -> str:
        return f"{self.icon} {self.name} ({self.model})"

    def preflight(self):
        """A warning to show at startup, or None."""
        return None

    async def stream(self, messages: Messages) -> AsyncIterator[str]:
        raise NotImplementedError
        yield ""  # pragma: no cover  (marks this as an async generator)

    async def list_models(self):
        """Names of the models this backend can use, for /models."""
        raise BackendError(f"{self.name} cannot list models")


class OllamaBackend(Backend):
    """A model running locally through Ollama. Needs the `ollama` extra."""
    name = "ollama"
    icon = "🦙"
    tagline = "Local • Private"
    default_model = "deepseek-r1:14b"

    def __init__(self, model=None, host=None, **options):
        super().__init__(model or self.default_model, **options)
        self.host = host

    def _client(self):
        try:
            import ollama
        except ImportError as exc:
            raise BackendError("Install the ollama extra: pip install 'tuichat[ollama]'") from exc
        kwargs = {"host": self.host}
        if self.timeout:
            kwargs["timeout"] = self.timeout
        return ollama.AsyncClient(**kwargs)

    def _options(self):
        options = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        return options

    async def stream(self, messages):
        client = self._client()
        extra = {"options": self._options()} if self._options() else {}
        try:
            response = await client.chat(model=self.model, messages=messages, stream=True, **extra)
            async for chunk in response:
                yield chunk["message"]["content"]
        except Exception as exc:  # connection refused, unknown model, ...
            if _is_timeout(exc):
                raise self._timeout_error() from exc
            raise BackendError(f"Ollama error: {exc}. Is `ollama serve` running and is "
                               f"`{self.model}` pulled?") from exc


    async def list_models(self):
        client = self._client()
        try:
            response = await client.list()
        except Exception as exc:
            raise BackendError(f"Could not list Ollama models: {exc}. Is `ollama serve` running?") from exc
        items = response.get("models", []) if isinstance(response, dict) else getattr(response, "models", [])
        names = []
        for item in items:
            name = (item.get("model") or item.get("name")) if isinstance(item, dict) else (getattr(item, "model", None) or getattr(item, "name", None))
            if name:
                names.append(name)
        return sorted(names)


class OpenAIBackend(Backend):
    """OpenAI or any OpenAI-compatible server (set base_url). Needs the `openai` extra."""
    name = "openai"
    icon = "🤖"
    tagline = "Cloud API • /model to change"
    default_model = "gpt-4o"

    def __init__(self, model=None, base_url=None, api_key=None, **options):
        super().__init__(model or self.default_model, **options)
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def preflight(self):
        if not self.api_key:
            return "OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY='your-key'"
        return None

    def _client(self):
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise BackendError("Install the openai extra: pip install 'tuichat[openai]'") from exc
        kwargs = {"api_key": self.api_key, "base_url": self.base_url}
        if self.timeout:
            kwargs["timeout"] = self.timeout
        return AsyncOpenAI(**kwargs)

    async def list_models(self):
        if not self.api_key:
            raise BackendError("OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY='your-key'")
        client = self._client()
        try:
            response = await client.models.list()
        except Exception as exc:
            raise BackendError(f"Could not list models: {exc}") from exc
        return sorted(getattr(item, "id", "") for item in getattr(response, "data", []) if getattr(item, "id", ""))

    async def stream(self, messages):
        if not self.api_key:
            raise BackendError("OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY='your-key'")
        client = self._client()
        extra = {"temperature": self.temperature} if self.temperature is not None else {}
        try:
            stream = await client.chat.completions.create(model=self.model, messages=messages, stream=True, **extra)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as exc:
            if _is_timeout(exc):
                raise self._timeout_error() from exc
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

    def __init__(self, model=None, delay=0.0, think=False, **options):
        super().__init__(model or self.default_model, **options)
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


    async def list_models(self):
        return [self.model]


def create_backend(kind: str, model=None, base_url=None, timeout=None, temperature=None, num_ctx=None) -> Backend:
    options = {"timeout": timeout, "temperature": temperature}
    if kind == "ollama":
        return OllamaBackend(model, num_ctx=num_ctx, **options)
    if kind == "openai":
        return OpenAIBackend(model, base_url=base_url, **options)
    if kind == "echo":
        return EchoBackend(model, delay=0.03, think=True, **options)
    raise ValueError(f"Unknown backend: {kind}")
