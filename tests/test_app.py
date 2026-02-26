"""Headless tests of the chat application: no terminal, fake backends."""
import asyncio

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from tuichat.app import ChatApp
from tuichat.backends import Backend, BackendError, EchoBackend


class ScriptedBackend(Backend):
    name = "scripted"

    def __init__(self, chunks=None, error=None, gate=None):
        super().__init__("m")
        self.chunks, self.error, self.gate = chunks or ["ok"], error, gate
        self.seen = []

    async def stream(self, messages):
        self.seen.append([dict(m) for m in messages])
        if self.gate:
            await self.gate.wait()
        for chunk in self.chunks:
            yield chunk
        if self.error:
            raise self.error


def run(coro_factory):
    async def go():
        with create_pipe_input() as pipe:
            return await coro_factory(pipe)
    return asyncio.run(go())


def make(backend, pipe):
    return ChatApp(backend, input=pipe, output=DummyOutput(), columns=lambda: 80)


async def settle(chat):
    while chat._task and not chat._task.done():
        await asyncio.sleep(0.01)
    await chat._task if chat._task else None


def test_reply_is_streamed_shown_and_recorded():
    async def scenario(pipe):
        chat = make(EchoBackend(), pipe)
        chat.submit("ping")
        await settle(chat)
        return chat

    chat = run(scenario)
    assert "You said: ping" in chat.output_area.text
    assert "👤 You" in chat.output_area.text
    assert [m["role"] for m in chat.messages] == ["user", "assistant"]
    assert chat.last_user_msg == "ping" and "ping" in chat.last_ai_msg
    assert "thinking" not in chat.output_area.text


def test_think_blocks_are_hidden_from_output_and_history():
    async def scenario(pipe):
        chat = make(ScriptedBackend(["<think>secret</think>", "visible"]), pipe)
        chat.submit("q")
        await settle(chat)
        return chat

    chat = run(scenario)
    assert "visible" in chat.output_area.text
    assert "secret" not in chat.output_area.text
    assert chat.messages[-1] == {"role": "assistant", "content": "visible"}


def test_second_message_sends_the_whole_history():
    async def scenario(pipe):
        backend = ScriptedBackend(["a"])
        chat = make(backend, pipe)
        chat.submit("one")
        await settle(chat)
        chat.submit("two")
        await settle(chat)
        return backend

    backend = run(scenario)
    assert [m["content"] for m in backend.seen[1]] == ["one", "a", "two"]


def test_messages_sent_while_streaming_are_queued_and_answered_in_order():
    async def scenario(pipe):
        gate = asyncio.Event()
        backend = ScriptedBackend(["r"], gate=gate)
        chat = make(backend, pipe)
        chat.submit("first")
        await asyncio.sleep(0.05)
        assert chat.streaming
        chat.submit("second")
        chat.submit("third")
        assert "Queued (2 pending)" in chat.output_area.text
        gate.set()
        await settle(chat)
        return chat, backend

    chat, backend = run(scenario)
    assert [m["content"] for m in chat.messages if m["role"] == "user"] == ["first", "second", "third"]
    assert len(backend.seen) == 3
    assert not chat.streaming and not chat.queue


def test_backend_error_is_shown_and_history_stays_clean():
    async def scenario(pipe):
        chat = make(ScriptedBackend([], error=BackendError("model not pulled")), pipe)
        chat.submit("hello")
        await settle(chat)
        return chat

    chat = run(scenario)
    assert "❌ model not pulled" in chat.output_area.text
    assert chat.messages == []
    assert not chat.streaming


def test_unexpected_exception_does_not_kill_the_app():
    async def scenario(pipe):
        chat = make(ScriptedBackend([], error=RuntimeError("boom")), pipe)
        chat.submit("hello")
        await settle(chat)
        chat.backend = ScriptedBackend(["fine"])
        chat.submit("again")
        await settle(chat)
        return chat

    chat = run(scenario)
    assert "Unexpected error: boom" in chat.output_area.text
    assert chat.messages[-1]["content"] == "fine"


def test_model_command_changes_and_reports_the_model():
    async def scenario(pipe):
        chat = make(ScriptedBackend(), pipe)
        chat.submit("/model gpt-x")
        chat.submit("/model")
        return chat

    chat = run(scenario)
    assert chat.backend.model == "gpt-x"
    assert "Model changed to: gpt-x" in chat.output_area.text
    assert "Current model: gpt-x" in chat.output_area.text
    assert chat.messages == []


def test_clear_command_resets_the_conversation():
    async def scenario(pipe):
        chat = make(ScriptedBackend(), pipe)
        chat.submit("hi")
        await settle(chat)
        chat.submit("/clear")
        return chat

    chat = run(scenario)
    assert chat.messages == []
    assert "Conversation cleared" in chat.output_area.text


def test_read_command_loads_files_into_context(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("remember the milk")

    async def scenario(pipe):
        chat = make(ScriptedBackend(), pipe)
        chat.submit(f"/read {f} {tmp_path / 'missing.txt'}")
        return chat

    chat = run(scenario)
    assert "📁 notes.txt (17 chars)" in chat.output_area.text
    assert "❌ Not found" in chat.output_area.text
    assert "remember the milk" in chat.messages[0]["content"]


def test_missing_api_key_warning_is_shown_at_startup(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from tuichat.backends import OpenAIBackend

    async def scenario(pipe):
        return make(OpenAIBackend(), pipe)

    assert "OPENAI_API_KEY is not set" in run(scenario).output_area.text


def test_blank_input_is_ignored():
    async def scenario(pipe):
        chat = make(ScriptedBackend(), pipe)
        chat.submit("   ")
        return chat

    chat = run(scenario)
    assert chat.messages == [] and chat._task is None


def test_full_application_loop_with_keystrokes():
    """Type a message, press Enter, wait for the reply, then Ctrl-D to exit."""
    async def scenario(pipe):
        chat = make(EchoBackend(), pipe)
        runner = asyncio.ensure_future(chat.run())
        await asyncio.sleep(0.2)
        pipe.send_text("hello there\r")
        for _ in range(100):
            await asyncio.sleep(0.05)
            if chat.last_ai_msg:
                break
        pipe.send_text("\x04")  # Ctrl-D
        await asyncio.wait_for(runner, timeout=3)
        return chat

    chat = run(scenario)
    assert "hello there" in chat.last_ai_msg


# ---- stopping a reply (Ctrl+C) ---------------------------------------------------------------

class SlowBackend(Backend):
    name = "slow"

    def __init__(self):
        super().__init__("m")

    async def stream(self, messages):
        yield "part one "
        await asyncio.sleep(10)
        yield "never shown"


def test_stop_keeps_the_partial_reply_and_drops_the_queue():
    async def scenario(pipe):
        chat = make(SlowBackend(), pipe)
        chat.submit("first")
        await asyncio.sleep(0.05)
        chat.submit("second")
        chat.submit("third")
        assert chat.stop() is True
        await settle(chat)
        return chat

    chat = run(scenario)
    assert "part one" in chat.output_area.text and "never shown" not in chat.output_area.text
    assert "⏹ Stopped (2 queued messages dropped)" in chat.output_area.text
    assert [m["role"] for m in chat.messages] == ["user", "assistant"]
    assert chat.messages[-1]["content"].strip() == "part one"
    assert chat.queue == [] and not chat.streaming


def test_stop_before_any_text_removes_the_unanswered_question():
    async def scenario(pipe):
        chat = make(ScriptedBackend(gate=asyncio.Event()), pipe)
        chat.submit("hello")
        await asyncio.sleep(0.05)
        chat.stop()
        await settle(chat)
        return chat

    chat = run(scenario)
    assert chat.messages == []
    assert "⏹ Stopped" in chat.output_area.text and "dropped" not in chat.output_area.text


def test_stop_when_idle_does_nothing_and_the_app_can_continue_afterwards():
    async def scenario(pipe):
        chat = make(ScriptedBackend(["fine"]), pipe)
        assert chat.stop() is False
        chat.submit("a")
        await settle(chat)
        chat.backend = SlowBackend()
        chat.submit("b")
        await asyncio.sleep(0.05)
        chat.stop()
        await settle(chat)
        chat.backend = ScriptedBackend(["again"])
        chat.submit("c")
        await settle(chat)
        return chat

    chat = run(scenario)
    assert chat.messages[-1] == {"role": "assistant", "content": "again"}
    assert not chat.streaming
