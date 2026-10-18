"""Headless tests of the chat application: no terminal, fake backends."""
import asyncio
import types

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from tuichat.app import ChatApp
from tuichat.files import build_context_message
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


# ---- system prompt --------------------------------------------------------------------------------

def test_the_system_prompt_is_sent_first_and_never_stored_in_the_history():
    async def scenario(pipe):
        backend = ScriptedBackend(["ok"])
        chat = ChatApp(backend, input=pipe, output=DummyOutput(), columns=lambda: 80, system="Answer in French.")
        chat.submit("hi")
        await settle(chat)
        return chat, backend

    chat, backend = run(scenario)
    assert backend.seen[0][0] == {"role": "system", "content": "Answer in French."}
    assert backend.seen[0][1] == {"role": "user", "content": "hi"}
    assert all(m["role"] != "system" for m in chat.messages)
    assert "System prompt set (17 characters)" in chat.output_area.text


def test_system_command_shows_sets_and_removes_the_prompt():
    async def scenario(pipe):
        backend = ScriptedBackend(["ok"])
        chat = make(backend, pipe)
        chat.submit("/system")
        assert "No system prompt" in chat.output_area.text
        chat.submit("/system You are a pirate.")
        chat.submit("/system")
        chat.submit("go")
        await settle(chat)
        chat.submit("/system off")
        chat.submit("again")
        await settle(chat)
        return chat, backend

    chat, backend = run(scenario)
    assert "System prompt (17 characters):\nYou are a pirate." in chat.output_area.text
    assert "System prompt removed" in chat.output_area.text
    assert backend.seen[0][0] == {"role": "system", "content": "You are a pirate."}
    assert backend.seen[1][0]["role"] == "user"                     # removed again


def test_clear_keeps_the_system_prompt_and_an_overlong_one_is_refused():
    async def scenario(pipe):
        chat = make(ScriptedBackend(), pipe)
        chat.submit("/system Be brief.")
        chat.submit("hi")
        await settle(chat)
        chat.submit("/clear")
        chat.submit("/system " + "x" * 5000)
        return chat

    chat = run(scenario)
    assert chat.system == "Be brief."
    assert "limited to 4000 characters" in chat.output_area.text


def test_cli_reads_the_system_prompt_from_text_or_file(tmp_path):
    import pytest
    from tuichat.cli import build_parser, resolve_system
    parser = build_parser()
    assert resolve_system(parser, parser.parse_args(["--system", "Hello"])) == "Hello"
    f = tmp_path / "s.txt"
    f.write_text("From file\n", encoding="utf-8")
    assert resolve_system(parser, parser.parse_args(["--system-file", str(f)])) == "From file\n"
    assert resolve_system(parser, parser.parse_args([])) == ""
    for bad in (["--system", "a", "--system-file", str(f)], ["--system-file", str(tmp_path / "missing")]):
        with pytest.raises(SystemExit):
            resolve_system(parser, parser.parse_args(bad))
    empty = tmp_path / "e.txt"
    empty.write_text("  \n", encoding="utf-8")
    with pytest.raises(SystemExit):
        resolve_system(parser, parser.parse_args(["--system-file", str(empty)]))


# ---- /save, /load, /saves ---------------------------------------------------------------------------

@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("TUICHAT_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def test_save_then_load_restores_the_conversation_and_the_system_prompt(home):
    async def scenario(pipe):
        chat = make(ScriptedBackend(["answer one"]), pipe)
        chat.submit("/system Be brief.")
        chat.submit("question one")
        await settle(chat)
        chat.submit("/save talk")
        assert "Saved as 'talk' (2 messages)" in chat.output_area.text
        chat.submit("/clear")
        chat.submit("/system off")
        assert chat.messages == [] and chat.system == ""
        chat.submit("/load talk")
        return chat

    chat = run(scenario)
    assert [m["content"] for m in chat.messages] == ["question one", "answer one"]
    assert chat.system == "Be brief."
    assert "question one" in chat.output_area.text and "answer one" in chat.output_area.text
    assert "Loaded 'talk' (2 messages)" in chat.output_area.text
    assert chat.last_user_msg == "question one" and chat.last_ai_msg == "answer one"


def test_a_loaded_conversation_is_sent_to_the_model_with_the_next_question(home):
    async def scenario(pipe):
        first = make(ScriptedBackend(["a1"]), pipe)
        first.submit("q1")
        await settle(first)
        first.submit("/save s")
        backend = ScriptedBackend(["a2"])
        second = make(backend, pipe)
        second.submit("/load s")
        second.submit("q2")
        await settle(second)
        return backend

    backend = run(scenario)
    assert [m["content"] for m in backend.seen[0]] == ["q1", "a1", "q2"]


def test_save_refuses_to_overwrite_and_reports_bad_names_and_empty_chats(home):
    async def scenario(pipe):
        chat = make(ScriptedBackend(["x"]), pipe)
        chat.submit("/save nothing-yet")
        chat.submit("hi")
        await settle(chat)
        chat.submit("/save keep")
        chat.submit("/save keep")
        chat.submit("/save -f keep")
        chat.submit("/save ../evil")
        chat.submit("/load missing")
        chat.submit("/load")
        return chat

    text = run(scenario).output_area.text
    assert "nothing to save" in text
    assert "'keep' already exists" in text
    assert text.count("Saved as 'keep'") == 2
    assert "A name can use letters" in text and "No saved conversation called 'missing'" in text
    assert "Give a name: /load NAME" in text


def test_saves_lists_conversations_and_load_waits_for_a_running_reply(home):
    async def scenario(pipe):
        chat = make(ScriptedBackend(["x"]), pipe)
        chat.submit("/saves")
        chat.submit("hi")
        await settle(chat)
        chat.submit("/save one")
        chat.submit("/saves")
        busy = make(ScriptedBackend(gate=asyncio.Event()), pipe)
        busy.submit("hello")
        await asyncio.sleep(0.05)
        busy.submit("/load one")
        busy.stop()
        await settle(busy)
        return chat, busy

    chat, busy = run(scenario)
    assert "No saved conversations yet" in chat.output_area.text
    assert "Saved conversations:\n  one  (2 messages" in chat.output_area.text
    assert "Wait for the reply to finish" in busy.output_area.text


def test_loading_a_conversation_with_loaded_files_shows_a_short_note_not_the_file_text(home):
    async def scenario(pipe):
        chat = make(ScriptedBackend(["x"]), pipe)
        chat.messages = [{"role": "user", "content": build_context_message([("a.txt", "SECRET BODY")])},
                         {"role": "user", "content": "what is in it?"}, {"role": "assistant", "content": "text"}]
        chat.submit("/save files")
        other = make(ScriptedBackend(), pipe)
        other.submit("/load files")
        return other

    other = run(scenario)
    assert "files loaded into the conversation" in other.output_area.text
    assert "SECRET BODY" not in other.output_area.text
    assert other.last_user_msg == "what is in it?"


# ---- context limit ---------------------------------------------------------------------------------

def test_old_messages_are_left_out_of_the_request_but_stay_on_screen_and_in_the_history():
    async def scenario(pipe):
        backend = ScriptedBackend(["r" * 30])
        chat = ChatApp(backend, input=pipe, output=DummyOutput(), columns=lambda: 80, max_context=150)
        for question in ("first " + "a" * 40, "second " + "b" * 40, "third " + "c" * 40):
            chat.submit(question)
            await settle(chat)
        return chat, backend

    chat, backend = run(scenario)
    assert [m["content"][:5] for m in backend.seen[0]] == ["first"]
    assert len(backend.seen[1]) == 3                                # still fits on the second turn
    assert [m["content"][:6] for m in backend.seen[2]] == ["second", "rrrrrr", "third "]       # the first pair is left out
    assert len(chat.messages) == 6                                  # the full conversation is still kept
    assert "left out so the request stays under 150 characters" in chat.output_area.text
    assert "first aaaa" in chat.output_area.text


def test_the_notice_is_shown_once_per_change_and_a_limit_of_zero_never_trims():
    async def scenario(pipe):
        backend = ScriptedBackend(["r" * 30])
        chat = ChatApp(backend, input=pipe, output=DummyOutput(), columns=lambda: 80, max_context=0)
        for n in range(4):
            chat.submit("q" * 60)
            await settle(chat)
        return chat, backend

    chat, backend = run(scenario)
    assert len(backend.seen[-1]) == 7 and "left out" not in chat.output_area.text


# ---- code blocks ----------------------------------------------------------------------------------------

def test_the_chat_lexer_styles_fences_and_code_but_not_normal_text():
    from prompt_toolkit.document import Document
    from tuichat.app import ChatLexer
    document = Document("plain **bold**\n```py\nx = **1**\n```\nafter")
    style = ChatLexer().lex_document(document)
    assert style(0)[1] == ("class:bold", "bold")
    assert style(1) == [("class:code-fence", "```py")]
    assert style(2) == [("class:code", "x = **1**")]            # no bold inside code
    assert style(3) == [("class:code-fence", "```")]
    assert style(4) == [("", "after")]


def test_ctrl_k_copies_the_last_code_block_of_the_last_reply(monkeypatch):
    copied = []
    monkeypatch.setattr("tuichat.app.copy_to_clipboard", lambda text: copied.append(text) or True)

    async def scenario(pipe):
        chat = make(ScriptedBackend(["Try:\n```sh\nls\n```\nor\n```sh\npwd\n```\n"]), pipe)
        chat.submit("how?")
        await settle(chat)
        binding = next(b for b in chat.app.key_bindings.bindings if b.keys and str(b.keys[0]) in ("c-k", "Keys.ControlK"))
        binding.handler(types.SimpleNamespace(app=chat.app))
        return chat

    chat = run(scenario)
    assert copied == ["pwd"]
    assert "Code block copied to clipboard" in chat.output_area.text


def test_ctrl_k_says_so_when_there_is_no_code(monkeypatch):
    monkeypatch.setattr("tuichat.app.copy_to_clipboard", lambda text: True)

    async def scenario(pipe):
        chat = make(ScriptedBackend(["just words"]), pipe)
        chat.submit("hi")
        await settle(chat)
        binding = next(b for b in chat.app.key_bindings.bindings if b.keys and str(b.keys[0]) in ("c-k", "Keys.ControlK"))
        binding.handler(types.SimpleNamespace(app=chat.app))
        return chat

    assert "No code block in the last reply" in run(scenario).output_area.text


# ---- /help, /models and command-line options -----------------------------------------------------------

class ListingBackend(ScriptedBackend):
    def __init__(self, names=None, error=None):
        super().__init__()
        self.names, self.listing_error = names, error

    async def list_models(self):
        if self.listing_error:
            raise self.listing_error
        return self.names


def test_help_lists_commands_and_keys():
    async def scenario(pipe):
        chat = make(ScriptedBackend(), pipe)
        chat.submit("/help")
        return chat

    text = run(scenario).output_area.text
    for expected in ("/models", "/save", "/system", "Ctrl+K", "Ctrl+C stop the reply"):
        assert expected in text


def test_models_lists_the_choices_and_marks_the_current_one():
    async def scenario(pipe):
        chat = make(ListingBackend(["m", "other"]), pipe)
        chat.submit("/models")
        await asyncio.sleep(0.05)
        return chat

    text = run(scenario).output_area.text
    assert "Models on scripted:" in text and "  m  (current)" in text and "  other\n" in text


def test_models_failure_is_shown_not_raised():
    async def scenario(pipe):
        chat = make(ListingBackend(error=BackendError("server down")), pipe)
        chat.submit("/models")
        await asyncio.sleep(0.05)
        chat.backend = ListingBackend(error=RuntimeError("weird"))
        chat.submit("/models")
        await asyncio.sleep(0.05)
        chat.backend = ListingBackend([])
        chat.submit("/models")
        await asyncio.sleep(0.05)
        return chat

    text = run(scenario).output_area.text
    assert "❌ server down" in text and "Could not list models: weird" in text and "No models found" in text


def test_command_line_options_are_checked():
    import pytest
    from tuichat.cli import build_parser, check_options
    parser = build_parser()

    def ok(*argv):
        check_options(parser, parser.parse_args(list(argv)))

    ok("--timeout", "0", "--temperature", "0", "--num-ctx", "4096")
    for bad in (["--timeout", "-1"], ["--temperature", "2.5"], ["--temperature", "-0.1"],
                ["--num-ctx", "100"], ["-b", "openai", "--num-ctx", "4096"]):
        with pytest.raises(SystemExit):
            ok(*bad)


def test_clear_while_streaming_stops_the_reply_drops_the_queue_and_leaves_nothing_behind():
    async def scenario(pipe):
        chat = make(SlowBackend(), pipe)
        chat.submit("first")
        await asyncio.sleep(0.05)
        chat.submit("second")
        assert "part one" in chat.output_area.text
        chat.submit("/clear")
        await settle(chat)
        assert chat.messages == [] and chat.queue == [] and not chat.streaming
        assert "part one" not in chat.output_area.text
        assert "Conversation cleared" in chat.output_area.text
        chat.backend = ScriptedBackend(["fresh"])
        chat.submit("again")
        await settle(chat)
        return chat

    chat = run(scenario)
    assert [m["content"] for m in chat.messages] == ["again", "fresh"]
    assert [m["content"] for m in chat.backend.seen[0]] == ["again"]
    assert "never shown" not in chat.output_area.text and not chat.streaming
