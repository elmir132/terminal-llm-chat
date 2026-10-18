"""The full-screen chat application (prompt_toolkit)."""
import asyncio
import shutil

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer  # noqa: F401  (re-exported for tests)
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import FormattedText  # noqa: F401
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea

from .backends import Backend, BackendError
from .clipboard import copy_to_clipboard
from .commands import parse_command
from . import store
from .context import DEFAULT_MAX_CHARS, fit_to_budget
from .files import FILE_CONTEXT_PREFIX, build_context_message, load_files, parse_paths
from .text import (MARGIN, SEPARATOR, code_line_states, extract_code_blocks, lex_line,
                   mark_user_lines, strip_user_marks, wrap_text, wrap_width)
from .thinking import ThinkFilter

BOTTOM_PADDING = 2
MAX_SYSTEM_CHARS = 4000
HELP_TEXT = """
Commands
  /model [NAME]      show or change the model        /models         list the models you can use
  /system [TEXT|off] show, set or remove the system prompt
  /read FILE...      load text files into the conversation
  /save [-f] [NAME]  save this conversation          /load NAME      load one      /saves  list them
  /clear             start over (keeps the system prompt)
  /help              this list

Keys
  Enter send        Alt+Enter (Esc, Enter) new line      Tab switch input / scroll area
  Ctrl+C stop the reply (quits when idle)                Ctrl+D quit
  Ctrl+Y copy last reply   Ctrl+K copy last code block   Ctrl+U copy your last message   Ctrl+A copy the chat
"""
THINK_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
ACCENTS = {"ollama": "#00d7ff", "openai": "#10a37f"}


class ChatLexer(Lexer):
    def lex_document(self, document):
        lines = document.lines
        states = code_line_states(lines)

        def style_line(number):
            if states[number] == "fence":
                return [("class:code-fence", lines[number])]
            if states[number] == "code":
                return [("class:code", lines[number])]
            return lex_line(lines[number])
        return style_line


class ChatApp:
    def __init__(self, backend: Backend, input=None, output=None, columns=None, system="", max_context=DEFAULT_MAX_CHARS):
        self.backend = backend
        self.system = system.strip()
        self.max_context = max_context
        self._noted_dropped = 0
        self.messages = []
        self.queue = []
        self.streaming = False
        self.last_user_msg = ""
        self.last_ai_msg = ""
        self._task = None
        self._clear_pending = False
        self._spinner = None
        self._base = ""
        self._frame = ""
        self._columns = columns or (lambda: shutil.get_terminal_size().columns)

        welcome = "\nWelcome! Type your message below.\n"
        if self.system:
            welcome += f"\n⚙ System prompt set ({len(self.system)} characters). /system shows it.\n"
        warning = backend.preflight()
        if warning:
            welcome += f"\n⚠️  {warning}\n"
        self.output_area = TextArea(text=welcome + "\n", read_only=True, scrollbar=True,
                                    focusable=True, wrap_lines=True, lexer=ChatLexer())
        self.input_area = TextArea(height=Dimension(min=3, max=8), multiline=True,
                                   prompt="❯ ", wrap_lines=True, focusable=True)
        self.app = Application(layout=self._build_layout(), key_bindings=self._build_keys(),
                               style=self._build_style(), full_screen=True,
                               mouse_support=True, input=input, output=output)

    # ---- construction -------------------------------------------------
    def _build_layout(self):
        spacer = lambda: Window(width=MARGIN, char=" ")  # noqa: E731
        header = Window(height=1, content=FormattedTextControl(
            lambda: [("class:header", f"  {self.backend.title} "),
                     ("class:dim", f"• {self.backend.tagline} ")]))
        rule = lambda: Window(height=1, char="─", style="class:separator")  # noqa: E731
        return Layout(HSplit([
            header, rule(),
            VSplit([spacer(), self.output_area, spacer()]),
            rule(),
            VSplit([spacer(), Frame(self.input_area,
                    title="Enter=send │ Tab=scroll │ ^A=copy all │ ^Y=copy reply │ ^C=stop/exit",
                    style="class:input-frame"), spacer()]),
            Window(height=BOTTOM_PADDING, char=" "),
        ]), focused_element=self.input_area)

    def _build_style(self):
        accent = ACCENTS.get(self.backend.name, "#aaaaaa")
        return Style.from_dict({
            "header": f"{accent} bold", "dim": "#666666", "separator": "#333333",
            "input-frame": "#444444", "frame.border": "#444444", "frame.label": accent,
            "text-area": "#ffffff", "user-msg": "bg:#373737", "bold": "bold #ffffff",
            "code": "bg:#1e1e1e #e6e6b8", "code-fence": "#666666",
        })

    def _build_keys(self):
        kb = KeyBindings()

        @kb.add("c-c")
        def _stop_or_exit(event):
            # While a reply is streaming, Ctrl+C stops it; when idle it quits.
            if not self.stop():
                event.app.exit()

        @kb.add("c-d")
        def _exit(event):
            event.app.exit()

        @kb.add("enter")
        def _submit(event):
            self.submit(self.input_area.text)

        @kb.add("escape", "enter")
        def _newline(event):
            self.input_area.buffer.insert_text("\n")

        @kb.add("pageup")
        def _up(event):
            self.output_area.buffer.cursor_up(count=10)

        @kb.add("pagedown")
        def _down(event):
            self.output_area.buffer.cursor_down(count=10)

        @kb.add("tab")
        def _focus(event):
            target = self.output_area if event.app.layout.has_focus(self.input_area) else self.input_area
            event.app.layout.focus(target)

        @kb.add("c-y")
        def _copy_ai(event):
            self._copy(self.last_ai_msg, "AI response")

        @kb.add("c-k")
        def _copy_code(event):
            blocks = extract_code_blocks(self.last_ai_msg)
            if blocks:
                self._copy(blocks[-1], "Code block")
            else:
                self.append_output("\nNo code block in the last reply\n\n")

        @kb.add("c-u")
        def _copy_user(event):
            self._copy(self.last_user_msg, "Your message")

        @kb.add("c-a")
        def _copy_all(event):
            self._copy(strip_user_marks(self.output_area.text), "Chat history")

        return kb

    # ---- output helpers ----------------------------------------------
    def append_output(self, text):
        if self._spinner is not None:
            # The spinner redraws the screen from `_base`, so a note added meanwhile must go into it.
            self._base += text
            self.set_output(self._base + self._frame)
            return
        self.set_output(self.output_area.text + text)

    def set_output(self, text):
        self.output_area.buffer.set_document(
            Document(text=text, cursor_position=len(text)), bypass_readonly=True)

    def _copy(self, text, label):
        if not text:
            return
        if copy_to_clipboard(text):
            self.append_output(f"\n✓ {label} copied to clipboard\n\n")
        else:
            self.append_output("\n❌ No clipboard tool found\n\n")

    # ---- input handling ----------------------------------------------
    def submit(self, raw):
        text = raw.strip()
        if not text:
            return
        self.input_area.buffer.reset()
        command = parse_command(text)
        if command:
            self._run_command(*command)
            return
        self.queue.append(text)
        if self.streaming:
            self.append_output(f"\n📋 Queued ({len(self.queue)} pending)\n")
        else:
            self._task = asyncio.ensure_future(self._drain())

    def _run_command(self, name, arg):
        if name == "model":
            if not arg:
                self.append_output(f"\nCurrent model: {self.backend.model}\n\n")
            else:
                self.backend.model = arg
                self.append_output(f"\n✓ Model changed to: {arg}\n\n")
        elif name == "clear":
            if self.stop():
                # The cancelled turn may still append its partial reply; _drain clears once it has finished.
                self._clear_pending = True
            else:
                self._clear()
        elif name == "read":
            self._read_files(arg)
        elif name == "system":
            self._system_command(arg)
        elif name == "save":
            self._save(arg)
        elif name == "load":
            self._load(arg)
        elif name == "saves":
            self._list_saves()
        elif name == "help":
            self.append_output(HELP_TEXT)
        elif name == "models":
            asyncio.ensure_future(self._list_models())

    def _clear(self):
        self.messages.clear()
        self._noted_dropped = 0
        self.set_output("\nConversation cleared.\n\n")

    async def _list_models(self):
        try:
            names = await self.backend.list_models()
        except BackendError as exc:
            self.append_output(f"\n❌ {exc}\n\n")
            return
        except Exception as exc:
            self.append_output(f"\n❌ Could not list models: {exc}\n\n")
            return
        if not names:
            self.append_output("\nNo models found.\n\n")
            return
        shown = "\n".join(f"  {name}{'  (current)' if name == self.backend.model else ''}" for name in names[:60])
        more = f"\n  … and {len(names) - 60} more" if len(names) > 60 else ""
        self.append_output(f"\nModels on {self.backend.name}:\n{shown}{more}\nSwitch with: /model NAME\n\n")

    def _save(self, arg):
        overwrite = False
        if arg.startswith("-f ") or arg == "-f":
            overwrite, arg = True, arg[2:].strip()
        name = arg or store.default_name()
        try:
            path = store.save(name, self.messages, self.system, self.backend.name, self.backend.model, overwrite)
        except store.StoreError as exc:
            self.append_output(f"\n❌ {exc}\n\n")
            return
        self.append_output(f"\n✓ Saved as '{name}' ({len(self.messages)} messages) in {path.parent}\n\n")

    def _load(self, name):
        if not name:
            self.append_output("\n❌ Give a name: /load NAME (see /saves)\n\n")
            return
        if self.streaming:
            self.append_output("\n❌ Wait for the reply to finish, or press Ctrl+C, before loading\n\n")
            return
        try:
            data = store.load(name)
        except store.StoreError as exc:
            self.append_output(f"\n❌ {exc}\n\n")
            return
        self.messages = data["messages"]
        self.system = data["system"]
        self._noted_dropped = 0
        users = [m["content"] for m in self.messages if m["role"] == "user" and not m["content"].startswith(FILE_CONTEXT_PREFIX)]
        assistants = [m["content"] for m in self.messages if m["role"] == "assistant"]
        self.last_user_msg = users[-1] if users else ""
        self.last_ai_msg = assistants[-1] if assistants else ""
        self.set_output(self._render_history() + f"\n✓ Loaded '{name}' ({len(self.messages)} messages)"
                        + (f", saved with {data['model']}" if data["model"] else "") + "\n\n")

    def _list_saves(self):
        rows = store.listing()
        if not rows:
            self.append_output(f"\nNo saved conversations yet. /save NAME stores one in {store.folder()}\n\n")
            return
        lines = "\n".join(f"  {name}  ({count} messages, {saved[:16].replace('T', ' ')})" for name, saved, count in rows[:30])
        self.append_output(f"\nSaved conversations:\n{lines}\n\n")

    def _render_history(self):
        out = "\n"
        for message in self.messages:
            if message["role"] == "user" and message["content"].startswith(FILE_CONTEXT_PREFIX):
                out += "\n📁 (files loaded into the conversation)\n"
            elif message["role"] == "user":
                body = mark_user_lines(wrap_text(message["content"], wrap_width(self._columns())))
                out += f"\n{SEPARATOR}\n👤 You\n{body}\n"
            else:
                out += f"\n{self.backend.icon} {self.backend.name}\n{message['content']}\n\n"
        return out

    def _outgoing(self):
        """What is sent to the model: the system prompt (if any) followed by the conversation, trimmed to the context limit.

        The full conversation stays on screen and in self.messages (so it can be saved and copied)."""
        messages, dropped, too_big = fit_to_budget(self.messages, self.max_context, len(self.system))
        if dropped != self._noted_dropped:
            if dropped:
                self.append_output(f"\n✂ The oldest {dropped} of {len(self.messages)} messages are left out so the request stays "
                                   f"under {self.max_context:,} characters. /clear starts fresh.\n")
            self._noted_dropped = dropped
        if too_big:
            self.append_output("\n⚠️  Your latest message alone is longer than the context limit; sending it anyway.\n")
        if not self.system:
            return messages
        return [{"role": "system", "content": self.system}] + messages

    def _system_command(self, arg):
        if not arg:
            if self.system:
                shown = self.system if len(self.system) <= 400 else self.system[:400] + "…"
                self.append_output(f"\n⚙ System prompt ({len(self.system)} characters):\n{shown}\n\n")
            else:
                self.append_output("\n⚙ No system prompt. Set one with: /system You are a ...\n\n")
        elif arg.lower() == "off":
            self.system = ""
            self.append_output("\n✓ System prompt removed (applies from the next message)\n\n")
        elif len(arg) > MAX_SYSTEM_CHARS:
            self.append_output(f"\n❌ The system prompt is limited to {MAX_SYSTEM_CHARS} characters\n\n")
        else:
            self.system = arg
            self.append_output(f"\n✓ System prompt set ({len(arg)} characters, applies from the next message)\n\n")

    def _read_files(self, raw):
        loaded, errors = load_files(parse_paths(raw))
        for name, content in loaded:
            self.append_output(f"\n📁 {name} ({len(content)} chars)")
        for error in errors:
            self.append_output(f"\n❌ {error}")
        if loaded:
            self.messages.append({"role": "user", "content": build_context_message(loaded)})
            self.append_output(f"\n✓ {len(loaded)} file(s) loaded into context\n\n")
        else:
            self.append_output("\n")

    # ---- conversation -------------------------------------------------
    def stop(self):
        """Stop the reply that is streaming and drop the queued messages. False if nothing is running."""
        if not self.streaming or self._task is None or self._task.done():
            return False
        self._dropped_on_stop = len(self.queue)
        self.queue.clear()
        self._task.cancel()
        return True

    async def _drain(self):
        self.streaming = True
        try:
            while self.queue:
                if len(self.queue) > 1:
                    self.append_output(f"📋 Processing queued request ({len(self.queue) - 1} remaining)\n")
                await self._turn(self.queue.pop(0))
        finally:
            self.streaming = False
            if self._clear_pending:
                self._clear_pending = False
                self._clear()

    async def _turn(self, user_text):
        self.last_user_msg = user_text
        body = mark_user_lines(wrap_text(user_text, wrap_width(self._columns())))
        self.append_output(f"\n{SEPARATOR}\n👤 You\n{body}\n")
        self.messages.append({"role": "user", "content": user_text})
        self.append_output(f"\n{self.backend.icon} {self.backend.name}\n")

        self._base = self.output_area.text
        self._start_spinner()
        think = ThinkFilter()
        reply, shown = "", False

        def show(text):
            nonlocal reply, shown
            if not text:
                return
            if not shown:
                self._stop_spinner()
                self.set_output(self._base)
                shown = True
            reply += text
            self.append_output(text)
            self.app.invalidate()

        try:
            async for chunk in self.backend.stream(self._outgoing()):
                show(think.feed(chunk))
            show(think.flush())
            self._stop_spinner()
            if not shown:
                self.set_output(self._base)
            self.last_ai_msg = reply
            self.messages.append({"role": "assistant", "content": reply})
            self.append_output("\n\n")
        except asyncio.CancelledError:
            self._stopped(reply)        # the user pressed Ctrl+C; do not re-raise, the queue is already empty
        except BackendError as exc:
            self._fail(str(exc))
        except Exception as exc:  # keep the UI alive whatever a backend raises
            self._fail(f"Unexpected error: {exc}")

    def _stopped(self, reply):
        self._stop_spinner()
        if reply:
            # Keep what was already shown, and in the history, so the next question has the context.
            self.append_output("\n")
            self.last_ai_msg = reply
            self.messages.append({"role": "assistant", "content": reply})
        else:
            self.set_output(self._base)   # remove the spinner line
            if self.messages and self.messages[-1]["role"] == "user":
                self.messages.pop()       # nothing was answered: a retry should not be duplicated
        dropped = getattr(self, "_dropped_on_stop", 0)
        note = f" ({dropped} queued message{'s' if dropped != 1 else ''} dropped)" if dropped else ""
        self.append_output(f"\n⏹ Stopped{note}\n\n")
        self._dropped_on_stop = 0

    def _fail(self, message):
        self._stop_spinner()
        self.set_output(self._base)
        self.append_output(f"\n❌ {message}\n\n")
        # Drop the user message that got no answer so a retry is not duplicated.
        if self.messages and self.messages[-1]["role"] == "user":
            self.messages.pop()

    # ---- spinner ------------------------------------------------------
    def _start_spinner(self):
        self._spinner = asyncio.ensure_future(self._spin())

    def _stop_spinner(self):
        if self._spinner:
            self._spinner.cancel()
            self._spinner = None

    async def _spin(self):
        frame = 0
        while True:
            self._frame = f"{THINK_FRAMES[frame % len(THINK_FRAMES)]} thinking..."
            self.set_output(self._base + self._frame)
            self.app.invalidate()
            frame += 1
            await asyncio.sleep(0.1)

    async def run(self):
        await self.app.run_async()
