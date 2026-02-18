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
from .files import build_context_message, load_files, parse_paths
from .text import (MARGIN, SEPARATOR, lex_line, mark_user_lines, strip_user_marks,
                   wrap_text, wrap_width)
from .thinking import ThinkFilter

BOTTOM_PADDING = 2
THINK_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
ACCENTS = {"ollama": "#00d7ff", "openai": "#10a37f"}


class ChatLexer(Lexer):
    def lex_document(self, document):
        lines = document.lines
        return lambda number: lex_line(lines[number])


class ChatApp:
    def __init__(self, backend: Backend, input=None, output=None, columns=None):
        self.backend = backend
        self.messages = []
        self.queue = []
        self.streaming = False
        self.last_user_msg = ""
        self.last_ai_msg = ""
        self._task = None
        self._spinner = None
        self._base = ""
        self._columns = columns or (lambda: shutil.get_terminal_size().columns)

        welcome = "\nWelcome! Type your message below.\n"
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
                    title="Enter=send │ Tab=scroll │ ^A=copy all │ ^Y=copy reply │ ^C=exit",
                    style="class:input-frame"), spacer()]),
            Window(height=BOTTOM_PADDING, char=" "),
        ]), focused_element=self.input_area)

    def _build_style(self):
        accent = ACCENTS.get(self.backend.name, "#aaaaaa")
        return Style.from_dict({
            "header": f"{accent} bold", "dim": "#666666", "separator": "#333333",
            "input-frame": "#444444", "frame.border": "#444444", "frame.label": accent,
            "text-area": "#ffffff", "user-msg": "bg:#373737", "bold": "bold #ffffff",
        })

    def _build_keys(self):
        kb = KeyBindings()

        @kb.add("c-c")
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

        @kb.add("c-u")
        def _copy_user(event):
            self._copy(self.last_user_msg, "Your message")

        @kb.add("c-a")
        def _copy_all(event):
            self._copy(strip_user_marks(self.output_area.text), "Chat history")

        return kb

    # ---- output helpers ----------------------------------------------
    def append_output(self, text):
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
            self.messages.clear()
            self.set_output("\nConversation cleared.\n\n")
        elif name == "read":
            self._read_files(arg)

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
    async def _drain(self):
        self.streaming = True
        try:
            while self.queue:
                if len(self.queue) > 1:
                    self.append_output(f"📋 Processing queued request ({len(self.queue) - 1} remaining)\n")
                await self._turn(self.queue.pop(0))
        finally:
            self.streaming = False

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
            async for chunk in self.backend.stream(self.messages):
                show(think.feed(chunk))
            show(think.flush())
            self._stop_spinner()
            if not shown:
                self.set_output(self._base)
            self.last_ai_msg = reply
            self.messages.append({"role": "assistant", "content": reply})
            self.append_output("\n\n")
        except BackendError as exc:
            self._fail(str(exc))
        except Exception as exc:  # keep the UI alive whatever a backend raises
            self._fail(f"Unexpected error: {exc}")

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
            self.set_output(f"{self._base}{THINK_FRAMES[frame % len(THINK_FRAMES)]} thinking...")
            self.app.invalidate()
            frame += 1
            await asyncio.sleep(0.1)

    async def run(self):
        await self.app.run_async()
