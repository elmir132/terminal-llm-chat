"""Pure text helpers: wrapping, bold markup, user-line marking."""
import re
import textwrap

MARGIN = 2
USER_MARK = "[USER]"
SEPARATOR = "─" * 30
_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")


def wrap_width(columns: int) -> int:
    """Usable text width for a terminal that is `columns` wide."""
    return max(40, columns - MARGIN * 2 - 4)  # margins plus scrollbar


def wrap_text(text: str, width: int, indent: int = 0) -> str:
    """Wrap at word boundaries without breaking words or hyphenated words.
    Blank lines are preserved."""
    prefix = " " * indent
    out = []
    for line in text.split("\n"):
        if not line.strip():
            out.append("")
        else:
            out.append(textwrap.fill(line, width=width, initial_indent=prefix,
                                     subsequent_indent=prefix,
                                     break_long_words=False, break_on_hyphens=False))
    return "\n".join(out)


def mark_user_lines(text: str) -> str:
    """Prefix every line so the lexer can give user messages a background."""
    return "\n".join(f"{USER_MARK}{line}" for line in text.split("\n"))


def strip_user_marks(text: str) -> str:
    return text.replace(USER_MARK, "")


def lex_line(line: str):
    """Split one output line into (style, text) tokens.
    User lines get the `user-msg` style; **bold** and __bold__ get `bold`."""
    if line.startswith(USER_MARK):
        return [("class:user-msg", line[len(USER_MARK):])]
    tokens, last = [], 0
    for m in _BOLD.finditer(line):
        if m.start() > last:
            tokens.append(("", line[last:m.start()]))
        tokens.append(("class:bold", m.group(1) or m.group(2)))
        last = m.end()
    if last < len(line):
        tokens.append(("", line[last:]))
    return tokens or [("", line)]


_FENCE = re.compile(r"^\s*```")


def code_line_states(lines):
    """For each output line: 'fence' (a ``` line), 'code' (inside a fenced block) or None.

    User messages are skipped, and a block that is still open (a reply that is streaming) counts as code to the end."""
    states, inside = [], False
    for line in lines:
        if line.startswith(USER_MARK):
            states.append(None)
        elif _FENCE.match(line):
            states.append("fence")
            inside = not inside
        else:
            states.append("code" if inside else None)
    return states


def extract_code_blocks(text: str):
    """The contents of every ``` block in a reply, without the fences. An unclosed last block is included."""
    blocks, current = [], None
    for line in text.split("\n"):
        if _FENCE.match(line):
            if current is None:
                current = []
            else:
                blocks.append("\n".join(current))
                current = None
        elif current is not None:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks
