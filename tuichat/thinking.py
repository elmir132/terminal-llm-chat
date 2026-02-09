"""Hide <think>…</think> reasoning blocks from a streamed reply.

Reasoning models (for example deepseek-r1) emit their chain of thought inside
<think> tags. Streamed chunks can split a tag anywhere ("<th" + "ink>"), so the
filter keeps a small buffer instead of searching each chunk on its own.
"""

OPEN, CLOSE = "<think>", "</think>"


def _partial_suffix(text: str, tag: str) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of `tag`."""
    for n in range(min(len(tag) - 1, len(text)), 0, -1):
        if tag.startswith(text[-n:]):
            return n
    return 0


class ThinkFilter:
    def __init__(self):
        self.in_think = False
        self._buf = ""

    def feed(self, chunk: str) -> str:
        """Return the part of `chunk` that should be shown."""
        self._buf += chunk
        shown = []
        while True:
            if self.in_think:
                end = self._buf.find(CLOSE)
                if end == -1:
                    keep = _partial_suffix(self._buf, CLOSE)
                    self._buf = self._buf[len(self._buf) - keep:] if keep else ""
                    break
                self._buf = self._buf[end + len(CLOSE):]
                self.in_think = False
            else:
                start = self._buf.find(OPEN)
                if start == -1:
                    keep = _partial_suffix(self._buf, OPEN)
                    cut = len(self._buf) - keep
                    shown.append(self._buf[:cut])
                    self._buf = self._buf[cut:]
                    break
                shown.append(self._buf[:start])
                self._buf = self._buf[start + len(OPEN):]
                self.in_think = True
        return "".join(shown)

    def flush(self) -> str:
        """Call at end of stream: release text held back as a possible tag."""
        rest = "" if self.in_think else self._buf
        self._buf = ""
        return rest
