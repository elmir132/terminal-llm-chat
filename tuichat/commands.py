"""Slash commands typed into the input box."""

KNOWN = {"read", "model", "clear", "system"}


def parse_command(text: str):
    """('read', 'a.txt') for '/read a.txt'; None if `text` is not a known command."""
    text = text.strip()
    if not text.startswith("/"):
        return None
    name, _, arg = text[1:].partition(" ")
    name = name.lower()
    return (name, arg.strip()) if name in KNOWN else None
