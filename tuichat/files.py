"""The /read command: load text files into the conversation."""
import os
import shlex


def parse_paths(raw: str):
    """Split the argument of /read into file paths.

    Handles shell-style quoting, backslash-escaped spaces, a leading '@' (added
    by some terminals on drag and drop) and '~'.
    """
    try:
        parts = shlex.split(raw)
    except ValueError:
        parts = [raw.strip()]
    paths = []
    for part in parts:
        if part.startswith("@"):
            part = part[1:]
        paths.append(os.path.expanduser(part.replace("\\ ", " ")))
    return paths


def load_files(paths):
    """Return ([(name, content)], [error message])."""
    loaded, errors = [], []
    for path in paths:
        if not os.path.isfile(path):
            errors.append(f"Not found: {path}")
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                loaded.append((os.path.basename(path), handle.read()))
        except OSError as exc:
            errors.append(f"Error reading {path}: {exc}")
    return loaded, errors


def build_context_message(files) -> str:
    blocks = [f"FILE: {name}\nCONTENT:\n{content}\n[End of {name}]" for name, content in files]
    return (f"I'm providing you with the contents of {len(files)} file(s). "
            "Please read and understand these files so you can answer questions about them.\n\n"
            + "\n".join(blocks))
