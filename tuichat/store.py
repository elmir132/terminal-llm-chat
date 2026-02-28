"""Saving and loading conversations as small JSON files in ~/.tuichat/conversations (or $TUICHAT_HOME)."""
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,59}$")
MAX_BYTES = 5_000_000
ROLES = ("user", "assistant")
VERSION = 1


class StoreError(Exception):
    """Something the user can fix: a bad name, a missing file, a damaged file."""


def folder() -> Path:
    base = os.environ.get("TUICHAT_HOME")
    return (Path(base) if base else Path.home() / ".tuichat") / "conversations"


def default_name() -> str:
    return datetime.now().strftime("chat-%Y%m%d-%H%M%S")


def _path(name: str) -> Path:
    if not NAME_RE.fullmatch(name):
        raise StoreError("A name can use letters, digits, dots, dashes and underscores (up to 60), and cannot start with a symbol.")
    return folder() / f"{name}.json"


def save(name, messages, system, backend, model, overwrite=False) -> Path:
    path = _path(name)
    if path.exists() and not overwrite:
        raise StoreError(f"'{name}' already exists. Use /save -f {name} to replace it.")
    if not messages:
        raise StoreError("There is nothing to save yet.")
    folder().mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(folder(), 0o700)
    payload = {"version": VERSION, "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "backend": backend, "model": model, "system": system or "", "messages": messages}
    fd, tmp = tempfile.mkstemp(dir=folder(), suffix=".tmp")          # write fully, then swap in: no half-written file
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.chmod(tmp, 0o600)                                         # conversations can be private
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def load(name) -> dict:
    path = _path(name)
    if not path.exists():
        raise StoreError(f"No saved conversation called '{name}'. /saves lists them.")
    if path.stat().st_size > MAX_BYTES:
        raise StoreError(f"'{name}' is too large to load.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        raise StoreError(f"'{name}' could not be read as a saved conversation.") from None
    if not isinstance(data, dict):
        raise StoreError(f"'{name}' is not a valid saved conversation.")
    messages = data.get("messages")
    ok = (data.get("version") == VERSION and isinstance(messages, list) and messages
          and all(isinstance(m, dict) and m.get("role") in ROLES and isinstance(m.get("content"), str) for m in messages)
          and isinstance(data.get("system", ""), str))
    if not ok:
        raise StoreError(f"'{name}' is not a valid saved conversation.")
    return {"messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "system": data.get("system", ""), "backend": data.get("backend", ""), "model": data.get("model", ""),
            "saved_at": data.get("saved_at", "")}


def listing():
    """[(name, saved_at, message_count)], newest first. Damaged files are skipped."""
    rows = []
    if folder().is_dir():
        for path in folder().glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                rows.append((path.stem, str(data.get("saved_at", "")), len(data["messages"])))
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return sorted(rows, key=lambda row: row[1], reverse=True)
