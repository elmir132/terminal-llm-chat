import subprocess

from tuichat import clipboard
from tuichat.commands import parse_command
from tuichat.files import build_context_message, load_files, parse_paths


def test_parse_command():
    assert parse_command("/read a.txt b.txt") == ("read", "a.txt b.txt")
    assert parse_command("/MODEL gpt-4o") == ("model", "gpt-4o")
    assert parse_command("/clear") == ("clear", "")
    assert parse_command("/unknown x") is None
    assert parse_command("hello /read") is None


def test_parse_paths_quotes_escapes_at_and_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert parse_paths("'my file.txt' other.txt") == ["my file.txt", "other.txt"]
    assert parse_paths("@drag.txt") == ["drag.txt"]
    assert parse_paths("with\\ space.txt") == ["with space.txt"]
    assert parse_paths("~/n.txt") == [str(tmp_path / "n.txt")]


def test_parse_paths_survives_unbalanced_quote():
    assert parse_paths("'broken") == ["'broken"]


def test_load_files_reports_missing_and_reads_the_rest(tmp_path):
    good = tmp_path / "a.txt"
    good.write_text("hello")
    loaded, errors = load_files([str(good), str(tmp_path / "nope.txt")])
    assert loaded == [("a.txt", "hello")]
    assert len(errors) == 1 and "Not found" in errors[0]


def test_load_files_rejects_directories(tmp_path):
    _, errors = load_files([str(tmp_path)])
    assert errors


def test_context_message_names_each_file():
    msg = build_context_message([("a.txt", "AAA"), ("b.txt", "BBB")])
    assert "2 file(s)" in msg
    assert "FILE: a.txt" in msg and "[End of b.txt]" in msg


def test_clipboard_uses_the_detected_tool(monkeypatch):
    monkeypatch.setattr(clipboard, "clipboard_command", lambda: ["faketool"])
    seen = {}

    def fake_run(cmd, input, check):
        seen.update(cmd=cmd, input=input)

    assert clipboard.copy_to_clipboard("héllo", run=fake_run)
    assert seen == {"cmd": ["faketool"], "input": "héllo".encode("utf-8")}


def test_clipboard_failure_and_missing_tool(monkeypatch):
    monkeypatch.setattr(clipboard, "clipboard_command", lambda: None)
    assert clipboard.copy_to_clipboard("x") is False
    monkeypatch.setattr(clipboard, "clipboard_command", lambda: ["faketool"])

    def boom(*a, **k):
        raise subprocess.CalledProcessError(1, "faketool")

    assert clipboard.copy_to_clipboard("x", run=boom) is False
