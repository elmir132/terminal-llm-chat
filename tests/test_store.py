import json
import os
import stat

import pytest

from tuichat import store

MESSAGES = [{"role": "user", "content": "héllo ✓"}, {"role": "assistant", "content": "hi"}]


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("TUICHAT_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def test_a_conversation_round_trips_with_its_details():
    path = store.save("chat1", MESSAGES, "Be brief.", "ollama", "llama3")
    data = store.load("chat1")
    assert path.name == "chat1.json"
    assert data["messages"] == MESSAGES
    assert (data["system"], data["backend"], data["model"]) == ("Be brief.", "ollama", "llama3")
    assert "héllo ✓" in path.read_text(encoding="utf-8")            # readable, not escaped


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_files_and_folder_are_private():
    path = store.save("secret", MESSAGES, "", "echo", "m")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_saving_never_overwrites_unless_asked_and_leaves_no_temp_files():
    store.save("a", MESSAGES, "", "echo", "m")
    with pytest.raises(store.StoreError, match="already exists"):
        store.save("a", MESSAGES, "", "echo", "m")
    store.save("a", MESSAGES[:1], "", "echo", "m", overwrite=True)
    assert len(store.load("a")["messages"]) == 1
    assert [p.name for p in store.folder().iterdir()] == ["a.json"]


@pytest.mark.parametrize("bad", ["", "../escape", "a/b", ".hidden", "x" * 61, "a b", "..", "a\\b", "name\n"])
def test_unsafe_names_are_refused_before_touching_the_disk(bad):
    with pytest.raises(store.StoreError):
        store.save(bad, MESSAGES, "", "echo", "m")
    with pytest.raises(store.StoreError):
        store.load(bad)
    assert not store.folder().exists()


def test_an_empty_conversation_is_not_saved():
    with pytest.raises(store.StoreError, match="nothing to save"):
        store.save("empty", [], "", "echo", "m")


@pytest.mark.parametrize("content", [
    "{not json", "[]", "null", json.dumps({"version": 2, "messages": [{"role": "user", "content": "x"}]}),
    json.dumps({"version": 1, "messages": []}), json.dumps({"version": 1, "messages": "no"}),
    json.dumps({"version": 1, "messages": [{"role": "system", "content": "x"}]}),
    json.dumps({"version": 1, "messages": [{"role": "user", "content": 5}]}),
    json.dumps({"version": 1, "messages": [{"role": "user", "content": "x"}], "system": 3}),
])
def test_damaged_or_foreign_files_are_reported_not_loaded(content):
    store.folder().mkdir(parents=True)
    (store.folder() / "bad.json").write_text(content, encoding="utf-8")
    with pytest.raises(store.StoreError):
        store.load("bad")


def test_missing_and_oversized_files_have_clear_errors():
    with pytest.raises(store.StoreError, match="No saved conversation"):
        store.load("nothing")
    store.folder().mkdir(parents=True)
    (store.folder() / "big.json").write_text(" " * (store.MAX_BYTES + 1), encoding="utf-8")
    with pytest.raises(store.StoreError, match="too large"):
        store.load("big")


def test_listing_is_newest_first_and_skips_damaged_files():
    store.save("old", MESSAGES, "", "echo", "m")
    store.save("new", MESSAGES, "", "echo", "m")
    (store.folder() / "junk.json").write_text("{", encoding="utf-8")
    (store.folder() / "old.json").write_text(json.dumps({"version": 1, "saved_at": "2020-01-01T00:00:00", "messages": MESSAGES}), encoding="utf-8")
    assert [row[0] for row in store.listing()] == ["new", "old"]
    assert store.listing()[0][2] == 2


def test_listing_without_a_folder_is_empty():
    assert store.listing() == []
