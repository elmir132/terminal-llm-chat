import pytest

from tuichat.thinking import ThinkFilter


def run(chunks):
    f = ThinkFilter()
    out = "".join(f.feed(c) for c in chunks) + f.flush()
    return out, f


def test_plain_text_passes_through():
    assert run(["hello ", "world"])[0] == "hello world"


def test_block_is_removed():
    assert run(["<think>secret</think>answer"])[0] == "answer"


def test_text_around_block_is_kept():
    assert run(["before <think>x</think> after"])[0] == "before  after"


def test_tags_split_across_chunks():
    chunks = ["<th", "ink>reasoning", " more</thi", "nk>The answer"]
    assert run(chunks)[0] == "The answer"


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7])
def test_any_chunking_gives_the_same_result(size):
    text = "Hi <think>long\nthought</think>there <think>again</think>done"
    chunks = [text[i:i + size] for i in range(0, len(text), size)]
    assert run(chunks)[0] == "Hi there done"


def test_lookalike_text_is_not_swallowed():
    out, _ = run(["a < b and <thing> and </thin"])
    assert out == "a < b and <thing> and </thin"


def test_unterminated_block_hides_the_rest():
    out, f = run(["ok <think>never closes"])
    assert out == "ok "
    assert f.in_think


def test_in_think_flag_tracks_state():
    f = ThinkFilter()
    f.feed("<think>")
    assert f.in_think
    f.feed("x</think>")
    assert not f.in_think
