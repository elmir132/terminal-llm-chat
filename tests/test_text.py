from tuichat.text import code_line_states, extract_code_blocks, lex_line, mark_user_lines, strip_user_marks, wrap_text, wrap_width


def test_wrap_keeps_words_whole_and_blank_lines():
    out = wrap_text("alpha beta gamma\n\ndelta", width=10)
    assert out.split("\n") == ["alpha beta", "gamma", "", "delta"]


def test_wrap_never_breaks_long_words_or_hyphens():
    out = wrap_text("a-very-long-hyphenated-word", width=8)
    assert out == "a-very-long-hyphenated-word"


def test_wrap_indent():
    assert wrap_text("hello world", width=20, indent=2) == "  hello world"


def test_wrap_width_has_a_floor_and_accounts_for_margins():
    assert wrap_width(20) == 40
    assert wrap_width(100) == 100 - 4 - 4


def test_user_marks_roundtrip():
    marked = mark_user_lines("one\ntwo")
    assert marked == "[USER]one\n[USER]two"
    assert strip_user_marks(marked) == "one\ntwo"


def test_lex_plain_line():
    assert lex_line("hello") == [("", "hello")]


def test_lex_empty_line():
    assert lex_line("") == [("", "")]


def test_lex_bold_both_syntaxes():
    assert lex_line("a **b** c __d__") == [
        ("", "a "), ("class:bold", "b"), ("", " c "), ("class:bold", "d")]


def test_lex_user_line_ignores_bold_markup():
    assert lex_line("[USER]**not bold**") == [("class:user-msg", "**not bold**")]


# ---- fenced code blocks ------------------------------------------------------------------------------

def test_fence_and_code_lines_are_recognised():
    lines = ["Here:", "```python", "print('x')", "", "```", "done"]
    assert code_line_states(lines) == [None, "fence", "code", "code", "fence", None]


def test_an_open_block_counts_as_code_while_a_reply_is_still_streaming():
    assert code_line_states(["```", "a = 1"]) == ["fence", "code"]


def test_user_lines_never_start_a_code_block():
    from tuichat.text import USER_MARK
    assert code_line_states([USER_MARK + "```", "normal text"]) == [None, None]


def test_blocks_are_extracted_without_fences_in_order():
    reply = "one\n```sh\nls -la\ncd ..\n```\ntext\n```\nsecond\n```\nend"
    assert extract_code_blocks(reply) == ["ls -la\ncd ..", "second"]
    assert extract_code_blocks("no code here") == []
    assert extract_code_blocks("```\nunclosed block\nstill code") == ["unclosed block\nstill code"]
    assert extract_code_blocks("```\n```") == [""]


def test_indented_fences_still_count():
    assert extract_code_blocks("  ```py\n  x = 1\n  ```") == ["  x = 1"]
