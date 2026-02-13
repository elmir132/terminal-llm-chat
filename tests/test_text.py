from tuichat.text import lex_line, mark_user_lines, strip_user_marks, wrap_text, wrap_width


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
