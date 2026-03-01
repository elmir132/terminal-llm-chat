from tuichat.context import fit_to_budget


def conv(*pairs):
    return [{"role": role, "content": text} for role, text in pairs]


def test_everything_is_kept_when_it_fits_or_trimming_is_off():
    messages = conv(("user", "a" * 10), ("assistant", "b" * 10))
    assert fit_to_budget(messages, 100) == (messages, 0, False)
    assert fit_to_budget(messages, 0) == (messages, 0, False)
    assert fit_to_budget([], 10) == ([], 0, False)


def test_the_oldest_messages_go_first_and_the_newest_stays():
    messages = conv(("user", "1" * 40), ("assistant", "2" * 40), ("user", "3" * 40), ("assistant", "4" * 40), ("user", "5" * 10))
    kept, dropped, too_big = fit_to_budget(messages, 100)
    assert [m["content"][0] for m in kept] == ["3", "4", "5"] and dropped == 2 and not too_big


def test_what_is_kept_starts_with_a_question_not_a_dangling_answer():
    messages = conv(("user", "1" * 30), ("assistant", "2" * 30), ("user", "3" * 30), ("assistant", "4" * 30), ("user", "5" * 5))
    kept, dropped, _ = fit_to_budget(messages, 70)
    assert kept[0]["role"] == "user"
    assert [m["content"][0] for m in kept] == ["3", "4", "5"]


def test_a_latest_message_over_the_limit_is_still_kept_and_flagged():
    messages = conv(("user", "old"), ("assistant", "reply"), ("user", "x" * 500))
    kept, dropped, too_big = fit_to_budget(messages, 100)
    assert [m["content"] for m in kept] == ["x" * 500] and dropped == 2 and too_big


def test_the_reserved_room_for_the_system_prompt_counts():
    messages = conv(("user", "a" * 30), ("assistant", "b" * 30), ("user", "c" * 30))
    assert fit_to_budget(messages, 100)[1] == 0
    assert fit_to_budget(messages, 100, reserved=40)[1] == 2
