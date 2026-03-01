"""Keep the part of the conversation that is sent to the model within a size limit."""

DEFAULT_MAX_CHARS = 48_000       # roughly 12,000 tokens; many local models allow far less than cloud ones


def fit_to_budget(messages, limit, reserved=0):
    """Return (messages_to_send, dropped_count, still_too_big).

    The oldest messages are left out first, the newest message is always kept, and what remains starts
    with a question (not a dangling answer). `reserved` is room already taken, for example by the system prompt.
    A limit of 0 turns trimming off."""
    if limit <= 0 or not messages:
        return messages, 0, False
    budget = limit - reserved
    total = sum(len(m["content"]) for m in messages)
    start = 0
    while total > budget and start < len(messages) - 1:
        total -= len(messages[start]["content"])
        start += 1
    while start < len(messages) - 1 and messages[start]["role"] != "user":
        total -= len(messages[start]["content"])
        start += 1
    return messages[start:], start, total > budget
