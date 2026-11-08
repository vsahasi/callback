"""redact_turn: no full bank number leaves the agent in a forwarded turn (ruling 2.16)."""

import pytest
from callback_shared.contract import Role, Turn, Word

from callback_agent.redact import redact_turn


def _turn(text: str, role: Role = Role.USER, words: list[Word] | None = None) -> Turn:
    return Turn(role=role, text=text, start_ms=1000, end_ms=5000, words=words or [])


def _words(*entries: str) -> list[Word]:
    return [
        Word(w=w, start_ms=1000 + 100 * i, end_ms=1090 + 100 * i) for i, w in enumerate(entries)
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("my account is 0 0 0 1 2 3 4 5 6 7 8 9", "my account is [NUM]"),
        ("routing zero two one, zero zero zero, zero two one", "routing [NUM]"),
        ("021000021", "[NUM]"),
        ("021-000-021", "[NUM]"),
        ("four two four two four two", "[NUM]"),
        ("double four five six", "[NUM]"),
        ("oh okay I have 3 bills", "oh okay I have 3 bills"),
        ("call me at 555 then 1 2 3 later", "call me at 555 then 1 2 3 later"),
        ("forty two forty two", "[NUM]"),
        ("it is 4 2 4 and 2 4 2", "it is 4 2 4 and 2 4 2"),
    ],
)
def test_not_exempt(text: str, expected: str) -> None:
    assert redact_turn(_turn(text), exempt_short_runs=False).text == expected


@pytest.mark.parametrize(
    "text",
    [
        "four two four two four two",
        "four seven two nine one three amber pillow rocket",
        "4 7 2, 9 1 3",
    ],
)
def test_exempt_keeps_short_runs(text: str) -> None:
    turn = _turn(text, words=_words(*text.split()))
    assert redact_turn(turn, exempt_short_runs=True) == turn


def test_exempt_still_redacts_a_long_run() -> None:
    out = redact_turn(_turn("four seven two nine one three and 12345678"), exempt_short_runs=True)
    assert out.text == "four seven two nine one three and [NUM]"


def test_agent_turn_is_redacted() -> None:
    out = redact_turn(_turn("the account ending 6789", role=Role.AGENT), exempt_short_runs=False)
    assert out.text == "the account ending [NUM]"


def test_words_collapse_into_one_entry() -> None:
    words = _words("my", "account", "is", "one", "two,", "three", "four", "five", "six", "seven")
    out = redact_turn(
        _turn("my account is one two, three four five six seven"), exempt_short_runs=False
    )
    assert out.words == []  # no words in, none out
    turn = _turn("my account is one two, three four five six seven", words=words)
    out = redact_turn(turn, exempt_short_runs=False)
    assert [w.w for w in out.words] == ["my", "account", "is", "[NUM]"]
    assert out.words[3].start_ms == words[3].start_ms
    assert out.words[3].end_ms == words[-1].end_ms
    assert out.text == "my account is [NUM]"


def test_words_only_digit_entries_join_a_run() -> None:
    words = _words("one", "two", "and", "three", "four")
    out = redact_turn(_turn("one two and three four", words=words), exempt_short_runs=False)
    assert [w.w for w in out.words] == ["one", "two", "and", "three", "four"]


def test_words_single_entry_with_separators() -> None:
    words = _words("routing", "021-000-021")
    out = redact_turn(_turn("routing 021-000-021", words=words), exempt_short_runs=False)
    assert [w.w for w in out.words] == ["routing", "[NUM]"]


def test_words_short_run_exempt_and_not() -> None:
    words = _words("four", "two", "four", "two")
    turn = _turn("four two four two", words=words)
    assert [w.w for w in redact_turn(turn, exempt_short_runs=False).words] == ["[NUM]"]
    assert redact_turn(turn, exempt_short_runs=True).words == words


def test_other_fields_survive_and_output_validates() -> None:
    turn = Turn(role=Role.USER, text="123456789", start_ms=1, end_ms=9, words=[], interrupted=True)
    out = redact_turn(turn, exempt_short_runs=False)
    assert out.interrupted is True
    assert (out.start_ms, out.end_ms) == (1, 9)
    assert Turn.model_validate(out.model_dump()) == out


def test_the_input_is_not_changed() -> None:
    turn = _turn("123456789")
    redact_turn(turn, exempt_short_runs=False)
    assert turn.text == "123456789"
