"""redact_turn: no number reaches the detector in a forwarded turn (ruling 2.16, as amended).

Outside a challenge answer, every numeral becomes [NUM], and so does every run of two or more
spoken digit words. Inside the exempt window, a challenge answer is forwarded whole unless a run
is long enough to be a bank number (7 or more digits).
"""

import pytest
from callback_shared.contract import Role, Turn, Word

from callback_agent.redact import redact_turn


def _turn(text: str, role: Role = Role.USER, words: list[Word] | None = None) -> Turn:
    return Turn(role=role, text=text, start_ms=1000, end_ms=5000, words=words or [])


def _words(*entries: str) -> list[Word]:
    return [
        Word(w=w, start_ms=1000 + 100 * i, end_ms=1090 + 100 * i) for i, w in enumerate(entries)
    ]


def _both(text: str, exempt: bool = False) -> Turn:
    """Redact `text` as a turn whose words are its whitespace-separated pieces."""
    return redact_turn(_turn(text, words=_words(*text.split())), exempt_short_runs=exempt)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("my account is 0 0 0 1 2 3 4 5 6 7 8 9", "my account is [NUM]"),
        ("routing zero two one, zero zero zero, zero two one", "routing [NUM]"),
        ("021000021", "[NUM]"),
        ("021-000-021", "[NUM]"),
        ("four two four two four two", "[NUM]"),
        ("double four five six", "[NUM]"),
        ("oh four two", "[NUM]"),
        ("zero two one", "[NUM]"),
        ("forty two forty two", "[NUM]"),
        # Any numeral goes, of any length.
        ("I have 3 bills", "I have [NUM] bills"),
        ("call at 555 then 1 2 3 later", "call at [NUM] then [NUM] later"),
        ("it is 4 2 4 and 2 4 2", "it is [NUM] and [NUM]"),
        # A single spoken digit word stays, so ordinary speech survives.
        ("I have one bill", "I have one bill"),
        ("oh okay I have one bill", "oh okay I have one bill"),
        ("forty bills", "forty bills"),
        ("one of them is late", "one of them is late"),
        # Punctuation on a number does not let it through.
        ("my account is 000123456789?", "my account is [NUM]?"),
        ("routing #021000021", "routing #[NUM]"),
        ("it's $12,345,678", "it's $[NUM]"),
        ("number 021000021!", "number [NUM]!"),
        ("4/7/2/9/1/3/5/5", "[NUM]"),
        ("4.7.2.9.1.3", "[NUM]"),
        ("(021000021)", "([NUM])"),
        ("acct: 000123456789", "acct: [NUM]"),
        # Chunks with filler between them are each redacted.
        (
            "my routing is zero two one and zero zero zero and zero two one",
            "my routing is [NUM] and [NUM] and [NUM]",
        ),
        (
            "account one two three, uh, four five six, uh, seven eight nine",
            "account [NUM], uh, [NUM], uh, [NUM]",
        ),
    ],
)
def test_not_exempt(text: str, expected: str) -> None:
    assert redact_turn(_turn(text), exempt_short_runs=False).text == expected


@pytest.mark.parametrize(
    "words",
    [
        ["my", "account", "is", "000123456789?"],
        ["routing", "#021000021"],
        ["it's", "$12,345,678"],
        ["number", "021000021!"],
        ["acct000123456789"],
        ["(021000021)"],
        ["a:", "000123456789"],
        ["4/7/2/9/1/3/5/5"],
    ],
)
def test_words_with_punctuation_are_redacted(words: list[str]) -> None:
    out = redact_turn(_turn(" ".join(words), words=_words(*words)), exempt_short_runs=False)
    assert len(out.words) == len(words)
    assert [w.w for w in out.words if any(c.isdigit() for c in w.w)] == []
    assert "[NUM]" in [w.w for w in out.words]
    assert [(w.start_ms, w.end_ms) for w in out.words] == [
        (w.start_ms, w.end_ms) for w in _words(*words)
    ]


def test_a_number_said_in_chunks_is_redacted_in_the_words_too() -> None:
    text = "my routing is zero two one and zero zero zero and zero two one"
    out = _both(text)
    assert [w.w for w in out.words] == [
        "my",
        "routing",
        "is",
        "[NUM]",
        "and",
        "[NUM]",
        "and",
        "[NUM]",
    ]


def test_three_separate_turns_are_each_redacted() -> None:
    for chunk in ("zero two one", "zero zero zero", "zero two one"):
        assert _both(chunk).text == "[NUM]"
        assert [w.w for w in _both(chunk).words] == ["[NUM]"]


def test_single_digit_word_survives_in_the_words() -> None:
    out = _both("I have one bill")
    assert [w.w for w in out.words] == ["I", "have", "one", "bill"]


@pytest.mark.parametrize(
    "text",
    [
        "four two four two four two",
        "four seven two nine one three amber pillow rocket",
        "4 7 2, 9 1 3",
        "4 7 2 9 1 3",
        "amber pillow rocket",
    ],
)
def test_a_challenge_answer_in_the_exempt_window_is_forwarded_whole(text: str) -> None:
    turn = _turn(text, words=_words(*text.split()))
    assert redact_turn(turn, exempt_short_runs=True) == turn


def test_exempt_still_redacts_a_bank_sized_run() -> None:
    out = redact_turn(_turn("four seven two nine one three and 12345678"), exempt_short_runs=True)
    assert out.text == "four seven two nine one three and [NUM]"
    out = redact_turn(_turn("a 4/7/2/9/1/3/5/5"), exempt_short_runs=True)
    assert out.text == "a [NUM]"


def test_agent_turn_is_redacted() -> None:
    out = redact_turn(_turn("the account ending 6789", role=Role.AGENT), exempt_short_runs=False)
    assert out.text == "the account ending [NUM]"


def test_words_collapse_into_one_entry_keeping_the_times() -> None:
    words = _words("my", "account", "is", "one", "two,", "three", "four", "five", "six", "seven")
    text = "my account is one two, three four five six seven"
    out = redact_turn(_turn(text, words=words), exempt_short_runs=False)
    assert [w.w for w in out.words] == ["my", "account", "is", "[NUM]"]
    assert out.words[3].start_ms == words[3].start_ms
    assert out.words[3].end_ms == words[-1].end_ms
    assert out.text == "my account is [NUM]"


def test_a_turn_without_words_stays_without_words() -> None:
    assert redact_turn(_turn("123456789"), exempt_short_runs=False).words == []


def test_text_and_words_agree_on_which_numbers_survive() -> None:
    samples = [
        "my routing is zero two one and zero zero zero",
        "it is 4 2 4 and I have one bill",
        "acct 000123456789? and #021000021",
        "call me at 555 then forty bills",
    ]
    for text in samples:
        out = _both(text)
        leaked_text = any(c.isdigit() for c in out.text)
        leaked_words = any(c.isdigit() for w in out.words for c in w.w)
        assert leaked_text == leaked_words == False, text  # noqa: E712


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
