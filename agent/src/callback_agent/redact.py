"""Redact digit runs in a turn before it is forwarded: no full bank number reaches the detector.

Account numbers run from 4 to 17 digits and routing numbers are 9, so the rule is by length. A run
of 7 or more digits always becomes [NUM]. A run of 4 to 6 digits does too, except in the caller's
answer to a voice challenge or an enroll prompt, where the digits are the challenge itself. Runs of
1 to 3 digits pass. The agent's own turns get no exemption.

A run is a stretch of digit tokens (numerals, digit words, "oh" next to a digit, tens words,
"double X" and "triple X", the vocabulary of spoken.py) separated only by whitespace, "-", "," or
".". In `words` it is consecutive entries that are digit tokens. This is a plain function of the
text: no model is involved.
"""

from collections.abc import Callable

from callback_shared.contract import Turn, Word

from callback_agent.spoken import DigitToken, digit_tokens, is_digit_token, word_spans

REDACTED = "[NUM]"
_SEPARATORS = " \t\r\n-,."
_LONG_RUN = 7  # digits: from here a run is always redacted
_SHORT_RUN = 4  # digits: from here a run is redacted unless the turn is exempt
# Words that can sit in a run without being a digit token on their own: "oh" and "o" next to a digit,
# and "double" and "triple" before one. digit_tokens decides whether they count.
_DIGIT_LIKE = {"oh", "o", "double", "triple"}
_BREAK = "\0"  # stands for a word that is no digit token, in a list of words


def _runs(
    tokens: list[DigitToken], joined: Callable[[DigitToken, DigitToken], bool]
) -> list[tuple[int, int, int]]:
    """Group adjacent tokens into runs: (first token index, last token index, digits).

    Two tokens are in one run when no other word lies between them (the second starts where the
    first ends) and `joined` says only separators do.
    """
    runs: list[tuple[int, int, int]] = []
    for index, token in enumerate(tokens):
        if runs and tokens[runs[-1][1]].end == token.start and joined(tokens[runs[-1][1]], token):
            first, _, digits = runs[-1]
            runs[-1] = (first, index, digits + token.digits)
        else:
            runs.append((index, index, token.digits))
    return runs


def _redacted(digits: int, exempt_short_runs: bool) -> bool:
    if digits >= _LONG_RUN:
        return True
    return digits >= _SHORT_RUN and not exempt_short_runs


def _redact_text(text: str, exempt_short_runs: bool) -> str:
    spans = word_spans(text)
    tokens = digit_tokens([w for w, _, _ in spans])

    def joined(a: DigitToken, b: DigitToken) -> bool:
        gap = text[spans[a.end - 1][2] : spans[b.start][1]]
        return not gap.strip(_SEPARATORS)

    pieces: list[str] = []
    cursor = 0
    for first, last, digits in _runs(tokens, joined):
        if not _redacted(digits, exempt_short_runs):
            continue
        start = spans[tokens[first].start][1]
        end = spans[tokens[last].end - 1][2]
        pieces.append(text[cursor:start])
        pieces.append(REDACTED)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _redact_words(words: list[Word], exempt_short_runs: bool) -> list[Word]:
    # One flat list of words over all the entries. An entry made only of digit tokens and
    # separators adds its tokens' words. An entry with anything else adds a break, and an entry of
    # only separators adds nothing. `entry_of` maps a flat word back to its entry.
    flat: list[str] = []
    entry_of: list[int] = []
    for index, word in enumerate(words):
        spans = word_spans(word.w)
        names = [w for w, _, _ in spans]
        rest = word.w.lower()
        for _, start, end in reversed(spans):
            rest = rest[:start] + rest[end:]
        digits_only = (
            bool(spans)
            and all(is_digit_token(n) or n in _DIGIT_LIKE for n in names)
            and not rest.strip(_SEPARATORS)
        )
        if digits_only:
            flat.extend(names)
            entry_of.extend([index] * len(names))
        elif spans or word.w.strip(_SEPARATORS):
            flat.append(_BREAK)
            entry_of.append(index)
    tokens = digit_tokens(flat)
    out: list[Word] = []
    cursor = 0
    for first, last, digits in _runs(tokens, lambda a, b: True):
        if not _redacted(digits, exempt_short_runs):
            continue
        start = entry_of[tokens[first].start]
        end = entry_of[tokens[last].end - 1]
        out.extend(words[cursor:start])
        out.append(Word(w=REDACTED, start_ms=words[start].start_ms, end_ms=words[end].end_ms))
        cursor = end + 1
    out.extend(words[cursor:])
    return out


def redact_turn(turn: Turn, *, exempt_short_runs: bool) -> Turn:
    """`turn` with every digit run that must not leave the agent replaced by [NUM].

    `exempt_short_runs` lets a run of 4 to 6 digits through: it is for the caller's answer to a
    secret ask, and for nothing else.
    """
    return Turn(
        role=turn.role,
        text=_redact_text(turn.text, exempt_short_runs),
        start_ms=turn.start_ms,
        end_ms=turn.end_ms,
        words=_redact_words(turn.words, exempt_short_runs),
        interrupted=turn.interrupted,
    )
