"""Parsers for what a caller says out loud: the digits of an SMS code, and a yes or no.

Both are plain functions of the transcript. No model reads the answer, because the LLM never
decides authorization. Flux may give digits as numerals or as words, so both are handled.

A yes is only ever a short, plain agreement. An answer that merely contains an agreement word, like
"I'm sure it's a scam", "Are you sure?" or "I refuse to confirm", is not one. So parse_yes_no checks
the whole answer against an allowlist instead of looking for a single agreement word.
"""

import re
from itertools import pairwise, product
from typing import NamedTuple

# Lowercase words and runs of ASCII digits. Punctuation and hyphens only separate them, and an
# apostrophe inside a word stays in it, so "don't" is one word.
_TOKEN = re.compile(r"[0-9]+|[a-z]+(?:'[a-z]+)*")

_UNITS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}
_DIGIT_WORDS = {**_UNITS, "zero": "0"}
# Said for zero in a string of digits ("four oh two"), and a word of its own anywhere else ("oh
# okay"). Where it counts is _oh_positions's business.
_OH_WORDS = {"oh", "o"}
_TENS = {
    "twenty": "2",
    "thirty": "3",
    "forty": "4",
    "fifty": "5",
    "sixty": "6",
    "seventy": "7",
    "eighty": "8",
    "ninety": "9",
}
_REPEATS = {"double": 2, "triple": 3}

# The word lists behind parse_yes_no. A yes uses only _CORE_WORDS and _FILLERS, and has a core word
# or one of _CORE_PHRASES. A word in _NEGATIONS, or any word ending in "n't", makes a no.
_NEGATIONS = {
    "no",
    "nope",
    "not",
    "never",
    "cannot",
    "don't",
    "didn't",
    "wasn't",
    "refuse",
    "refused",
    "decline",
    "declined",
    "unable",
    "unwilling",
    "reject",
    "rejected",
    "deny",
    "denied",
}
_CORE_WORDS = {
    "yes",
    "yeah",
    "yep",
    "yup",
    "correct",
    "right",
    "sure",
    "agree",
    "agreed",
    "confirm",
    "confirmed",
    "consent",
    "okay",
    "ok",
    "absolutely",
    "definitely",
}
_CORE_PHRASES = ("that was me", "it was me", "i did", "i do", "of course")
_FILLERS = {
    "i",
    "i'm",
    "im",
    "it",
    "that",
    "that's",
    "thats",
    "was",
    "is",
    "me",
    "do",
    "did",
    "of",
    "course",
    "please",
    "uh",
    "um",
    "mm",
    "mhm",
    "hmm",
    "oh",
    "well",
    "so",
    "go",
    "ahead",
    "true",
}
_ALLOWED = _CORE_WORDS | _FILLERS
# Words that start a question. An answer whose first real word is one of these (see
# _LEADING_FILLERS) is a question, even when the transcript has no "?".
_QUESTION_STARTERS = {
    "is",
    "are",
    "do",
    "does",
    "can",
    "could",
    "would",
    "will",
    "should",
    "did",
    "have",
    "has",
    "was",
    "were",
}
# Fillers that can come before a question word ("So is that right"). They are skipped to find the
# first real word. "that" and "I" are not skipped, or "I do" would look like a question.
_LEADING_FILLERS = {"uh", "um", "mm", "mhm", "hmm", "oh", "well", "so"}
# Two words in a row that leave an answer unclear.
_UNCLEAR_PAIRS = {
    *product(_QUESTION_STARTERS, ("that", "it", "i", "you")),  # a question: "is that", "do I"
    *product(("i'm", "im"), ("okay", "ok")),  # "I'm okay" can mean "no thanks"
}
# Phrases that wave something off, and so answer none of the questions asked here, although each
# word is allowed (R80): "that's okay" can be "no thanks, that's okay". They make an answer unclear
# wherever they are in it, like the pairs above. Plain "okay" is still a yes. The spellings with
# and without the apostrophe are both here, because the transcript may have either.
_WAVED_OFF = tuple(
    f"{lead} {okay}"
    for lead in ("that's", "thats", "that is", "it's", "its", "it is")
    for okay in ("okay", "ok")
)
_MAX_YES_WORDS = 12


def _words(text: str) -> list[str]:
    # A curly apostrophe (U+2019) in typed text must read the same as a straight one.
    return _TOKEN.findall(text.lower().replace("\u2019", "'"))


def _negates(word: str) -> bool:
    # Any word ending in "n't" negates, so "I can't agree" is a no although "can't" is not in
    # _NEGATIONS.
    return word in _NEGATIONS or word.endswith("n't")


def _is_digit(word: str) -> bool:
    """Whether `word` is a numeral, a digit word, "zero" or a tens word: a number said for sure."""
    return word.isdigit() or word in _DIGIT_WORDS or word in _TENS


def _oh_positions(words: list[str]) -> set[int]:
    """Where in `words` an "oh" or an "o" is a zero: right next to a digit, and nowhere else.

    The word right before it or right after it has to be a numeral, a digit word, "zero" or a tens
    word. A run of them ("oh oh seven") counts together, by the word before and the word after the
    run. So the "oh" of "Oh okay, it's 4 2 4 2 4 2" is dropped, and that of "four oh two" is not.
    """
    zeros: set[int] = set()
    start = 0
    while start < len(words):
        if words[start] not in _OH_WORDS:
            start += 1
            continue
        end = start
        while end < len(words) and words[end] in _OH_WORDS:
            end += 1
        before = words[start - 1] if start > 0 else ""
        after = words[end] if end < len(words) else ""
        if _is_digit(before) or _is_digit(after):
            zeros.update(range(start, end))
        start = end
    return zeros


def parse_spoken_digits(text: str) -> str:
    """The digits in a spoken answer, as one string.

    Digit words and "zero" become digits, and numerals are kept. "Oh" and "o" are zeros only next
    to a digit, as in "four oh two" (402); anywhere else they are dropped, so "Oh okay, it's 4 2 4
    2 4 2" is six digits and not seven. "Double" and "triple" repeat the number right after them.
    A tens word takes the unit word right after it, so "forty two" is 42, and a tens word alone
    gives the tens digit and 0. Everything else is dropped, and that includes the numbers that
    are not said digit by digit: "ten", the teens ("eleven" to "nineteen") and "hundred" are not
    parsed. Only whole words count: the "one" in "someone" and the "oh" in "john" are not digits.
    The length is not checked here.
    """
    words = _words(text)
    ohs = _oh_positions(words)
    digits: list[str] = []
    repeat = 1
    i = 0
    while i < len(words):
        word = words[i]
        i += 1
        if word.isdigit():
            number = word
        elif word in _DIGIT_WORDS:
            number = _DIGIT_WORDS[word]
        elif (i - 1) in ohs:
            number = "0"
        elif word in _TENS:
            number = _TENS[word]
            if i < len(words) and words[i] in _UNITS:
                number += _UNITS[words[i]]
                i += 1
            else:
                number += "0"
        else:
            # "Double" and "triple" wait for the very next word. Any other word cancels them.
            repeat = _REPEATS.get(word, 1)
            continue
        digits.append(number * repeat)
        repeat = 1
    return "".join(digits)


def parse_yes_no(text: str) -> bool | None:
    """True for a short, plain yes, False for a no, otherwise None.

    A negation anywhere makes it False: a word in _NEGATIONS, or any word ending in "n't". So "I'm
    not sure", "I can't agree" and "I refuse to confirm" are all a no. Without one, it is None if
    it is a question or unclear: it has a question mark, its first word after any _LEADING_FILLERS
    is in _QUESTION_STARTERS ("So is that right"), two words in a row are in _UNCLEAR_PAIRS
    ("Yes, is that correct", "I'm okay"), or it has one of _WAVED_OFF ("that's okay", "it is
    ok"). Plain "okay" is a yes. Anything else is True only if there are at most
    _MAX_YES_WORDS words, every word is in _CORE_WORDS or _FILLERS, and the answer has a core word
    or one of _CORE_PHRASES. An answer that merely contains an agreement word, like "I'm sure it's
    a scam", is None. None is not a no, it is no usable answer: where a yes lets something happen,
    only `is True` counts.
    """
    words = _words(text)
    if any(_negates(word) for word in words):
        return False
    first = next((word for word in words if word not in _LEADING_FILLERS), None)
    sentence = f" {' '.join(words)} "  # padded, so a phrase matches whole words only
    unclear = any(pair in _UNCLEAR_PAIRS for pair in pairwise(words)) or any(
        f" {phrase} " in sentence for phrase in _WAVED_OFF
    )
    if "?" in text or first in _QUESTION_STARTERS or unclear:
        return None
    if len(words) > _MAX_YES_WORDS or not _ALLOWED.issuperset(words):
        return None
    has_phrase = any(f" {phrase} " in sentence for phrase in _CORE_PHRASES)
    if _CORE_WORDS.intersection(words) or has_phrase:
        return True
    return None


def is_digit_token(word: str) -> bool:
    """Whether the lowercase `word` is a numeral, a digit word, "zero" or a tens word.

    "Oh" and "o" are not: they are zeros only next to a digit (see oh_positions).
    """
    return _is_digit(word)


def oh_positions(words: list[str]) -> set[int]:
    """Where in the lowercase `words` an "oh" or "o" is a zero, as parse_spoken_digits reads it."""
    return _oh_positions(words)


def word_spans(text: str) -> list[tuple[str, int, int]]:
    """The words of `text`, lowercased, with where each starts and ends in it.

    These are the words parse_spoken_digits reads: numeral runs and letter runs, with an
    apostrophe kept inside a word. A curly apostrophe counts as a straight one.
    """
    return [
        (m.group(0), m.start(), m.end())
        for m in _TOKEN.finditer(text.lower().replace("\u2019", "'"))
    ]


class DigitToken(NamedTuple):
    """One spoken number inside a list of words: words[start:end] say `digits` digits."""

    start: int
    end: int
    digits: int


def digit_tokens(words: list[str]) -> list[DigitToken]:
    """The digit tokens of the lowercase `words`, in order, each with how many digits it says.

    - A numeral counts its length, a digit word and "zero" count 1.
    - "Oh" and "o" count 1, only where oh_positions says they are a zero.
    - A tens word counts 2, with or without the unit word right after it ("forty two" is one
      token of 2 digits).
    - "Double" and "triple" take the digit token right after them, and count 2 or 3 for it.

    Any other word is no token, and it ends the run a caller builds from adjacent tokens.
    """
    ohs = oh_positions(words)

    def single(i: int) -> DigitToken | None:
        word = words[i]
        if word.isdigit():
            return DigitToken(i, i + 1, len(word))
        if word in _DIGIT_WORDS or i in ohs:
            return DigitToken(i, i + 1, 1)
        if word in _TENS:
            if i + 1 < len(words) and words[i + 1] in _UNITS:
                return DigitToken(i, i + 2, 2)
            return DigitToken(i, i + 1, 2)
        return None

    tokens: list[DigitToken] = []
    i = 0
    while i < len(words):
        repeat = _REPEATS.get(words[i], 0)
        if repeat and i + 1 < len(words):
            # "double oh seven": the "oh" is a zero here although it stands next to "double".
            is_oh = words[i + 1] in _OH_WORDS
            target = DigitToken(i + 1, i + 2, 1) if is_oh else single(i + 1)
            if target is not None:
                tokens.append(DigitToken(i, target.end, repeat * target.digits))
                i = target.end
                continue
        token = single(i)
        if token is None:
            i += 1
            continue
        tokens.append(token)
        i = token.end
    return tokens
