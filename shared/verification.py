"""Does what the caller said out loud match what the vendor emailed?

This is the half of Callback that the detector does not do. The detector asks
whether the *voice* is genuine. This module asks whether the *details* match,
and it does so under one hard constraint that shapes every decision below:

    Callback never reads the expected account number to the caller.

The caller has to produce the digits unprompted. An attacker who intercepted
the vendor's email still has to say them; an attacker who did not must not be
able to learn them by failing. That asymmetry is the whole anti social
engineering property, so it is enforced here in code rather than left to
convention:

* the expected digits only ever live inside `RedactedDigits`, whose `repr`,
  `str`, `format`, JSON serialisation and pickle support all refuse to render
  them. `reveal()` is the single, grep-able place the plaintext is read.
* no field on any result model carries a digit, a position, a length or a
  count derived from the expected value. See "What a near miss may say" below.
* every free-form string that reaches a result is checked by `leaks()` before
  it is returned, and a leak raises instead of answering.

Vocabulary. The product speaks `verified` / `mismatch` / `needs_review`
(`Verdict` here). The detector service speaks `allow` / `deny` / `challenge`
(`shared.contract.Decision`). They answer different questions and both are
needed for a real outcome; `detector.verification` holds the table that
combines them.

What a near miss may say
------------------------
A near miss (one substituted digit, one adjacent transposition, one missing or
one extra digit) is both a plausible speech-to-text error and a plausible
fraud attempt, so it resolves to `needs_review` and never to `verified`.

Saying *why* is where the leaks hide. "The last four digits differ" discloses
a position. "You said twelve digits, we expected eleven" discloses a length.
Even the bare fact "you were one digit away" discloses that the expected value
lies in the edit-distance-1 neighbourhood of what the caller just said, which
is roughly 10n candidates instead of 10^n. So the disclosure is split in two:

* `VerificationResult` is the operator record attached to the bill. It names
  the *class* of the discrepancy with no digit, position or length, because
  the human approving the payment needs to know whether this looked like a
  slip of the tongue or a different account entirely. It travels
  server-to-server behind the bearer token and is never spoken.
* `CallerDisclosure` is everything the caller is allowed to hear, and it has
  exactly two values. `mismatch` and `needs_review` produce the *same* words,
  so failing tells the caller nothing except that a human will look. The
  `verified` wording reveals only what the caller themselves supplied.

The residual leak is one bit per attempt ("were those the emailed digits"),
which is unavoidable: it is the product. It is bounded by the attempt cap in
`detector.verification`, without which the never-reveal property is worthless.

Snapshot drift is caught by tests/contract/test_verification.py.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from secrets import compare_digest
from typing import Any, Final

from pydantic import (
    BaseModel,
    Field,
    GetCoreSchemaHandler,
    ValidationError,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema

from shared.contract import Decision, Score, StrictModel

# Bump whenever any model or policy constant in this module changes shape.
# Tracked separately from CONTRACT_VERSION: the detector wire contract and the
# details check can move independently.
VERIFICATION_VERSION: Final = "0.1.0-provisional"

# What any redacted value renders as, everywhere.
REDACTED_PLACEHOLDER: Final = "<redacted>"

# A near miss is only forgiven down to `needs_review` on numbers long enough
# for "one digit away" to still be a large search space. Below this, any
# difference at all is a plain mismatch: in a four digit number the
# edit-distance-1 neighbourhood is a meaningful fraction of the whole space,
# so treating it as a near miss would hand an attacker too much.
NEAR_MISS_MIN_DIGITS: Final = 6

# One edit, and only one. Two substitutions are a mismatch, not a slip.
MAX_NEAR_MISS_EDITS: Final = 1

# Longest run of expected digits that may appear in any outbound string. Four
# is the usual "last four" convention and is already more than this service is
# willing to confirm, so the guard treats any run of this length as a leak.
LEAK_SUBSTRING_DIGITS: Final = 4

# Transcripts are a few spoken sentences, not documents.
MAX_TRANSCRIPT_CHARS: Final = 2000

# Bounds on the expected value itself, before any target-specific rule.
MIN_EXPECTED_DIGITS: Final = 1
MAX_EXPECTED_DIGITS: Final = 40


class Verdict(StrEnum):
    """The product's vocabulary, exactly as the README states it."""

    VERIFIED = "verified"
    MISMATCH = "mismatch"
    NEEDS_REVIEW = "needs_review"


class VerificationTarget(StrEnum):
    """Which field is being confirmed.

    Both halves of a bank change get read back, so the account number is not
    the only thing compared.
    """

    ACCOUNT_NUMBER = "account_number"
    SORT_CODE = "sort_code"
    ROUTING_NUMBER = "routing_number"


class MatchClass(StrEnum):
    """The shape of the discrepancy. Never a digit, a position or a length."""

    EXACT = "exact"
    SINGLE_DIGIT_SUBSTITUTION = "single_digit_substitution"
    ADJACENT_TRANSPOSITION = "adjacent_transposition"
    MISSING_DIGIT = "missing_digit"
    EXTRA_DIGIT = "extra_digit"
    UNRELATED = "unrelated"
    UNPARSEABLE = "unparseable"
    NOT_COMPARED = "not_compared"


class CallerDisclosure(StrEnum):
    """Everything the caller is allowed to hear. Deliberately two valued."""

    CONFIRMED = "confirmed"
    NOT_CONFIRMED = "not_confirmed"


class PaymentAction(StrEnum):
    """What happens to the held payment.

    Note that `verified` does not pay anyone. A person approves before money
    moves, so the best outcome this service can produce is "a human may now
    approve it".
    """

    RELEASE_FOR_APPROVAL = "release_for_approval"
    HOLD_FOR_REVIEW = "hold_for_review"
    BLOCK = "block"


# Near misses are the classes that resolve to needs_review rather than being
# silently forgiven as a match or hardened into a mismatch.
NEAR_MISS_CLASSES: Final[frozenset[MatchClass]] = frozenset(
    {
        MatchClass.SINGLE_DIGIT_SUBSTITUTION,
        MatchClass.ADJACENT_TRANSPOSITION,
        MatchClass.MISSING_DIGIT,
        MatchClass.EXTRA_DIGIT,
    }
)

# Exactly one class means verified. Everything that is not an exact match is
# either a mismatch or a referral, never a pass.
_VERDICT_BY_MATCH: Final[dict[MatchClass, Verdict]] = {
    MatchClass.EXACT: Verdict.VERIFIED,
    MatchClass.SINGLE_DIGIT_SUBSTITUTION: Verdict.NEEDS_REVIEW,
    MatchClass.ADJACENT_TRANSPOSITION: Verdict.NEEDS_REVIEW,
    MatchClass.MISSING_DIGIT: Verdict.NEEDS_REVIEW,
    MatchClass.EXTRA_DIGIT: Verdict.NEEDS_REVIEW,
    MatchClass.UNRELATED: Verdict.MISMATCH,
    MatchClass.UNPARSEABLE: Verdict.NEEDS_REVIEW,
    MatchClass.NOT_COMPARED: Verdict.NEEDS_REVIEW,
}

# Operator-facing wording. Constant strings, so no reason line can ever be
# built out of the expected value.
_REASON_BY_MATCH: Final[dict[MatchClass, str]] = {
    MatchClass.EXACT: "The caller read back the emailed details exactly.",
    MatchClass.SINGLE_DIGIT_SUBSTITUTION: (
        "One digit differs. That is as consistent with a transcription error "
        "as with a wrong account, so it goes to a person."
    ),
    MatchClass.ADJACENT_TRANSPOSITION: (
        "Two neighbouring digits are swapped. That is as consistent with a "
        "transcription error as with a wrong account, so it goes to a person."
    ),
    MatchClass.MISSING_DIGIT: (
        "One digit fewer than expected. That is as consistent with a dropped "
        "word as with a wrong account, so it goes to a person."
    ),
    MatchClass.EXTRA_DIGIT: (
        "One digit more than expected. That is as consistent with a repeated "
        "word as with a wrong account, so it goes to a person."
    ),
    MatchClass.UNRELATED: (
        "What the caller said is not the emailed account. Payment blocked."
    ),
    MatchClass.UNPARSEABLE: (
        "No usable digits could be taken from the recording, or the wording "
        "was ambiguous. Nothing was compared."
    ),
    MatchClass.NOT_COMPARED: (
        "The attempt limit for this call was reached, so no comparison was "
        "performed. Failing closed to review."
    ),
}

# What the agent is permitted to say on the line. Identical wording for
# mismatch and needs_review on purpose: a failing caller learns only that a
# human will look at it.
CALLER_SCRIPT: Final[dict[CallerDisclosure, str]] = {
    CallerDisclosure.CONFIRMED: (
        "Thank you, that matches what we have on file. Someone on our side "
        "still has to approve the payment before anything moves."
    ),
    CallerDisclosure.NOT_CONFIRMED: (
        "Thank you. We are not able to confirm that over the phone, so a "
        "member of our team will follow up before anything is paid."
    ),
}

# Digit counts a well-formed expected value must have, per target. Sort codes
# are six digits and ABA routing numbers are nine; account numbers vary by
# country, so the range is wide and NEAR_MISS_MIN_DIGITS does the careful part.
TARGET_DIGIT_RULES: Final[dict[VerificationTarget, tuple[int, int]]] = {
    VerificationTarget.ACCOUNT_NUMBER: (4, 20),
    VerificationTarget.SORT_CODE: (6, 6),
    VerificationTarget.ROUTING_NUMBER: (9, 9),
}


class LeakGuardError(RuntimeError):
    """Raised when an outbound string would have carried expected digits.

    Failing the request is the safe direction: a 500 discloses nothing, a
    leaked account number cannot be taken back.
    """


# --------------------------------------------------------------- secret type


class RedactedDigits:
    """A digit string that must never be rendered.

    Everything that normally turns an object into text is overridden to emit
    `REDACTED_PLACEHOLDER` instead: `repr`, `str`, f-strings, JSON
    serialisation. There is no `__len__` and no digit-count property, because
    the length of the expected value is itself a disclosure. Instances are
    unhashable and unpicklable so they cannot be quietly used as a dict key or
    written into a cache.

    `reveal()` is the one place the plaintext comes back out. Grep for it.
    """

    __slots__ = ("_digits",)

    def __init__(self, raw: str) -> None:
        digits = normalise_expected(raw)
        if len(digits) < MIN_EXPECTED_DIGITS:
            raise ValueError("expected value contains no digits")
        if len(digits) > MAX_EXPECTED_DIGITS:
            raise ValueError(
                f"expected value has more than {MAX_EXPECTED_DIGITS} digits"
            )
        object.__setattr__(self, "_digits", digits)

    # Rendering. Every one of these is a leak path in an ordinary class.

    def __repr__(self) -> str:
        return f"RedactedDigits({REDACTED_PLACEHOLDER})"

    __str__ = __repr__

    def __format__(self, format_spec: str) -> str:
        return repr(self)

    def __reduce__(self) -> Any:
        raise TypeError(
            "RedactedDigits cannot be pickled, so it cannot be written into a "
            "cache or a log by accident"
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> RedactedDigits:
        # Explicit, because blocking __reduce__ also blocks the default
        # deepcopy path that pydantic's model_copy(deep=True) uses.
        return RedactedDigits(self._digits)

    def __copy__(self) -> RedactedDigits:
        return RedactedDigits(self._digits)

    # Comparison. Constant time, and only against things that are digits.

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RedactedDigits):
            return compare_digest(self._digits, other._digits)
        if isinstance(other, str):
            return compare_digest(self._digits, other)
        return NotImplemented

    # Unhashable on purpose: a hash of an eight digit number is trivially
    # reversible, so it must not become a dict key or land in a log.
    __hash__ = None  # type: ignore[assignment]

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("RedactedDigits is immutable")

    def reveal(self) -> str:
        """The plaintext digits. The only audited exit from this class."""

        return self._digits

    # ------------------------------------------------------------- pydantic

    @classmethod
    def _validate(cls, value: object) -> RedactedDigits:
        if isinstance(value, RedactedDigits):
            return value
        if isinstance(value, str):
            # Any ValueError raised here is rendered by pydantic without the
            # offending input only if the caller asks for errors with
            # include_input=False. detector.verification does exactly that,
            # and never hands the body to FastAPI's own validator.
            return cls(value)
        raise ValueError("expected a string of digits")

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda _: REDACTED_PLACEHOLDER,
                return_schema=core_schema.str_schema(),
                when_used="always",
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, schema: CoreSchema, handler: Any
    ) -> JsonSchemaValue:
        return {
            "type": "string",
            "title": "RedactedDigits",
            "description": (
                "Digits supplied inbound and never echoed. Serialises as "
                f"{REDACTED_PLACEHOLDER!r}."
            ),
            "writeOnly": True,
        }


# ------------------------------------------------------------- leak guard


# Separators a bank, or a mail client, puts between groups of digits. The
# dash range is U+2010 to U+2015, built with chr() rather than written out so
# the source holds no character that reads as an ASCII hyphen but is not one.
_SEPARATOR_CHARS: Final[frozenset[str]] = frozenset(
    "-./" + "".join(chr(code) for code in range(0x2010, 0x2016))
)


def normalise_expected(raw: str) -> str:
    """Strip the formatting a bank writes in, keep the digits.

    Spaces, hyphens, dots and slashes are separators. Anything else is a sign
    the caller passed something other than an account identifier, so it fails
    loudly rather than being silently discarded.
    """

    stripped = "".join(
        ch for ch in raw if not ch.isspace() and ch not in _SEPARATOR_CHARS
    )
    if not stripped.isdigit():
        raise ValueError(
            "expected value must contain only digits and separators "
            "(space, hyphen, dot, slash)"
        )
    return stripped


def _digit_runs(digits: str, length: int) -> set[str]:
    if len(digits) <= length:
        return {digits}
    return {digits[i : i + length] for i in range(len(digits) - length + 1)}


def leaks(text: str, secret: RedactedDigits) -> bool:
    """Whether `text` discloses a useful run of the expected digits.

    Checks the text as written and the text with every non-digit removed, so
    "1 2 3 4" and "1-2-3-4" are caught alongside "1234".
    """

    digits = secret.reveal()
    haystacks = (text, re.sub(r"\D", "", text))
    for run in _digit_runs(digits, LEAK_SUBSTRING_DIGITS):
        if not run:
            continue
        if any(run in haystack for haystack in haystacks):
            return True
    return False


def scrub(text: str, secret: RedactedDigits) -> str:
    """Replace any disclosed run of the expected digits with the placeholder.

    Used on strings this module did not author. Everything it does author is a
    constant, which is why `leaks` is an assertion rather than a filter.
    """

    cleaned = text
    for run in sorted(
        _digit_runs(secret.reveal(), LEAK_SUBSTRING_DIGITS), key=len, reverse=True
    ):
        if run:
            cleaned = cleaned.replace(run, REDACTED_PLACEHOLDER)
    return cleaned


def safe_validation_detail(exc: ValidationError, *, limit: int = 3) -> str:
    """Render a validation failure without the input that caused it.

    This exists because `str(exc)` is a leak. pydantic attaches the offending
    value to every error, and for `VerificationRequest` that value can be the
    expected account number: a field error reports the raw string, and the
    model-level target rule reports the whole raw request dict. `str(exc)`
    truncates long inputs, which hides the problem rather than fixing it.

    `msg` is a template with no value interpolated into it, so location plus
    message is safe. Use this anywhere a validation failure is about to become
    a response body or a log line, and never the exception itself.
    """

    parts = [
        f"{'.'.join(str(bit) for bit in error['loc']) or 'body'}: {error['msg']}"
        for error in exc.errors(
            include_url=False, include_context=False, include_input=False
        )[:limit]
    ]
    return "; ".join(parts)


def _guard(text: str, secret: RedactedDigits) -> str:
    if leaks(text, secret):
        raise LeakGuardError(
            "refusing to return a string that discloses the expected digits"
        )
    return text


# --------------------------------------------------- spoken digit parsing

_UNITS: Final[dict[str, str]] = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "nought": "0",
    "naught": "0",
    "nil": "0",
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

_TEENS: Final[dict[str, str]] = {
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
}

_TENS: Final[dict[str, str]] = {
    "twenty": "2",
    "thirty": "3",
    "forty": "4",
    # Misspelling common enough in speech-to-text output to be worth taking.
    "fourty": "4",
    "fifty": "5",
    "sixty": "6",
    "seventy": "7",
    "eighty": "8",
    "ninety": "9",
}

_MULTIPLIERS: Final[dict[str, int]] = {
    "double": 2,
    "dubble": 2,
    "triple": 3,
    "treble": 3,
    "quadruple": 4,
}

# Magnitude words are refused rather than guessed. "Four hundred and twenty"
# could be 420, and "four hundred twenty three" could be 423 or 400 23, and a
# bank number spoken in magnitudes is rare enough that guessing buys nothing.
# Refusing lands the call in needs_review, which is where an ambiguous
# recording belongs.
_MAGNITUDES: Final[frozenset[str]] = frozenset(
    {"hundred", "thousand", "million", "billion", "grand", "k"}
)

# Words that carry no digit. Homophones of digits ("for", "to", "too", "ate",
# "won", "ate") are deliberately absent from _UNITS and present here only
# where they are unambiguous English, because reading "for" as 4 would invent
# digits out of ordinary speech.
_FILLER: Final[frozenset[str]] = frozenset(
    {
        "a",
        "aba",
        "account",
        "accounts",
        "after",
        "again",
        "ah",
        "all",
        "an",
        "and",
        "another",
        "are",
        "as",
        "at",
        "back",
        "bank",
        "be",
        "break",
        "bic",
        "by",
        "can",
        "code",
        "comma",
        "dash",
        "details",
        "did",
        "digit",
        "digits",
        "do",
        "dot",
        "em",
        "er",
        "erm",
        "ending",
        "ends",
        "figure",
        "figures",
        "first",
        "for",
        "from",
        "get",
        "give",
        "go",
        "going",
        "got",
        "have",
        "here",
        "hm",
        "hmm",
        "hold",
        "hyphen",
        "i",
        "iban",
        "if",
        "in",
        "is",
        "it",
        "its",
        "just",
        "last",
        "let",
        "like",
        "line",
        "me",
        "mm",
        "my",
        "need",
        "new",
        "next",
        "no",
        "number",
        "numbers",
        "of",
        "off",
        "ok",
        "okay",
        "on",
        "once",
        "one's",
        "our",
        "pause",
        "please",
        "point",
        "read",
        "repeat",
        "right",
        "routing",
        "say",
        "second",
        "see",
        "slash",
        "so",
        "sorry",
        "sort",
        "space",
        "starts",
        "stop",
        "sure",
        "swift",
        "thank",
        "thanks",
        "that",
        "thats",
        "the",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "too",
        "uh",
        "um",
        "us",
        "was",
        "we",
        "well",
        "which",
        "with",
        "yeah",
        "yep",
        "yes",
        "you",
        "your",
    }
)

_TOKEN_RE: Final = re.compile(r"[a-z]+|\d+")


@dataclass(frozen=True, slots=True)
class SpokenDigits:
    """What a transcript yielded.

    `ambiguity` is set when the wording could mean more than one number. The
    digits are then empty and nothing is compared, because a guess that
    happens to match would be the worst possible outcome here. A transcript
    that simply held no digits yields empty digits with no ambiguity: there
    was nothing to misread, only nothing to read.
    """

    digits: str
    ignored_token_count: int
    ambiguity: str | None = None

    @property
    def is_usable(self) -> bool:
        return self.ambiguity is None and bool(self.digits)


def join_utterances(parts: str | Sequence[str]) -> str:
    """Join speech-to-text fragments before parsing them.

    Spoken numbers run across utterance boundaries ("forty" / "five" arriving
    as two fragments is one number, 45), so the fragments are joined and then
    parsed as a whole rather than parsed one at a time and concatenated.
    """

    if isinstance(parts, str):
        return parts
    return " ".join(parts)


def parse_spoken_digits(transcript: str) -> SpokenDigits:
    """Turn a messy transcript into a digit string.

    Accepted, with examples:

    * bare digits, grouped or not: "4567", "45 67", "45-67"
    * digit words: "four five six seven"
    * zero spoken as "oh", "o", "nought", "naught" or "nil"
    * teens as one token: "nineteen" is 1 then 9
    * tens compounded with a following unit: "forty five" is 4 then 5, and
      "forty five six" is 4, 5, 6. A bare tens word keeps its zero, so
      "sixty" is 6 then 0
    * repeat words: "double seven" is 77, "triple eight" is 888, "double oh"
      is 00
    * filler, politeness and field names ("um", "sort code", "the account
      number is") are dropped
    * hyphens, dots, slashes, commas and pauses are separators

    Refused as ambiguous, which lands the call in needs_review:

    * magnitude words: "four hundred and twenty" could be 420 or 4 100 20
    * a repeat word with nothing single to repeat: "double", "double forty"

    A transcript with no digits in it at all is not ambiguous, just empty. It
    yields no digits, which `classify_match` reports as `unparseable`.

    Not accepted as digits, on purpose: homophones. "For", "to", "too", "ate"
    and "won" stay ordinary English words, because reading them as digits
    would invent numbers out of normal speech.

    Anything else unrecognised is dropped and counted in
    `ignored_token_count`, since real transcripts contain words no list
    anticipates.
    """

    tokens = _TOKEN_RE.findall(transcript.lower())
    out: list[str] = []
    ignored = 0
    index = 0

    while index < len(tokens):
        token = tokens[index]

        if token.isdigit():
            out.append(token)
            index += 1
            continue

        if token in _MAGNITUDES:
            return SpokenDigits(
                digits="",
                ignored_token_count=ignored,
                ambiguity=(
                    "transcript uses a magnitude word, which could mean more "
                    "than one number"
                ),
            )

        if token in _MULTIPLIERS:
            repeat = _MULTIPLIERS[token]
            following = tokens[index + 1] if index + 1 < len(tokens) else None
            if following is not None and following in _UNITS:
                out.append(_UNITS[following] * repeat)
                index += 2
                continue
            if following is not None and following.isdigit() and len(following) == 1:
                out.append(following * repeat)
                index += 2
                continue
            return SpokenDigits(
                digits="",
                ignored_token_count=ignored,
                ambiguity=("transcript repeats something that is not a single digit"),
            )

        if token in _TENS:
            following = tokens[index + 1] if index + 1 < len(tokens) else None
            # "forty five" is the pair 4 5, which is also what a digit by
            # digit reading gives. A following zero word is kept separate so
            # "forty oh" stays three digits rather than losing one.
            if following in _UNITS and _UNITS[following] != "0":
                out.append(_TENS[token] + _UNITS[following])
                index += 2
                continue
            out.append(_TENS[token] + "0")
            index += 1
            continue

        if token in _TEENS:
            out.append(_TEENS[token])
            index += 1
            continue

        if token in _UNITS:
            out.append(_UNITS[token])
            index += 1
            continue

        if token in _FILLER:
            index += 1
            continue

        ignored += 1
        index += 1

    return SpokenDigits(digits="".join(out), ignored_token_count=ignored)


# ------------------------------------------------------------- comparison


def _is_one_deletion(longer: str, shorter: str) -> bool:
    """Whether deleting exactly one character from `longer` gives `shorter`."""

    if len(longer) != len(shorter) + 1:
        return False
    i = 0
    while i < len(shorter) and longer[i] == shorter[i]:
        i += 1
    return longer[i + 1 :] == shorter[i:]


def classify_match(expected: RedactedDigits, spoken: str) -> MatchClass:
    """How the spoken digits differ from the expected ones.

    The policy, in order, and deliberately not a generic edit distance so
    that each forgiven shape is named and testable:

    1. identical, compared in constant time, is `exact`
    2. nothing usable from the recording is `unparseable`
    3. an expected value shorter than NEAR_MISS_MIN_DIGITS admits no near
       miss at all: anything but exact is `unrelated`
    4. same length, one differing position: `single_digit_substitution`
    5. same length, two differing positions that are neighbours and hold each
       other's digits: `adjacent_transposition`. A swap of two digits that are
       not neighbours is two substitutions, so it is `unrelated`
    6. one digit shorter, obtainable by deleting one digit: `missing_digit`
    7. one digit longer, obtainable by deleting one digit: `extra_digit`
    8. anything else: `unrelated`

    Rules 4 to 7 are the near misses. They are the only interesting cases:
    each is exactly as consistent with a speech-to-text error as with someone
    reading a number that is nearly right, so none of them may resolve to
    `verified` and none of them may be hardened into `mismatch` either.
    """

    if not spoken:
        return MatchClass.UNPARSEABLE

    target = expected.reveal()
    if compare_digest(target, spoken):
        return MatchClass.EXACT

    if len(target) < NEAR_MISS_MIN_DIGITS:
        return MatchClass.UNRELATED

    if len(spoken) == len(target):
        differing = [
            i for i, (a, b) in enumerate(zip(target, spoken, strict=True)) if a != b
        ]
        if len(differing) == MAX_NEAR_MISS_EDITS:
            return MatchClass.SINGLE_DIGIT_SUBSTITUTION
        if (
            len(differing) == 2
            and differing[1] == differing[0] + 1
            and target[differing[0]] == spoken[differing[1]]
            and target[differing[1]] == spoken[differing[0]]
        ):
            return MatchClass.ADJACENT_TRANSPOSITION
        return MatchClass.UNRELATED

    if len(spoken) == len(target) - 1 and _is_one_deletion(target, spoken):
        return MatchClass.MISSING_DIGIT

    if len(spoken) == len(target) + 1 and _is_one_deletion(spoken, target):
        return MatchClass.EXTRA_DIGIT

    return MatchClass.UNRELATED


def verdict_for_match(match: MatchClass) -> Verdict:
    """The product verdict for one discrepancy shape."""

    return _VERDICT_BY_MATCH[match]


def disclosure_for(verdict: Verdict) -> CallerDisclosure:
    """What the caller may hear.

    `mismatch` and `needs_review` collapse to the same disclosure. That is the
    point: a caller who gets it wrong learns nothing about how wrong.
    """

    if verdict is Verdict.VERIFIED:
        return CallerDisclosure.CONFIRMED
    return CallerDisclosure.NOT_CONFIRMED


# ----------------------------------------------------------------- models


class VerificationRequest(StrictModel):
    """POST body for the details check.

    One field per request. Asking for the account number and the sort code in
    separate exchanges is how the call actually runs, and it keeps the
    transcript unambiguous: two numbers in one utterance cannot be split
    reliably.
    """

    session_id: str = Field(min_length=8, max_length=128)
    target: VerificationTarget
    expected: RedactedDigits = Field(
        description=(
            "The digits from the vendor's email. Inbound only: this never "
            "appears in a response, a log or an error."
        )
    )
    transcript: str = Field(
        max_length=MAX_TRANSCRIPT_CHARS,
        description=(
            "What the caller said, speech-to-text fragments joined with whitespace."
        ),
    )
    voice_decision: Decision | None = Field(
        default=None,
        description=(
            "The detector's verdict for this session, if the agent already "
            "has it. Absent is treated as pending, which cannot verify."
        ),
    )
    voice_risk_score: Score | None = Field(
        default=None, description="The detector's blended risk, 0.0 safest."
    )

    @model_validator(mode="after")
    def _check_target_shape(self) -> VerificationRequest:
        low, high = TARGET_DIGIT_RULES[self.target]
        count = len(self.expected.reveal())
        if not low <= count <= high:
            # The message states the policy, which is a public constant, and
            # never the value or its actual length. pydantic still attaches
            # the raw request dict to the error it wraps this in, so render
            # the failure with safe_validation_detail, never str(exc).
            raise ValueError(
                f"expected {self.target.value} must have between {low} and "
                f"{high} digits"
            )
        return self


class VerificationResult(StrictModel):
    """The details verdict. The operator record that is attached to the bill.

    Carries no digit, no position and no length. `match_class` names the shape
    of the discrepancy because the person approving the payment needs it;
    `caller_disclosure` is the only part of this that may be spoken.
    """

    verification_version: str = VERIFICATION_VERSION
    session_id: str
    target: VerificationTarget
    verdict: Verdict
    match_class: MatchClass
    caller_disclosure: CallerDisclosure
    attempts_used: int = Field(ge=0)
    attempts_remaining: int = Field(ge=0)
    rate_limited: bool = Field(
        default=False,
        description="True when the attempt cap stopped the comparison.",
    )
    transcript_ambiguous: bool = Field(
        default=False,
        description="True when the wording could mean more than one number.",
    )
    reason: str = Field(max_length=500, description="Operator-facing, digit free.")

    def caller_script(self) -> str:
        """The exact words the agent may use on the line."""

        return CALLER_SCRIPT[self.caller_disclosure]


class CombinedVerdict(StrictModel):
    """Voice and details together.

    Both inputs are kept so the bill shows which dimension failed, since
    `mismatch` on its own does not say whether the digits were wrong or the
    caller was.
    """

    verification_version: str = VERIFICATION_VERSION
    session_id: str
    voice_decision: Decision
    voice_risk_score: Score | None = None
    details_verdict: Verdict
    verdict: Verdict
    payment_action: PaymentAction
    caller_disclosure: CallerDisclosure
    reason: str = Field(max_length=500)

    def caller_script(self) -> str:
        return CALLER_SCRIPT[self.caller_disclosure]


class VerificationResponse(StrictModel):
    """200 response from the details endpoint."""

    verification_version: str = VERIFICATION_VERSION
    details: VerificationResult
    combined: CombinedVerdict


def build_result(
    *,
    session_id: str,
    target: VerificationTarget,
    expected: RedactedDigits,
    match: MatchClass,
    attempts_used: int,
    attempts_remaining: int,
    rate_limited: bool = False,
    transcript_ambiguous: bool = False,
) -> VerificationResult:
    """Assemble a result, checking it for disclosure on the way out.

    The reason lines are constants, so `_guard` can never fire in production.
    It is here so that a future edit which interpolates the expected value
    into a message fails the test suite instead of shipping.
    """

    return VerificationResult(
        session_id=session_id,
        target=target,
        verdict=verdict_for_match(match),
        match_class=match,
        caller_disclosure=disclosure_for(verdict_for_match(match)),
        attempts_used=attempts_used,
        attempts_remaining=attempts_remaining,
        rate_limited=rate_limited,
        transcript_ambiguous=transcript_ambiguous,
        reason=_guard(_REASON_BY_MATCH[match], expected),
    )


def verify_request(
    request: VerificationRequest,
    *,
    attempts_used: int,
    attempts_remaining: int,
) -> VerificationResult:
    """Parse the transcript, compare it, and build the result."""

    spoken = parse_spoken_digits(request.transcript)
    if not spoken.is_usable:
        return build_result(
            session_id=request.session_id,
            target=request.target,
            expected=request.expected,
            match=MatchClass.UNPARSEABLE,
            attempts_used=attempts_used,
            attempts_remaining=attempts_remaining,
            transcript_ambiguous=spoken.ambiguity is not None,
        )

    return build_result(
        session_id=request.session_id,
        target=request.target,
        expected=request.expected,
        match=classify_match(request.expected, spoken.digits),
        attempts_used=attempts_used,
        attempts_remaining=attempts_remaining,
    )


def refused_result(
    *,
    session_id: str,
    target: VerificationTarget,
    expected: RedactedDigits,
    attempts_used: int,
) -> VerificationResult:
    """The result for an attempt the cap refused to compare."""

    return build_result(
        session_id=session_id,
        target=target,
        expected=expected,
        match=MatchClass.NOT_COMPARED,
        attempts_used=attempts_used,
        attempts_remaining=0,
        rate_limited=True,
    )


# ------------------------------------------------------------- schema bundle

SCHEMA_MODELS: tuple[type[BaseModel], ...] = (
    VerificationRequest,
    VerificationResult,
    CombinedVerdict,
    VerificationResponse,
)


def verification_schema_bundle() -> dict[str, Any]:
    """Deterministic JSON-schema bundle for the details check.

    Enum members and the policy constants are listed explicitly, so loosening
    a threshold or renaming a verdict shows up as a snapshot diff even when no
    model field changes.
    """

    enums: dict[str, list[str]] = {
        enum_cls.__name__: [member.value for member in enum_cls]
        for enum_cls in (
            Verdict,
            VerificationTarget,
            MatchClass,
            CallerDisclosure,
            PaymentAction,
        )
    }

    return {
        "verification_version": VERIFICATION_VERSION,
        "policy": {
            "near_miss_min_digits": NEAR_MISS_MIN_DIGITS,
            "max_near_miss_edits": MAX_NEAR_MISS_EDITS,
            "near_miss_classes": sorted(m.value for m in NEAR_MISS_CLASSES),
            "verdict_by_match": {
                match.value: verdict.value
                for match, verdict in sorted(_VERDICT_BY_MATCH.items())
            },
            "disclosure_by_verdict": {
                verdict.value: disclosure_for(verdict).value for verdict in Verdict
            },
            "target_digit_rules": {
                target.value: list(rule)
                for target, rule in sorted(TARGET_DIGIT_RULES.items())
            },
            "leak_substring_digits": LEAK_SUBSTRING_DIGITS,
            "max_transcript_chars": MAX_TRANSCRIPT_CHARS,
            "redacted_placeholder": REDACTED_PLACEHOLDER,
        },
        "caller_script": {
            disclosure.value: CALLER_SCRIPT[disclosure]
            for disclosure in CallerDisclosure
        },
        "enums": enums,
        "models": {
            model.__name__: model.model_json_schema(mode="serialization")
            for model in SCHEMA_MODELS
        },
    }
