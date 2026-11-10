"""Voice challenges for /v1 (ruling section 2.7).

A challenge is issued with a `say` the agent speaks word for word, expires 45 s
after issue, and is resolved lazily on read from timestamps on the registry
clock: it stays `pending` until the first user turn *received* after it was
issued, and turns into its result 700 ms after that receive time. With no user
turn before it expires it becomes `expired`, reason `no_answer`. A new
challenge while one is pending supersedes it: `expired`, reason `superseded`.

A digits, phrase or liveness challenge binds only to a turn that is an attempt
at the answer (`answers`): a stray "okay" or "one moment" between the prompt
and the answer leaves it pending. An enroll prompt binds the first user turn,
as it always did: it records a voice, and the turn's clip is what matters.

In mock mode the result is fixed at issue: the scenario's script. In real mode
a verification challenge `awaits_evidence`: the service settles it as it
resolves, from whether the bound turn carried the expected words
(`words_right`, a bool decided at bind time so no text is kept) and the voice
signals POST /events had measured by then. Until `outcome_fixed` it reads
`pending`, and once fixed it never changes, so the agent and the policy see
one result. With nothing heard it stays `failed`/`no_audio`, which the policy
never counts as a voice fail (ruling F1).

`say` and the expected answer never reach a log line, and `/end` drops both.
"""

from __future__ import annotations

import random
import re
import secrets
from collections.abc import Sequence
from dataclasses import dataclass

from callback_shared.contract import ChallengeKind, ChallengeState, ChallengeStatus

from detector.analysis.voiceprint import VoicePrint
from detector.v1.scenarios import ChallengeOutcome

# What an `enroll` prompt answered with usable audio resolves to. No
# phrase_match and no speaker_score, because an enroll prompt verifies
# nothing: it records a voice. The policy layer reads those two fields to
# count voice failures, and an enrollment must not feed that count.
ENROLL_HEARD = ChallengeOutcome(ChallengeState.PASSED, None, None, None)

CHALLENGE_TTL_S = 45.0
# A whole enrollment takes at most 21 (7 prompts, 3 tries each), and a call a
# handful more. Past this, a create is refused: records are never dropped,
# because the policy reads every failure and refusal in the session.
MAX_CHALLENGES_PER_SESSION = 50
RESOLVE_DELAY_S = 0.7
ENROLL_PROMPTS = 7

# The kinds the policy reads: a pass, a replay failure, a speaker failure, a
# refusal. `liveness` behaves as `digits`. `enroll` is none of these: it
# records a voice, it does not verify one.
VERIFICATION_KINDS = frozenset(
    {ChallengeKind.DIGITS, ChallengeKind.PHRASE, ChallengeKind.LIVENESS}
)

# Two-syllable concrete nouns and colours. No homophones of digits, no proper
# nouns, no two words one edit apart (N:596), and nothing that opens with a
# digit's sound ("tulip", "forest"), so a word never parses as a digit.
# tests/v1/test_challenges.py checks the edit distances and the count.
WORDS: tuple[str, ...] = (
    "amber", "anchor", "apple", "apron", "basket", "beacon", "bucket",
    "button", "cabin", "cactus", "camel", "candle", "canvas", "carpet",
    "castle", "cherry", "cobalt", "coffee", "copper", "coral", "cotton",
    "crayon", "crimson", "dolphin", "falcon", "feather", "finger", "garden",
    "goblet", "golden", "hammer", "harbor", "helmet", "island", "jacket",
    "kettle", "kitten", "ladder", "lantern", "lemon", "lilac", "lizard",
    "magnet", "maple", "marble", "maroon", "meadow", "mirror", "monkey",
    "muffin", "napkin", "needle", "nickel", "orange", "oyster", "paddle",
    "panda", "parrot", "peanut", "pebble", "pencil", "penguin", "pepper",
    "pickle", "pigeon", "pillow", "planet", "pumpkin", "puppet", "purple",
    "rabbit", "ribbon", "river", "rocket", "rubber", "salmon", "scarlet",
    "shovel", "silver", "spider", "statue", "sugar", "sweater", "tablet",
    "teapot", "thimble", "tiger", "timber", "tractor", "trumpet", "turtle",
    "velvet", "wagon", "waffle", "walnut", "whistle", "window", "yellow",
    "zebra",
)  # fmt: skip

# Prompt 7 of an enrollment: a free answer, so the print is not built only
# from read digits and words.
FREE_ANSWER_QUESTIONS: tuple[str, ...] = (
    "In a sentence or two, tell me what you had for breakfast today.",
    "In a sentence or two, tell me how you usually get to work.",
    "In a sentence or two, tell me about the weather where you are.",
)

_RNG: random.Random = secrets.SystemRandom()

# How speech-to-text spells a digit, and the digit itself.
_DIGIT_WORDS: dict[str, str] = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}  # fmt: skip
# A word (with an inner apostrophe, so "didn't" is one token) or one digit.
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d")

# What makes a user turn an attempt at the answer rather than a stray remark.
# Six digits is the shortest answer, so a turn that long is an attempt whatever
# it says; a shorter one is an attempt when two of its tokens are expected
# ones. One is not enough: "one moment" hits a digit six times in ten.
MIN_ANSWER_TOKENS = 6
MIN_EXPECTED_HITS = 2


def _spoken_tokens(text: str) -> list[str]:
    """Lowercase words and single digits, digit words turned into digits."""

    return [_DIGIT_WORDS.get(match, match) for match in _TOKEN.findall(text.lower())]


def words_match(expected: Sequence[str] | None, text: str) -> bool:
    """Whether the caller's turn carries the expected answer, in order.

    Every expected token has to appear in the spoken tokens as an ordered
    subsequence; words around them ("sure, ...", "... okay") are allowed, a
    missing, swapped or wrong token is not. Digits may be spoken ("four") or
    written ("4", or a run "472913"). An empty expectation never matches,
    because a challenge always expects something.
    """

    if not expected:
        return False
    spoken = _spoken_tokens(text)
    position = 0
    for token in expected:
        wanted = token.lower()
        while position < len(spoken) and spoken[position] != wanted:
            position += 1
        if position == len(spoken):
            return False
        position += 1
    return True


def answers(expected: Sequence[str] | None, text: str) -> bool:
    """Whether a user turn is an attempt at the answer, right or wrong.

    The first such turn after issue binds the challenge. "okay", "um" and
    "one moment" are not attempts; six wrong digits are, and fail it.
    """

    spoken = _spoken_tokens(text)
    if len(spoken) >= MIN_ANSWER_TOKENS:
        return True
    wanted = {token.lower() for token in expected or ()}
    return sum(1 for token in spoken if token in wanted) >= MIN_EXPECTED_HITS


def new_challenge_id() -> str:
    return "chl_" + secrets.token_hex(8)


def _digits(rng: random.Random) -> tuple[list[str], str]:
    digits = [str(rng.randrange(10)) for _ in range(6)]
    return digits, f"{' '.join(digits[:3])}, {' '.join(digits[3:])}"


def _words(rng: random.Random) -> list[str]:
    return rng.sample(WORDS, 3)


def make_say(
    kind: ChallengeKind,
    *,
    enroll_prompt: int | None = None,
    rng: random.Random | None = None,
) -> tuple[str, tuple[str, ...]]:
    """The `say` text and the expected answer tokens for one challenge."""

    source = rng or _RNG
    if kind is ChallengeKind.PHRASE:
        digits, spoken = _digits(source)
        words = _words(source)
        return f"{spoken}, {', '.join(words)}", (*digits, *words)
    if kind in (ChallengeKind.DIGITS, ChallengeKind.LIVENESS):
        digits, spoken = _digits(source)
        return spoken, tuple(digits)
    # enroll: a full sentence, because the agent speaks `say` with no lead-in.
    prompt = enroll_prompt or 1
    if prompt <= 3:
        digits, spoken = _digits(source)
        return f"Please repeat after me: {spoken}.", tuple(digits)
    if prompt <= 6:
        words = _words(source)
        return f"Please repeat after me: {', '.join(words)}.", tuple(words)
    return source.choice(FREE_ANSWER_QUESTIONS), ()


@dataclass(frozen=True)
class Resolution:
    """A challenge's state at one moment."""

    status: ChallengeState
    phrase_match: bool | None = None
    speaker_score: float | None = None
    reason: str | None = None
    resolved_at: float | None = None  # registry clock, None while pending

    def wire(self) -> ChallengeStatus:
        return ChallengeStatus(
            status=self.status,
            phrase_match=self.phrase_match,
            speaker_score=self.speaker_score,
            reason=self.reason,
        )


_PENDING = Resolution(ChallengeState.PENDING)


@dataclass
class ChallengeRecord:
    challenge_id: str
    kind: ChallengeKind
    issued_at: float  # registry clock, epoch seconds
    outcome: ChallengeOutcome  # fixed at issue
    say: str | None  # dropped on /end
    expected: tuple[str, ...] | None  # dropped on /end
    # The session's pending_bind when a digits or phrase challenge was issued.
    bind: str | None = None
    enroll_prompt: int | None = None
    answered_at: float | None = None  # receive time of the first user turn
    superseded_at: float | None = None
    # A tier 2 allow consumes the phrase pass it rested on.
    consumed: bool = False
    # Enrollment bookkeeping: the try has been counted.
    settled: bool = False
    # Real mode, verification kinds: `outcome` is not the result but a
    # placeholder (no_audio) until the service settles it from the evidence as
    # the challenge resolves. `resolve` reads pending until `outcome_fixed`.
    awaits_evidence: bool = False
    outcome_fixed: bool = False
    # Whether the bound turn's words matched `expected`, decided when the turn
    # arrived. Never the words themselves.
    words_right: bool | None = None
    # The voice measured from this prompt's clip, for an enroll challenge the
    # caller answered with audio. A print, never the audio: the clip is read
    # once, reduced, and dropped. Only ever set for ChallengeKind.ENROLL.
    voice: VoicePrint | None = None

    @property
    def expires_at(self) -> float:
        return self.issued_at + CHALLENGE_TTL_S

    @property
    def expires_at_ms(self) -> int:
        return round(self.issued_at * 1000) + round(CHALLENGE_TTL_S * 1000)

    def resolve(self, now: float) -> Resolution:
        if self.superseded_at is not None:
            return Resolution(
                ChallengeState.EXPIRED,
                reason="superseded",
                resolved_at=self.superseded_at,
            )
        if self.answered_at is None:
            if now >= self.expires_at:
                return Resolution(
                    ChallengeState.EXPIRED,
                    reason="no_answer",
                    resolved_at=self.expires_at,
                )
            return _PENDING
        done_at = self.answered_at + RESOLVE_DELAY_S
        if now < done_at or (self.awaits_evidence and not self.outcome_fixed):
            return _PENDING
        return Resolution(
            self.outcome.status,
            phrase_match=self.outcome.phrase_match,
            speaker_score=self.outcome.speaker_score,
            reason=self.outcome.reason,
            resolved_at=done_at,
        )

    def note_user_turn(self, received_at: float) -> bool:
        """The first user turn received after issue and before expiry answers it.

        True when this turn is the one that did.
        """

        if self.answered_at is not None or self.superseded_at is not None:
            return False
        if self.issued_at <= received_at < self.expires_at:
            self.answered_at = received_at
            return True
        return False

    def due_to_settle(self, now: float) -> bool:
        """Answered, past its 700 ms, and still waiting on the evidence."""

        return (
            self.awaits_evidence
            and not self.outcome_fixed
            and self.answered_at is not None
            and self.superseded_at is None
            and now >= self.answered_at + RESOLVE_DELAY_S
        )

    def supersede(self, now: float) -> bool:
        """Expire this challenge if it is still pending. True if it was."""

        if self.resolve(now).status is ChallengeState.PENDING:
            self.superseded_at = now
            return True
        return False

    def refused(self, now: float) -> bool:
        """Ended expired (no answer or superseded) with no user turn after issue."""

        state = self.resolve(now)
        return state.status is ChallengeState.EXPIRED and self.answered_at is None

    def forget_secrets(self) -> None:
        self.say = None
        self.expected = None
