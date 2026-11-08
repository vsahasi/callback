"""Voice challenges for /v1 (ruling section 2.7).

A challenge is issued with a `say` the agent speaks word for word, expires 45 s
after issue, and is resolved lazily on read from timestamps on the registry
clock: it stays `pending` until the first user turn *received* after it was
issued, and turns into its result 700 ms after that receive time. With no user
turn before it expires it becomes `expired`, reason `no_answer`. A new
challenge while one is pending supersedes it: `expired`, reason `superseded`.

The result itself is fixed at issue: the scenario's script in mock mode, and
`failed`/`no_audio` in real mode until the listener exists.

`say` and the expected answer never reach a log line, and `/end` drops both.
"""

from __future__ import annotations

import random
import secrets
from dataclasses import dataclass

from callback_shared.contract import ChallengeKind, ChallengeState, ChallengeStatus

from detector.v1.scenarios import ChallengeOutcome

CHALLENGE_TTL_S = 45.0
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
        if now < done_at:
            return _PENDING
        return Resolution(
            self.outcome.status,
            phrase_match=self.outcome.phrase_match,
            speaker_score=self.outcome.speaker_score,
            reason=self.outcome.reason,
            resolved_at=done_at,
        )

    def note_user_turn(self, received_at: float) -> None:
        """The first user turn received after issue and before expiry answers it."""

        if self.answered_at is not None or self.superseded_at is not None:
            return
        if self.issued_at <= received_at < self.expires_at:
            self.answered_at = received_at

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
