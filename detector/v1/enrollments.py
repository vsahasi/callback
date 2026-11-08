"""Voiceprint enrollment and consent for /v1 (ruling section 2.15).

The flow: the caller is identified (a session exists), passes an SMS code in
that session, and says yes to the consent text. Then `POST /v1/enrollments`
records the consent and opens an enrollment of 7 prompts, which the agent runs
as `enroll` challenges. `complete` answers `enrolled` when at least 5 of the 7
prompts passed.

Prompts: 1-3 repeat six digits, 4-6 repeat three words, 7 is a free answer.
Each prompt gets up to 3 tries, like the agent's 1 + 2 retries: a failed try
reissues the same prompt number as a new challenge, and the next prompt comes
after a pass or after the third failed try.

Consent is an append-only, in-memory record, never an application log line.
`DELETE /v1/enrollments/{user_id}` erases the prints for both channels and
appends a `revoke` row.

Prints: real mode writes into PR 3's VoicePrintStore under
"{user_id}:{channel}", from listener clips only, which is stretch work. In
Round 2 real mode has no audio, every prompt fails with no_audio, and nothing
is written. Mock mode always enrolls and writes no print.
"""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass, field
from typing import Literal

from callback_shared.contract import ChallengeState, Channel, EnrollmentResult

from detector.v1.challenges import ENROLL_PROMPTS, ChallengeRecord
from detector.v1.errors import NoOpenEnrollment

MAX_TRIES_PER_PROMPT = 3
PASSES_TO_ENROLL = 5


def new_enrollment_id() -> str:
    return "enr_" + secrets.token_hex(16)


def print_identity(user_id: str, channel: Channel) -> str:
    """The VoicePrintStore key of one user's print on one channel."""

    return f"{user_id}:{channel.value}"


@dataclass
class EnrollmentRecord:
    enrollment_id: str
    session_id: str
    user_id: str
    channel: Channel
    consent_id: str
    started_at: float
    prompts_required: int = ENROLL_PROMPTS
    current_prompt: int = 1
    tries_on_prompt: int = 0
    passed_prompts: set[int] = field(default_factory=set)
    last_challenge: ChallengeRecord | None = None
    result: EnrollmentResult | None = None

    def _advance(self) -> None:
        self.current_prompt += 1
        self.tries_on_prompt = 0

    def settle(self, now: float, *, superseding: bool) -> None:
        """Count the last try once it has an outcome.

        `superseding` is True when a new enroll challenge is about to replace
        it, so a try still pending counts as failed: it is about to expire.
        """

        challenge = self.last_challenge
        if challenge is None or challenge.settled:
            return
        state = challenge.resolve(now)
        if state.status is ChallengeState.PENDING and not superseding:
            return
        challenge.settled = True
        if state.status is ChallengeState.PASSED and challenge.enroll_prompt:
            self.passed_prompts.add(challenge.enroll_prompt)
            self._advance()
        elif self.tries_on_prompt >= MAX_TRIES_PER_PROMPT:
            self._advance()

    def next_prompt(self, now: float) -> int:
        """The prompt number the next enroll challenge asks."""

        self.settle(now, superseding=True)
        if self.result is not None or self.current_prompt > self.prompts_required:
            raise NoOpenEnrollment(detail="every prompt has been asked")
        return self.current_prompt

    def issued(self, challenge: ChallengeRecord) -> None:
        self.last_challenge = challenge
        self.tries_on_prompt += 1

    def finish(self, now: float) -> EnrollmentResult:
        self.settle(now, superseding=False)
        passed = len(self.passed_prompts)
        self.result = EnrollmentResult(
            status="enrolled" if passed >= PASSES_TO_ENROLL else "failed",
            prompts_passed=passed,
        )
        return self.result


@dataclass(frozen=True)
class ConsentRow:
    consent_id: str
    kind: Literal["consent", "revoke"]
    user_id: str
    ts_ms: int
    channel: Channel | None = None
    version: str | None = None
    # The caller's answer, word for word. Kept out of repr so a stray print
    # of a row cannot put it in a log.
    verbatim: str | None = field(default=None, repr=False)


# Any user_id can be revoked, so revoke rows are bounded: past this many, the
# oldest revoke row goes. Consent rows are never dropped.
MAX_REVOKE_ROWS = 1000


class ConsentLog:
    """Append-only, except that old revoke rows age out. Never a log line."""

    def __init__(self, *, max_revokes: int = MAX_REVOKE_ROWS) -> None:
        self._rows: list[ConsentRow] = []
        self._lock = threading.Lock()
        self.max_revokes = max_revokes

    def append_consent(
        self,
        *,
        user_id: str,
        channel: Channel,
        version: str,
        verbatim: str,
        ts_ms: int,
    ) -> ConsentRow:
        row = ConsentRow(
            consent_id="cns_" + secrets.token_hex(16),
            kind="consent",
            user_id=user_id,
            channel=channel,
            version=version,
            verbatim=verbatim,
            ts_ms=ts_ms,
        )
        with self._lock:
            self._rows.append(row)
        return row

    def append_revoke(self, *, user_id: str, ts_ms: int) -> ConsentRow:
        row = ConsentRow(
            consent_id="cns_" + secrets.token_hex(16),
            kind="revoke",
            user_id=user_id,
            ts_ms=ts_ms,
        )
        with self._lock:
            self._rows.append(row)
            revokes = [i for i, r in enumerate(self._rows) if r.kind == "revoke"]
            for index in reversed(revokes[: max(0, len(revokes) - self.max_revokes)]):
                del self._rows[index]
        return row

    def rows(self) -> tuple[ConsentRow, ...]:
        with self._lock:
            return tuple(self._rows)

    def __len__(self) -> int:
        with self._lock:
            return len(self._rows)
