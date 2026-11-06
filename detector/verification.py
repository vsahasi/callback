"""The details check as an HTTP endpoint, and the two verdicts combined.

Three things live here.

**The endpoint.** `POST /verify/bank-details`. One field per call, mounted by
whoever builds the app (see `build_verification_router`). It validates the body
itself instead of letting FastAPI do it, which looks odd until you notice that
pydantic attaches the offending input to every validation error and the app's
own handler stringifies `exc.errors()` into the response body. For a malformed
expected account number that would print the account number in a 422. So the
body arrives raw, is validated here, and errors are rendered from `loc` and
`msg` with `include_input=False`.

**The combination layer.** `shared.contract.Decision` answers "is the voice
genuine". `shared.verification.Verdict` answers "do the details match". Neither
is a verdict on its own: a genuine voice reading the wrong account is the exact
attack this product exists to catch, and a perfect read-back by a cloned voice
is the other half of it. `COMBINATION_TABLE` holds all twelve cells.

    voice        details        product        payment
    -----------  -------------  -------------  --------------------
    allow        verified       verified       release_for_approval
    allow        needs_review   needs_review   hold_for_review
    allow        mismatch       mismatch       block
    challenge    verified       needs_review   hold_for_review
    challenge    needs_review   needs_review   hold_for_review
    challenge    mismatch       mismatch       block
    pending      verified       needs_review   hold_for_review
    pending      needs_review   needs_review   hold_for_review
    pending      mismatch       mismatch       block
    deny         verified       mismatch       block
    deny         needs_review   mismatch       block
    deny         mismatch       mismatch       block

Read the shape of it rather than the cells:

* exactly one cell is `verified`, and it needs both halves to be clean
* the `mismatch` column is `mismatch` throughout, so wrong digits dominate any
  voice result, however clean
* the `deny` row never produces `verified`, so a caller the detector has
  refused cannot be rescued by reading the digits back correctly
* `pending` and `challenge` mean the voice question is still open, so they
  cannot verify. They fall to `needs_review`, because a human approves anyway
  and "unsure" is a cheaper mistake than "verified"
* `verified` still does not pay anyone. It means a person may now approve it

`deny` plus `verified` is the cell worth arguing about. The digits were right,
so calling it `mismatch` reads oddly, but it is a concrete contradiction (the
callback found the caller is not the vendor) and it must block. The result
keeps both input verdicts so the bill shows which dimension failed.

The risk score is used as a consistency check rather than a fourth input. An
`allow` that arrives with a risk score the detector's own thresholds would not
allow is a contradiction, so the voice side is downgraded before the table is
consulted. Same for a risk score at or above the deny threshold.

**The attempt cap.** Never revealing the expected number is worthless if the
caller can keep guessing: each attempt returns one bit ("were those the emailed
digits"), and a near miss narrows the search space to the edit-distance-1
neighbourhood. So attempts are capped per call and, separately, per expected
value across calls, and at the cap no comparison is performed at all.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
from collections import OrderedDict
from typing import Any, Final

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from detector import scoring
from detector.config import Settings
from detector.errors import DetectorError
from shared.contract import (
    Decision,
    ErrorBody,
    ErrorCode,
    ErrorResponse,
)
from shared.verification import (
    CallerDisclosure,
    CombinedVerdict,
    MatchClass,
    PaymentAction,
    RedactedDigits,
    Verdict,
    VerificationRequest,
    VerificationResponse,
    VerificationResult,
    VerificationTarget,
    disclosure_for,
    refused_result,
    safe_validation_detail,
    verify_request,
)

logger = logging.getLogger(__name__)

# Guesses allowed per call, per field. Matches the detector's OTP allowance,
# and is per (session, target) so confirming the account number does not spend
# the sort code's attempts.
MAX_ATTEMPTS_PER_SESSION: Final = 3

# Guesses allowed against one expected value across all calls. Without this, a
# caller who can open new sessions has no cap at all.
MAX_ATTEMPTS_PER_TARGET: Final = 6

# Ledger size. Attempt counters are process local, like the default session
# store, so this is a bound on memory rather than a security property.
MAX_TRACKED_KEYS: Final = 4096


class AttemptOutcome:
    """Immutable record of what the ledger decided for one attempt."""

    __slots__ = ("allowed", "attempts_remaining", "attempts_used")

    def __init__(self, *, allowed: bool, used: int, remaining: int) -> None:
        self.allowed = allowed
        self.attempts_used = used
        self.attempts_remaining = remaining

    def __repr__(self) -> str:
        return (
            f"AttemptOutcome(allowed={self.allowed}, "
            f"used={self.attempts_used}, remaining={self.attempts_remaining})"
        )


class AttemptLedger:
    """Counts guesses, per call and per expected value.

    The per-value counter is keyed by an HMAC under a key generated when the
    ledger is built. A plain digest of an eight digit number is reversible by
    anyone who can read the ledger, which would turn this defence into the leak
    it exists to prevent; an HMAC under a process-local key is not, and it does
    not need to survive a restart because the counter does not either.

    There is no time based decay. A call that burns its attempts stays burnt
    for its lifetime, which is the fail-closed direction and keeps the thing
    deterministic in tests.

    When the ledger is full the oldest entries are evicted, which resets their
    counters. The alternative, refusing new sessions, lets anyone lock out all
    verification by filling the table, so eviction is the lesser problem. The
    real fix is to keep these counters in the shared session store next to the
    OTP attempt counts.
    """

    def __init__(
        self,
        *,
        max_per_session: int = MAX_ATTEMPTS_PER_SESSION,
        max_per_target: int = MAX_ATTEMPTS_PER_TARGET,
        max_tracked: int = MAX_TRACKED_KEYS,
    ) -> None:
        self.max_per_session = max_per_session
        self.max_per_target = max_per_target
        self.max_tracked = max_tracked
        self._per_session: OrderedDict[tuple[str, str], int] = OrderedDict()
        self._per_target: OrderedDict[str, int] = OrderedDict()
        self._hmac_key = secrets.token_bytes(32)

    def _target_key(self, target: VerificationTarget, expected: RedactedDigits) -> str:
        return hmac.new(
            self._hmac_key,
            f"{target.value}:{expected.reveal()}".encode(),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def _bump(
        counts: OrderedDict[Any, int], key: Any, ceiling: int, max_tracked: int
    ) -> int:
        current = counts.pop(key, 0)
        # Stop counting once well past the ceiling: the answer is the same and
        # the integer stays small.
        nxt = current + 1 if current <= ceiling else current
        counts[key] = nxt
        while len(counts) > max_tracked:
            counts.popitem(last=False)
        return nxt

    def record(
        self,
        *,
        session_id: str,
        target: VerificationTarget,
        expected: RedactedDigits,
    ) -> AttemptOutcome:
        """Count this attempt and say whether it may be compared.

        The attempt is counted whether or not it is allowed, so a refused
        caller cannot reset the counter by trying again.
        """

        session_used = self._bump(
            self._per_session,
            (session_id, target.value),
            self.max_per_session,
            self.max_tracked,
        )
        target_used = self._bump(
            self._per_target,
            self._target_key(target, expected),
            self.max_per_target,
            self.max_tracked,
        )

        allowed = (
            session_used <= self.max_per_session and target_used <= self.max_per_target
        )
        # Zero when refused, including when it was the cross-call cap that
        # refused it: reporting attempts still available on a refused attempt
        # would invite a caller to keep trying.
        remaining = max(0, self.max_per_session - session_used) if allowed else 0
        return AttemptOutcome(allowed=allowed, used=session_used, remaining=remaining)


# ----------------------------------------------------------- combination

COMBINATION_TABLE: Final[dict[tuple[Decision, Verdict], Verdict]] = {
    (Decision.ALLOW, Verdict.VERIFIED): Verdict.VERIFIED,
    (Decision.ALLOW, Verdict.NEEDS_REVIEW): Verdict.NEEDS_REVIEW,
    (Decision.ALLOW, Verdict.MISMATCH): Verdict.MISMATCH,
    (Decision.CHALLENGE, Verdict.VERIFIED): Verdict.NEEDS_REVIEW,
    (Decision.CHALLENGE, Verdict.NEEDS_REVIEW): Verdict.NEEDS_REVIEW,
    (Decision.CHALLENGE, Verdict.MISMATCH): Verdict.MISMATCH,
    (Decision.PENDING, Verdict.VERIFIED): Verdict.NEEDS_REVIEW,
    (Decision.PENDING, Verdict.NEEDS_REVIEW): Verdict.NEEDS_REVIEW,
    (Decision.PENDING, Verdict.MISMATCH): Verdict.MISMATCH,
    (Decision.DENY, Verdict.VERIFIED): Verdict.MISMATCH,
    (Decision.DENY, Verdict.NEEDS_REVIEW): Verdict.MISMATCH,
    (Decision.DENY, Verdict.MISMATCH): Verdict.MISMATCH,
}

PAYMENT_ACTION_BY_VERDICT: Final[dict[Verdict, PaymentAction]] = {
    Verdict.VERIFIED: PaymentAction.RELEASE_FOR_APPROVAL,
    Verdict.NEEDS_REVIEW: PaymentAction.HOLD_FOR_REVIEW,
    Verdict.MISMATCH: PaymentAction.BLOCK,
}

# Operator-facing wording, keyed by the pair that produced the verdict. Every
# line is a constant: none of them can be built out of the expected value.
_COMBINED_REASON: Final[dict[tuple[Decision, Verdict], str]] = {
    (Decision.ALLOW, Verdict.VERIFIED): (
        "Voice and details both check out. Ready for a person to approve."
    ),
    (Decision.ALLOW, Verdict.NEEDS_REVIEW): (
        "Voice checks out but the read-back was not conclusive. Held for review."
    ),
    (Decision.ALLOW, Verdict.MISMATCH): (
        "Voice checks out but the caller did not read back the emailed "
        "account. Wrong details block the payment whatever the voice says."
    ),
    (Decision.CHALLENGE, Verdict.VERIFIED): (
        "Details read back correctly but the voice check is still open. Held "
        "for review."
    ),
    (Decision.CHALLENGE, Verdict.NEEDS_REVIEW): (
        "Neither the voice check nor the read-back concluded. Held for review."
    ),
    (Decision.CHALLENGE, Verdict.MISMATCH): (
        "The caller did not read back the emailed account. Payment blocked."
    ),
    (Decision.PENDING, Verdict.VERIFIED): (
        "Details read back correctly but no voice analysis has concluded, so "
        "there is nothing corroborating who said them. Held for review."
    ),
    (Decision.PENDING, Verdict.NEEDS_REVIEW): (
        "No voice analysis has concluded and the read-back was not "
        "conclusive. Held for review."
    ),
    (Decision.PENDING, Verdict.MISMATCH): (
        "The caller did not read back the emailed account. Payment blocked."
    ),
    (Decision.DENY, Verdict.VERIFIED): (
        "The digits were right but the caller is not who they claim to be, "
        "which is consistent with someone who has read the email. Payment "
        "blocked."
    ),
    (Decision.DENY, Verdict.NEEDS_REVIEW): (
        "The caller failed the voice check. Payment blocked."
    ),
    (Decision.DENY, Verdict.MISMATCH): (
        "The caller failed the voice check and did not read back the emailed "
        "account. Payment blocked."
    ),
}


def reconcile_voice(
    decision: Decision | None, risk_score: float | None
) -> tuple[Decision, str | None]:
    """Fold the risk score into the voice decision before combining.

    The score is not a fourth input to the table. It is a consistency check on
    the other three: an `allow` carrying a risk the detector's own thresholds
    would not allow means the two halves of the caller's claim disagree, and
    the honest response to that is to stop treating it as an allow. Both
    directions fail closed.
    """

    if decision is None:
        return Decision.PENDING, "No voice decision was supplied."
    if risk_score is None:
        return decision, None
    if risk_score >= scoring.DENY_AT_OR_ABOVE and decision is not Decision.DENY:
        return Decision.DENY, (
            "Voice decision disagreed with its own risk score, which is at or "
            "above the deny threshold. Taking the stricter reading."
        )
    if decision is Decision.ALLOW and risk_score >= scoring.ALLOW_BELOW:
        return Decision.CHALLENGE, (
            "Voice allow disagreed with its own risk score, which is above "
            "the allow threshold. Taking the stricter reading."
        )
    return decision, None


def combine_verdicts(
    *,
    session_id: str,
    details: VerificationResult,
    voice_decision: Decision | None,
    voice_risk_score: float | None = None,
) -> CombinedVerdict:
    """The product verdict from the voice verdict and the details verdict."""

    resolved, note = reconcile_voice(voice_decision, voice_risk_score)
    key = (resolved, details.verdict)
    # Not .get(): an unmapped pair is a bug, and guessing one would be the
    # kind of silent default this table exists to remove.
    verdict = COMBINATION_TABLE[key]
    reason = _COMBINED_REASON[key]
    if note is not None:
        reason = f"{reason} {note}"

    return CombinedVerdict(
        session_id=session_id,
        voice_decision=resolved,
        voice_risk_score=voice_risk_score,
        details_verdict=details.verdict,
        verdict=verdict,
        payment_action=PAYMENT_ACTION_BY_VERDICT[verdict],
        caller_disclosure=disclosure_for(verdict),
        reason=reason,
    )


def run_verification(
    request: VerificationRequest,
    *,
    ledger: AttemptLedger,
    voice_decision: Decision | None = None,
    voice_risk_score: float | None = None,
) -> VerificationResponse:
    """Compare, combine, and answer. The endpoint is a thin wrapper on this.

    Importable in process, so the agent can run the same policy without an
    HTTP hop.
    """

    outcome = ledger.record(
        session_id=request.session_id,
        target=request.target,
        expected=request.expected,
    )

    if outcome.allowed:
        details = verify_request(
            request,
            attempts_used=outcome.attempts_used,
            attempts_remaining=outcome.attempts_remaining,
        )
    else:
        # Fail closed at the cap: no comparison at all, so no further bit of
        # the expected value is disclosed by this attempt.
        details = refused_result(
            session_id=request.session_id,
            target=request.target,
            expected=request.expected,
            attempts_used=outcome.attempts_used,
        )

    combined = combine_verdicts(
        session_id=request.session_id,
        details=details,
        voice_decision=voice_decision
        if voice_decision is not None
        else request.voice_decision,
        voice_risk_score=voice_risk_score
        if voice_risk_score is not None
        else request.voice_risk_score,
    )

    # Digit free by construction: the session id, the field name and the
    # verdicts only. The transcript is never logged, because a transcript of a
    # successful read-back contains the emailed account number and a log file
    # is exactly where it must not end up.
    logger.info(
        "details check session=%s target=%s match=%s details=%s combined=%s "
        "attempt=%d rate_limited=%s",
        request.session_id,
        request.target.value,
        details.match_class.value,
        details.verdict.value,
        combined.verdict.value,
        details.attempts_used,
        details.rate_limited,
    )

    return VerificationResponse(details=details, combined=combined)


# -------------------------------------------------------------- the router


def _error(
    status: int, code: ErrorCode, message: str, detail: str | None
) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message, detail=detail))
    return JSONResponse(status_code=status, content=body.model_dump(mode="json"))


def _authorised(header: str | None, settings: Settings | None) -> bool:
    if settings is None:
        # No settings means nothing to compare against, so nobody is
        # authorised. The alternative is an open endpoint.
        return False
    if not header or not header.lower().startswith("bearer "):
        return False
    presented = header.split(" ", 1)[1].strip()
    # Bytes, not str: compare_digest raises TypeError on non-ASCII str and
    # Starlette decodes headers as latin-1, so one stray byte would turn a 401
    # into a 500. Same reasoning as detector.app.
    return hmac.compare_digest(
        presented.encode("utf-8"), settings.api_token.encode("utf-8")
    )


def _voice_from_engine(
    app_state: Any, session_id: str
) -> tuple[Decision | None, float | None]:
    """Read the detector's own verdict for this session, if it has one.

    Convenience only: the agent may pass the decision in the body instead, and
    a session the detector has never seen is treated as pending, which cannot
    verify.
    """

    engine = getattr(app_state, "engine", None)
    if engine is None:
        return None, None
    try:
        snapshot = engine.factors_for(session_id)
    except DetectorError:
        return None, None
    return snapshot.decision, snapshot.risk_score


def build_verification_router(
    settings: Settings | None = None,
    *,
    ledger: AttemptLedger | None = None,
) -> APIRouter:
    """Build the router.

    Pass `settings` to pin the bearer token, or leave it out and the token is
    read from `app.state.settings` on each request, which is what the detector
    app puts there. Mount it with:

        from detector.verification import build_verification_router
        app.include_router(build_verification_router(resolved))

    The ledger is per router, so one is created here unless a test injects
    its own.
    """

    router = APIRouter(prefix="/verify", tags=["verification"])
    attempts = ledger if ledger is not None else AttemptLedger()

    @router.post("/bank-details", response_model=VerificationResponse)
    async def verify_bank_details(request: Request) -> Response:
        """Compare spoken bank details against the emailed ones.

        Body: `shared.verification.VerificationRequest`. The body is validated
        here rather than by FastAPI, so that a malformed `expected` value
        cannot be echoed back inside a 422. See the module docstring.
        """

        resolved = settings or getattr(request.app.state, "settings", None)
        if not _authorised(request.headers.get("authorization"), resolved):
            return _error(
                401,
                ErrorCode.UNAUTHORIZED,
                "missing or invalid bearer token",
                None,
            )

        raw = await request.body()
        try:
            payload = json.loads(raw or b"null")
        except json.JSONDecodeError:
            return _error(
                422,
                ErrorCode.VALIDATION_ERROR,
                "request body is not valid JSON",
                None,
            )
        if not isinstance(payload, dict):
            return _error(
                422,
                ErrorCode.VALIDATION_ERROR,
                "request body must be a JSON object",
                None,
            )

        try:
            parsed = VerificationRequest.model_validate(payload)
        except ValidationError as exc:
            return _error(
                422,
                ErrorCode.VALIDATION_ERROR,
                "request body failed contract validation",
                safe_validation_detail(exc),
            )

        voice_decision = parsed.voice_decision
        voice_risk = parsed.voice_risk_score
        if voice_decision is None:
            voice_decision, voice_risk = _voice_from_engine(
                request.app.state, parsed.session_id
            )

        answer = run_verification(
            parsed,
            ledger=attempts,
            voice_decision=voice_decision,
            voice_risk_score=voice_risk,
        )
        return JSONResponse(content=answer.model_dump(mode="json"))

    return router


# Mountable without arguments; the token then comes from app.state.settings.
router = build_verification_router()


__all__ = [
    "COMBINATION_TABLE",
    "MAX_ATTEMPTS_PER_SESSION",
    "MAX_ATTEMPTS_PER_TARGET",
    "PAYMENT_ACTION_BY_VERDICT",
    "AttemptLedger",
    "AttemptOutcome",
    "CallerDisclosure",
    "MatchClass",
    "build_verification_router",
    "combine_verdicts",
    "reconcile_voice",
    "router",
    "run_verification",
]
