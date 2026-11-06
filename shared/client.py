"""Reference client for the LiveKit agent side. The integration kit.

The point of this module is the fail-closed rule: if the detector is slow,
unreachable or broken, the agent must treat the call as DENY. Veer can import
this directly or copy the handful of lines that matter.

Two layers, pick one:

1. `DetectorClient` is the thin transport. One method per endpoint, you own the
   sequence numbers and the idempotency keys.

2. `CallSession` is the layer you probably want. It owns one call: monotonic
   sequence numbering, deterministic idempotency keys, one method per step of
   the real flow, and a guard so a finished session cannot be driven further.

        from shared.client import CallSession, DetectorClient

        async with DetectorClient(base_url=..., token=...) as detector:
            call = CallSession(detector, claimed_user_id="vendor-42")

            outcome = await call.start()
            outcome = await call.claim_identity("vendor-42")
            while outcome.should_continue_listening:
                outcome = await call.submit_speech()

            if outcome.should_request_otp:
                code = await ask_caller_for_code()
                outcome = await call.submit_otp(code)

            if outcome.is_allow:
                await release_payment_for_human_approval()
            else:
                await refuse_and_hang_up(outcome.reason)

Every method returns an `Outcome`. `Outcome.decision` is always populated, and
it is `DENY` whenever the detector could not be trusted to answer. There is no
path through this module that turns a failure into an ALLOW.

See docs/AGENT-INTEGRATION.md for the copy-paste guide and
docs/CONTRACT.md for the wire reference.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import ValidationError

from shared.contract import (
    CLIENT_TIMEOUT_MS,
    CONTRACT_VERSION,
    FAIL_CLOSED_DECISION,
    IDEMPOTENCY_HEADER,
    SCENARIO_HEADER,
    AudioRef,
    Decision,
    ErrorCode,
    ErrorResponse,
    EventResponse,
    EventType,
    FactorsResponse,
    HealthResponse,
    NextAction,
    OtpChallenge,
    Scenario,
    SessionState,
)

log = logging.getLogger("callback.client")

# A guard against the contract and the client drifting apart. If someone ever
# changes FAIL_CLOSED_DECISION to something that is not a deny, this module
# refuses to import rather than quietly failing open.
assert Decision(FAIL_CLOSED_DECISION) is Decision.DENY, (
    "shared.contract.FAIL_CLOSED_DECISION must be the deny decision; "
    "this client is only safe if failing closed means denying."
)

__all__ = [
    "CallSession",
    "DetectorClient",
    "FactorsOutcome",
    "Outcome",
    "RetryPolicy",
    "SessionClosed",
    "new_session_id",
]


class SessionClosed(RuntimeError):
    """Raised when a finished `CallSession` is driven further.

    This is a programming error, not a detector failure: the session already
    reached a terminal decision (or already failed closed) and no further event
    can change it. The last outcome is attached, so a caller that catches this
    still has the decision it must act on.

    Check `CallSession.closed` first if you want to avoid the exception.
    """

    def __init__(self, message: str, outcome: Outcome | None = None) -> None:
        super().__init__(message)
        self.outcome = outcome


# --------------------------------------------------------------------- outcomes


@dataclass(frozen=True)
class Outcome:
    """What the agent should act on.

    `decision` is always populated. `response` is None when the detector failed
    and the decision was synthesised by the fail-closed rule.

    The convenience properties below are the point of this type: they read the
    same whether the detector answered or not, so the agent needs one branch
    rather than two.
    """

    decision: Decision
    response: EventResponse | None
    failed_closed: bool = False
    failure_reason: str | None = None
    # HTTP status of the final attempt, when there was a response at all.
    status_code: int | None = None
    # Parsed `error.code` from a non-2xx body, when the body was well formed.
    error_code: ErrorCode | None = None
    # How many HTTP attempts this logical event cost, including the first.
    attempts: int = 1

    # ---------------------------------------------------------- what to do now

    @property
    def next_action(self) -> NextAction:
        """The instruction to act on. `complete_deny` when we failed closed."""

        if self.response is None:
            return NextAction.COMPLETE_DENY
        return self.response.next_action

    @property
    def state(self) -> SessionState:
        """Server-side session state. `denied` when we failed closed."""

        if self.response is None:
            return SessionState.DENIED
        return self.response.state

    @property
    def is_allow(self) -> bool:
        return self.decision is Decision.ALLOW

    @property
    def is_deny(self) -> bool:
        return self.decision is Decision.DENY

    @property
    def should_request_otp(self) -> bool:
        """True when the caller must be asked for a one-time code.

        False on a hard deny even though the decision is not final-looking,
        because `replay` and `ai_agent` never issue a challenge.
        """

        return self.next_action is NextAction.REQUEST_OTP

    @property
    def should_continue_listening(self) -> bool:
        return self.next_action is NextAction.CONTINUE_LISTENING

    @property
    def is_final(self) -> bool:
        """True when no further event can change the answer."""

        if self.response is None:
            return True
        return self.response.state.is_terminal

    # ------------------------------------------------------------ passthroughs

    @property
    def otp(self) -> OtpChallenge | None:
        """The live challenge, or None.

        None is normal, not exceptional: a hard deny never issues one. Never
        assume step-up is available.
        """

        if self.response is None:
            return None
        return self.response.otp

    @property
    def risk_score(self) -> float | None:
        """Diagnostics only. Branch on `next_action`, never on this."""

        if self.response is None:
            return None
        return self.response.risk_score

    @property
    def reason(self) -> str | None:
        if self.response is None:
            return self.failure_reason
        return self.response.reason

    @property
    def session_id(self) -> str | None:
        if self.response is None:
            return None
        return self.response.session_id


@dataclass(frozen=True)
class FactorsOutcome:
    """Result of reading `GET /factors/{session_id}`.

    Read-only: it never advances the state machine. It still carries a
    `decision` so an unreadable session cannot be mistaken for a safe one. Use
    it for operator display and debugging, and drive the call from the
    `Outcome` that `/events` returned.
    """

    decision: Decision
    factors: FactorsResponse | None
    failed_closed: bool = False
    failure_reason: str | None = None
    status_code: int | None = None
    error_code: ErrorCode | None = None


# ----------------------------------------------------------------- retry policy


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retry, deliberately narrow.

    Three rules make a retry safe here, and all three are enforced below.

    1. **Same idempotency key, byte-identical body.** The detector caches a
       response per idempotency key and rejects a reused key whose payload
       changed (409 `sequence_conflict`). So a retry can only ever land as a
       fresh application or as an idempotent replay of the first attempt. The
       body, `occurred_at` included, is built once per logical event and the
       same bytes are resent. Nothing re-drives a state transition.

    2. **Only on failures that carry no decision.** Timeouts, transport errors
       and 5xx. Never on a 4xx: those are deterministic answers from the
       service (401, 404, 409, 410, 422) and a retry would only hide them.
       Never on a malformed 200 either, since the cached reply would be just as
       malformed.

    3. **Inside the original latency budget, not on top of it.** The whole
       logical event, retries and backoff included, stays inside
       `total_budget_ms`, which defaults to the contract's 800 ms. Per-attempt
       timeouts shrink to fit. A retry is skipped when too little budget is
       left to be worth it, and the client fails closed instead of making a
       live call wait.

    Note what rule 3 implies: a read timeout usually consumes the whole budget
    on its own, so a timeout is usually not retried. That is correct. Out of
    time means deny now. Retries exist for the cheap failures, a refused
    connection or a cold-start 503, which fail in milliseconds.
    """

    # Total HTTP attempts for one logical event, the first one included.
    max_attempts: int = 2
    # Wall-clock ceiling for the whole logical event.
    total_budget_ms: int = CLIENT_TIMEOUT_MS
    # Backoff before the second attempt, doubled each time, capped.
    backoff_initial_ms: int = 25
    backoff_multiplier: float = 2.0
    backoff_max_ms: int = 100
    # Random extra delay added to each backoff, to avoid synchronised retries.
    jitter_ms: int = 25
    # Do not start another attempt unless at least this much budget remains.
    min_attempt_ms: int = 100
    retry_on_server_error: bool = True

    @classmethod
    def none(cls) -> RetryPolicy:
        """A single attempt. The default for `DetectorClient`."""

        return cls(max_attempts=1)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.total_budget_ms <= 0:
            raise ValueError("total_budget_ms must be positive")

    def backoff_seconds(self, attempt: int, rng: random.Random | None = None) -> float:
        """Delay before attempt number `attempt + 1`, in seconds."""

        base = self.backoff_initial_ms * (self.backoff_multiplier ** (attempt - 1))
        base = min(base, float(self.backoff_max_ms))
        source = rng or random
        jitter = source.uniform(0.0, float(self.jitter_ms)) if self.jitter_ms else 0.0
        return (base + jitter) / 1000.0


# ---------------------------------------------------------------------- client


class DetectorClient:
    """Thin, fail-closed transport over the three detector endpoints.

    Default behaviour is unchanged from the first version of this file: one
    attempt, 800 ms, deny on anything that is not a well-formed 200. Pass
    `retry=RetryPolicy()` to opt into bounded retry, or use `CallSession`,
    which opts in for you.
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_ms: int = CLIENT_TIMEOUT_MS,
        client: httpx.AsyncClient | None = None,
        retry: RetryPolicy | None = None,
    ) -> None:
        self._timeout_s = timeout_ms / 1000.0
        self._owns_client = client is None
        self._retry = retry or RetryPolicy.none()
        self._warned_about_version = False
        self._client = client or httpx.AsyncClient(
            base_url=base_url,
            timeout=self._timeout_s,
            headers={"Authorization": f"Bearer {token}"},
        )

    @property
    def retry(self) -> RetryPolicy:
        return self._retry

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> DetectorClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    # ----------------------------------------------------------- POST /events

    async def push_event(
        self,
        *,
        session_id: str,
        event: EventType,
        sequence: int,
        claimed_user_id: str | None = None,
        otp_code: str | None = None,
        audio: AudioRef | None = None,
        scenario: Scenario | None = None,
        idempotency_key: str | None = None,
        metadata: dict[str, Any] | None = None,
        retry: RetryPolicy | None = None,
    ) -> Outcome:
        """Push one event and get the decision to act on.

        Never raises for a detector problem. A timeout, a refused connection, a
        5xx, a 4xx or a 200 whose body does not parse all come back as a
        fail-closed DENY with `failed_closed=True`.
        """

        # Built once. Every retry resends these exact bytes, which is what
        # makes the detector's idempotency cache able to protect us.
        body: dict[str, Any] = {
            "session_id": session_id,
            "event": event.value,
            "sequence": sequence,
            "occurred_at": datetime.now(UTC).isoformat(),
            "metadata": metadata or {},
        }
        if claimed_user_id is not None:
            body["claimed_user_id"] = claimed_user_id
        if otp_code is not None:
            body["otp_code"] = otp_code
        if audio is not None:
            body["audio"] = audio.model_dump(mode="json")

        headers: dict[str, str] = {}
        if scenario is not None:
            headers[SCENARIO_HEADER] = scenario.value
        if idempotency_key is not None:
            headers[IDEMPOTENCY_HEADER] = idempotency_key

        policy = retry or self._retry
        if policy.max_attempts > 1 and idempotency_key is None:
            # Without a key the detector falls back to "seq:<sequence>", which
            # still dedupes per session, but being explicit is the contract's
            # advice and keeps a retry diagnosable in the service logs.
            log.debug(
                "retrying without an explicit %s; the detector will key on the "
                "sequence number instead",
                IDEMPOTENCY_HEADER,
            )

        return await self._send_with_retry(
            body=body, headers=headers, policy=policy, event=event
        )

    async def _send_with_retry(
        self,
        *,
        body: dict[str, Any],
        headers: dict[str, str],
        policy: RetryPolicy,
        event: EventType,
    ) -> Outcome:
        started = time.monotonic()
        budget_s = policy.total_budget_ms / 1000.0
        last: Outcome | None = None

        for attempt in range(1, policy.max_attempts + 1):
            spent_s = time.monotonic() - started
            if policy.max_attempts == 1:
                # Preserve the single-attempt contract exactly: the full
                # per-call timeout, no budget arithmetic.
                attempt_timeout_s = self._timeout_s
            else:
                attempt_timeout_s = min(self._timeout_s, budget_s - spent_s)
                if attempt_timeout_s <= 0.0:
                    return self._fail_closed(
                        reason=(
                            f"latency budget of {policy.total_budget_ms}ms exhausted "
                            f"after {attempt - 1} attempt(s)"
                        ),
                        attempts=attempt - 1 or 1,
                        previous=last,
                    )

            outcome, retryable = await self._attempt(
                body=body,
                headers=headers,
                timeout_s=attempt_timeout_s,
                attempt=attempt,
                policy=policy,
            )
            last = outcome
            if not retryable or attempt == policy.max_attempts:
                return replace(outcome, attempts=attempt)

            backoff_s = policy.backoff_seconds(attempt)
            spent_s = time.monotonic() - started
            remaining_s = budget_s - spent_s - backoff_s
            if remaining_s < policy.min_attempt_ms / 1000.0:
                log.debug(
                    "not retrying %s: only %.0fms of the %dms budget would remain",
                    event.value,
                    remaining_s * 1000.0,
                    policy.total_budget_ms,
                )
                return replace(outcome, attempts=attempt)

            log.info(
                "retrying %s after a retryable failure (%s); attempt %d of %d",
                event.value,
                outcome.failure_reason,
                attempt + 1,
                policy.max_attempts,
            )
            await asyncio.sleep(backoff_s)

        # Unreachable: the loop always returns. Fail closed regardless.
        return self._fail_closed(
            reason="retry loop ended without a decision",
            attempts=policy.max_attempts,
            previous=last,
        )

    async def _attempt(
        self,
        *,
        body: dict[str, Any],
        headers: dict[str, str],
        timeout_s: float,
        attempt: int,
        policy: RetryPolicy,
    ) -> tuple[Outcome, bool]:
        """One HTTP attempt. Returns the outcome and whether to retry it.

        Total by construction: there is no exception this can raise. The
        handlers below are ordered narrowest first, and the last one is a
        blind catch on purpose, because the guarantee this module advertises
        must not depend on httpx's exception hierarchy keeping the shape it
        has today. `asyncio.CancelledError` is a BaseException and so is not
        swallowed, which is what we want: a cancelled call is not a decision.
        """

        try:
            http_response = await self._client.post(
                "/events", json=body, headers=headers, timeout=timeout_s
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            # Failed at the network. Cheap failures are worth one retry.
            return (
                Outcome(
                    decision=Decision.DENY,
                    response=None,
                    failed_closed=True,
                    failure_reason=f"detector unreachable: {type(exc).__name__}",
                    attempts=attempt,
                ),
                True,
            )
        except httpx.HTTPError as exc:
            # The other half of the httpx hierarchy: DecodingError (a proxy
            # setting a Content-Encoding the body does not match, decoded
            # eagerly inside post()), TooManyRedirects, and anything else that
            # is not a transport failure. Bytes arrived but are unusable, so
            # this is deterministic and is not retried, exactly like a
            # malformed 200.
            return (
                Outcome(
                    decision=Decision.DENY,
                    response=None,
                    failed_closed=True,
                    failure_reason=(
                        f"detector reply unusable: {type(exc).__name__}: {exc}"
                    ),
                    attempts=attempt,
                ),
                False,
            )
        except Exception as exc:
            # Belt and braces. An exception is not a DENY: the agent's
            # `if outcome.is_allow: ... else: hang_up()` never runs, and an
            # agent with a broad except around its turn handler would carry on
            # with the call. So nothing escapes from here.
            log.exception("unexpected failure talking to the detector")
            return (
                Outcome(
                    decision=Decision.DENY,
                    response=None,
                    failed_closed=True,
                    failure_reason=(
                        f"unexpected client failure: {type(exc).__name__}: {exc}"
                    ),
                    attempts=attempt,
                ),
                False,
            )

        status = http_response.status_code

        if status >= 500:
            return (
                Outcome(
                    decision=Decision.DENY,
                    response=None,
                    failed_closed=True,
                    failure_reason=f"detector returned {status}",
                    status_code=status,
                    error_code=_error_code_of(http_response),
                    attempts=attempt,
                ),
                policy.retry_on_server_error,
            )

        if status >= 400:
            # Deterministic. The service answered and said no. Retrying would
            # only hide a real 401, 409, 410 or 422.
            return (
                Outcome(
                    decision=Decision.DENY,
                    response=None,
                    failed_closed=True,
                    failure_reason=(
                        f"detector rejected the request with "
                        f"{status}: {_safe_text(http_response)}"
                    ),
                    status_code=status,
                    error_code=_error_code_of(http_response),
                    attempts=attempt,
                ),
                False,
            )

        try:
            parsed = EventResponse.model_validate(http_response.json())
        except (ValidationError, ValueError, httpx.HTTPError) as exc:
            # A 200 we cannot read is worse than no answer, not better. The
            # cached reply to a retry would be identical, so do not retry.
            # httpx.HTTPError covers a body that will not decode at all.
            return (
                Outcome(
                    decision=Decision.DENY,
                    response=None,
                    failed_closed=True,
                    failure_reason=(
                        f"detector returned {status} with a body this client "
                        f"cannot read: {type(exc).__name__}"
                    ),
                    status_code=status,
                    attempts=attempt,
                ),
                False,
            )
        except Exception as exc:
            log.exception("unexpected failure parsing a detector response")
            return (
                Outcome(
                    decision=Decision.DENY,
                    response=None,
                    failed_closed=True,
                    failure_reason=(
                        f"unexpected client failure: {type(exc).__name__}: {exc}"
                    ),
                    status_code=status,
                    attempts=attempt,
                ),
                False,
            )

        self._check_contract_version(parsed.contract_version)
        return (
            Outcome(
                decision=parsed.decision,
                response=parsed,
                status_code=status,
                attempts=attempt,
            ),
            False,
        )

    # ----------------------------------------------- GET /factors/{session_id}

    async def get_factors(self, session_id: str) -> FactorsOutcome:
        """Read the current factor breakdown. Never advances the state machine.

        Not retried: it is a diagnostic read, and a call must not wait on it.
        Like `push_event`, it cannot raise for a detector problem.
        """

        try:
            http_response = await self._client.get(
                f"/factors/{session_id}", timeout=self._timeout_s
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            return FactorsOutcome(
                decision=Decision.DENY,
                factors=None,
                failed_closed=True,
                failure_reason=f"detector unreachable: {type(exc).__name__}",
            )
        except httpx.HTTPError as exc:
            return FactorsOutcome(
                decision=Decision.DENY,
                factors=None,
                failed_closed=True,
                failure_reason=f"detector reply unusable: {type(exc).__name__}: {exc}",
            )
        except Exception as exc:
            log.exception("unexpected failure reading /factors")
            return FactorsOutcome(
                decision=Decision.DENY,
                factors=None,
                failed_closed=True,
                failure_reason=f"unexpected client failure: {type(exc).__name__}: {exc}",
            )

        status = http_response.status_code
        if status >= 400:
            return FactorsOutcome(
                decision=Decision.DENY,
                factors=None,
                failed_closed=True,
                failure_reason=(
                    f"detector returned {status}: {_safe_text(http_response)}"
                ),
                status_code=status,
                error_code=_error_code_of(http_response),
            )

        try:
            parsed = FactorsResponse.model_validate(http_response.json())
        except (ValidationError, ValueError, httpx.HTTPError) as exc:
            return FactorsOutcome(
                decision=Decision.DENY,
                factors=None,
                failed_closed=True,
                failure_reason=(
                    f"detector returned {status} with a body this client "
                    f"cannot read: {type(exc).__name__}"
                ),
                status_code=status,
            )
        except Exception as exc:
            log.exception("unexpected failure parsing /factors")
            return FactorsOutcome(
                decision=Decision.DENY,
                factors=None,
                failed_closed=True,
                failure_reason=f"unexpected client failure: {type(exc).__name__}: {exc}",
                status_code=status,
            )

        self._check_contract_version(parsed.contract_version)
        return FactorsOutcome(
            decision=parsed.decision, factors=parsed, status_code=status
        )

    # ---------------------------------------------------------- GET /healthz

    async def health(self) -> HealthResponse | None:
        """Liveness probe. None when the detector did not answer usefully.

        Catches a wrong URL, a detector that is not up yet and a contract
        version you did not expect. Call it at agent start-up.

        It does **not** check your token. `/healthz` carries no auth, by
        design, so a platform liveness probe needs no secret, which means a
        200 here says nothing at all about your credentials: a completely
        wrong token still gets `status: ok` and the first real event then
        denies with 401. Use `verify_credentials()` for that.
        """

        try:
            http_response = await self._client.get("/healthz", timeout=self._timeout_s)
            if http_response.status_code != 200:
                return None
            parsed = HealthResponse.model_validate(http_response.json())
        except Exception:  # noqa: BLE001
            # A probe that cannot answer is a None, never an exception.
            return None
        self._check_contract_version(parsed.contract_version)
        return parsed

    async def verify_credentials(self) -> bool | None:
        """Check the bearer token against an endpoint that actually requires it.

        True when the token is accepted, False when it is rejected, None when
        the detector could not be reached to ask.

        Side-effect free: it reads `/factors` for a session id that cannot
        exist. The auth dependency runs before the handler, so a good token
        gets `404 session_not_found` and a bad one gets `401 unauthorized`.

        Pair it with `health()` at start-up. Between them you catch a wrong
        URL, a detector that is down, and a bad token, which is the whole set
        of misconfigurations that would otherwise surface as a denied call in
        front of a real vendor.
        """

        probe = await self.get_factors(f"preflight-{uuid.uuid4().hex[:16]}")

        if probe.status_code is None:
            return None  # never reached the service
        if probe.status_code == 401:
            return False
        if probe.status_code == 404 or 200 <= probe.status_code < 300:
            return True
        return None  # a 5xx tells us nothing about the token

    # ------------------------------------------------------------- internals

    def _check_contract_version(self, seen: str) -> None:
        """Warn once if the service speaks a different contract version.

        Deliberately a warning and not a denial: the detector already answered,
        and refusing a well-formed decision over a version string would fail a
        call closed for a reason that is not a security signal. Pin versions in
        CI instead, where a mismatch is cheap to fix.
        """

        if seen == CONTRACT_VERSION or self._warned_about_version:
            return
        self._warned_about_version = True
        log.warning(
            "contract version mismatch: this client was built against %s, the "
            "detector reports %s. Re-check docs/CONTRACT.md.",
            CONTRACT_VERSION,
            seen,
        )

    @staticmethod
    def _fail_closed(
        *, reason: str, attempts: int, previous: Outcome | None
    ) -> Outcome:
        return Outcome(
            decision=Decision.DENY,
            response=None,
            failed_closed=True,
            failure_reason=reason,
            status_code=previous.status_code if previous else None,
            error_code=previous.error_code if previous else None,
            attempts=attempts,
        )


def _error_code_of(response: httpx.Response) -> ErrorCode | None:
    """Pull `error.code` out of a non-2xx body, if it is the shape we expect.

    Best effort and total. It is called from the branches whose entire job is
    to fail closed, so it must never raise: a body that will not decode at
    all, say a proxy with a Content-Encoding the bytes do not match, simply
    has no code to report.
    """

    try:
        return ErrorResponse.model_validate(response.json()).error.code
    except Exception:  # noqa: BLE001
        return None


def _safe_text(response: httpx.Response, limit: int = 200) -> str:
    """The response body as text for a log line. Never raises."""

    try:
        return response.text[:limit]
    except Exception:  # noqa: BLE001
        return "<body could not be decoded>"


# ----------------------------------------------------------------- call session


class CallSession:
    """One call. Owns the sequence numbers, the keys and the terminal guard.

    The sequence number must strictly increase per session and an idempotency
    key must not be reused with a different payload. Both are easy to get
    wrong by hand and both produce a 409 that looks like a detector fault, so
    this class owns them.

    Everything returns an `Outcome`. Nothing here can produce an ALLOW the
    detector did not give.
    """

    def __init__(
        self,
        client: DetectorClient,
        *,
        session_id: str | None = None,
        claimed_user_id: str | None = None,
        scenario: Scenario | None = None,
        metadata: dict[str, Any] | None = None,
        retry: RetryPolicy | None = None,
        key_prefix: str | None = None,
    ) -> None:
        self._client = client
        self.session_id = session_id or new_session_id()
        self.claimed_user_id = claimed_user_id
        self.scenario = scenario
        self.metadata = dict(metadata or {})
        # The session opts into bounded retry by default. See RetryPolicy for
        # why that is safe and why it cannot extend the latency budget.
        self._retry = retry if retry is not None else RetryPolicy()
        self._key_prefix = key_prefix or self.session_id

        self._sequence = 0
        self._closed = False
        self._ended = False
        self._last: Outcome | None = None

    # ------------------------------------------------------------- properties

    @property
    def sequence(self) -> int:
        """The last sequence number sent. 0 before the first event."""

        return self._sequence

    @property
    def closed(self) -> bool:
        """True once the session is final, by decision or by failing closed.

        Check this instead of catching `SessionClosed`.
        """

        return self._closed

    @property
    def last_outcome(self) -> Outcome | None:
        return self._last

    @property
    def decision(self) -> Decision:
        """The current decision. DENY until the detector says otherwise."""

        if self._last is None:
            return Decision.PENDING
        return self._last.decision

    # ------------------------------------------------------------- call steps

    async def start(self, *, metadata: dict[str, Any] | None = None) -> Outcome:
        """`session_start`. Send this first, when the call connects."""

        return await self._push(EventType.SESSION_START, metadata=metadata)

    async def claim_identity(
        self, claimed_user_id: str, *, metadata: dict[str, Any] | None = None
    ) -> Outcome:
        """`caller_claim`. The caller stated who they are.

        The claim sticks for the rest of the session and, in mock mode, also
        selects a scenario when no `X-Callback-Scenario` header is set.
        """

        self.claimed_user_id = claimed_user_id
        return await self._push(EventType.CALLER_CLAIM, metadata=metadata)

    async def submit_speech(
        self,
        audio: AudioRef | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Outcome:
        """`speech_segment`. A span of speech is ready to analyse.

        Audio is referenced, never inlined. Two segments are needed before the
        detector commits to a decision; until then you get `pending` and
        `continue_listening`.
        """

        return await self._push(
            EventType.SPEECH_SEGMENT, audio=audio, metadata=metadata
        )

    async def request_otp(self, *, metadata: dict[str, Any] | None = None) -> Outcome:
        """`otp_request`. Ask for a step-up challenge to be issued or reissued.

        Usually unnecessary: a `challenge` decision already carries a live
        `otp`. And it is refused on a hard deny, because `replay` and
        `ai_agent` cannot be rescued by a code.
        """

        return await self._push(EventType.OTP_REQUEST, metadata=metadata)

    async def submit_otp(
        self, code: str, *, metadata: dict[str, Any] | None = None
    ) -> Outcome:
        """`otp_submit`. The caller read a code back.

        Three attempts. A wrong code with attempts left comes back as another
        `challenge` with `otp.attempts_remaining` decremented; the third wrong
        code is a terminal deny.
        """

        return await self._push(EventType.OTP_SUBMIT, otp_code=code, metadata=metadata)

    async def end_call(self, *, metadata: dict[str, Any] | None = None) -> Outcome:
        """`session_end`. Always send this, including after a deny.

        A session that just stops without a decision is denied server-side.
        This is the one event a terminal session still accepts, and calling it
        twice is a no-op that costs no request.
        """

        if self._ended and self._last is not None:
            return self._last
        self._ended = True
        return await self._push(EventType.SESSION_END, metadata=metadata)

    # ---------------------------------------------------------------- helpers

    async def run_speech_until_decision(
        self,
        *,
        max_segments: int = 4,
        audio: AudioRef | None = None,
    ) -> Outcome:
        """Send speech segments until the detector stops saying `pending`.

        Bounded on purpose. Running out of segments is treated as a failure to
        verify, so it fails closed rather than looping forever on a call.
        """

        if max_segments < 1:
            raise ValueError("max_segments must be at least 1")

        outcome: Outcome | None = None
        for _ in range(max_segments):
            outcome = await self.submit_speech(audio)
            if not outcome.should_continue_listening:
                return outcome

        assert outcome is not None
        log.warning(
            "session %s still pending after %d speech segments; failing closed",
            self.session_id,
            max_segments,
        )
        self._closed = True
        failed = Outcome(
            decision=Decision.DENY,
            response=None,
            failed_closed=True,
            failure_reason=(
                f"no decision after {max_segments} speech segments; "
                "failing closed rather than waiting"
            ),
        )
        self._last = failed
        return failed

    async def factors(self) -> FactorsOutcome:
        """Read this session's factor breakdown. Safe on a closed session."""

        return await self._client.get_factors(self.session_id)

    # -------------------------------------------------------------- internals

    def idempotency_key_for(self, sequence: int) -> str:
        """Deterministic, so a retry of the same logical event reuses it.

        Sequence numbers already strictly increase per session, so this is
        unique per logical event without any extra bookkeeping, and it is
        greppable in the detector's logs.
        """

        return f"{self._key_prefix}:{sequence:06d}"

    def _guard(self, event: EventType) -> None:
        if not self._closed:
            return
        if event is EventType.SESSION_END:
            return
        raise SessionClosed(
            f"session {self.session_id} is already final "
            f"(decision={self.decision.value}, state={_state_name(self._last)}); "
            f"{event.value} cannot change it",
            outcome=self._last,
        )

    async def _push(
        self,
        event: EventType,
        *,
        otp_code: str | None = None,
        audio: AudioRef | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Outcome:
        self._guard(event)

        self._sequence += 1
        merged = {**self.metadata, **(metadata or {})}

        outcome = await self._client.push_event(
            session_id=self.session_id,
            event=event,
            sequence=self._sequence,
            claimed_user_id=self.claimed_user_id,
            otp_code=otp_code,
            audio=audio,
            scenario=self.scenario,
            idempotency_key=self.idempotency_key_for(self._sequence),
            metadata=merged,
            retry=self._retry,
        )

        self._last = outcome
        if outcome.is_final:
            # Covers both a terminal server state and any fail-closed DENY.
            self._closed = True
        return outcome


def _state_name(outcome: Outcome | None) -> str:
    if outcome is None:
        return "unknown"
    return outcome.state.value


def new_session_id(prefix: str = "sess") -> str:
    """A fresh session id in the contract's 8 to 128 character range."""

    return f"{prefix}-{uuid.uuid4().hex[:16]}"
