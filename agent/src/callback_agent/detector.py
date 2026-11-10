"""The agent's HTTP client for the detection service.

authorize fails closed: any failure, including an answer that comes too late, returns a deny
Decision instead of raising, so a broken detector can never read as a green light. Every other call
raises DetectorError and leaves the choice to the caller. The endpoints are in the Contract tab of
docs/spec/2026-11-04-notion-snapshot.md, plus the paths added by D3 to D5 in the plan.
"""

import asyncio
import logging
from functools import cache
from typing import Any

import httpx
from callback_shared.contract import (
    AUTHORIZE_TIMEOUT_S,
    CONTRACT_VERSION,
    CONTRACT_VERSION_HEADER,
    SCENARIO_HEADER,
    Action,
    AuthorizeRequest,
    CallerUpdate,
    ChallengeCreate,
    ChallengeIssued,
    ChallengeKind,
    ChallengeStatus,
    Decision,
    DecisionKind,
    EnrollmentResult,
    EnrollmentStart,
    EnrollmentStarted,
    FactorReport,
    OtpCheck,
    OtpCheckResult,
    OtpSent,
    Scenario,
    SessionCreate,
    SessionCreated,
    SpeechEvent,
    Turn,
    UserOnFile,
    UserSummary,
)
from pydantic import BaseModel, TypeAdapter, ValidationError

from callback_agent.audio_segments import AudioSegment, speech_segment_body

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_S = 5.0  # OTP calls wait on Twilio
CHALLENGE_POLL_TIMEOUT_S = 1.0  # get_challenge is polled several times a second: no long waits


class DetectorError(Exception):
    """A call to the detection service failed: no answer, a non-2xx status or a bad body.

    The message names the call and the failure, never the response body: the service's validation
    errors echo their input, and the input can be a one-time code.
    """


def fail_closed_decision(session_id: str, action: Action, reason: str) -> Decision:
    """The decision to act on when the detector can't give one: deny."""
    return Decision(
        session_id=session_id,
        action=action,
        decision=DecisionKind.DENY,
        reasons=[f"detector_unavailable: {reason}"],
    )


@cache
def _adapter(tp: Any) -> TypeAdapter[Any]:
    return TypeAdapter(tp)


def _problems(error: ValidationError) -> str:
    """The fields that failed and why, without their values: a body can hold a phone number."""
    return ", ".join(
        f"{'.'.join(map(str, e['loc'])) or 'body'}: {e['type']}"
        for e in error.errors(include_input=False)
    )


class DetectorClient:
    """Calls the detection service. One instance serves a whole call, from any task.

    `scenario` picks a scripted call when the service runs in mock mode. `authorize_deadline_s`
    bounds the whole of an authorize request. `http` replaces the client the instance would create;
    aclose() closes it either way.
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        scenario: Scenario | None = None,
        authorize_deadline_s: float = AUTHORIZE_TIMEOUT_S,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        # Sent with each request rather than set on the client, so an `http` passed in works as is.
        self._headers = {
            "Authorization": f"Bearer {token}",
            CONTRACT_VERSION_HEADER: CONTRACT_VERSION,
        }
        self._scenario = scenario
        self._authorize_deadline_s = authorize_deadline_s
        self._http = http if http is not None else httpx.AsyncClient()

    async def create_session(self, req: SessionCreate) -> SessionCreated:
        # Mock mode only: the service plays the scripted call named in the header.
        headers = {SCENARIO_HEADER: self._scenario.value} if self._scenario is not None else None
        return await self._fetch(SessionCreated, "POST", "/v1/sessions", body=req, headers=headers)

    async def update_caller(self, session_id: str, req: CallerUpdate) -> None:
        await self._send("POST", f"/v1/sessions/{session_id}/caller", body=req)

    async def post_turn(self, session_id: str, turn: Turn) -> None:
        await self._send("POST", f"/v1/sessions/{session_id}/turns", body=turn)

    async def post_event(self, session_id: str, event: SpeechEvent) -> None:
        await self._send("POST", f"/v1/sessions/{session_id}/events", body=event)

    async def post_speech_segment(
        self,
        session_id: str,
        segment: AudioSegment,
        *,
        sequence: int,
        claimed_user_id: str | None,
    ) -> None:
        """Give the detector one segment of the caller's speech to measure, inline.

        This is the detector's own /events intake, not a /v1 route: the only way audio reaches it
        without a hidden listener or a bucket. The session id is the /v1 session's, so the detector
        can line the two up; `sequence` counts up per session, as the intake requires.
        """
        body = speech_segment_body(
            session_id, segment, sequence=sequence, claimed_user_id=claimed_user_id
        )
        await self._send("POST", "/events", body=body)

    async def create_challenge(self, session_id: str, kind: ChallengeKind) -> ChallengeIssued:
        return await self._fetch(
            ChallengeIssued,
            "POST",
            f"/v1/sessions/{session_id}/challenges",
            body=ChallengeCreate(kind=kind),
        )

    async def get_challenge(self, session_id: str, challenge_id: str) -> ChallengeStatus:
        return await self._fetch(
            ChallengeStatus,
            "GET",
            f"/v1/sessions/{session_id}/challenges/{challenge_id}",
            timeout=CHALLENGE_POLL_TIMEOUT_S,
        )

    async def send_otp(self, session_id: str) -> OtpSent:
        # No phone number to pass: the service texts the number on file, never one the caller gave.
        return await self._fetch(OtpSent, "POST", f"/v1/sessions/{session_id}/otp/send", body={})

    async def check_otp(self, session_id: str, code: str) -> OtpCheckResult:
        return await self._fetch(
            OtpCheckResult,
            "POST",
            f"/v1/sessions/{session_id}/otp/check",
            body=OtpCheck(code=code),
        )

    async def authorize(self, session_id: str, req: AuthorizeRequest) -> Decision:
        """Ask the detector whether the action may go ahead.

        Never raises. An error of any kind, a non-2xx status, a body that is not a Decision, a
        Decision for another session or action, and an answer later than authorize_deadline_s all
        come back as a deny. Cancellation is the one thing that gets through, so a call that ends
        mid-request is not turned into a decision.
        """
        try:
            # httpx timeouts apply per phase (connect, write, read), so only this bounds the total.
            async with asyncio.timeout(self._authorize_deadline_s):
                decision = await self._fetch(
                    Decision,
                    "POST",
                    f"/v1/sessions/{session_id}/authorize",
                    body=req,
                    timeout=self._authorize_deadline_s,
                )
        except TimeoutError:
            reason = f"no answer within {self._authorize_deadline_s:g}s"
        except DetectorError as exc:
            reason = str(exc)
        except Exception as exc:
            logger.exception("authorize failed in an unexpected way, so it is denied")
            reason = type(exc).__name__
        else:
            # An answer for another session or action must never pass, whatever it says.
            if decision.session_id != session_id:
                reason = "decision mismatch (session_id)"
            elif decision.action != req.action:
                reason = "decision mismatch (action)"
            else:
                return decision
        return fail_closed_decision(session_id, req.action, reason)

    async def report_factor(self, session_id: str, report: FactorReport) -> None:
        await self._send("POST", f"/v1/sessions/{session_id}/factors", body=report)

    async def end_session(self, session_id: str) -> None:
        await self._send("POST", f"/v1/sessions/{session_id}/end")

    async def find_users(self, name: str) -> list[UserSummary]:
        return await self._fetch(list[UserSummary], "GET", "/v1/users", params={"name": name})

    async def get_user(self, user_id: str) -> UserOnFile:
        return await self._fetch(UserOnFile, "GET", f"/v1/users/{user_id}")

    async def start_enrollment(self, req: EnrollmentStart) -> EnrollmentStarted:
        return await self._fetch(EnrollmentStarted, "POST", "/v1/enrollments", body=req)

    async def complete_enrollment(self, enrollment_id: str) -> EnrollmentResult:
        return await self._fetch(
            EnrollmentResult, "POST", f"/v1/enrollments/{enrollment_id}/complete"
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _send(
        self,
        method: str,
        path: str,
        *,
        body: BaseModel | dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = REQUEST_TIMEOUT_S,
    ) -> httpx.Response:
        """Send one request. Raises DetectorError unless the service answers with a 2xx."""
        json_body = body.model_dump(mode="json") if isinstance(body, BaseModel) else body
        try:
            response = await self._http.request(
                method,
                self._base_url + path,
                json=json_body,
                params=params,
                headers={**self._headers, **(headers or {})},
                timeout=timeout,
            )
        except httpx.HTTPError as exc:
            raise DetectorError(f"{method} {path}: {exc!r}") from exc
        if not response.is_success:
            raise DetectorError(f"{method} {path}: HTTP {response.status_code}")
        return response

    async def _fetch[T](self, tp: type[T], method: str, path: str, **kwargs: Any) -> T:
        """Send one request and parse the answer as `tp`."""
        response = await self._send(method, path, **kwargs)
        try:
            return _adapter(tp).validate_json(response.content)
        except ValidationError as exc:
            # from None, so the rejected body stays out of tracebacks and logs
            raise DetectorError(
                f"{method} {path}: the answer breaks the contract ({_problems(exc)})"
            ) from None
