"""The /v1 router: callback_shared 1.1.0, the agent contract.

Every route needs the bearer token and `X-Contract-Version` (auth first, then
the version), and every response carries `X-Contract-Version: 1.1.0`, errors
included. A non-2xx body is always callback_shared's `ErrorResponse`.

The routes are thin. All /v1 state lives in one `V1Service`
(detector/v1/service.py), which create_app keeps on `app.state.v1`. Nothing
here goes through the /events engine, the internal scoring or the /events
mock scripts.

Mock jitter is slept on every route except /turns, /events and the challenge
poll (ruling F4), so background turn posts never delay a challenge result.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Annotated, Any

import httpx
from callback_shared.contract import (
    CONTRACT_VERSION as WIRE_CONTRACT_VERSION,
)
from callback_shared.contract import (
    CONTRACT_VERSION_HEADER,
    SCENARIO_HEADER,
    AuthorizeRequest,
    CallerUpdate,
    ChallengeCreate,
    ChallengeIssued,
    ChallengeStatus,
    EnrollmentResult,
    EnrollmentStart,
    EnrollmentStarted,
    FactorReport,
    OtpCheck,
    OtpCheckResult,
    OtpSent,
    SessionCreate,
    SessionCreated,
    SpeechEvent,
    Turn,
    UserOnFile,
    UserSummary,
)
from callback_shared.contract import (
    Decision as WireDecision,
)
from fastapi import APIRouter, Depends, Header, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException as StarletteHTTPException

from detector.audio import AudioPolicy
from detector.config import Settings
from detector.enrolment import VoicePrintStore
from detector.errors import DetectorError
from detector.v1.clips import ClipSource
from detector.v1.errors import (
    ContractVersionMismatch,
    InternalError,
    UserNotFound,
    V1ValidationError,
)
from detector.v1.service import V1Service
from detector.v1.sessions import MAX_TURN_TEXT_CHARS, MAX_TURN_WORDS, MAX_WORD_CHARS
from detector.v1.users import UserDirectory

WIRE_MAJOR = WIRE_CONTRACT_VERSION.split(".", 1)[0]


def _no_content() -> Response:
    return Response(status_code=204)


def build_v1_router(
    settings: Settings,
    *,
    voiceprints: VoicePrintStore,
    clock: Callable[[], float] | None = None,
    require_token: Callable[..., None],
    directory: UserDirectory | None = None,
    clips: ClipSource | None = None,
    audio_policy: AudioPolicy | None = None,
    client_of: Callable[[], httpx.AsyncClient] | None = None,
) -> APIRouter:
    """The /v1 routes over a fresh `V1Service`, which is `router.service`.

    `clips` is where enrollment audio comes from. None, which is every
    deployment today, means nothing is heard and no print is written; see
    `detector/v1/clips.py` for what the listener has to provide.
    """

    service = V1Service(
        settings,
        voiceprints=voiceprints,
        clock=clock,
        directory=directory,
        clips=clips,
        audio_policy=audio_policy,
        client_of=client_of,
    )

    class V1Route(APIRoute):
        """Stamps the version header on every answer and keeps 500s in shape.

        An exception that is not a contract error becomes `internal_error`
        here, inside the route, so the app's DetectorError handler renders it
        as an `ErrorResponse` like every other /v1 failure. The authorize
        route never gets this far: it turns its own failures, including one
        while recording the answer, into the 200 internal_timeout deny.
        """

        def get_route_handler(
            self,
        ) -> Callable[[Request], Coroutine[Any, Any, Response]]:
            handler = super().get_route_handler()
            template = self.path

            async def v1_handler(request: Request) -> Response:
                try:
                    response = await handler(request)
                except (DetectorError, RequestValidationError, StarletteHTTPException):
                    raise
                except Exception as exc:
                    service.log(
                        "v1_internal_error",
                        level="error",
                        endpoint=template,
                        exception=type(exc).__name__,
                    )
                    raise InternalError() from exc
                response.headers[CONTRACT_VERSION_HEADER] = WIRE_CONTRACT_VERSION
                return response

            return v1_handler

    def require_contract_version(
        response: Response,
        x_contract_version: Annotated[
            str | None, Header(alias=CONTRACT_VERSION_HEADER)
        ] = None,
    ) -> None:
        presented = (x_contract_version or "").strip()
        if not presented or presented.split(".", 1)[0] != WIRE_MAJOR:
            raise ContractVersionMismatch(
                detail=f"this detector serves {WIRE_CONTRACT_VERSION}"
            )
        if presented != WIRE_CONTRACT_VERSION:
            service.warn_version_once(presented)
        response.headers[CONTRACT_VERSION_HEADER] = WIRE_CONTRACT_VERSION

    router = APIRouter(
        prefix="/v1",
        dependencies=[Depends(require_token), Depends(require_contract_version)],
        route_class=V1Route,
    )

    # ------------------------------------------------------- 1, 2: users

    @router.get("/users", response_model=list[UserSummary])
    async def search_users(name: Annotated[str, Query()]) -> list[UserSummary]:
        # The name is a caller's words. It is never logged, and uvicorn runs
        # with --no-access-log so the query string is not either.
        await service.jitter()
        return service.directory.search(name)

    @router.get("/users/{user_id}", response_model=UserOnFile)
    async def get_user(user_id: str) -> UserOnFile:
        await service.jitter()
        user = service.directory.get(user_id)
        if user is None:
            raise UserNotFound()
        return user

    # ---------------------------------------------------- 3-6: sessions

    @router.post("/sessions", response_model=SessionCreated)
    async def create_session(
        body: SessionCreate,
        x_callback_scenario: Annotated[
            str | None, Header(alias=SCENARIO_HEADER)
        ] = None,
    ) -> SessionCreated:
        # The only route that reads the scenario header.
        await service.jitter()
        return service.create_session(body, x_callback_scenario)

    @router.post("/sessions/{session_id}/caller", status_code=204)
    async def update_caller(session_id: str, body: CallerUpdate) -> Response:
        await service.jitter()
        service.update_caller(session_id, body)
        return _no_content()

    @router.post("/sessions/{session_id}/turns", status_code=204)
    async def post_turn(session_id: str, body: Turn) -> Response:
        # Bounded here, before anything is stored. The detail names the limit,
        # never the text.
        if len(body.text) > MAX_TURN_TEXT_CHARS:
            raise V1ValidationError(
                detail=f"body.text: at most {MAX_TURN_TEXT_CHARS} characters"
            )
        if len(body.words) > MAX_TURN_WORDS:
            raise V1ValidationError(
                detail=f"body.words: at most {MAX_TURN_WORDS} entries"
            )
        if any(len(word.w) > MAX_WORD_CHARS for word in body.words):
            raise V1ValidationError(
                detail=f"body.words.w: at most {MAX_WORD_CHARS} characters"
            )
        # No jitter. Awaited because a turn may carry an enrollment clip, and
        # the prompt it answers has to be settled before this returns.
        await service.add_turn(session_id, body)
        return _no_content()

    @router.post("/sessions/{session_id}/events", status_code=204)
    async def post_event(session_id: str, body: SpeechEvent) -> Response:
        service.add_event(session_id, body)  # no jitter
        return _no_content()

    # ------------------------------------------------- 7, 8: challenges

    @router.post("/sessions/{session_id}/challenges", response_model=ChallengeIssued)
    async def create_challenge(
        session_id: str, body: ChallengeCreate
    ) -> ChallengeIssued:
        await service.jitter()
        return service.create_challenge(session_id, body.kind)

    @router.get(
        "/sessions/{session_id}/challenges/{challenge_id}",
        response_model=ChallengeStatus,
    )
    async def get_challenge(session_id: str, challenge_id: str) -> ChallengeStatus:
        # No jitter: the agent polls this every 0.25 s with a 1.0 s budget.
        return service.challenge_status(session_id, challenge_id)

    # ------------------------------------------------------- 9, 10: OTP

    @router.post("/sessions/{session_id}/otp/send", response_model=OtpSent)
    async def otp_send(session_id: str) -> OtpSent:
        # Any body is ignored: the code goes to the number on file, never to
        # one the caller gave, so there is nothing to send here.
        await service.jitter()
        return await service.otp_send(session_id)

    @router.post("/sessions/{session_id}/otp/check", response_model=OtpCheckResult)
    async def otp_check(session_id: str, body: OtpCheck) -> OtpCheckResult:
        await service.jitter()
        return await service.otp_check(session_id, body.code)

    # --------------------------------------------------- 11: authorize

    @router.post("/sessions/{session_id}/authorize", response_model=WireDecision)
    async def authorize(session_id: str, body: AuthorizeRequest) -> WireDecision:
        # The jitter is slept inside, under the 2.2 s deadline, and never on an
        # idempotent replay. Past the deadline, or on any internal failure,
        # this answers 200 deny internal_timeout rather than an error.
        return await service.authorize(session_id, body)

    # ------------------------------------------------ 12, 13: factors, end

    @router.post("/sessions/{session_id}/factors", status_code=204)
    async def report_factor(session_id: str, body: FactorReport) -> Response:
        await service.jitter()
        service.report_factor(session_id, body)
        return _no_content()

    @router.post("/sessions/{session_id}/end", status_code=204)
    async def end_session(session_id: str) -> Response:
        await service.jitter()
        service.end_session(session_id)
        return _no_content()

    # ------------------------------------------------ 14-16: enrollments

    @router.post("/enrollments", response_model=EnrollmentStarted)
    async def start_enrollment(body: EnrollmentStart) -> EnrollmentStarted:
        await service.jitter()
        return service.start_enrollment(body)

    @router.post(
        "/enrollments/{enrollment_id}/complete", response_model=EnrollmentResult
    )
    async def complete_enrollment(enrollment_id: str) -> EnrollmentResult:
        await service.jitter()
        return service.complete_enrollment(enrollment_id)

    @router.delete("/enrollments/{user_id}", status_code=204)
    async def delete_enrollment(user_id: str) -> Response:
        # Operator use; the agent never calls it. Idempotent.
        await service.jitter()
        service.delete_enrollment(user_id)
        return _no_content()

    router.service = service  # type: ignore[attr-defined]
    return router
