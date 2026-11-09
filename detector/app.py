"""FastAPI surface for the detector.

Endpoints:
    POST /events                 push a call event, get a decision
    GET  /factors/{session_id}   current factor breakdown for a session
    GET  /healthz                liveness, contract version, mock flag

The agent contract, callback_shared 1.1.0, is served under /v1 by
detector.v1. Its routes share the bearer check below and also need
X-Contract-Version.

Mounted here, defined elsewhere:
    POST /verify/bank-details    detector.verification, only when
                                 CALLBACK_BANK_VERIFY_ENABLED is on
    POST /enrol/voice            detector.enrolment
    POST /enrol/voice/status     detector.enrolment
    POST /enrol/voice/erase      detector.enrolment
    GET  /metrics                detector.review
    GET  /review, /review/api/*  detector.review

Auth: a bearer token on every request except /healthz, compared in constant
time against DETECTOR_TOKEN (or its old name, CALLBACK_API_TOKEN).
"""

from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any, TextIO

from callback_shared.contract import (
    CONTRACT_VERSION as WIRE_CONTRACT_VERSION,
)
from callback_shared.contract import (
    CONTRACT_VERSION_HEADER as WIRE_CONTRACT_VERSION_HEADER,
)
from callback_shared.contract import (
    ErrorBody as WireErrorBody,
)
from callback_shared.contract import (
    ErrorResponse as WireErrorResponse,
)
from fastapi import Depends, FastAPI, Header, Request, Response
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from detector import mock
from detector.config import Settings
from detector.enrolment import (
    VoicePrintStore,
    build_enrolment_router,
    store_from_settings,
)
from detector.errors import DetectorError, Unauthorized
from detector.observability import V1_ENDPOINTS, ObservabilityConfig
from detector.pipeline import AudioPipeline
from detector.review import mount as mount_review
from detector.state import DetectorEngine
from detector.v1 import build_v1_router
from detector.v1.clips import ClipSource
from detector.v1.service import V1Service
from detector.verification import build_verification_router
from shared.contract import (
    CONTRACT_VERSION,
    ErrorBody,
    ErrorCode,
    ErrorResponse,
    EventRequest,
    EventResponse,
    FactorsResponse,
    HealthResponse,
)


def bearer_matches(authorization: str | None, expected: str) -> bool:
    """Whether an Authorization header carries the bearer token."""

    if not authorization or not authorization.lower().startswith("bearer "):
        return False
    presented = authorization.split(" ", 1)[1].strip()
    # Compare bytes, not str: secrets.compare_digest raises TypeError on
    # non-ASCII str, and Starlette decodes headers as latin-1, so one
    # non-ASCII byte in the header turned a 401 into an unhandled 500.
    return secrets.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


def make_require_token(settings: Settings) -> Callable[..., None]:
    """The bearer check, as a FastAPI dependency. /events and /v1 share it."""

    def require_token(
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise Unauthorized("missing bearer token")
        if not bearer_matches(authorization, settings.api_token):
            raise Unauthorized("invalid bearer token")

    return require_token


class V1AuthGate:
    """Auth first for every /v1 request (ruling section 2.3).

    FastAPI routes a request and decodes its body before it runs a router's
    dependencies, so without this an unauthenticated caller got a 422 for a
    malformed body, a 404 for an unknown path or a 405 for a wrong method,
    which maps the routes and gives validation feedback. Pure ASGI, so the
    body is never read: a missing or wrong token is a 401 before any of that.
    The router keeps its own bearer dependency as a second line.
    """

    def __init__(self, app: Any, *, token: str) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") == "http":
            path = scope.get("path", "")
            if path == "/v1" or path.startswith("/v1/"):
                authorization = next(
                    (
                        value.decode("latin-1")
                        for name, value in scope.get("headers") or ()
                        if name.lower() == b"authorization"
                    ),
                    None,
                )
                if not bearer_matches(authorization, self.token):
                    response = _wire_error(
                        401, "unauthorized", "missing or invalid bearer token", None
                    )
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)


def _is_v1(request: Request) -> bool:
    return request.url.path.startswith("/v1/")


def _wire_error(
    status: int, code: str, message: str, detail: str | None
) -> JSONResponse:
    """A /v1 error: callback_shared's ErrorResponse, with the version header."""

    body = WireErrorResponse(
        contract_version=WIRE_CONTRACT_VERSION,
        error=WireErrorBody(code=code, message=message, detail=detail),
    )
    return JSONResponse(
        status_code=status,
        content=body.model_dump(mode="json"),
        headers={WIRE_CONTRACT_VERSION_HEADER: WIRE_CONTRACT_VERSION},
    )


def create_app(
    settings: Settings | None = None,
    *,
    observability: ObservabilityConfig | None = None,
    log_stream: TextIO | None = None,
    voiceprints: VoicePrintStore | None = None,
    v1_clock: Callable[[], float] | None = None,
    v1_clips: ClipSource | None = None,
) -> FastAPI:
    """Build the app. Tests call this directly with explicit settings.

    `observability` and `log_stream` are the same kind of seam as `settings`:
    the observability layer reads its switches from the environment by
    default, and a test that asserts on log lines needs to hand it a stream
    and a config it controls. Both default to the production behaviour, which
    is `ObservabilityConfig.from_env()` writing JSON lines to stdout.

    `voiceprints` is the same seam for the enrolment store, so a test can
    hand over a pre-populated one instead of enrolling over HTTP first.

    `v1_clock` is the /v1 registry clock, epoch seconds. It defaults to
    `time.time`; the /v1 tests hand in a fake one to step through session
    expiry, challenge timing and the cloned-voice spoof rise.

    `v1_clips` is where /v1 enrollment audio comes from. It is None in every
    deployment today, because the hidden listener that would supply it is
    stretch work, so a /v1 enrollment records consent and writes no print.
    See `detector/v1/clips.py`.
    """

    resolved = settings or Settings.from_env()
    engine = DetectorEngine(resolved)

    # The reference voices voice_match compares callers against. One store per
    # app, shared by the enrolment routes that write it and the pipeline that
    # reads it.
    prints = voiceprints if voiceprints is not None else store_from_settings(resolved)

    # Fetch, decode and analyse real call audio. Inert in mock mode and inert
    # with CALLBACK_AUDIO_ENABLED unset, which is the default, so in both of
    # those the route pays one boolean for it.
    audio = AudioPipeline(
        resolved,
        voiceprints=prints,
        # Only called for a span that is about to be analysed and whose event
        # did not repeat the claim, so most calls never pay for it.
        identity_of=engine.claimed_identity,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # The pipeline holds a keep-alive httpx client. Without this the
        # connections leak on reload and every test that builds an app warns.
        try:
            yield
        finally:
            await audio.aclose()
            await v1.aclose()

    app = FastAPI(
        lifespan=lifespan,
        title="Callback detector",
        version=CONTRACT_VERSION,
        description=(
            "Caller authenticity detection for the Callback voice agent. "
            "Mock mode runs five scripted scenarios through the real state "
            "machine."
        ),
        # The interactive docs and the OpenAPI document are unauthenticated by
        # default, which publishes the whole contract to anyone who can reach
        # the service. It is shared deliberately via docs/CONTRACT.md and
        # schemas/contract.schema.json instead.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = resolved
    app.state.engine = engine
    app.state.audio = audio
    app.state.voiceprints = prints

    # Bank-detail verification. Mounted after app.state.engine is set, because
    # a request that omits voice_decision has the detector read its own verdict
    # off the session. The endpoint exists to receive a full account number,
    # so it is not mounted at all unless CALLBACK_BANK_VERIFY_ENABLED is on.
    if resolved.bank_verify_enabled:
        app.include_router(build_verification_router(resolved))

    # Enrolment. Shares the pipeline's httpx client rather than opening a
    # second one, because it fetches under the same policy with a longer
    # timeout, and a timeout is per request in detector.audio.
    app.include_router(
        build_enrolment_router(resolved, store=prints, client_of=audio.http)
    )

    require_token = make_require_token(resolved)

    # The agent contract, callback_shared 1.1.0. After the enrolment router so
    # /v1 enrollments write into the same voice print store, and before the
    # review mount so the observability middleware wraps it.
    v1_router = build_v1_router(
        resolved,
        voiceprints=prints,
        clock=v1_clock or time.time,
        require_token=require_token,
        clips=v1_clips,
        # So a /v1 turn can point at its own audio and enrol a voice from it
        # (contract 1.2.0). The same policy and the same connection pool the
        # call path uses, so there is one allowlist and one set of limits.
        audio_policy=audio.policy,
        client_of=audio.http,
    )
    v1: V1Service = v1_router.service  # type: ignore[attr-defined]
    app.include_router(v1_router)
    app.state.v1 = v1
    # Auth before routing and body parsing for /v1. Added before the review
    # mount, so the observability middleware still wraps and logs each 401.
    app.add_middleware(V1AuthGate, token=resolved.api_token)

    if resolved.api_token_generated:
        # Loud on purpose: the service is up but can authenticate nobody.
        print(
            "WARNING: DETECTOR_TOKEN was not set (nor its old name "
            "CALLBACK_API_TOKEN), so a random token was generated and every "
            "request will be rejected with 401. Set DETECTOR_TOKEN (Modal "
            "secret callback-detector) and restart.",
            flush=True,
        )

    # ------------------------------------------------------------- handlers

    @app.exception_handler(DetectorError)
    async def _detector_error(request: Request, exc: DetectorError) -> JSONResponse:
        if _is_v1(request):
            return _wire_error(exc.http_status, str(exc.code), exc.message, exc.detail)
        body = ErrorResponse(
            error=ErrorBody(code=exc.code, message=exc.message, detail=exc.detail)
        )
        return JSONResponse(
            status_code=exc.http_status, content=body.model_dump(mode="json")
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Never echo the rejected input. Pydantic attaches it to every error
        # entry, so rendering errors() verbatim hands the caller back whatever
        # they sent and writes it into any proxy log on the way out. A rejected
        # otp_code, or a spoken account number on the verification endpoint,
        # must not make that round trip. loc and msg say what is wrong without
        # repeating the value, and ctx is dropped too because it can carry
        # context values rather than just the constraint.
        def _detail_line(error: dict[str, object]) -> str:
            location = ".".join(str(part) for part in error.get("loc", ())) or "body"
            message = error.get("msg", "invalid")
            return f"{location}: {message}"

        detail = "; ".join(_detail_line(error) for error in exc.errors()[:3])
        if _is_v1(request):
            return _wire_error(
                422,
                "validation_error",
                "request failed contract validation",
                detail or None,
            )
        body = ErrorResponse(
            error=ErrorBody(
                code=ErrorCode.VALIDATION_ERROR,
                message="request body failed contract validation",
                detail=detail or None,
            )
        )
        return JSONResponse(status_code=422, content=body.model_dump(mode="json"))

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> Response:
        # Routing errors: an unknown /v1 path or a wrong method. Under /v1 they
        # get the contract's error body too; everywhere else, FastAPI's own.
        if _is_v1(request):
            code = {404: "not_found", 405: "method_not_allowed"}.get(
                exc.status_code, f"http_{exc.status_code}"
            )
            return _wire_error(exc.status_code, code, "no such /v1 route", None)
        return await http_exception_handler(request, exc)

    # ------------------------------------------------------------ endpoints

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return HealthResponse(mock_mode=resolved.mock_mode)

    @app.post(
        "/events",
        response_model=EventResponse,
        dependencies=[Depends(require_token)],
    )
    async def post_events(
        payload: EventRequest,
        response: Response,
        x_callback_scenario: Annotated[str | None, Header()] = None,
        idempotency_key: Annotated[str | None, Header()] = None,
    ) -> EventResponse:
        started = time.perf_counter()

        # Before the state machine, because this is the only layer that may
        # block on I/O. A span that cannot be fetched or decoded is not an
        # error: it yields no factors and a note saying why, and the scoring
        # layer refuses to call an unscored caller safe.
        span = await audio.for_event(payload)

        result = engine.handle_event(
            payload,
            scenario_header=x_callback_scenario,
            idempotency_key=idempotency_key,
            observed_factors=span.factors,
            audio_note=span.note,
        )

        if resolved.mock_mode and not result.idempotent_replay:
            await asyncio.sleep(
                mock.jitter_seconds(
                    resolved.mock_jitter_min_ms, resolved.mock_jitter_max_ms
                )
            )

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        response.headers["X-Callback-Contract-Version"] = CONTRACT_VERSION
        return result.model_copy(update={"latency_ms": elapsed_ms})

    @app.get(
        "/factors/{session_id}",
        response_model=FactorsResponse,
        dependencies=[Depends(require_token)],
    )
    async def get_factors(session_id: str) -> FactorsResponse:
        return engine.factors_for(session_id)

    # ------------------------------------------------- observability + review
    #
    # Last, because it installs ASGI middleware that wraps every route above
    # and it registers the 401 handler the review routes share with them.
    # Returns the Observability instance: the journal a reviewer reads and the
    # metric registry a scraper reads both hang off it.
    app.state.observability = mount_review(
        app,
        engine=engine,
        settings=resolved,
        config=observability,
        stream=log_stream,
        v1=v1,
    )
    # /v1 logs through the same structured logger and redactor, and its route
    # templates become known endpoints, so ids never become metric labels.
    v1.bind_logger(app.state.observability.logger)
    app.state.observability.add_known_endpoints(V1_ENDPOINTS)

    return app


app = create_app()
