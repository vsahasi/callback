"""FastAPI surface for the detector.

Endpoints:
    POST /events                 push a call event, get a decision
    GET  /factors/{session_id}   current factor breakdown for a session
    GET  /healthz                liveness, contract version, mock flag

Mounted here, defined elsewhere:
    POST /verify/bank-details    detector.verification
    GET  /metrics                detector.review
    GET  /review, /review/api/*  detector.review

Auth: a bearer token on every request except /healthz, compared in constant
time against CALLBACK_API_TOKEN.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from typing import Annotated, TextIO

from fastapi import Depends, FastAPI, Header, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from detector import mock
from detector.config import Settings
from detector.errors import DetectorError, Unauthorized
from detector.observability import ObservabilityConfig
from detector.review import mount as mount_review
from detector.state import DetectorEngine
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


def create_app(
    settings: Settings | None = None,
    *,
    observability: ObservabilityConfig | None = None,
    log_stream: TextIO | None = None,
) -> FastAPI:
    """Build the app. Tests call this directly with explicit settings.

    `observability` and `log_stream` are the same kind of seam as `settings`:
    the observability layer reads its switches from the environment by
    default, and a test that asserts on log lines needs to hand it a stream
    and a config it controls. Both default to the production behaviour, which
    is `ObservabilityConfig.from_env()` writing JSON lines to stdout.
    """

    resolved = settings or Settings.from_env()
    engine = DetectorEngine(resolved)

    app = FastAPI(
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

    # Bank-detail verification. Mounted after app.state.engine is set, because
    # a request that omits voice_decision has the detector read its own verdict
    # off the session.
    app.include_router(build_verification_router(resolved))

    if resolved.api_token_generated:
        # Loud on purpose: the service is up but can authenticate nobody.
        print(
            "WARNING: CALLBACK_API_TOKEN was not set, so a random token was "
            "generated and every request will be rejected with 401. Set the "
            "token (Modal secret callback-detector) and restart.",
            flush=True,
        )

    def require_token(
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        expected = resolved.api_token
        if not authorization or not authorization.lower().startswith("bearer "):
            raise Unauthorized("missing bearer token")
        presented = authorization.split(" ", 1)[1].strip()
        # Compare bytes, not str: secrets.compare_digest raises TypeError on
        # non-ASCII str, and Starlette decodes headers as latin-1, so one
        # non-ASCII byte in the header turned a 401 into an unhandled 500.
        if not secrets.compare_digest(
            presented.encode("utf-8"), expected.encode("utf-8")
        ):
            raise Unauthorized("invalid bearer token")

    # ------------------------------------------------------------- handlers

    @app.exception_handler(DetectorError)
    async def _detector_error(_: Request, exc: DetectorError) -> JSONResponse:
        body = ErrorResponse(
            error=ErrorBody(code=exc.code, message=exc.message, detail=exc.detail)
        )
        return JSONResponse(
            status_code=exc.http_status, content=body.model_dump(mode="json")
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        _: Request, exc: RequestValidationError
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
        body = ErrorResponse(
            error=ErrorBody(
                code=ErrorCode.VALIDATION_ERROR,
                message="request body failed contract validation",
                detail=detail or None,
            )
        )
        return JSONResponse(status_code=422, content=body.model_dump(mode="json"))

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

        result = engine.handle_event(
            payload,
            scenario_header=x_callback_scenario,
            idempotency_key=idempotency_key,
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
    )

    return app


app = create_app()
