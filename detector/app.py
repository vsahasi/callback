"""FastAPI surface for the detector.

Endpoints:
    POST /events                 push a call event, get a decision
    GET  /factors/{session_id}   current factor breakdown for a session
    GET  /healthz                liveness, contract version, mock flag

Auth: a bearer token on every request except /healthz, compared in constant
time against CALLBACK_API_TOKEN.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from typing import Annotated

from fastapi import Depends, FastAPI, Header, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from detector import mock
from detector.config import Settings
from detector.errors import DetectorError, Unauthorized
from detector.state import DetectorEngine
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


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. Tests call this directly with explicit settings."""

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
        body = ErrorResponse(
            error=ErrorBody(
                code=ErrorCode.VALIDATION_ERROR,
                message="request body failed contract validation",
                detail=str(exc.errors()[:3]),
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

    return app


app = create_app()
