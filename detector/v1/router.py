"""The /v1 router: callback_shared 1.1.0, the agent contract.

Every route needs the bearer token and `X-Contract-Version` (auth first, then
the version), and every response carries `X-Contract-Version: 1.1.0`, errors
included. A non-2xx body is always callback_shared's `ErrorResponse`.

The router holds all /v1 state through one `V1Service`, which create_app keeps
on `app.state.v1`. Nothing here goes through the /events engine, the internal
scoring or the /events mock scripts.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine
from typing import Annotated, Any

from callback_shared.contract import (
    CONTRACT_VERSION as WIRE_CONTRACT_VERSION,
)
from callback_shared.contract import (
    CONTRACT_VERSION_HEADER,
    UserOnFile,
    UserSummary,
)
from fastapi import APIRouter, Depends, Header, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException as StarletteHTTPException

from detector.config import Settings
from detector.enrolment import VoicePrintStore
from detector.errors import DetectorError
from detector.mock import jitter_seconds
from detector.observability import StructuredLogger
from detector.v1.errors import ContractVersionMismatch, InternalError, UserNotFound
from detector.v1.users import UserDirectory

WIRE_MAJOR = WIRE_CONTRACT_VERSION.split(".", 1)[0]

# Distinct version strings warned about, at most. Auth runs first, so only a
# caller holding the token can add one, but the set is bounded anyway.
_MAX_WARNED_VERSIONS = 32


class V1Service:
    """Every piece of /v1 state, in one process, in memory."""

    def __init__(
        self,
        settings: Settings,
        *,
        voiceprints: VoicePrintStore,
        clock: Callable[[], float] | None = None,
        directory: UserDirectory | None = None,
    ) -> None:
        self.settings = settings
        self.clock: Callable[[], float] = clock or time.time
        self.voiceprints = voiceprints
        self.directory = (
            directory
            if directory is not None
            else UserDirectory.from_settings(settings)
        )
        self.logger: StructuredLogger | None = None
        self._warned_versions: set[str] = set()

    # ------------------------------------------------------------ plumbing

    def bind_logger(self, logger: StructuredLogger) -> None:
        """create_app hands over the observability logger once it exists."""

        self.logger = logger

    def log(self, event: str, *, level: str = "info", **fields: Any) -> None:
        # Callers pass ids, codes, counts and tags only. The structured
        # logger's redactor is a second line, not the plan.
        if self.logger is not None:
            self.logger.log(event, level=level, **fields)

    async def jitter(self) -> None:
        """Mock-mode latency. Every route but turns, events and the poll."""

        if not self.settings.mock_mode:
            return
        delay = jitter_seconds(
            self.settings.mock_jitter_min_ms, self.settings.mock_jitter_max_ms
        )
        if delay > 0:
            await asyncio.sleep(delay)

    def warn_version_once(self, version: str) -> None:
        if version in self._warned_versions:
            return
        if len(self._warned_versions) >= _MAX_WARNED_VERSIONS:
            return
        self._warned_versions.add(version)
        self.log(
            "v1_contract_version_differs",
            level="warning",
            presented=version[:32],
            served=WIRE_CONTRACT_VERSION,
        )

    async def aclose(self) -> None:
        return None


def build_v1_router(
    settings: Settings,
    *,
    voiceprints: VoicePrintStore,
    clock: Callable[[], float] | None = None,
    require_token: Callable[..., None],
    directory: UserDirectory | None = None,
) -> APIRouter:
    """The /v1 routes over a fresh `V1Service`, which is `router.service`."""

    service = V1Service(
        settings, voiceprints=voiceprints, clock=clock, directory=directory
    )

    class V1Route(APIRoute):
        """Stamps the version header on every answer and keeps 500s in shape.

        An exception that is not a contract error becomes `internal_error`
        here, inside the route, so the app's DetectorError handler renders it
        as an `ErrorResponse` like every other /v1 failure. The authorize
        route never gets this far: it turns its own failures into a deny.
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

    router.service = service  # type: ignore[attr-defined]
    return router
