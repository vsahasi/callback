"""Fixtures for the /v1 agent contract (callback_shared 1.2.0).

The app runs in process over ASGI with jitter off, a fake registry clock, a
captured log stream, and the users_on_file example from the final ruling
(section 2.9) as inline JSON.
"""

from __future__ import annotations

import dataclasses
import io
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from callback_shared.contract import CONTRACT_VERSION, CONTRACT_VERSION_HEADER
from fastapi import FastAPI

from detector.app import create_app
from detector.config import Settings
from detector.observability import ObservabilityConfig

V1_TOKEN = "v1-test-token-not-a-real-secret"

EXAMPLE_USERS: dict[str, Any] = {
    "users": [
        {
            "user_id": "u_eric",
            "display_name": "Eric Catalano",
            # How speech-to-text may spell the name (integration review M9).
            "aliases": [
                "Erik Catalano",
                "Eric Catalono",
                "Erik Catalono",
                "Eric Catalina",
                "Eric Cattalano",
            ],
            "role": "CFO",
            "phone_e164": "+15555550100",
        }
    ]
}

HEADERS = {
    "Authorization": f"Bearer {V1_TOKEN}",
    CONTRACT_VERSION_HEADER: CONTRACT_VERSION,
}

# A start time well clear of zero, so "session age" and epoch ms are realistic.
CLOCK_START = 1_800_000_000.0


class FakeClock:
    """The /v1 registry clock, in epoch seconds, moved by hand."""

    def __init__(self, start: float = CLOCK_START) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def ms(self) -> int:
        return int(self.now * 1000)


def base_settings(**overrides: Any) -> Settings:
    settings = Settings(
        api_token=V1_TOKEN,
        mock_mode=True,
        session_ttl_seconds=300,
        otp_max_attempts=3,
        otp_ttl_seconds=120,
        mock_jitter_min_ms=0,
        mock_jitter_max_ms=0,
        speech_segments_before_decision=2,
        users_on_file_json=json.dumps(EXAMPLE_USERS),
    )
    return dataclasses.replace(settings, **overrides)


def build_app(
    settings: Settings, clock: FakeClock, log_stream: io.StringIO, **kwargs: Any
) -> FastAPI:
    return create_app(
        settings,
        v1_clock=clock,
        log_stream=log_stream,
        observability=ObservabilityConfig(),
        **kwargs,
    )


def client_for(
    app: FastAPI, headers: dict[str, str] | None = None
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://detector.test",
        headers=HEADERS if headers is None else headers,
    )


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def log_stream() -> io.StringIO:
    return io.StringIO()


@pytest.fixture
def v1_settings() -> Settings:
    return base_settings()


@pytest.fixture
def v1_app(v1_settings: Settings, clock: FakeClock, log_stream: io.StringIO) -> FastAPI:
    return build_app(v1_settings, clock, log_stream)


@pytest.fixture
async def v1(v1_app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with client_for(v1_app) as client:
        yield client


# ------------------------------------------------------------- helpers


def session_body(
    room: str = "room-1",
    caller_identity: str = "sip_caller",
    claimed_user_id: str = "u_eric",
    channel: str = "phone",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "room": room,
        "caller_identity": caller_identity,
        "claimed_user_id": claimed_user_id,
        "channel": channel,
        **extra,
    }


async def create_session(
    client: httpx.AsyncClient,
    scenario: str | None = None,
    **body: Any,
) -> str:
    headers = {"X-Callback-Scenario": scenario} if scenario is not None else None
    response = await client.post(
        "/v1/sessions", json=session_body(**body), headers=headers
    )
    assert response.status_code == 200, response.text
    return response.json()["session_id"]


def turn_body(
    clock: FakeClock,
    text: str = "four seven two nine one three",
    role: str = "user",
    **extra: Any,
) -> dict[str, Any]:
    start = clock.ms()
    return {
        "role": role,
        "text": text,
        "start_ms": start,
        "end_ms": start + 1200,
        "words": [],
        **extra,
    }
