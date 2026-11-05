"""Test wiring.

By default these tests run the app in process over an ASGI transport. Point
them at a deployed mock instead by exporting:

    CALLBACK_BASE_URL=https://<your-modal-url>
    CALLBACK_API_TOKEN=<the token that deployment expects>
    uv run pytest tests/contract -v

That is how Veer validates his LiveKit agent against the real mock without
needing this repository's Python environment to match his.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import httpx
import pytest

from detector.app import create_app
from detector.config import Settings

TEST_TOKEN = "test-token-do-not-use-in-production"


@pytest.fixture
def settings() -> Settings:
    """Jitter off, so the suite is fast. One test re-enables it explicitly."""

    return Settings(
        api_token=TEST_TOKEN,
        mock_mode=True,
        session_ttl_seconds=300,
        otp_max_attempts=3,
        otp_ttl_seconds=120,
        mock_jitter_min_ms=0,
        mock_jitter_max_ms=0,
        speech_segments_before_decision=2,
    )


@pytest.fixture
def base_url() -> str | None:
    return os.environ.get("CALLBACK_BASE_URL")


@pytest.fixture
def auth_token(base_url: str | None) -> str:
    if base_url:
        token = os.environ.get("CALLBACK_API_TOKEN")
        if not token:
            pytest.skip("CALLBACK_BASE_URL is set but CALLBACK_API_TOKEN is not")
        return token
    return TEST_TOKEN


@pytest.fixture
async def client(
    settings: Settings, base_url: str | None, auth_token: str
) -> AsyncIterator[httpx.AsyncClient]:
    headers = {"Authorization": f"Bearer {auth_token}"}
    if base_url:
        async with httpx.AsyncClient(
            base_url=base_url, headers=headers, timeout=10.0
        ) as remote:
            yield remote
        return

    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://detector.test", headers=headers
    ) as local:
        yield local
