"""The read-only review token: what it opens, and what it must not.

`CALLBACK_REVIEW_TOKEN` exists so the Vercel dashboard stops holding
`DETECTOR_TOKEN`. That is worth something only if the second token really is
confined to the review surface, so most of this file is the negative case. The
review token is pointed at `/v1`, `/events` and `/metrics`, and every one of
them has to refuse it. If any of those ever starts accepting it, the setting
has become a second way to authorize a payment rather than a smaller
credential, which is worse than not having it at all.

`/metrics` is in that list on purpose. It is read-only too, so it looks
harmless, but its labels carry the decision mix: a caller who has just been
denied can watch the deny counter move and learn that they were caught.
"""

from __future__ import annotations

import base64
import io
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

import httpx
import pytest

from detector.app import create_app
from detector.config import Settings
from detector.observability import ObservabilityConfig
from tests.contract.conftest import TEST_TOKEN

REVIEW_TOKEN = "review-token-read-only-do-not-use-in-production"

SERVICE = {"Authorization": f"Bearer {TEST_TOKEN}"}
REVIEW = {"Authorization": f"Bearer {REVIEW_TOKEN}"}
WIRE = {"X-Contract-Version": "1.2.0"}

#: Everything the dashboard actually fetches.
REVIEW_ROUTES: tuple[str, ...] = (
    "/review",
    "/review/api/sessions",
    "/review/api/v1/sessions",
)


def basic(token: str) -> dict[str, str]:
    """Any username is accepted; the token may go in either field."""

    raw = base64.b64encode(f"reviewer:{token}".encode()).decode()
    return {"Authorization": f"Basic {raw}"}


@pytest.fixture
def review_settings(settings: Settings) -> Settings:
    return replace(settings, review_token=REVIEW_TOKEN)


@pytest.fixture
def review_app(review_settings: Settings) -> Any:
    return create_app(
        review_settings,
        observability=ObservabilityConfig(
            log_enabled=True, metrics_enabled=True, journal_enabled=True
        ),
        log_stream=io.StringIO(),
    )


@pytest.fixture
async def client(review_app: Any) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=review_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://review.test"
    ) as local:
        yield local


# --------------------------------------------------------------- what it opens


@pytest.mark.parametrize("path", REVIEW_ROUTES)
async def test_the_review_token_opens_every_review_route(
    client: httpx.AsyncClient, path: str
) -> None:
    response = await client.get(path, headers=REVIEW)
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("path", REVIEW_ROUTES)
async def test_the_service_token_still_opens_every_review_route(
    client: httpx.AsyncClient, path: str
) -> None:
    """Adding a second token must not take the first one away."""

    response = await client.get(path, headers=SERVICE)
    assert response.status_code == 200, response.text


async def test_the_review_token_works_over_basic(client: httpx.AsyncClient) -> None:
    """A plain navigation cannot set a bearer header, which is why Basic exists."""

    response = await client.get("/review", headers=basic(REVIEW_TOKEN))
    assert response.status_code == 200, response.text


# ------------------------------------------------------------ what it must not


async def test_the_review_token_is_refused_on_v1(client: httpx.AsyncClient) -> None:
    """The whole point. /v1 can authorize a payment."""

    response = await client.get(
        "/v1/users", params={"name": "Eric Catalano"}, headers={**REVIEW, **WIRE}
    )
    assert response.status_code == 401, response.text


async def test_the_review_token_is_refused_on_a_v1_authorize(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post(
        "/v1/sessions/any-session/authorize", json={}, headers={**REVIEW, **WIRE}
    )
    assert response.status_code == 401, response.text


async def test_the_review_token_is_refused_on_events(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post("/events", json={}, headers=REVIEW)
    assert response.status_code == 401, response.text


async def test_the_review_token_is_refused_on_metrics(
    client: httpx.AsyncClient,
) -> None:
    """Read-only, but an oracle: the deny counter says whether a call was caught."""

    response = await client.get("/metrics", headers=REVIEW)
    assert response.status_code == 401, response.text


async def test_the_service_token_still_opens_metrics(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/metrics", headers=SERVICE)
    assert response.status_code == 200, response.text


async def test_the_review_token_is_refused_on_metrics_over_basic(
    client: httpx.AsyncClient,
) -> None:
    """Basic is not a way around the guard that does not hold the token."""

    response = await client.get("/metrics", headers=basic(REVIEW_TOKEN))
    assert response.status_code == 401, response.text


# ------------------------------------------------------------------ fail closed


async def test_an_unknown_token_opens_nothing(client: httpx.AsyncClient) -> None:
    headers = {"Authorization": "Bearer not-either-of-them"}
    for path in REVIEW_ROUTES:
        assert (await client.get(path, headers=headers)).status_code == 401


async def test_a_bare_bearer_authenticates_nothing(
    client: httpx.AsyncClient,
) -> None:
    """compare_digest("", "") is True, so an empty accepted token would pass."""

    response = await client.get("/review", headers={"Authorization": "Bearer "})
    assert response.status_code == 401, response.text


async def test_without_a_review_token_the_service_token_is_the_only_one(
    settings: Settings,
) -> None:
    """Unset is the default, and it has to leave the old behaviour exactly."""

    assert settings.review_token == ""
    transport = httpx.ASGITransport(app=create_app(settings))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://review.test"
    ) as local:
        assert (await local.get("/review", headers=SERVICE)).status_code == 200
        assert (await local.get("/review", headers=REVIEW)).status_code == 401


def test_config_refuses_a_review_token_equal_to_the_service_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured but identical is the one failure that would look fine."""

    monkeypatch.setenv("DETECTOR_TOKEN", "the-same-value-in-both")
    monkeypatch.delenv("CALLBACK_API_TOKEN", raising=False)
    monkeypatch.setenv("CALLBACK_REVIEW_TOKEN", "the-same-value-in-both")
    with pytest.raises(ValueError, match="must differ"):
        Settings.from_env()


def test_config_reads_a_distinct_review_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DETECTOR_TOKEN", "the-service-token")
    monkeypatch.delenv("CALLBACK_API_TOKEN", raising=False)
    monkeypatch.setenv("CALLBACK_REVIEW_TOKEN", "  the-review-token  ")
    resolved = Settings.from_env()
    assert resolved.api_token == "the-service-token"
    assert resolved.review_token == "the-review-token"


def test_config_defaults_the_review_token_to_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DETECTOR_TOKEN", "the-service-token")
    monkeypatch.delenv("CALLBACK_API_TOKEN", raising=False)
    monkeypatch.delenv("CALLBACK_REVIEW_TOKEN", raising=False)
    assert Settings.from_env().review_token == ""


# ----------------------------------------------------------------- never logged


def test_the_review_token_is_scrubbed_like_the_service_token(
    review_app: Any,
) -> None:
    """It is a credential too, so it must not reach a log line either."""

    redactor = review_app.state.observability.redactor
    assert redactor.contains_secret(f"authorization=Bearer {REVIEW_TOKEN}")
    assert redactor.contains_secret(f"authorization=Bearer {TEST_TOKEN}")
