"""Auth, idempotency, sequencing, jitter and the client-side fail-closed rule."""

from __future__ import annotations

import time
from datetime import UTC, datetime

import httpx
import pytest

from detector import scoring
from detector.app import create_app
from detector.config import Settings
from shared.client import DetectorClient
from shared.contract import (
    SCENARIO_HEADER,
    Decision,
    EventType,
    Factor,
    FactorName,
    FactorVerdict,
    Scenario,
)
from tests.contract.conftest import TEST_TOKEN
from tests.contract.journeys import Journey, new_session_id

# --------------------------------------------------------------------- auth


async def test_healthz_needs_no_auth(client: httpx.AsyncClient) -> None:
    response = await client.get("/healthz", headers={"Authorization": ""})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert body["mock_mode"] is True


async def test_missing_bearer_is_401(client: httpx.AsyncClient) -> None:
    response = await client.post("/events", json={}, headers={"Authorization": ""})
    assert response.status_code == 401, response.text
    assert response.json()["error"]["code"] == "unauthorized"


async def test_wrong_bearer_is_401(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/events", json={}, headers={"Authorization": "Bearer nope-not-the-token"}
    )
    assert response.status_code == 401, response.text
    assert response.json()["error"]["code"] == "unauthorized"


async def test_factors_requires_auth(client: httpx.AsyncClient) -> None:
    response = await client.get("/factors/whatever-1234", headers={"Authorization": ""})
    assert response.status_code == 401, response.text


# -------------------------------------------------------------- validation


async def test_body_violating_the_contract_is_422(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/events",
        json={"session_id": "short", "event": "session_start", "sequence": -1},
        headers={SCENARIO_HEADER: Scenario.REAL_CFO.value},
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "validation_error"


async def test_unknown_field_is_rejected(client: httpx.AsyncClient) -> None:
    """extra="forbid" means a typo in Veer's payload fails loudly."""

    response = await client.post(
        "/events",
        json={
            "session_id": new_session_id(),
            "event": "session_start",
            "sequence": 1,
            "occurred_at": datetime.now(UTC).isoformat(),
            "claimed_user_idd": "typo",
        },
        headers={SCENARIO_HEADER: Scenario.REAL_CFO.value},
    )
    assert response.status_code == 422, response.text


# ------------------------------------------------------------- idempotency


def _body(session_id: str, sequence: int, event: EventType) -> dict:
    return {
        "session_id": session_id,
        "event": event.value,
        "sequence": sequence,
        "occurred_at": "2026-11-04T21:00:00+00:00",
        "metadata": {},
    }


async def test_identical_retry_is_an_idempotent_replay(
    client: httpx.AsyncClient,
) -> None:
    session_id = new_session_id()
    body = _body(session_id, 1, EventType.SESSION_START)
    headers = {SCENARIO_HEADER: Scenario.REAL_CFO.value}

    first = await client.post("/events", json=body, headers=headers)
    assert first.status_code == 200, first.text
    assert first.json()["idempotent_replay"] is False

    second = await client.post("/events", json=body, headers=headers)
    assert second.status_code == 200, second.text
    replay = second.json()
    assert replay["idempotent_replay"] is True
    assert replay["decision"] == first.json()["decision"]
    assert replay["state"] == first.json()["state"]


async def test_same_key_different_payload_is_a_conflict(
    client: httpx.AsyncClient,
) -> None:
    session_id = new_session_id()
    headers = {
        SCENARIO_HEADER: Scenario.REAL_CFO.value,
        "Idempotency-Key": "agent-retry-1",
    }

    first = await client.post(
        "/events", json=_body(session_id, 1, EventType.SESSION_START), headers=headers
    )
    assert first.status_code == 200, first.text

    second = await client.post(
        "/events", json=_body(session_id, 2, EventType.SPEECH_SEGMENT), headers=headers
    )
    assert second.status_code == 409, second.text
    assert second.json()["error"]["code"] == "sequence_conflict"


async def test_non_monotonic_sequence_is_a_conflict(
    client: httpx.AsyncClient,
) -> None:
    journey = Journey(client, Scenario.REAL_CFO)
    await journey.send(EventType.SESSION_START, sequence=5)

    response = await journey.send(EventType.SPEECH_SEGMENT, sequence=3)
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "sequence_conflict"


async def test_events_after_a_terminal_decision_are_rejected(
    client: httpx.AsyncClient,
) -> None:
    journey = Journey(client, Scenario.REAL_CFO)
    decided = (await journey.run_to_decision()).json()
    assert decided["state"] == "allowed"

    response = await journey.send(EventType.SPEECH_SEGMENT)
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "session_terminal"


async def test_otp_submit_without_a_challenge_is_rejected(
    client: httpx.AsyncClient,
) -> None:
    journey = Journey(client, Scenario.REAL_CFO)
    await journey.send(EventType.SESSION_START)

    response = await journey.send(EventType.OTP_SUBMIT, otp_code="424242")
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "session_terminal"


# ------------------------------------------------------------------ jitter


async def test_mock_jitter_falls_in_the_configured_window() -> None:
    """The brief asks for 150-400 ms of simulated latency."""

    settings = Settings(
        api_token=TEST_TOKEN,
        mock_mode=True,
        session_ttl_seconds=300,
        otp_max_attempts=3,
        otp_ttl_seconds=120,
        mock_jitter_min_ms=150,
        mock_jitter_max_ms=400,
        speech_segments_before_decision=2,
    )
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://detector.test",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    ) as local:
        session_id = new_session_id()
        started = time.perf_counter()
        response = await local.post(
            "/events",
            json=_body(session_id, 1, EventType.SESSION_START),
            headers={SCENARIO_HEADER: Scenario.REAL_CFO.value},
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

    assert response.status_code == 200, response.text
    assert elapsed_ms >= 150, f"no jitter applied, took {elapsed_ms:.0f}ms"
    assert elapsed_ms < 1500, f"jitter far outside the window, took {elapsed_ms:.0f}ms"
    assert response.json()["latency_ms"] >= 150


# ------------------------------------------------- client-side fail-closed


def _client_with(transport: httpx.MockTransport) -> DetectorClient:
    http = httpx.AsyncClient(
        transport=transport,
        base_url="http://detector.test",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    return DetectorClient(
        base_url="http://detector.test", token=TEST_TOKEN, client=http
    )


async def test_timeout_is_treated_as_deny() -> None:
    """The headline client rule: a slow detector means DENY, never ALLOW."""

    def _timeout(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("detector did not answer in time")

    async with _client_with(httpx.MockTransport(_timeout)) as detector:
        outcome = await detector.push_event(
            session_id=new_session_id(), event=EventType.SPEECH_SEGMENT, sequence=1
        )

    assert outcome.decision is Decision.DENY
    assert outcome.failed_closed is True
    assert outcome.response is None
    assert "ReadTimeout" in (outcome.failure_reason or "")


async def test_connection_error_is_treated_as_deny() -> None:
    def _refused(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    async with _client_with(httpx.MockTransport(_refused)) as detector:
        outcome = await detector.push_event(
            session_id=new_session_id(), event=EventType.SPEECH_SEGMENT, sequence=1
        )

    assert outcome.decision is Decision.DENY
    assert outcome.failed_closed is True


async def test_server_error_is_treated_as_deny() -> None:
    def _broken(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "unavailable"})

    async with _client_with(httpx.MockTransport(_broken)) as detector:
        outcome = await detector.push_event(
            session_id=new_session_id(), event=EventType.SPEECH_SEGMENT, sequence=1
        )

    assert outcome.decision is Decision.DENY
    assert outcome.failed_closed is True
    assert "503" in (outcome.failure_reason or "")


async def test_reference_client_round_trips_a_real_allow() -> None:
    """The same client must pass through a genuine ALLOW unchanged."""

    settings = Settings(
        api_token=TEST_TOKEN,
        mock_mode=True,
        session_ttl_seconds=300,
        otp_max_attempts=3,
        otp_ttl_seconds=120,
        mock_jitter_min_ms=0,
        mock_jitter_max_ms=0,
        speech_segments_before_decision=1,
    )
    app = create_app(settings)
    http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://detector.test",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    session_id = new_session_id()

    async with DetectorClient(
        base_url="http://detector.test", token=TEST_TOKEN, client=http
    ) as detector:
        await detector.push_event(
            session_id=session_id,
            event=EventType.SESSION_START,
            sequence=1,
            scenario=Scenario.REAL_CFO,
        )
        outcome = await detector.push_event(
            session_id=session_id,
            event=EventType.SPEECH_SEGMENT,
            sequence=2,
            scenario=Scenario.REAL_CFO,
        )

    assert outcome.failed_closed is False
    assert outcome.decision is Decision.ALLOW
    assert outcome.response is not None
    assert outcome.response.state.is_terminal


# ----------------------------------------------------------------- scoring


def test_otp_blending_matches_the_explicit_helper() -> None:
    """Rescaling weights and blend_risk_with_otp must agree.

    state.py rescales behavioural weights by (1 - OTP_WEIGHT) and appends the
    OTP factor, then calls blend_risk. That must equal the closed form.
    """

    behavioural = [
        Factor(
            name=FactorName.VOICE_MATCH,
            score=0.18,
            weight=0.35,
            verdict=FactorVerdict.FAIL,
        ),
        Factor(
            name=FactorName.SYNTHETIC_SPEECH,
            score=0.95,
            weight=0.25,
            verdict=FactorVerdict.PASS,
        ),
        Factor(
            name=FactorName.REPLAY, score=0.93, weight=0.15, verdict=FactorVerdict.PASS
        ),
        Factor(
            name=FactorName.AGENT_BEHAVIOUR,
            score=0.90,
            weight=0.15,
            verdict=FactorVerdict.PASS,
        ),
        Factor(
            name=FactorName.LIVENESS,
            score=0.94,
            weight=0.10,
            verdict=FactorVerdict.PASS,
        ),
    ]
    otp_pass = Factor(
        name=FactorName.OTP,
        score=1.0,
        weight=scoring.OTP_WEIGHT,
        verdict=FactorVerdict.PASS,
    )

    closed_form = scoring.blend_risk_with_otp(behavioural, otp_pass)

    rescaled = [f.model_copy(deep=True) for f in behavioural]
    for factor in rescaled:
        factor.weight = factor.weight * (1.0 - scoring.OTP_WEIGHT)
    rescaled_blend = scoring.blend_risk([*rescaled, otp_pass])

    assert rescaled_blend == pytest.approx(closed_form, abs=1e-9)


def test_fatal_factor_overrides_a_good_blended_score() -> None:
    factors = [
        Factor(
            name=FactorName.VOICE_MATCH,
            score=0.99,
            weight=0.85,
            verdict=FactorVerdict.PASS,
        ),
        Factor(
            name=FactorName.REPLAY, score=0.02, weight=0.15, verdict=FactorVerdict.FAIL
        ),
    ]
    assert scoring.blend_risk(factors) < scoring.ALLOW_BELOW
    tripped = scoring.fatal_factor(factors)
    assert tripped is not None
    assert tripped.name is FactorName.REPLAY


def test_middle_band_falls_closed_without_otp() -> None:
    midband = (scoring.ALLOW_BELOW + scoring.DENY_AT_OR_ABOVE) / 2
    assert scoring.decide(midband, otp_available=True) is Decision.CHALLENGE
    assert scoring.decide(midband, otp_available=False) is Decision.DENY
