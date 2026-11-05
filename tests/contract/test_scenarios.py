"""The five scripted scenarios, end to end through the real state machine.

These are the fixtures Veer runs against the mock URL. Set CALLBACK_BASE_URL
and CALLBACK_API_TOKEN to point them at a deployment.
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from shared.contract import (
    CONTRACT_VERSION,
    MOCK_OTP_CODE,
    SCENARIO_HEADER,
    Decision,
    EventType,
    FactorName,
    Scenario,
    SessionState,
)
from tests.contract.journeys import (
    EXPECTATIONS,
    Expectation,
    Journey,
    new_session_id,
)

WRONG_OTP = "000000"


@pytest.mark.parametrize("expected", EXPECTATIONS, ids=lambda e: e.scenario.value)
async def test_scenario_reaches_expected_decision(
    client: httpx.AsyncClient, expected: Expectation
) -> None:
    journey = Journey(client, expected.scenario)
    response = await journey.run_to_decision()

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["decision"] == expected.decision_after_speech.value
    assert body["state"] == expected.state_after_speech.value
    assert body["contract_version"] == CONTRACT_VERSION
    assert body["scenario"] == expected.scenario.value
    assert body["session_id"] == journey.session_id


@pytest.mark.parametrize("expected", EXPECTATIONS, ids=lambda e: e.scenario.value)
async def test_scenario_factor_shape_is_sane(
    client: httpx.AsyncClient, expected: Expectation
) -> None:
    journey = Journey(client, expected.scenario)
    body = (await journey.run_to_decision()).json()

    factors = body["factors"]
    assert factors, "a decision must be explained by at least one factor"

    names = [f["name"] for f in factors]
    assert len(names) == len(set(names)), f"duplicate factors: {names}"

    for factor in factors:
        assert 0.0 <= factor["score"] <= 1.0
        assert 0.0 <= factor["weight"] <= 1.0
        assert factor["verdict"] in {"pass", "fail", "inconclusive", "not_run"}

    assert 0.0 <= body["risk_score"] <= 1.0
    assert 0.0 <= body["confidence"] <= 1.0

    # Weights of a pre-OTP evaluation sum to 1.0.
    if FactorName.OTP.value not in names:
        assert sum(f["weight"] for f in factors) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "expected",
    [e for e in EXPECTATIONS if e.decision_with_correct_otp is not None],
    ids=lambda e: e.scenario.value,
)
async def test_correct_otp_resolves_step_up(
    client: httpx.AsyncClient, expected: Expectation
) -> None:
    journey = Journey(client, expected.scenario)
    challenged = (await journey.run_to_decision()).json()

    assert challenged["decision"] == Decision.CHALLENGE.value
    assert challenged["otp"] is not None
    assert challenged["otp"]["attempts_used"] == 0
    assert challenged["otp"]["attempts_remaining"] == 3
    assert challenged["next_action"] == "request_otp"

    response = await journey.send(EventType.OTP_SUBMIT, otp_code=MOCK_OTP_CODE)
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["decision"] == expected.decision_with_correct_otp.value
    assert body["state"] == SessionState.ALLOWED.value
    assert body["next_action"] == "complete_allow"

    otp_factors = [f for f in body["factors"] if f["name"] == FactorName.OTP.value]
    assert len(otp_factors) == 1
    assert otp_factors[0]["verdict"] == "pass"
    assert otp_factors[0]["score"] == 1.0


@pytest.mark.parametrize(
    "expected",
    [e for e in EXPECTATIONS if e.decision_with_wrong_otp is not None],
    ids=lambda e: e.scenario.value,
)
async def test_exhausting_otp_attempts_denies(
    client: httpx.AsyncClient, expected: Expectation
) -> None:
    journey = Journey(client, expected.scenario)
    await journey.run_to_decision()

    first = (await journey.send(EventType.OTP_SUBMIT, otp_code=WRONG_OTP)).json()
    assert first["decision"] == Decision.CHALLENGE.value
    assert first["otp"]["attempts_used"] == 1
    assert first["otp"]["attempts_remaining"] == 2

    second = (await journey.send(EventType.OTP_SUBMIT, otp_code=WRONG_OTP)).json()
    assert second["decision"] == Decision.CHALLENGE.value
    assert second["otp"]["attempts_remaining"] == 1

    third = (await journey.send(EventType.OTP_SUBMIT, otp_code=WRONG_OTP)).json()
    assert third["decision"] == expected.decision_with_wrong_otp.value
    assert third["state"] == SessionState.DENIED.value
    assert third["next_action"] == "complete_deny"
    assert third["otp"]["attempts_remaining"] == 0

    otp_factors = [f for f in third["factors"] if f["name"] == FactorName.OTP.value]
    assert len(otp_factors) == 1
    assert otp_factors[0]["verdict"] == "fail"


@pytest.mark.parametrize(
    "expected",
    [e for e in EXPECTATIONS if not e.otp_can_rescue],
    ids=lambda e: e.scenario.value,
)
async def test_hard_deny_scenarios_refuse_step_up(
    client: httpx.AsyncClient, expected: Expectation
) -> None:
    """Replay and an automated caller are refused outright, with no OTP path."""

    journey = Journey(client, expected.scenario)
    decided = (await journey.run_to_decision()).json()

    assert decided["decision"] == Decision.DENY.value
    assert decided["otp"] is None, "a hard deny must not issue an OTP challenge"

    # Any further event on a terminal session is a conflict.
    followup = await journey.send(EventType.OTP_SUBMIT, otp_code=MOCK_OTP_CODE)
    assert followup.status_code == 409, followup.text
    assert followup.json()["error"]["code"] == "session_terminal"


@pytest.mark.parametrize("expected", EXPECTATIONS, ids=lambda e: e.scenario.value)
async def test_factors_endpoint_agrees_with_events(
    client: httpx.AsyncClient, expected: Expectation
) -> None:
    journey = Journey(client, expected.scenario)
    event_body = (await journey.run_to_decision()).json()

    response = await client.get(f"/factors/{journey.session_id}")
    assert response.status_code == 200, response.text
    factors_body = response.json()

    assert factors_body["session_id"] == journey.session_id
    assert factors_body["decision"] == event_body["decision"]
    assert factors_body["state"] == event_body["state"]
    assert factors_body["risk_score"] == pytest.approx(event_body["risk_score"])
    assert factors_body["factors"] == event_body["factors"]
    assert factors_body["events_seen"] == 4  # start, claim, two speech segments
    assert factors_body["contract_version"] == CONTRACT_VERSION


async def test_factors_unknown_session_is_404(client: httpx.AsyncClient) -> None:
    response = await client.get("/factors/sess-does-not-exist-0000")
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "session_not_found"


@pytest.mark.parametrize("scenario", list(Scenario), ids=lambda s: s.value)
async def test_claimed_user_id_selects_scenario_without_header(
    client: httpx.AsyncClient, scenario: Scenario
) -> None:
    """Veer can drive scenarios by claimed_user_id alone, no custom header."""

    journey = Journey(
        client,
        scenario,
        claimed_user_id=scenario.value,
        pin_scenario_header=False,
    )
    body = (await journey.run_to_decision()).json()
    assert body["scenario"] == scenario.value


async def test_unknown_scenario_header_is_rejected(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post(
        "/events",
        json={
            "session_id": new_session_id(),
            "event": EventType.SESSION_START.value,
            "sequence": 1,
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {},
        },
        headers={SCENARIO_HEADER: "not_a_real_scenario"},
    )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "invalid_scenario"


async def test_pending_before_enough_speech(client: httpx.AsyncClient) -> None:
    journey = Journey(client, Scenario.REAL_CFO)
    await journey.send(EventType.SESSION_START)
    body = (await journey.send(EventType.SPEECH_SEGMENT)).json()

    assert body["decision"] == Decision.PENDING.value
    assert body["state"] == SessionState.LISTENING.value
    assert body["next_action"] == "continue_listening"


async def test_session_end_before_decision_fails_closed(
    client: httpx.AsyncClient,
) -> None:
    journey = Journey(client, Scenario.REAL_CFO)
    await journey.send(EventType.SESSION_START)
    body = (await journey.send(EventType.SESSION_END)).json()

    assert body["decision"] == Decision.DENY.value
    assert body["state"] == SessionState.DENIED.value
