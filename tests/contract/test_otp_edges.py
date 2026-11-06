"""OTP edge cases, pinned because they are not what you would guess.

These run in process only: they need settings the deployed service does not
expose, so they build their own app rather than using the shared client fixture.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from detector.app import create_app
from detector.config import Settings
from shared.contract import MOCK_OTP_CODE, SCENARIO_HEADER, Scenario

TOKEN = "otp-edge-token"


def settings_for(*, otp_ttl: int = 120, session_ttl: int = 300) -> Settings:
    return Settings(
        api_token=TOKEN,
        mock_mode=True,
        session_ttl_seconds=session_ttl,
        otp_max_attempts=3,
        otp_ttl_seconds=otp_ttl,
        mock_jitter_min_ms=0,
        mock_jitter_max_ms=0,
        speech_segments_before_decision=2,
    )


def body(session_id: str, event: str, sequence: int, otp: str | None = None) -> dict:
    payload: dict = {
        "session_id": session_id,
        "event": event,
        "sequence": sequence,
        "occurred_at": datetime.now(UTC).isoformat(),
        "metadata": {},
    }
    if otp is not None:
        payload["otp_code"] = otp
    return payload


def make_client(settings: Settings) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(settings)),
        base_url="http://otp-edge.test",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )


async def drive_to_challenge(
    client: httpx.AsyncClient, session_id: str, scenario: Scenario
) -> httpx.Response:
    headers = {SCENARIO_HEADER: scenario.value}
    response = None
    for sequence, event in enumerate(
        ["session_start", "speech_segment", "speech_segment"], start=1
    ):
        response = await client.post(
            "/events", json=body(session_id, event, sequence), headers=headers
        )
        response.raise_for_status()
    assert response is not None
    assert response.json()["decision"] == "challenge", response.text
    return response


async def test_fourth_wrong_code_is_session_terminal_not_otp_exhausted() -> None:
    """The third failure is terminal, so the fourth attempt hits that guard.

    Documented in docs/CONTRACT.md. The agent must not code against 429 here.
    """

    session_id = "edge-exhaust-00000001"
    headers = {SCENARIO_HEADER: Scenario.CLONED_VOICE.value}

    async with make_client(settings_for()) as client:
        await drive_to_challenge(client, session_id, Scenario.CLONED_VOICE)

        remaining = []
        for sequence in (4, 5, 6):
            response = await client.post(
                "/events",
                json=body(session_id, "otp_submit", sequence, otp="000000"),
                headers=headers,
            )
            assert response.status_code == 200, response.text
            remaining.append(response.json()["otp"]["attempts_remaining"])

        assert remaining == [2, 1, 0]
        assert response.json()["decision"] == "deny"
        assert response.json()["state"] == "denied"

        fourth = await client.post(
            "/events",
            json=body(session_id, "otp_submit", 7, otp="000000"),
            headers=headers,
        )

    assert fourth.status_code == 409, fourth.text
    assert fourth.json()["error"]["code"] == "session_terminal"


async def test_submitting_against_an_expired_challenge_is_410() -> None:
    session_id = "edge-expired-00000001"
    headers = {SCENARIO_HEADER: Scenario.HUMAN_IMPOSTOR.value}

    async with make_client(settings_for(otp_ttl=0)) as client:
        await drive_to_challenge(client, session_id, Scenario.HUMAN_IMPOSTOR)
        expired = await client.post(
            "/events",
            json=body(session_id, "otp_submit", 4, otp=MOCK_OTP_CODE),
            headers=headers,
        )

    assert expired.status_code == 410, expired.text
    assert expired.json()["error"]["code"] == "otp_expired"


def test_expired_session_fails_closed() -> None:
    """Past its TTL a session is terminal, and /factors reports why.

    Driven through the engine so the clock can be moved without sleeping. A
    TTL of 0 is degenerate (it expires a session on its own first event), so
    this backdates a real session instead.
    """

    from datetime import timedelta

    from detector.errors import SessionTerminal
    from detector.state import DetectorEngine
    from shared.contract import EventRequest, EventType

    engine = DetectorEngine(settings_for(session_ttl=60))
    session_id = "edge-ttl-00000001"

    def request(event: EventType, sequence: int) -> EventRequest:
        return EventRequest(
            session_id=session_id,
            event=event,
            sequence=sequence,
            occurred_at=datetime.now(UTC),
            metadata={},
        )

    opened = engine.handle_event(
        request(EventType.SESSION_START, 1),
        scenario_header=Scenario.REAL_CFO.value,
        idempotency_key=None,
    )
    assert opened.state.value == "listening"

    # Move this session past its time to live.
    engine.store.get(session_id).created_at -= timedelta(seconds=120)

    with pytest.raises(SessionTerminal):
        engine.handle_event(
            request(EventType.SPEECH_SEGMENT, 2),
            scenario_header=Scenario.REAL_CFO.value,
            idempotency_key=None,
        )

    readback = engine.factors_for(session_id)
    assert readback.state.value == "expired"
    assert readback.decision.value == "deny"
    assert readback.next_action.value == "complete_deny"


@pytest.mark.parametrize("scenario", [Scenario.REPLAY, Scenario.AI_AGENT])
async def test_otp_request_on_a_hard_denied_session_is_rejected(
    scenario: Scenario,
) -> None:
    """You cannot talk your way out of a replay by asking for a challenge."""

    session_id = f"edge-hard-{scenario.value}-01"
    headers = {SCENARIO_HEADER: scenario.value}

    async with make_client(settings_for()) as client:
        response = None
        for sequence, event in enumerate(
            ["session_start", "speech_segment", "speech_segment"], start=1
        ):
            response = await client.post(
                "/events", json=body(session_id, event, sequence), headers=headers
            )
        assert response is not None
        assert response.json()["decision"] == "deny"

        asked = await client.post(
            "/events", json=body(session_id, "otp_request", 4), headers=headers
        )

    assert asked.status_code == 409, asked.text
    assert asked.json()["error"]["code"] == "session_terminal"
