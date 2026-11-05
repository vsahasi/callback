"""Regression tests for fail-open defects found by an adversarial review.

Each test here corresponds to a specific way the service could be made to
authenticate, or to allow, when it should have refused. They are grouped in one
file so the fail-closed guarantees are legible as a set rather than scattered.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import httpx
import pytest

from detector import scoring
from detector.app import create_app
from detector.config import Settings
from detector.mock import resolve_scenario
from shared.contract import (
    SCENARIO_HEADER,
    Decision,
    Factor,
    FactorName,
    FactorVerdict,
    Scenario,
)

TOKEN = "hardening-token-not-a-real-secret"


def settings_with(token: str = TOKEN) -> Settings:
    return Settings(
        api_token=token,
        mock_mode=True,
        session_ttl_seconds=300,
        otp_max_attempts=3,
        otp_ttl_seconds=120,
        mock_jitter_min_ms=0,
        mock_jitter_max_ms=0,
        speech_segments_before_decision=2,
    )


def client_for(settings: Settings, token: str = TOKEN) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(settings)),
        base_url="http://hardening.test",
        headers={"Authorization": f"Bearer {token}"},
    )


# ------------------------------------------------- the API token default


def test_absent_token_is_generated_not_defaulted(monkeypatch) -> None:
    """A missing secret must not fall back to a committed literal.

    The old default was the string "dev-token", which lives in this repo. A
    misnamed Modal secret would have left the service accepting a token any
    reader of the source already knows.
    """

    monkeypatch.delenv("CALLBACK_API_TOKEN", raising=False)
    settings = Settings.from_env()

    assert settings.api_token_generated is True
    assert settings.api_token != "dev-token"
    assert len(settings.api_token) >= 32


def test_empty_token_does_not_authenticate_a_bare_bearer(monkeypatch) -> None:
    """compare_digest("", "") is True, so an empty secret accepted "Bearer ".

    This is the nastiest form of the bug, because `modal secret create ...
    CALLBACK_API_TOKEN=` produces exactly it and the service looks healthy.
    """

    monkeypatch.setenv("CALLBACK_API_TOKEN", "")
    settings = Settings.from_env()

    assert settings.api_token_generated is True
    assert settings.api_token != ""


async def test_generated_token_rejects_everything() -> None:
    """With no configured token the service must refuse all callers."""

    monkeypatch_free = Settings(
        api_token="x" * 43,  # stand-in for the generated value
        mock_mode=True,
        session_ttl_seconds=300,
        otp_max_attempts=3,
        otp_ttl_seconds=120,
        mock_jitter_min_ms=0,
        mock_jitter_max_ms=0,
        speech_segments_before_decision=2,
        api_token_generated=True,
    )
    async with client_for(monkeypatch_free, token="") as c:
        response = await c.post("/events", json={})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


# ------------------------------------------- no evidence is maximum risk


def test_unscored_factors_blend_to_maximum_risk() -> None:
    """blend_risk returned 0.0 when nothing was scored, which decide() allows.

    "No model could run" and "every signal is perfect" must never produce the
    same number.
    """

    assert scoring.blend_risk([]) == 1.0
    assert scoring.decide(scoring.blend_risk([])) is Decision.DENY

    all_unrun = [
        Factor(
            name=FactorName.VOICE_MATCH,
            score=0.0,
            weight=0.5,
            verdict=FactorVerdict.NOT_RUN,
        ),
        Factor(
            name=FactorName.SYNTHETIC_SPEECH,
            score=0.0,
            weight=0.5,
            verdict=FactorVerdict.NOT_RUN,
        ),
    ]
    assert scoring.blend_risk(all_unrun) == 1.0
    assert scoring.decide(scoring.blend_risk(all_unrun)) is Decision.DENY


def test_zero_weight_factors_blend_to_maximum_risk() -> None:
    """Weights that sum to zero carry no information either."""

    zero_weight = [
        Factor(
            name=FactorName.VOICE_MATCH,
            score=1.0,
            weight=0.0,
            verdict=FactorVerdict.PASS,
        )
    ]
    assert scoring.blend_risk(zero_weight) == 1.0


def test_real_scenarios_are_unaffected_by_the_empty_case() -> None:
    """The guard must not disturb the shipped scripts."""

    from detector.mock import SCRIPTS

    assert scoring.blend_risk(SCRIPTS[Scenario.REAL_CFO].factor_list()) < 0.05


# ------------------------------------------------ non-ASCII bearer token


def _raw_asgi_request(app, headers: list[tuple[bytes, bytes]]) -> dict:
    """Call the ASGI app directly, so arbitrary header bytes can be sent.

    httpx will not send a non-ASCII header value, but a real client can, and
    Starlette decodes header bytes as latin-1.
    """

    captured: dict = {}

    async def run() -> None:
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.1"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/factors/does-not-exist-1234",
            "raw_path": b"/factors/does-not-exist-1234",
            "query_string": b"",
            "root_path": "",
            "headers": headers,
            "client": ("127.0.0.1", 12345),
            "server": ("hardening.test", 80),
        }

        async def receive() -> dict:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict) -> None:
            if message["type"] == "http.response.start":
                captured["status"] = message["status"]
            elif message["type"] == "http.response.body":
                captured.setdefault("body", b"")
                captured["body"] += message.get("body", b"")

        await app(scope, receive, send)

    asyncio.run(run())
    return captured


def test_non_ascii_bearer_token_is_401_not_500() -> None:
    """compare_digest raises TypeError on non-ASCII str.

    One non-ASCII byte in the Authorization header used to crash the auth path
    into an unhandled 500 with a non-contract body, masking a 401 as a server
    fault.
    """

    app = create_app(settings_with())
    headers = [
        (b"host", b"hardening.test"),
        (b"authorization", "Bearer ü".encode("latin-1")),
    ]
    result = _raw_asgi_request(app, headers)

    assert result["status"] == 401, result
    assert b"unauthorized" in result["body"]


def test_ascii_token_still_authenticates() -> None:
    """The byte comparison must not break the normal path."""

    app = create_app(settings_with())
    headers = [
        (b"host", b"hardening.test"),
        (b"authorization", f"Bearer {TOKEN}".encode("ascii")),
    ]
    result = _raw_asgi_request(app, headers)

    # Authenticated, so it reaches the handler and 404s on the unknown session.
    assert result["status"] == 404, result
    assert b"session_not_found" in result["body"]


# ------------------------------------------- the contract is not published


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
async def test_schema_endpoints_are_not_served(path: str) -> None:
    """FastAPI publishes these unauthenticated by default."""

    async with client_for(settings_with()) as c:
        response = await c.get(path)
    assert response.status_code == 404, f"{path} is still public"


# --------------------------------------- scenario selection is exact match


def test_claimed_user_id_is_matched_exactly() -> None:
    """Substring matching scored a real vendor as an attack.

    A genuine vendor id that merely contained "replay" used to be routed to the
    replay script and hard-denied over its name.
    """

    scenario, source = resolve_scenario(None, "acme-replay-parts-ltd")
    assert scenario is Scenario.REAL_CFO
    assert source == "default"

    scenario, source = resolve_scenario(None, "ai_agent_solutions_inc")
    assert scenario is Scenario.REAL_CFO

    # Exact names still work, which is how the fixtures drive scenarios.
    scenario, source = resolve_scenario(None, "replay")
    assert scenario is Scenario.REPLAY
    assert source == "claimed_user_id exact"


async def test_substring_vendor_id_is_not_denied_end_to_end() -> None:
    """The same thing through the real service."""

    async with client_for(settings_with()) as c:
        session_id = "hardening-vendor-substring-1"
        response = None
        for sequence, event in enumerate(
            ["session_start", "caller_claim", "speech_segment", "speech_segment"],
            start=1,
        ):
            response = await c.post(
                "/events",
                json={
                    "session_id": session_id,
                    "event": event,
                    "sequence": sequence,
                    "occurred_at": datetime.now(UTC).isoformat(),
                    "claimed_user_id": "replay-audio-systems-gmbh",
                    "metadata": {},
                },
            )
            assert response.status_code == 200, response.text

    body = response.json()
    assert body["scenario"] == Scenario.REAL_CFO.value
    assert body["decision"] == Decision.ALLOW.value


# ------------------------- the step-up cannot launder a hard deny


async def _drive(
    c: httpx.AsyncClient, session_id: str, scenario: Scenario, events: list[tuple]
) -> httpx.Response:
    headers = {SCENARIO_HEADER: scenario.value}
    response = None
    for sequence, (event, otp) in enumerate(events, start=1):
        payload = {
            "session_id": session_id,
            "event": event,
            "sequence": sequence,
            "occurred_at": datetime.now(UTC).isoformat(),
            "metadata": {},
        }
        if otp is not None:
            payload["otp_code"] = otp
        response = await c.post("/events", json=payload, headers=headers)
        if response.status_code != 200:
            break
    assert response is not None
    return response


@pytest.mark.parametrize("scenario", [Scenario.REPLAY, Scenario.AI_AGENT])
async def test_early_otp_request_cannot_launder_a_hard_deny(
    scenario: Scenario,
) -> None:
    """The headline fail-open this service exists to prevent.

    Sending otp_request BEFORE any speech used to open a challenge with an
    empty factor list. Every later speech_segment was then discarded, and the
    published mock code returned allow on a replay attack with risk 0.0.

    Now the speech is still analysed while the challenge is open, so the fatal
    factor lands and the session is terminal before any code is submitted.
    """

    events = [("session_start", None), ("otp_request", None)]
    events += [("speech_segment", None)] * 5
    events += [("otp_submit", "424242")]

    async with client_for(settings_with()) as c:
        response = await _drive(
            c, f"hardening-launder-{scenario.value}", scenario, events
        )

    if response.status_code == 200:
        body = response.json()
        assert body["decision"] == Decision.DENY.value, body
    else:
        # Already terminal, an even stronger refusal.
        assert response.status_code == 409, response.text
        assert response.json()["error"]["code"] == "session_terminal"


async def test_correct_code_on_an_unanalysed_session_is_refused() -> None:
    """A code alone must never verify a caller whose voice was not examined."""

    async with client_for(settings_with()) as c:
        response = await _drive(
            c,
            "hardening-no-analysis-1",
            Scenario.REAL_CFO,
            [
                ("session_start", None),
                ("otp_request", None),
                ("otp_submit", "424242"),
            ],
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["decision"] == Decision.DENY.value, body
    assert "No voice analysis" in (body["reason"] or "")


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        (Scenario.REAL_CFO, Decision.ALLOW),
        (Scenario.CLONED_VOICE, Decision.ALLOW),
        (Scenario.HUMAN_IMPOSTOR, Decision.ALLOW),
        (Scenario.REPLAY, Decision.DENY),
        (Scenario.AI_AGENT, Decision.DENY),
    ],
)
async def test_the_honest_path_is_unchanged(
    scenario: Scenario, expected: Decision
) -> None:
    """Closing the hole must not break legitimate step-up."""

    async with client_for(settings_with()) as c:
        session_id = f"hardening-honest-{scenario.value}"
        response = await _drive(
            c,
            session_id,
            scenario,
            [
                ("session_start", None),
                ("speech_segment", None),
                ("speech_segment", None),
            ],
        )
        assert response.status_code == 200, response.text
        body = response.json()

        if body["decision"] == Decision.CHALLENGE.value:
            payload = {
                "session_id": session_id,
                "event": "otp_submit",
                "sequence": 4,
                "occurred_at": datetime.now(UTC).isoformat(),
                "otp_code": "424242",
                "metadata": {},
            }
            response = await c.post(
                "/events",
                json=payload,
                headers={SCENARIO_HEADER: scenario.value},
            )
            assert response.status_code == 200, response.text
            body = response.json()

    assert body["decision"] == expected.value, body


async def test_resent_session_start_does_not_clear_a_challenge() -> None:
    """A reconnecting agent must not be able to reset an open step-up."""

    async with client_for(settings_with()) as c:
        session_id = "hardening-resend-start-1"
        challenged = await _drive(
            c,
            session_id,
            Scenario.CLONED_VOICE,
            [
                ("session_start", None),
                ("speech_segment", None),
                ("speech_segment", None),
            ],
        )
        assert challenged.json()["decision"] == Decision.CHALLENGE.value

        again = await c.post(
            "/events",
            json={
                "session_id": session_id,
                "event": "session_start",
                "sequence": 4,
                "occurred_at": datetime.now(UTC).isoformat(),
                "metadata": {},
            },
            headers={SCENARIO_HEADER: Scenario.CLONED_VOICE.value},
        )

    assert again.status_code == 200, again.text
    body = again.json()
    assert body["state"] == "challenging", body
    assert body["decision"] == Decision.CHALLENGE.value, body
    assert body["next_action"] == "request_otp", body


async def test_explicit_scenario_header_still_wins() -> None:
    """The header remains the primary selector for fixtures."""

    async with client_for(settings_with()) as c:
        response = await c.post(
            "/events",
            json={
                "session_id": "hardening-header-wins-1",
                "event": "session_start",
                "sequence": 1,
                "occurred_at": datetime.now(UTC).isoformat(),
                "claimed_user_id": "some-unrelated-vendor",
                "metadata": {},
            },
            headers={SCENARIO_HEADER: Scenario.AI_AGENT.value},
        )
    assert response.status_code == 200, response.text
    assert response.json()["scenario"] == Scenario.AI_AGENT.value
