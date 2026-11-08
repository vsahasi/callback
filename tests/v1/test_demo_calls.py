"""The five demo calls of the final ruling, section 5, replayed over HTTP.

Raw httpx and callback_shared models, in process over ASGI, on a fake
registry clock: 0.7 s per challenge, and 6 s before cloned_voice's first
request. Veer's agent/tests/test_conformance.py drives his real Gate through
the same sequences against a running detector.

Every status, decision, required_factors and reasons[0] is asserted, and the
score lines by membership.
"""

from __future__ import annotations

import httpx
import pytest
from callback_shared.contract import (
    CONTRACT_VERSION_HEADER,
    ChallengeState,
    Decision,
    DecisionKind,
    EnrollmentResult,
    EnrollmentStarted,
    ErrorResponse,
    Factor,
)
from fastapi import FastAPI

from tests.v1.conftest import V1_TOKEN, FakeClock, client_for
from tests.v1.demo_driver import DemoCall

ALL_THREE = [Factor.VOICE_CHALLENGE, Factor.SMS_OTP, Factor.CALLBACK_ON_FILE]
VOICE = [Factor.VOICE_CHALLENGE]


def assert_step_up(decision: Decision, factors: list[Factor]) -> None:
    assert decision.decision is DecisionKind.STEP_UP
    assert decision.required_factors == factors
    assert decision.ttl_s == 120


def assert_deny(decision: Decision, code: str, *lines: str) -> None:
    assert decision.decision is DecisionKind.DENY
    assert decision.reasons[0] == code
    assert decision.required_factors == []
    assert decision.ttl_s == 0
    for line in lines:
        assert line in decision.reasons, (line, decision.reasons)


async def assert_hold_allowed(call: DemoCall) -> None:
    hold = await call.hold()
    assert hold.decision is DecisionKind.ALLOW
    assert hold.ttl_s == 0
    assert hold.reasons == ["tier0_hold"]


async def assert_locked(call: DemoCall, code: str) -> None:
    for decision in (await call.read_accounts(), await call.bank_change()):
        assert decision.decision is DecisionKind.DENY
        assert decision.reasons[:2] == ["session_locked", code]
    # A locked session refuses every new step-up: a code cannot rescue it.
    for path, body in (
        ("challenges", {"kind": "phrase"}),
        ("otp/send", {}),
    ):
        response = await call.client.post(call._path(path), json=body)
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "session_locked"


# ----------------------------------------------------- call 1: real_cfo


async def test_call_1_real_cfo_steps_up_then_allows(
    v1: httpx.AsyncClient, clock: FakeClock
) -> None:
    call = DemoCall(v1, clock, "real_cfo")
    await call.start()

    # 1. Round 1 asks for every factor.
    assert_step_up(await call.bank_change(), ALL_THREE)

    # 2. A phrase challenge passes.
    _, status = await call.challenge("phrase")
    assert status.status is ChallengeState.PASSED
    assert status.phrase_match is True
    assert status.speaker_score == 0.83

    # 3. The code: sent to the number on file, 424242 verifies.
    assert (await call.sms_otp()).verified is True

    # 4. The callback to the number on file confirms.
    clock.advance(10)
    await call.callback("confirmed")

    # 5. Round 2: the round 1 pass is older than /caller.
    assert_step_up(await call.bank_change(), VOICE)

    # 6. A fresh phrase challenge on the callback leg.
    _, status = await call.challenge("phrase")
    assert status.status is ChallengeState.PASSED

    # 7. Round 3 allows the change.
    allowed = await call.bank_change()
    assert allowed.decision is DecisionKind.ALLOW
    assert allowed.ttl_s == 60
    assert allowed.required_factors == []
    assert "voiceprint_pass 0.83≥0.62" in allowed.reasons
    assert "spoof_low 0.04" in allowed.reasons
    assert allowed.signals.otp == "verified"

    # 8. A follow-up read in the same session: plain allow, no step-up.
    read = await call.read_accounts()
    assert read.decision is DecisionKind.ALLOW
    assert read.ttl_s == 300
    assert read.required_factors == []

    # The pass was consumed: the same change again needs a new challenge.
    assert_step_up(await call.bank_change(), VOICE)

    await assert_hold_allowed(call)
    # 9.
    await call.end()


# ------------------------------------------------- call 2: cloned_voice


async def test_call_2_cloned_voice_is_denied(
    v1: httpx.AsyncClient, clock: FakeClock
) -> None:
    call = DemoCall(v1, clock, "cloned_voice")
    await call.start()
    clock.advance(6)

    # 1. The clone passes the voiceprint, and is denied anyway.
    denied = await call.bank_change()
    assert_deny(
        denied, "synthetic_speech", "spoof_high 0.92", "voiceprint_pass 0.78≥0.62"
    )
    assert denied.signals.voiceprint is not None
    assert denied.signals.voiceprint.passed is True

    # 2. The hold is allowed.
    await assert_hold_allowed(call)

    # 3. Lock check, before /end.
    await assert_locked(call, "synthetic_speech")
    await call.end()


async def test_call_2_before_the_rise_steps_up(
    v1: httpx.AsyncClient, clock: FakeClock
) -> None:
    call = DemoCall(v1, clock, "cloned_voice")
    await call.start()
    assert_step_up(await call.bank_change(), ALL_THREE)
    clock.advance(6)
    assert_deny(await call.bank_change(), "synthetic_speech", "spoof_high 0.92")
    await call.end()


async def test_call_2_browser_read_is_denied(
    v1: httpx.AsyncClient, clock: FakeClock
) -> None:
    call = DemoCall(v1, clock, "cloned_voice", channel="browser")
    await call.start()
    clock.advance(6.5)
    assert_deny(await call.read_accounts(), "synthetic_speech", "spoof_high 0.92")
    await assert_hold_allowed(call)
    await call.end()


# ------------------------------------------------------- call 3: replay


async def test_call_3_replay_is_denied(v1: httpx.AsyncClient, clock: FakeClock) -> None:
    call = DemoCall(v1, clock, "replay")
    await call.start()

    # 1.
    assert_step_up(await call.bank_change(), ALL_THREE)

    # 2. The recording cannot say the new words.
    _, status = await call.challenge("phrase")
    assert status.status is ChallengeState.FAILED
    assert status.phrase_match is False
    assert status.reason == "phrase_mismatch"

    # 3.
    assert_deny(
        await call.bank_change(),
        "challenge_failed_replay",
        "phrase_match false",
        "spoof_mid 0.45",
    )
    await assert_hold_allowed(call)
    await assert_locked(call, "challenge_failed_replay")
    await call.end()


async def test_call_3_browser_read_is_denied(
    v1: httpx.AsyncClient, clock: FakeClock
) -> None:
    call = DemoCall(v1, clock, "replay", channel="browser")
    await call.start()
    assert_step_up(await call.read_accounts(), VOICE)
    _, status = await call.challenge("digits")
    assert status.status is ChallengeState.FAILED
    assert status.phrase_match is False
    assert_deny(
        await call.read_accounts(), "challenge_failed_replay", "phrase_match false"
    )
    await assert_hold_allowed(call)
    await call.end()


# ----------------------------------------------------- call 4: ai_agent


async def test_call_4_ai_agent_is_denied(
    v1: httpx.AsyncClient, clock: FakeClock
) -> None:
    call = DemoCall(v1, clock, "ai_agent")
    await call.start()
    assert_deny(
        await call.bank_change(),
        "automated_caller",
        "spoof_high 0.85",
        "agent_likelihood 0.90",
    )
    await assert_hold_allowed(call)
    await assert_locked(call, "automated_caller")
    await call.end()


# ----------------------------------------------- call 5: human_impostor


async def test_call_5_human_impostor_is_denied(
    v1: httpx.AsyncClient, clock: FakeClock
) -> None:
    call = DemoCall(v1, clock, "human_impostor")
    await call.start()

    # 1. The passive voiceprint 0.29 is voice fail 1.
    assert_step_up(await call.bank_change(), ALL_THREE)

    # 2. The right words in the wrong voice: voice fail 2.
    _, status = await call.challenge("phrase")
    assert status.status is ChallengeState.FAILED
    assert status.phrase_match is True
    assert status.speaker_score == 0.29
    assert status.reason == "speaker_mismatch"

    # 3.
    assert_deny(
        await call.bank_change(),
        "speaker_mismatch",
        "voiceprint_fail 0.29<0.45",
        "text_flags: urgency, verification_refusal",
    )
    await assert_hold_allowed(call)
    await assert_locked(call, "speaker_mismatch")
    await call.end()


async def test_call_5_browser_read_is_denied(
    v1: httpx.AsyncClient, clock: FakeClock
) -> None:
    call = DemoCall(v1, clock, "human_impostor", channel="browser")
    await call.start()
    assert_step_up(await call.read_accounts(), VOICE)
    _, status = await call.challenge("digits")
    assert status.status is ChallengeState.FAILED
    assert status.phrase_match is True
    assert_deny(await call.read_accounts(), "speaker_mismatch")
    await assert_hold_allowed(call)
    await call.end()


# ------------------------------------------------ also required to pass


@pytest.mark.parametrize(
    "scenario", ["real_cfo", "cloned_voice", "replay", "ai_agent", "human_impostor"]
)
async def test_the_hold_is_allowed_in_every_scenario(
    v1: httpx.AsyncClient, clock: FakeClock, scenario: str
) -> None:
    call = DemoCall(v1, clock, scenario)
    await call.start()
    clock.advance(6.5)  # past the cloned voice's spoof rise
    await assert_hold_allowed(call)
    await call.bank_change()
    await assert_hold_allowed(call)  # also after a deny, where there was one
    await call.end()


async def test_enrollment_before_the_demo(
    v1: httpx.AsyncClient, clock: FakeClock
) -> None:
    call = DemoCall(v1, clock, "real_cfo")
    session_id = await call.start()
    # 1.
    assert (await call.sms_otp()).verified is True
    # 2.
    response = await v1.post(
        "/v1/enrollments",
        json={
            "session_id": session_id,
            "user_id": "u_eric",
            "channel": "phone",
            "consent": {
                "version": "2026-11-04",
                "verbatim": "yes I agree",
                "ts_ms": clock.ms(),
            },
        },
    )
    assert response.status_code == 200
    started = EnrollmentStarted.model_validate(response.json())
    assert started.prompts_required == 7
    # 3.
    for _ in range(7):
        _, status = await call.challenge("enroll")
        assert status.status is ChallengeState.PASSED
    # 4.
    response = await v1.post(f"/v1/enrollments/{started.enrollment_id}/complete")
    assert response.status_code == 200
    result = EnrollmentResult.model_validate(response.json())
    assert (result.status, result.prompts_passed) == ("enrolled", 7)
    await call.end()


async def test_a_wrong_token_is_401(v1_app: FastAPI) -> None:
    headers = {
        "Authorization": "Bearer not-the-token",
        CONTRACT_VERSION_HEADER: "1.1.0",
    }
    async with client_for(v1_app, headers) as client:
        response = await client.get("/v1/users", params={"name": "Eric Catalano"})
    assert response.status_code == 401
    assert ErrorResponse.model_validate(response.json()).error.code == "unauthorized"


async def test_a_missing_version_header_is_400(v1_app: FastAPI) -> None:
    async with client_for(v1_app, {"Authorization": f"Bearer {V1_TOKEN}"}) as client:
        response = await client.post(
            "/v1/sessions",
            json={
                "room": "r",
                "caller_identity": "c",
                "claimed_user_id": "u_eric",
                "channel": "phone",
            },
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "contract_version_mismatch"


# "With the detector stopped, every tier 1-2 request is denied with
# detector_unavailable and the vendor is held" is the agent's half: its
# DetectorClient turns a refused connection into that deny. Veer's
# agent/tests/test_conformance.py and test_detector.py cover it.
