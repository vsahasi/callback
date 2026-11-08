"""The review surface's read-only view of the /v1 registry.

`GET /review/api/v1/sessions` and `GET /review/api/v1/sessions/{id}`, behind
the review token guard. One real_cfo call runs a bank change through every
step-up to an allow, then both endpoints are checked for shape, order and
the absence of anything secret.
"""

from __future__ import annotations

import base64
import re

import httpx
from callback_shared.contract import SessionCreate
from fastapi import FastAPI

from tests.v1.conftest import V1_TOKEN, FakeClock, client_for
from tests.v1.demo_driver import DemoCall

SUMMARY_FIELDS = {
    "session_id",
    "created_at",
    "updated_at",
    "scenario",
    "claimed_user_id",
    "channel",
    "last_decision",
    "last_action",
    "last_reasons",
    "locked",
    "ended",
}
DETAIL_LISTS = {"decisions", "challenges", "otp", "factors", "events"}
DECISION_FIELDS = {
    "at",
    "action",
    "tier",
    "decision",
    "reasons",
    "required_factors",
    "ttl_s",
    "idempotency_key",
    "signals",
}
CHALLENGE_FIELDS = {
    "challenge_id",
    "kind",
    "status",
    "phrase_match",
    "speaker_score",
    "reason",
    "issued_at",
    "resolved_at",
}
ISO_UTC = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?\+00:00")


async def run_call_1(call: DemoCall) -> list[str]:
    """Call 1 up to the allow. Returns every challenge `say`."""

    says = []
    await call.start()
    await call.bank_change()
    issued, _ = await call.challenge("phrase")
    says.append(issued.say)
    assert (await call.sms_otp()).verified is True
    call.clock.advance(10)
    await call.callback("confirmed")
    await call.bank_change()
    issued, _ = await call.challenge("phrase")
    says.append(issued.say)
    assert (await call.bank_change()).decision == "allow"
    return says


async def test_both_endpoints_after_a_real_cfo_bank_change(
    v1: httpx.AsyncClient, clock: FakeClock
) -> None:
    older = DemoCall(v1, clock, "replay", room="older-room")
    await older.start()
    clock.advance(5)
    call = DemoCall(v1, clock, "real_cfo", room="newer-room")
    says = await run_call_1(call)

    # ------------------------------------------------------- the list
    listed = await v1.get("/review/api/v1/sessions")
    assert listed.status_code == 200
    sessions = listed.json()["sessions"]
    assert [s["session_id"] for s in sessions] == [call.session_id, older.session_id]
    newest = sessions[0]
    assert set(newest) == SUMMARY_FIELDS
    assert newest["scenario"] == "real_cfo"
    assert newest["claimed_user_id"] == "u_eric"
    assert newest["channel"] == "phone"
    assert newest["last_decision"] == "allow"
    assert newest["last_action"] == "update_vendor_bank_account"
    assert "voiceprint_pass 0.83\u22650.62" in newest["last_reasons"]
    assert newest["locked"] is False
    assert newest["ended"] is False
    assert ISO_UTC.fullmatch(newest["created_at"])
    assert ISO_UTC.fullmatch(newest["updated_at"])
    assert newest["updated_at"] > newest["created_at"]
    assert sessions[1]["last_decision"] is None

    # ----------------------------------------------------- the detail
    response = await v1.get(f"/review/api/v1/sessions/{call.session_id}")
    assert response.status_code == 200
    detail = response.json()
    assert set(detail) == SUMMARY_FIELDS | DETAIL_LISTS
    assert {k: detail[k] for k in SUMMARY_FIELDS} == newest

    decisions = detail["decisions"]
    assert [d["decision"] for d in decisions] == ["step_up", "step_up", "allow"]
    assert all(set(d) == DECISION_FIELDS for d in decisions)
    assert decisions[0]["required_factors"] == [
        "voice_challenge",
        "sms_otp",
        "callback_on_file",
    ]
    assert decisions[1]["required_factors"] == ["voice_challenge"]
    assert [d["tier"] for d in decisions] == [2, 2, 2]
    assert [d["ttl_s"] for d in decisions] == [120, 120, 60]
    assert [d["at"] for d in decisions] == sorted(d["at"] for d in decisions)
    # The signals object is exactly the one each Decision carried.
    assert [d["signals"] for d in decisions] == [
        sent.signals.model_dump(mode="json") for sent in call.decisions
    ]
    keys = [d["idempotency_key"] for d in decisions]
    assert len(set(keys)) == 3
    assert all(key.startswith("idem-") for key in keys)

    challenges = detail["challenges"]
    assert len(challenges) == 2
    assert all(set(c) == CHALLENGE_FIELDS for c in challenges)
    assert [c["status"] for c in challenges] == ["passed", "passed"]
    assert challenges[0]["kind"] == "phrase"
    assert challenges[0]["speaker_score"] == 0.83
    assert ISO_UTC.fullmatch(challenges[0]["resolved_at"])

    assert detail["otp"] == {
        "sent": True,
        "sent_to_last_four": "0100",
        "attempts": 1,
        "verified": True,
    }
    assert [(f["kind"], f["result"]) for f in detail["factors"]] == [
        ("callback", "confirmed")
    ]
    assert {e["type"] for e in detail["events"]} == {
        "agent_speech_start",
        "agent_speech_end",
        "user_speech_start",
    }
    assert all(set(e) == {"type", "at"} for e in detail["events"])

    # ------------------------------------------------ nothing secret
    for body in (listed.text, response.text):
        for needle in ("424242", "+15555550100", "5555550100", "Eric Catalano"):
            assert needle not in body, needle
        assert "four seven two" not in body  # the caller's answer turns
        for say in says:
            assert say not in body
            for group in re.findall(r"\d \d \d", say):
                assert group not in body
            for word in re.findall(r"[a-z]{4,}", say):
                assert not re.search(rf"\b{word}\b", body), word


async def test_a_locked_and_ended_session_shows_as_such(
    v1: httpx.AsyncClient, clock: FakeClock
) -> None:
    call = DemoCall(v1, clock, "ai_agent")
    await call.start()
    await call.bank_change()
    await call.hold()
    await call.end()
    response = await v1.get(f"/review/api/v1/sessions/{call.session_id}")
    detail = response.json()
    assert detail["locked"] is True
    assert detail["ended"] is True
    assert detail["last_action"] == "hold_vendor"
    assert detail["last_reasons"] == ["tier0_hold"]
    assert [d["tier"] for d in detail["decisions"]] == [2, 0]
    assert detail["decisions"][0]["reasons"][0] == "automated_caller"
    assert detail["otp"] == {
        "sent": False,
        "sent_to_last_four": None,
        "attempts": 0,
        "verified": None,
    }
    # /end drops speech events (ruling 2.6).
    assert detail["events"] == []


async def test_the_list_is_capped_at_50(v1_app: FastAPI, clock: FakeClock) -> None:
    registry = v1_app.state.v1.registry
    registry.max_live = 100
    for n in range(55):
        registry.create(
            SessionCreate(
                room=f"r{n}",
                caller_identity="c",
                claimed_user_id="u_eric",
                channel="phone",
            ),
            None,
        )
        clock.advance(1)
    async with client_for(v1_app) as client:
        sessions = (await client.get("/review/api/v1/sessions")).json()["sessions"]
    assert len(sessions) == 50
    created = [s["created_at"] for s in sessions]
    assert created == sorted(created, reverse=True)


async def test_an_unknown_session_is_404(v1: httpx.AsyncClient) -> None:
    response = await v1.get("/review/api/v1/sessions/ses_" + "0" * 32)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


async def test_401_without_a_token(v1_app: FastAPI) -> None:
    async with client_for(v1_app, {}) as client:
        for path in ("/review/api/v1/sessions", "/review/api/v1/sessions/ses_x"):
            response = await client.get(path)
            assert response.status_code == 401
            assert response.json()["error"]["code"] == "unauthorized"
    wrong = {"Authorization": "Bearer wrong"}
    async with client_for(v1_app, wrong) as client:
        assert (await client.get("/review/api/v1/sessions")).status_code == 401


async def test_basic_auth_works_as_on_the_other_review_routes(v1_app: FastAPI) -> None:
    basic = base64.b64encode(f"review:{V1_TOKEN}".encode()).decode()
    async with client_for(v1_app, {"Authorization": f"Basic {basic}"}) as client:
        response = await client.get("/review/api/v1/sessions")
    assert response.status_code == 200
    assert response.json() == {"sessions": []}


async def test_reading_does_not_keep_a_session_alive(
    v1: httpx.AsyncClient, clock: FakeClock
) -> None:
    call = DemoCall(v1, clock, "real_cfo")
    await call.start()
    for _ in range(3):
        clock.advance(4 * 60)
        await v1.get(f"/review/api/v1/sessions/{call.session_id}")
    detail = (await v1.get(f"/review/api/v1/sessions/{call.session_id}")).json()
    assert detail["ended"] is True  # 12 idle minutes: reviews are not /v1 calls
