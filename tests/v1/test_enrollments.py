"""Enrollment and consent on /v1 (ruling section 2.15)."""

from __future__ import annotations

import io
import re
from typing import Any

import httpx
from fastapi import FastAPI

from detector.enrolment import InMemoryVoicePrintStore
from detector.v1.challenges import FREE_ANSWER_QUESTIONS
from tests.v1.conftest import (
    FakeClock,
    base_settings,
    build_app,
    client_for,
    create_session,
    turn_body,
)

CONSENT_TEXT = "yes I agree, sentinel-consent-5f3a"


class SpyStore(InMemoryVoicePrintStore):
    def __init__(self) -> None:
        super().__init__()
        self.deleted: list[str] = []

    def delete(self, identity: str) -> bool:
        self.deleted.append(identity)
        return super().delete(identity)


def start_body(session_id: str, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "session_id": session_id,
        "user_id": "u_eric",
        "channel": "phone",
        "consent": {"version": "2026-11-04", "verbatim": CONSENT_TEXT, "ts_ms": 1},
    }
    body.update(overrides)
    return body


async def verified_session(client: httpx.AsyncClient, **kwargs: Any) -> str:
    session_id = await create_session(client, **kwargs)
    await client.post(f"/v1/sessions/{session_id}/otp/send", json={})
    checked = await client.post(
        f"/v1/sessions/{session_id}/otp/check", json={"code": "424242"}
    )
    assert checked.json() == {"verified": True}
    return session_id


async def start(
    client: httpx.AsyncClient, session_id: str, **overrides: Any
) -> httpx.Response:
    return await client.post(
        "/v1/enrollments", json=start_body(session_id, **overrides)
    )


async def enroll_prompt(
    client: httpx.AsyncClient,
    session_id: str,
    clock: FakeClock,
    *,
    answer: bool = True,
) -> dict[str, Any]:
    issued = await client.post(
        f"/v1/sessions/{session_id}/challenges", json={"kind": "enroll"}
    )
    assert issued.status_code == 200, issued.text
    if answer:
        await client.post(
            f"/v1/sessions/{session_id}/turns",
            json=turn_body(clock, text=f"answer {clock.now}"),
        )
    clock.advance(0.7)
    status = await client.get(
        f"/v1/sessions/{session_id}/challenges/{issued.json()['challenge_id']}"
    )
    return {"say": issued.json()["say"], **status.json()}


def code_of(response: httpx.Response) -> str:
    return response.json()["error"]["code"]


# --------------------------------------------------------------- gates


async def test_no_verified_code_is_409(v1: httpx.AsyncClient) -> None:
    session_id = await create_session(v1)
    response = await start(v1, session_id)
    assert response.status_code == 409
    assert code_of(response) == "otp_not_verified"
    # A sent but unverified code is not enough either.
    await v1.post(f"/v1/sessions/{session_id}/otp/send", json={})
    await v1.post(f"/v1/sessions/{session_id}/otp/check", json={"code": "000000"})
    assert code_of(await start(v1, session_id)) == "otp_not_verified"


async def test_claim_mismatch_for_the_user(v1: httpx.AsyncClient) -> None:
    session_id = await verified_session(v1)
    response = await start(v1, session_id, user_id="u_someone_else")
    assert response.status_code == 409
    assert code_of(response) == "claim_mismatch"


async def test_claim_mismatch_for_the_channel(v1: httpx.AsyncClient) -> None:
    session_id = await verified_session(v1)
    response = await start(v1, session_id, channel="browser")
    assert response.status_code == 409
    assert code_of(response) == "claim_mismatch"


async def test_a_locked_session_is_409(v1: httpx.AsyncClient, v1_app: FastAPI) -> None:
    session_id = await verified_session(v1)
    v1_app.state.v1.registry.get(session_id).locked_code = "speaker_mismatch"
    response = await start(v1, session_id)
    assert response.status_code == 409
    assert code_of(response) == "session_locked"


async def test_an_unknown_session_is_404(v1: httpx.AsyncClient) -> None:
    response = await start(v1, "ses_" + "0" * 32)
    assert response.status_code == 404
    assert code_of(response) == "session_not_found"


# ---------------------------------------------------------------- flow


async def test_seven_prompts_then_enrolled(
    v1: httpx.AsyncClient, v1_app: FastAPI, clock: FakeClock
) -> None:
    session_id = await verified_session(v1)
    started = await start(v1, session_id)
    assert started.status_code == 200
    body = started.json()
    assert re.fullmatch(r"enr_[0-9a-f]{32}", body["enrollment_id"])
    assert body["prompts_required"] == 7

    results = [await enroll_prompt(v1, session_id, clock) for _ in range(7)]
    assert all(r["status"] == "passed" for r in results)
    for r in results[:3]:
        assert re.fullmatch(r"Please repeat after me: \d \d \d, \d \d \d\.", r["say"])
    for r in results[3:6]:
        assert re.fullmatch(
            r"Please repeat after me: [a-z]+, [a-z]+, [a-z]+\.", r["say"]
        )
    assert results[6]["say"] in FREE_ANSWER_QUESTIONS

    # Every prompt has been asked.
    extra = await v1.post(
        f"/v1/sessions/{session_id}/challenges", json={"kind": "enroll"}
    )
    assert extra.status_code == 409
    assert code_of(extra) == "no_open_enrollment"

    done = await v1.post(f"/v1/enrollments/{body['enrollment_id']}/complete")
    assert done.status_code == 200
    assert done.json() == {"status": "enrolled", "prompts_passed": 7}
    # Completing again gives the same answer; the enrollment is closed.
    again = await v1.post(f"/v1/enrollments/{body['enrollment_id']}/complete")
    assert again.json() == done.json()
    closed = await v1.post(
        f"/v1/sessions/{session_id}/challenges", json={"kind": "enroll"}
    )
    assert code_of(closed) == "no_open_enrollment"
    # Mock mode writes no print.
    assert len(v1_app.state.v1.voiceprints) == 0


async def test_fewer_than_five_passes_is_failed(
    v1: httpx.AsyncClient, clock: FakeClock
) -> None:
    session_id = await verified_session(v1)
    enrollment_id = (await start(v1, session_id)).json()["enrollment_id"]
    for _ in range(4):
        await enroll_prompt(v1, session_id, clock)
    done = await v1.post(f"/v1/enrollments/{enrollment_id}/complete")
    assert done.json() == {"status": "failed", "prompts_passed": 4}


async def test_a_failed_try_reissues_the_same_prompt(
    v1: httpx.AsyncClient, v1_app: FastAPI, clock: FakeClock
) -> None:
    session_id = await verified_session(v1)
    enrollment_id = (await start(v1, session_id)).json()["enrollment_id"]
    for _ in range(3):
        await enroll_prompt(v1, session_id, clock)  # prompts 1-3 pass
    # Prompt 4 goes unanswered three times: the same prompt, then the next.
    prompts = []
    for _ in range(4):
        await enroll_prompt(v1, session_id, clock, answer=False)
        session = v1_app.state.v1.registry.get(session_id)
        current = session.challenges[session.current_challenge_id]
        prompts.append(current.enroll_prompt)
    assert prompts == [4, 4, 4, 5]
    # The unanswered first try of prompt 5 is superseded: prompt 5 again.
    for _ in range(3):
        await enroll_prompt(v1, session_id, clock)  # prompts 5, 6, 7 pass
    done = await v1.post(f"/v1/enrollments/{enrollment_id}/complete")
    # Prompts 1-3 and 5-7 passed; prompt 4 failed all three tries.
    assert done.json() == {"status": "enrolled", "prompts_passed": 6}


async def test_complete_counts_a_pass_not_yet_polled(
    v1: httpx.AsyncClient, clock: FakeClock
) -> None:
    session_id = await verified_session(v1)
    enrollment_id = (await start(v1, session_id)).json()["enrollment_id"]
    for _ in range(4):
        await enroll_prompt(v1, session_id, clock)
    await v1.post(f"/v1/sessions/{session_id}/challenges", json={"kind": "enroll"})
    await v1.post(
        f"/v1/sessions/{session_id}/turns", json=turn_body(clock, text="last")
    )
    clock.advance(0.7)
    done = await v1.post(f"/v1/enrollments/{enrollment_id}/complete")
    assert done.json() == {"status": "enrolled", "prompts_passed": 5}


async def test_a_new_start_replaces_the_open_enrollment(v1: httpx.AsyncClient) -> None:
    session_id = await verified_session(v1)
    first = (await start(v1, session_id)).json()["enrollment_id"]
    second = (await start(v1, session_id)).json()["enrollment_id"]
    assert first != second
    response = await v1.post(f"/v1/enrollments/{first}/complete")
    assert response.status_code == 404
    assert code_of(response) == "enrollment_not_found"


async def test_an_unknown_enrollment_is_404(v1: httpx.AsyncClient) -> None:
    response = await v1.post("/v1/enrollments/enr_" + "0" * 32 + "/complete")
    assert response.status_code == 404
    assert code_of(response) == "enrollment_not_found"


async def test_complete_after_end_or_lock_is_refused(
    v1: httpx.AsyncClient, v1_app: FastAPI
) -> None:
    ended = await verified_session(v1, room="a")
    enrollment_a = (await start(v1, ended)).json()["enrollment_id"]
    await v1.post(f"/v1/sessions/{ended}/end")
    assert (
        code_of(await v1.post(f"/v1/enrollments/{enrollment_a}/complete"))
        == "session_ended"
    )

    locked = await verified_session(v1, room="b")
    enrollment_b = (await start(v1, locked)).json()["enrollment_id"]
    v1_app.state.v1.registry.get(locked).locked_code = "synthetic_speech"
    assert (
        code_of(await v1.post(f"/v1/enrollments/{enrollment_b}/complete"))
        == "session_locked"
    )


async def test_real_mode_cannot_enroll(clock: FakeClock) -> None:
    store = SpyStore()
    app = build_app(
        base_settings(mock_mode=False), clock, io.StringIO(), voiceprints=store
    )
    service = app.state.v1
    async with client_for(app) as client:
        session_id = await create_session(client)
        # Real mode has no Twilio here, so mark the code verified by hand.
        service.registry.get(session_id).otp.verified = True
        enrollment_id = (await start(client, session_id)).json()["enrollment_id"]
        result = await enroll_prompt(client, session_id, clock)
        assert result["status"] == "failed"
        assert result["reason"] == "no_audio"
        done = await client.post(f"/v1/enrollments/{enrollment_id}/complete")
    assert done.json() == {"status": "failed", "prompts_passed": 0}
    assert len(store) == 0


# ------------------------------------------------------ consent, delete


async def test_consent_is_recorded(v1: httpx.AsyncClient, v1_app: FastAPI) -> None:
    session_id = await verified_session(v1)
    enrollment_id = (await start(v1, session_id)).json()["enrollment_id"]
    (row,) = v1_app.state.v1.consents.rows()
    assert row.kind == "consent"
    assert (row.user_id, row.channel, row.version, row.verbatim, row.ts_ms) == (
        "u_eric",
        "phone",
        "2026-11-04",
        CONSENT_TEXT,
        1,
    )
    assert CONSENT_TEXT not in repr(row)
    record = v1_app.state.v1.enrollments[enrollment_id]
    assert record.consent_id == row.consent_id


async def test_delete_erases_both_channels_and_is_idempotent(clock: FakeClock) -> None:
    store = SpyStore()
    app = build_app(base_settings(), clock, io.StringIO(), voiceprints=store)
    async with client_for(app) as client:
        for _ in range(2):
            response = await client.delete("/v1/enrollments/u_eric")
            assert response.status_code == 204
            assert response.content == b""
    assert store.deleted == ["u_eric:phone", "u_eric:browser"] * 2
    rows = app.state.v1.consents.rows()
    assert [r.kind for r in rows] == ["revoke", "revoke"]
    assert all(r.user_id == "u_eric" for r in rows)


async def test_verbatim_never_reaches_the_log(
    v1: httpx.AsyncClient, clock: FakeClock, log_stream: io.StringIO
) -> None:
    session_id = await verified_session(v1)
    enrollment_id = (await start(v1, session_id)).json()["enrollment_id"]
    for _ in range(7):
        await enroll_prompt(v1, session_id, clock)
    await v1.post(f"/v1/enrollments/{enrollment_id}/complete")
    text = log_stream.getvalue()
    assert "v1_enrollment_started" in text
    assert "v1_enrollment_completed" in text
    assert "sentinel-consent-5f3a" not in text
    assert "I agree" not in text
