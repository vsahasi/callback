"""Bounds on per-session and per-process state (integration review M8).

One process holds every session, so an agent bug that loops, or a leaked
token, must not be able to grow memory without limit.
"""

from __future__ import annotations

import httpx
from fastapi import FastAPI

from detector.v1.enrollments import ConsentLog
from tests.v1.conftest import FakeClock, create_session, turn_body
from tests.v1.test_enrollments import enroll_prompt, start, verified_session


def code_of(response: httpx.Response) -> str:
    return response.json()["error"]["code"]


async def test_at_most_50_challenges_per_session(v1: httpx.AsyncClient) -> None:
    session_id = await create_session(v1)
    for _ in range(50):
        response = await v1.post(
            f"/v1/sessions/{session_id}/challenges", json={"kind": "digits"}
        )
        assert response.status_code == 200
    response = await v1.post(
        f"/v1/sessions/{session_id}/challenges", json={"kind": "digits"}
    )
    assert response.status_code == 429
    assert code_of(response) == "too_many_challenges"


async def test_turn_text_is_capped_at_the_route(
    v1: httpx.AsyncClient, v1_app: FastAPI, clock: FakeClock
) -> None:
    session_id = await create_session(v1)
    ok = await v1.post(
        f"/v1/sessions/{session_id}/turns", json=turn_body(clock, text="x" * 4000)
    )
    assert ok.status_code == 204
    response = await v1.post(
        f"/v1/sessions/{session_id}/turns", json=turn_body(clock, text="x" * 4001)
    )
    assert response.status_code == 422
    assert code_of(response) == "validation_error"
    assert "xxxx" not in response.text  # the input is never echoed
    assert len(v1_app.state.v1.registry.get(session_id).turns) == 1


async def test_turn_word_count_is_capped_at_the_route(
    v1: httpx.AsyncClient, clock: FakeClock
) -> None:
    session_id = await create_session(v1)
    word = {"w": "four", "start_ms": 1, "end_ms": 2}
    ok = await v1.post(
        f"/v1/sessions/{session_id}/turns",
        json=turn_body(clock, text="a", words=[word] * 400),
    )
    assert ok.status_code == 204
    response = await v1.post(
        f"/v1/sessions/{session_id}/turns",
        json=turn_body(clock, text="b", words=[word] * 401),
    )
    assert response.status_code == 422
    assert code_of(response) == "validation_error"
    long_word = {"w": "y" * 65, "start_ms": 1, "end_ms": 2}
    response = await v1.post(
        f"/v1/sessions/{session_id}/turns",
        json=turn_body(clock, text="c", words=[long_word]),
    )
    assert response.status_code == 422


async def test_completed_enrollments_go_when_their_session_is_evicted(
    v1: httpx.AsyncClient, v1_app: FastAPI, clock: FakeClock
) -> None:
    service = v1_app.state.v1
    session_id = await verified_session(v1)
    enrollment_id = (await start(v1, session_id)).json()["enrollment_id"]
    for _ in range(7):
        await enroll_prompt(v1, session_id, clock)
    assert (
        await v1.post(f"/v1/enrollments/{enrollment_id}/complete")
    ).status_code == 200
    await v1.post(f"/v1/sessions/{session_id}/end")
    assert enrollment_id in service.enrollments
    clock.advance(60 * 60)
    await create_session(v1, room="later")  # any access sweeps the registry
    assert enrollment_id not in service.enrollments
    response = await v1.post(f"/v1/enrollments/{enrollment_id}/complete")
    assert response.status_code == 404
    assert code_of(response) == "enrollment_not_found"


def test_revoke_rows_are_bounded_and_consents_are_kept() -> None:
    log = ConsentLog(max_revokes=3)
    log.append_consent(
        user_id="u_eric", channel="phone", version="v", verbatim="yes", ts_ms=1
    )
    for n in range(10):
        log.append_revoke(user_id=f"u_{n}", ts_ms=n)
    rows = log.rows()
    assert [r.kind for r in rows] == ["consent", "revoke", "revoke", "revoke"]
    assert [r.user_id for r in rows[1:]] == ["u_7", "u_8", "u_9"]


async def test_the_service_bounds_revoke_rows_by_default(
    v1: httpx.AsyncClient, v1_app: FastAPI
) -> None:
    consents = v1_app.state.v1.consents
    assert 0 < consents.max_revokes <= 1000
    consents.max_revokes = 5
    for n in range(8):
        assert (await v1.delete(f"/v1/enrollments/u_{n}")).status_code == 204
    assert len(consents.rows()) == 5
