"""The /v1 session registry and its simple routes (ruling section 2.6)."""

from __future__ import annotations

import io
import json
import re

import httpx
from fastapi import FastAPI

from detector.v1.logsafe import tag
from detector.v1.service import V1Service
from tests.v1.conftest import (
    FakeClock,
    base_settings,
    build_app,
    client_for,
    create_session,
    session_body,
    turn_body,
)

TWO_USERS = json.dumps(
    {
        "users": [
            {
                "user_id": "u_eric",
                "display_name": "Eric Catalano",
                "role": "CFO",
                "phone_e164": "+15555550100",
            },
            {
                "user_id": "u_sam",
                "display_name": "Sam Ortiz",
                "role": "AP lead",
                "phone_e164": "+15555550102",
            },
        ]
    }
)


def service_of(app: FastAPI) -> V1Service:
    return app.state.v1


# ---------------------------------------------------------------- create


async def test_the_session_id_format(v1: httpx.AsyncClient) -> None:
    session_id = await create_session(v1)
    assert re.fullmatch(r"ses_[0-9a-f]{32}", session_id)


async def test_create_is_idempotent_per_live_room(v1: httpx.AsyncClient) -> None:
    first = await create_session(v1)
    again = await create_session(v1)
    other_channel = await create_session(v1, channel="browser")
    assert first == again == other_channel


async def test_a_new_room_is_a_new_session(v1: httpx.AsyncClient) -> None:
    assert await create_session(v1, room="a") != await create_session(v1, room="b")


async def test_same_room_another_caller_is_a_conflict(v1: httpx.AsyncClient) -> None:
    await create_session(v1)
    response = await v1.post(
        "/v1/sessions", json=session_body(caller_identity="sip_other")
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "session_conflict"


async def test_same_room_another_claim_is_a_conflict(clock: FakeClock) -> None:
    app = build_app(base_settings(users_on_file_json=TWO_USERS), clock, io.StringIO())
    async with client_for(app) as client:
        await create_session(client)
        response = await client.post(
            "/v1/sessions", json=session_body(claimed_user_id="u_sam")
        )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "session_conflict"


async def test_an_unknown_claim_is_404(v1: httpx.AsyncClient) -> None:
    response = await v1.post(
        "/v1/sessions", json=session_body(claimed_user_id="smoke-user")
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "user_not_found"


async def test_the_sixth_live_session_is_503(v1: httpx.AsyncClient) -> None:
    ids = [await create_session(v1, room=f"room-{n}") for n in range(5)]
    response = await v1.post("/v1/sessions", json=session_body(room="room-5"))
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "too_many_sessions"
    # Ending one frees a slot.
    assert (await v1.post(f"/v1/sessions/{ids[0]}/end")).status_code == 204
    assert await create_session(v1, room="room-5")


async def test_an_unknown_scenario_is_400(v1: httpx.AsyncClient) -> None:
    response = await v1.post(
        "/v1/sessions",
        json=session_body(),
        headers={"X-Callback-Scenario": "deepfake_ceo"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_scenario"
    assert "deepfake_ceo" not in response.text


async def test_the_scenario_header_pins_the_script(
    v1: httpx.AsyncClient, v1_app: FastAPI
) -> None:
    registry = service_of(v1_app).registry
    pinned = await create_session(v1, scenario="cloned_voice", room="a")
    default = await create_session(v1, room="b")
    shouted = await create_session(v1, scenario=" REPLAY ", room="c")
    assert registry.get(pinned).scenario == "cloned_voice"
    assert registry.get(default).scenario == "real_cfo"
    assert registry.get(shouted).scenario == "replay"


async def test_a_repeat_create_ignores_the_header(
    v1: httpx.AsyncClient, v1_app: FastAPI
) -> None:
    first = await create_session(v1, scenario="replay")
    again = await create_session(v1, scenario="ai_agent")
    assert first == again
    assert service_of(v1_app).registry.get(first).scenario == "replay"


async def test_the_header_is_ignored_on_other_routes(
    v1: httpx.AsyncClient, clock: FakeClock
) -> None:
    session_id = await create_session(v1)
    response = await v1.post(
        f"/v1/sessions/{session_id}/turns",
        json=turn_body(clock),
        headers={"X-Callback-Scenario": "not-a-scenario"},
    )
    assert response.status_code == 204


async def test_real_mode_ignores_the_header_with_one_warning(clock: FakeClock) -> None:
    stream = io.StringIO()
    app = build_app(base_settings(mock_mode=False), clock, stream)
    async with client_for(app) as client:
        a = await create_session(client, scenario="not-a-scenario", room="a")
        await create_session(client, scenario="replay", room="b")
    assert service_of(app).registry.get(a).scenario is None
    warnings = [
        line
        for line in stream.getvalue().splitlines()
        if "scenario_header_ignored" in line
    ]
    assert len(warnings) == 1


async def test_unknown_body_fields_are_accepted(
    v1: httpx.AsyncClient, clock: FakeClock
) -> None:
    session_id = await create_session(v1, added_later={"x": 1})
    response = await v1.post(
        f"/v1/sessions/{session_id}/turns", json=turn_body(clock, added_later=True)
    )
    assert response.status_code == 204


async def test_a_bad_body_is_a_422(v1: httpx.AsyncClient) -> None:
    response = await v1.post("/v1/sessions", json={"room": "r"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


# ------------------------------------------------------- caller, turns


async def test_caller_keeps_the_claim(
    v1: httpx.AsyncClient, v1_app: FastAPI, clock: FakeClock
) -> None:
    session_id = await create_session(v1)
    clock.advance(5)
    response = await v1.post(
        f"/v1/sessions/{session_id}/caller",
        json={"caller_identity": "callee-eric", "channel": "phone"},
    )
    assert response.status_code == 204
    session = service_of(v1_app).registry.get(session_id)
    assert session.claimed_user_id == "u_eric"
    assert session.caller_identity == "callee-eric"
    assert session.caller_changed_at == clock.now
    # Create stays idempotent on the identity it was created with.
    assert await create_session(v1) == session_id


async def test_turns_are_deduplicated(
    v1: httpx.AsyncClient, v1_app: FastAPI, clock: FakeClock
) -> None:
    session_id = await create_session(v1)
    turn = turn_body(clock)
    for _ in range(3):
        assert (
            await v1.post(f"/v1/sessions/{session_id}/turns", json=turn)
        ).status_code == 204
    other = turn_body(clock, text="something else")
    await v1.post(f"/v1/sessions/{session_id}/turns", json=other)
    session = service_of(v1_app).registry.get(session_id)
    assert len(session.turns) == 2
    assert all(record.received_at == clock.now for record in session.turns.values())


async def test_turns_are_capped_at_200(
    v1: httpx.AsyncClient, v1_app: FastAPI, clock: FakeClock
) -> None:
    session_id = await create_session(v1)
    for n in range(205):
        await v1.post(
            f"/v1/sessions/{session_id}/turns", json=turn_body(clock, text=f"t{n}")
        )
    assert len(service_of(v1_app).registry.get(session_id).turns) == 200


async def test_events_and_factors_are_stored(
    v1: httpx.AsyncClient, v1_app: FastAPI, clock: FakeClock
) -> None:
    session_id = await create_session(v1)
    event = {"type": "user_speech_start", "ts_ms": clock.ms()}
    assert (
        await v1.post(f"/v1/sessions/{session_id}/events", json=event)
    ).status_code == 204
    for result in ("no_answer", "confirmed"):
        response = await v1.post(
            f"/v1/sessions/{session_id}/factors",
            json={"kind": "callback", "result": result},
        )
        assert response.status_code == 204
    session = service_of(v1_app).registry.get(session_id)
    assert len(session.events) == 1
    assert session.callback is not None
    assert session.callback.result == "confirmed"  # the latest report wins


async def test_a_factor_of_another_kind_is_a_422(v1: httpx.AsyncClient) -> None:
    session_id = await create_session(v1)
    response = await v1.post(
        f"/v1/sessions/{session_id}/factors",
        json={"kind": "probe", "result": "confirmed"},
    )
    assert response.status_code == 422


# --------------------------------------------------------- end, expiry


async def test_after_end_routes_are_409_and_end_is_204(
    v1: httpx.AsyncClient, v1_app: FastAPI, clock: FakeClock
) -> None:
    session_id = await create_session(v1)
    await v1.post(f"/v1/sessions/{session_id}/turns", json=turn_body(clock))
    assert (await v1.post(f"/v1/sessions/{session_id}/end")).status_code == 204
    response = await v1.post(f"/v1/sessions/{session_id}/turns", json=turn_body(clock))
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "session_ended"
    for path, body in (
        ("caller", {"caller_identity": "x", "channel": "phone"}),
        ("events", {"type": "agent_speech_end", "ts_ms": 1}),
        ("factors", {"kind": "callback", "result": "confirmed"}),
    ):
        response = await v1.post(f"/v1/sessions/{session_id}/{path}", json=body)
        assert response.status_code == 409, path
    assert (await v1.post(f"/v1/sessions/{session_id}/end")).status_code == 204
    # /end dropped the turns.
    assert service_of(v1_app).registry.lookup(session_id).turns == {}


async def test_an_ended_room_gets_a_new_session(v1: httpx.AsyncClient) -> None:
    first = await create_session(v1)
    await v1.post(f"/v1/sessions/{first}/end")
    assert await create_session(v1) != first


async def test_an_unknown_id_is_404(v1: httpx.AsyncClient, clock: FakeClock) -> None:
    unknown = "ses_" + "0" * 32
    for path, body in (
        ("turns", turn_body(clock)),
        ("caller", {"caller_identity": "x", "channel": "phone"}),
        ("events", {"type": "agent_speech_end", "ts_ms": 1}),
        ("factors", {"kind": "callback", "result": "confirmed"}),
        ("end", None),
    ):
        response = await v1.post(f"/v1/sessions/{unknown}/{path}", json=body)
        assert response.status_code == 404, path
        assert response.json()["error"]["code"] == "session_not_found"


async def test_the_30_minute_cap(v1: httpx.AsyncClient, clock: FakeClock) -> None:
    session_id = await create_session(v1)
    for _ in range(5):  # stay active, so only the cap can end it
        clock.advance(5 * 60)
        response = await v1.post(
            f"/v1/sessions/{session_id}/turns", json=turn_body(clock)
        )
        assert response.status_code == 204
    clock.advance(5 * 60)  # 30 minutes after create
    response = await v1.post(f"/v1/sessions/{session_id}/turns", json=turn_body(clock))
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "session_ended"


async def test_the_10_minute_idle_limit(
    v1: httpx.AsyncClient, clock: FakeClock
) -> None:
    session_id = await create_session(v1)
    clock.advance(9 * 60 + 59)
    assert (
        await v1.post(f"/v1/sessions/{session_id}/turns", json=turn_body(clock))
    ).status_code == 204
    clock.advance(10 * 60)
    response = await v1.post(f"/v1/sessions/{session_id}/turns", json=turn_body(clock))
    assert response.status_code == 409


async def test_an_expired_session_frees_its_slot_and_room(
    v1: httpx.AsyncClient, clock: FakeClock
) -> None:
    first = await create_session(v1)
    for n in range(4):
        await create_session(v1, room=f"other-{n}")
    clock.advance(11 * 60)
    assert await create_session(v1) != first


async def test_eviction_an_hour_after_the_end(
    v1: httpx.AsyncClient, clock: FakeClock
) -> None:
    session_id = await create_session(v1)
    await v1.post(f"/v1/sessions/{session_id}/end")
    clock.advance(59 * 60)
    assert (await v1.post(f"/v1/sessions/{session_id}/end")).status_code == 204
    clock.advance(60)
    response = await v1.post(f"/v1/sessions/{session_id}/end")
    assert response.status_code == 404


async def test_logs_carry_tags_not_raw_rooms_or_identities(
    v1: httpx.AsyncClient, log_stream: io.StringIO
) -> None:
    room = "room-sentinel-7a1f"
    caller = "sip_+15555550199"
    session_id = await create_session(v1, room=room, caller_identity=caller)
    await v1.post(
        f"/v1/sessions/{session_id}/caller",
        json={"caller_identity": "callee-sentinel-0b2c", "channel": "phone"},
    )
    text = log_stream.getvalue()
    assert room not in text
    assert caller not in text
    assert "callee-sentinel-0b2c" not in text
    assert tag(room) in text
    assert tag(caller) in text
