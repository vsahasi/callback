"""Route 11, authorize: idempotency, the deadline, echoes and hygiene."""

from __future__ import annotations

import asyncio
import io
import time
from typing import Any

import httpx
import pytest
from callback_shared.contract import AUTHORIZE_SERVER_DEADLINE_S, Decision
from fastapi import FastAPI

from detector.v1 import service as service_module
from tests.v1.conftest import (
    FakeClock,
    base_settings,
    build_app,
    client_for,
    create_session,
)

BANK_PARAMS = {
    "vendor_id": "v1",
    "vendor_name": "Acme Supplies",
    "new_account_last4": "6789",
    "new_routing_last4": "0021",
    "account_fingerprint": "0123456789abcdef",
}


def bank_change(key: str, **extra: Any) -> dict[str, Any]:
    return {
        "action": "update_vendor_bank_account",
        "params": BANK_PARAMS,
        "idempotency_key": key,
        **extra,
    }


def read_accounts(key: str) -> dict[str, Any]:
    return {
        "action": "list_vendor_accounts",
        "params": {"vendor_id": "v1", "vendor_name": "Acme Supplies"},
        "idempotency_key": key,
    }


async def authorize(
    client: httpx.AsyncClient, session_id: str, body: dict[str, Any]
) -> httpx.Response:
    return await client.post(f"/v1/sessions/{session_id}/authorize", json=body)


async def test_session_id_and_action_are_echoed(v1: httpx.AsyncClient) -> None:
    session_id = await create_session(v1)
    response = await authorize(v1, session_id, bank_change("k1"))
    assert response.status_code == 200
    decision = Decision.model_validate(response.json())
    assert decision.session_id == session_id
    assert decision.action == "update_vendor_bank_account"
    assert decision.decision == "step_up"
    assert decision.agent_say == ""
    assert response.json()["agent_say"] == ""


async def test_a_replay_returns_the_cached_decision(
    v1: httpx.AsyncClient, v1_app: FastAPI
) -> None:
    session_id = await create_session(v1)
    first = await authorize(v1, session_id, bank_change("k1"))
    # Change the state so a fresh evaluation would answer differently.
    v1_app.state.v1.registry.get(session_id).callback = None
    v1_app.state.v1.registry.get(session_id).locked_code = "otp_failed"
    replay = await authorize(v1, session_id, bank_change("k1"))
    assert replay.status_code == 200
    assert replay.json() == first.json()
    fresh = await authorize(v1, session_id, bank_change("k2"))
    assert fresh.json()["reasons"][:2] == ["session_locked", "otp_failed"]


async def test_the_same_key_for_another_request_is_409(v1: httpx.AsyncClient) -> None:
    session_id = await create_session(v1)
    await authorize(v1, session_id, bank_change("k1"))
    other_params = {**bank_change("k1"), "params": {**BANK_PARAMS, "vendor_id": "v2"}}
    for body in (other_params, read_accounts("k1")):
        response = await authorize(v1, session_id, body)
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "idempotency_conflict"


async def test_keys_are_per_session(v1: httpx.AsyncClient) -> None:
    a = await create_session(v1, room="a")
    b = await create_session(v1, room="b")
    await authorize(v1, a, bank_change("shared"))
    response = await authorize(v1, b, read_accounts("shared"))
    assert response.status_code == 200
    assert response.json()["session_id"] == b


async def test_the_deadline_denies_with_internal_timeout(
    v1: httpx.AsyncClient, v1_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = v1_app.state.v1
    assert service.authorize_deadline_s == AUTHORIZE_SERVER_DEADLINE_S == 2.2
    service.authorize_deadline_s = 0.05

    async def slow_jitter() -> None:
        await asyncio.sleep(1.0)

    monkeypatch.setattr(service, "jitter", slow_jitter)
    session_id = await create_session(v1)
    started = time.perf_counter()
    response = await authorize(v1, session_id, read_accounts("k1"))
    elapsed = time.perf_counter() - started
    assert response.status_code == 200
    assert response.json()["decision"] == "deny"
    assert response.json()["reasons"] == ["internal_timeout"]
    assert elapsed < 0.5
    # The deny locks the session.
    monkeypatch.setattr(service, "jitter", lambda: asyncio.sleep(0))
    locked = await authorize(v1, session_id, bank_change("k2"))
    assert locked.json()["reasons"][:2] == ["session_locked", "internal_timeout"]
    hold = {
        "action": "hold_vendor",
        "params": {"vendor_id": "v1", "vendor_name": "Acme Supplies", "reason": "x"},
        "idempotency_key": "k3",
    }
    assert (await authorize(v1, session_id, hold)).json()["decision"] == "allow"


async def test_an_internal_exception_denies(
    v1: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_: Any, **__: Any) -> None:
        raise RuntimeError("policy bug")

    monkeypatch.setattr(service_module, "evaluate", broken)
    session_id = await create_session(v1)
    response = await authorize(v1, session_id, bank_change("k1"))
    assert response.status_code == 200
    assert response.json()["decision"] == "deny"
    assert response.json()["reasons"] == ["internal_timeout"]


async def test_a_replay_is_not_jittered(clock: FakeClock) -> None:
    settings = base_settings(mock_jitter_min_ms=200, mock_jitter_max_ms=200)
    app = build_app(settings, clock, io.StringIO())
    async with client_for(app) as client:
        session_id = await create_session(client)
        started = time.perf_counter()
        first = await authorize(client, session_id, read_accounts("k1"))
        first_s = time.perf_counter() - started
        started = time.perf_counter()
        replay = await authorize(client, session_id, read_accounts("k1"))
        replay_s = time.perf_counter() - started
    assert first.json() == replay.json()
    assert first_s >= 0.2
    assert replay_s < 0.1


async def test_unknown_and_ended_sessions(v1: httpx.AsyncClient) -> None:
    response = await authorize(v1, "ses_" + "0" * 32, read_accounts("k1"))
    assert response.status_code == 404
    session_id = await create_session(v1)
    await v1.post(f"/v1/sessions/{session_id}/end")
    response = await authorize(v1, session_id, read_accounts("k1"))
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "session_ended"


async def test_an_unknown_action_is_422(v1: httpx.AsyncClient) -> None:
    session_id = await create_session(v1)
    body = {**read_accounts("k1"), "action": "wire_money"}
    response = await authorize(v1, session_id, body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert "wire_money" not in response.text


async def test_a_decision_never_ends_the_session(v1: httpx.AsyncClient) -> None:
    session_id = await create_session(v1, scenario="ai_agent")
    denied = await authorize(v1, session_id, bank_change("k1"))
    assert denied.json()["decision"] == "deny"
    turn = {"role": "user", "text": "hello", "start_ms": 1, "end_ms": 2}
    assert (
        await v1.post(f"/v1/sessions/{session_id}/turns", json=turn)
    ).status_code == 204


async def test_reasons_and_logs_hold_no_secrets(
    v1: httpx.AsyncClient, log_stream: io.StringIO
) -> None:
    session_id = await create_session(v1, scenario="human_impostor")
    context = {
        "pending_action": "bank change for Acme Supplies",
        "vendor_name": "Acme Supplies",
        "vendor_bank_last4": "6789",
    }
    first = await authorize(v1, session_id, bank_change("k1", context=context))
    assert first.status_code == 200
    body = first.text
    for needle in ("Acme", "6789", "0021", "0123456789abcdef", "424242"):
        assert needle not in body
    text = log_stream.getvalue()
    assert "v1_decision" in text
    # Short digit runs like the last four are left out here: a timestamp or a
    # request id can hold them by chance.
    for needle in ("Acme", "0123456789abcdef", "bank change", "vendor_bank_last4"):
        assert needle not in text


# ------------------------------------------------- M1: holds always pass

HOLD = {
    "action": "hold_vendor",
    "params": {"vendor_id": "v1", "vendor_name": "Acme Supplies", "reason": "x"},
}


async def test_a_hold_skips_jitter_above_the_deadline(clock: FakeClock) -> None:
    settings = base_settings(mock_jitter_min_ms=3000, mock_jitter_max_ms=3000)
    app = build_app(settings, clock, io.StringIO())
    async with client_for(app) as client:
        session_id = await create_session(client)
        started = time.perf_counter()
        response = await authorize(
            client, session_id, {**HOLD, "idempotency_key": "h1"}
        )
        elapsed = time.perf_counter() - started
    assert response.status_code == 200
    assert response.json()["decision"] == "allow"
    assert response.json()["reasons"] == ["tier0_hold"]
    assert response.json()["ttl_s"] == 0
    assert elapsed < 0.5
    assert app.state.v1.registry.get(session_id).locked_code is None


async def test_a_hold_with_a_policy_bug_is_still_allowed_and_never_locks(
    v1: httpx.AsyncClient, v1_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_: Any, **__: Any) -> None:
        raise RuntimeError("policy bug")

    monkeypatch.setattr(service_module, "evaluate", broken)
    session_id = await create_session(v1)
    response = await authorize(v1, session_id, {**HOLD, "idempotency_key": "h1"})
    assert response.status_code == 200
    assert response.json()["decision"] == "allow"
    assert response.json()["reasons"] == ["tier0_hold"]
    assert v1_app.state.v1.registry.get(session_id).locked_code is None


# ------------------------------------- M2: _apply failures deny, not 500


async def test_a_failure_while_recording_the_decision_denies(
    v1: httpx.AsyncClient, v1_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = v1_app.state.v1

    def broken_apply(*_: Any, **__: Any) -> None:
        raise RuntimeError("bookkeeping bug")

    monkeypatch.setattr(service, "_apply", broken_apply)
    session_id = await create_session(v1)
    response = await authorize(v1, session_id, bank_change("k1"))
    assert response.status_code == 200
    assert response.json()["decision"] == "deny"
    assert response.json()["reasons"] == ["internal_timeout"]
    assert response.json()["session_id"] == session_id
    assert service.registry.get(session_id).locked_code == "internal_timeout"


async def test_a_failure_in_the_decision_log_denies(
    v1: httpx.AsyncClient, v1_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = v1_app.state.v1
    real_log = service.log

    def log(event: str, **fields: Any) -> None:
        if event == "v1_decision" and fields.get("decision") == "step_up":
            raise RuntimeError("log sink down")
        real_log(event, **fields)

    monkeypatch.setattr(service, "log", log)
    session_id = await create_session(v1)
    response = await authorize(v1, session_id, bank_change("k1"))
    assert response.status_code == 200
    assert response.json()["reasons"] == ["internal_timeout"]
    assert service.registry.get(session_id).locked_code == "internal_timeout"


async def test_a_hold_whose_recording_fails_is_still_allowed(
    v1: httpx.AsyncClient, v1_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = v1_app.state.v1

    def broken_apply(*_: Any, **__: Any) -> None:
        raise RuntimeError("bookkeeping bug")

    monkeypatch.setattr(service, "_apply", broken_apply)
    session_id = await create_session(v1)
    response = await authorize(v1, session_id, {**HOLD, "idempotency_key": "h1"})
    assert response.status_code == 200
    assert response.json()["decision"] == "allow"
    assert service.registry.get(session_id).locked_code is None
