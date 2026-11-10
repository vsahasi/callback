"""The cross-leg speaker check reaches /v1 authorize, end to end, in real mode.

The agent posts both legs of the call to POST /events under the session id
`/v1/sessions` returned: the `caller` track is whoever rang in, the `callback`
track is whoever answered the number on file. `/v1 .../authorize` reads what
the engine made of them through `detector.bridge`.

Two things the bridge has to get right, both found in review:

- A confirmed mismatch is the engine's own fatal deny. It must reach /v1 as
  `speaker_mismatch` and lock the session, not as a voiceprint under 0.45,
  which is one voice fail of the two the policy wants and steps the caller
  up for the one-time code and callback that the mismatch says prove nothing.
- An inconclusive comparison (the same talker, four semitones up) is not
  evidence against the caller. It must not lower the voiceprint /v1 reads,
  or main's challenge settlement turns a legitimate caller's correct answer
  into a `speaker_mismatch` fail. With nothing enrolled the voiceprint stays
  None, exactly as it would with no cross-leg check at all.
"""

from __future__ import annotations

import io
from collections.abc import AsyncIterator

import httpx
import pytest
from callback_shared.contract import Decision
from fastapi import FastAPI

from shared.contract import FactorVerdict
from tests.contract.test_cross_leg import FULL, HIGH, LOW, SHIFTED_4ST, Call, voice
from tests.v1.conftest import (
    FakeClock,
    base_settings,
    build_app,
    client_for,
    create_session,
    turn_body,
)

READ = {
    "action": "list_vendor_accounts",
    "params": {"vendor_id": "v1", "vendor_name": "Acme Supplies"},
}
BANK = {
    "action": "update_vendor_bank_account",
    "params": {
        "vendor_id": "v1",
        "vendor_name": "Acme Supplies",
        "new_account_last4": "6789",
        "new_routing_last4": "0021",
        "account_fingerprint": "0123456789abcdef",
    },
}


@pytest.fixture
def real_app(clock: FakeClock) -> FastAPI:
    return build_app(
        base_settings(mock_mode=False, audio_enabled=True, audio_allow_data_uris=True),
        clock,
        io.StringIO(),
    )


@pytest.fixture
async def client(real_app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with client_for(real_app) as http:
        yield http


async def authorize(
    client: httpx.AsyncClient, session_id: str, key: str, body: dict = READ
) -> Decision:
    response = await client.post(
        f"/v1/sessions/{session_id}/authorize",
        json={**body, "idempotency_key": key},
    )
    assert response.status_code == 200, response.text
    return Decision.model_validate(response.json())


async def both_legs(
    client: httpx.AsyncClient, session_id: str, caller, callback
) -> dict:
    """The whole call under the /v1 session id: start, caller leg, callback leg."""

    call = Call(client, session_id)
    await call.start()
    await call.speech("caller", caller)
    return await call.speech("callback", callback)


# ----------------------------------------------------- (a) two different voices


async def test_a_confirmed_mismatch_denies_speaker_mismatch_and_locks(
    client: httpx.AsyncClient, real_app: FastAPI
) -> None:
    """An impostor rang in, the real vendor answered the callback."""

    session_id = await create_session(client)
    body = await both_legs(
        client,
        session_id,
        voice(HIGH, seed=1, duration_s=FULL),
        voice(LOW, seed=2, duration_s=FULL),
    )
    assert body["decision"] == "deny"  # the engine's own verdict

    decision = await authorize(client, session_id, "k-bank", BANK)
    assert decision.decision == "deny", decision
    assert decision.reasons[0] == "speaker_mismatch", decision.reasons
    # No step-up on offer: the code would go to the phone the real vendor is
    # holding, so nothing it proves is about the caller.
    assert decision.required_factors == []

    assert real_app.state.v1.registry.get(session_id).locked_code == "speaker_mismatch"

    # The lock is sticky, for a read as much as for a money move.
    again = await authorize(client, session_id, "k-read", READ)
    assert again.decision == "deny"
    assert again.reasons[:2] == ["session_locked", "speaker_mismatch"]


async def test_the_mismatch_denies_without_any_voiceprint_fail_count(
    client: httpx.AsyncClient,
) -> None:
    """The deny comes from the bridged fatal, not from two voice fails.

    Nothing is enrolled and no challenge has run, so the only voice evidence
    is the cross-leg factor itself: one fail, which alone never reached the
    policy's two-fail speaker_mismatch rule. It denies anyway.
    """

    session_id = await create_session(client)
    await both_legs(
        client,
        session_id,
        voice(HIGH, seed=1, duration_s=FULL),
        voice(LOW, seed=2, duration_s=FULL),
    )
    decision = await authorize(client, session_id, "k1", BANK)
    assert decision.decision == "deny"
    assert decision.reasons[0] == "speaker_mismatch"
    assert decision.signals.voiceprint is not None
    assert decision.signals.voiceprint.passed is False


# ------------------------------------------ (b) the same talker, shifted


async def test_an_inconclusive_comparison_leaves_the_voiceprint_unset(
    client: httpx.AsyncClient, real_app: FastAPI
) -> None:
    """One talker, four semitones apart on the two legs: 0.3786 at 6 s.

    That is inside the comparator's inconclusive band. It used to be merged
    onto the wire as `voice_match` and read by /v1 as a voiceprint under
    0.45, a voice fail. Now it is recorded on the session and contributes
    no factor, so the voiceprint is what it is with no cross-leg check at
    all: None, because nothing is enrolled.
    """

    session_id = await create_session(client)
    body = await both_legs(
        client,
        session_id,
        voice(LOW, seed=1, duration_s=FULL),
        voice(SHIFTED_4ST, seed=2, duration_s=FULL),
    )
    assert body["decision"] != "deny"
    match = next(f for f in body["factors"] if f["name"] == "voice_match")
    assert match["verdict"] == FactorVerdict.NOT_RUN.value

    stored = real_app.state.engine.store.get(session_id)
    assert stored.cross_leg is not None  # recorded for the dashboard
    assert stored.cross_leg.verdict is FactorVerdict.INCONCLUSIVE
    assert not stored.cross_leg.confirmed

    decision = await authorize(client, session_id, "k1", BANK)
    assert decision.decision != "deny"
    assert "speaker_mismatch" not in decision.reasons
    assert decision.signals.voiceprint is None
    assert real_app.state.v1.registry.get(session_id).locked_code is None


async def test_an_inconclusive_comparison_equals_no_cross_leg_check(
    client: httpx.AsyncClient,
) -> None:
    """Side by side: the same two clips, with and without a callback leg.

    The control session hears the identical audio on the `caller` track only,
    so the cross-leg check never runs there. Everything /v1 reads must agree.
    """

    first = voice(LOW, seed=1, duration_s=FULL)
    second = voice(SHIFTED_4ST, seed=2, duration_s=FULL)

    shifted = await create_session(client, room="room-shifted")
    await both_legs(client, shifted, first, second)
    alone = await create_session(client, room="room-alone")
    call = Call(client, alone)
    await call.start()
    await call.speech("caller", first)
    await call.speech("caller", second)

    one = await authorize(client, shifted, "k1", BANK)
    other = await authorize(client, alone, "k1", BANK)
    assert one.decision == other.decision
    assert one.signals.voiceprint is None and other.signals.voiceprint is None
    assert one.signals == other.signals
    assert one.reasons == other.reasons
    assert one.required_factors == other.required_factors


async def test_the_legitimate_talker_passes_a_challenge_with_no_speaker_score(
    client: httpx.AsyncClient, clock: FakeClock
) -> None:
    """Main's settlement (#13) with the inconclusive check in the room.

    Words right and no voiceprint to fail: the challenge passes with
    speaker_score None, and the authorize that follows is not a deny.
    """

    session_id = await create_session(client)
    await both_legs(
        client,
        session_id,
        voice(LOW, seed=1, duration_s=FULL),
        voice(SHIFTED_4ST, seed=2, duration_s=FULL),
    )

    issued = await client.post(
        f"/v1/sessions/{session_id}/challenges", json={"kind": "phrase"}
    )
    assert issued.status_code == 200, issued.text
    cid = issued.json()["challenge_id"]
    said = await client.post(
        f"/v1/sessions/{session_id}/turns",
        json=turn_body(clock, text=issued.json()["say"]),
    )
    assert said.status_code == 204, said.text
    clock.advance(0.7)

    status = await client.get(f"/v1/sessions/{session_id}/challenges/{cid}")
    assert status.status_code == 200, status.text
    assert status.json()["status"] == "passed", status.json()
    assert status.json()["speaker_score"] is None
    assert status.json().get("reason") is None

    decision = await authorize(client, session_id, "k1", READ)
    assert decision.decision != "deny", decision
    assert "speaker_mismatch" not in decision.reasons


# --------------------------------------------------- (c) the same voice twice


async def test_the_same_voice_on_both_legs_is_not_a_speaker_mismatch(
    client: httpx.AsyncClient, real_app: FastAPI
) -> None:
    session_id = await create_session(client)
    body = await both_legs(
        client,
        session_id,
        voice(LOW, seed=1, duration_s=FULL),
        voice(LOW, seed=2, duration_s=FULL),
    )
    match = next(f for f in body["factors"] if f["name"] == "voice_match")
    assert match["verdict"] == FactorVerdict.PASS.value

    decision = await authorize(client, session_id, "k1", BANK)
    assert decision.decision != "deny"
    assert "speaker_mismatch" not in decision.reasons
    # The consistent legs are the one voice evidence there is, and it passes.
    assert decision.signals.voiceprint is not None
    assert decision.signals.voiceprint.passed is True
    assert real_app.state.v1.registry.get(session_id).locked_code is None
