"""Log hygiene for /v1 (ruling section 2.16, E9).

Call 1 and an enrollment run with sentinel values for the room, the caller
and callee identities, the turn text and the consent answer. Nothing the
detector must never write down may appear in the captured log stream: the
mock code, the number on file, the name searched for, any challenge `say`,
or a sentinel.
"""

from __future__ import annotations

import io
import json
import re

import httpx

from tests.v1.conftest import FakeClock
from tests.v1.demo_driver import DemoCall

ROOM = "room-sentinel-4c1d9e"
CALLER = "sip_sentinel_+15555550188"
CALLEE = "callee-sentinel-7b2f0a"
TURN_TEXT = "turn-sentinel-3e8a51 four seven two nine one three"
CONSENT = "consent-sentinel-9d4b26 yes I agree"

FORBIDDEN = (
    "424242",
    "+15555550100",
    "5555550100",
    "Eric Catalano",
    "Catalano",
    ROOM,
    CALLER,
    CALLEE,
    "turn-sentinel-3e8a51",
    "consent-sentinel-9d4b26",
)


async def test_call_1_and_enrollment_leave_nothing_secret_in_the_log(
    v1: httpx.AsyncClient, clock: FakeClock, log_stream: io.StringIO
) -> None:
    call = DemoCall(
        v1,
        clock,
        "real_cfo",
        room=ROOM,
        caller_identity=CALLER,
        answer_text=TURN_TEXT,
    )
    session_id = await call.start()
    says: list[str] = []

    # Enrollment first, as before the demo.
    assert (await call.sms_otp()).verified is True
    response = await v1.post(
        "/v1/enrollments",
        json={
            "session_id": session_id,
            "user_id": "u_eric",
            "channel": "phone",
            "consent": {
                "version": "2026-11-04",
                "verbatim": CONSENT,
                "ts_ms": clock.ms(),
            },
        },
    )
    enrollment_id = response.json()["enrollment_id"]
    for _ in range(7):
        issued, _ = await call.challenge("enroll")
        says.append(issued.say)
    await v1.post(f"/v1/enrollments/{enrollment_id}/complete")

    # Then Call 1.
    await call.bank_change()
    issued, _ = await call.challenge("phrase")
    says.append(issued.say)
    clock.advance(31)
    await call.sms_otp()
    clock.advance(10)
    await call.callback("confirmed", callee=CALLEE)
    await call.bank_change()
    issued, _ = await call.challenge("phrase")
    says.append(issued.say)
    assert (await call.bank_change()).decision == "allow"
    await call.read_accounts()
    await call.end()

    text = log_stream.getvalue()
    lines = [json.loads(line) for line in text.splitlines()]
    events = {line["event"] for line in lines}
    # The detector did log the call: these are the lines under test.
    assert {"http_request", "v1_session_created", "v1_challenge_issued"} <= events
    assert {"v1_otp_checked", "v1_decision", "v1_enrollment_started"} <= events

    for needle in FORBIDDEN:
        assert needle not in text, needle
    for say in says:
        assert say not in text
        for group in re.findall(r"\d \d \d", say):
            assert group not in text, group
        for word in re.findall(r"\b[a-z]{4,}\b", say.split(":")[-1]):
            if say.startswith("In a sentence"):
                break  # the free-answer question has no secret words
            assert not re.search(rf"\b{word}\b", text), word
