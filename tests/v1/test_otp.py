"""SMS codes on /v1 (ruling section 2.8). Twilio is mocked with MockTransport.

No request in this file leaves the process: the Twilio backend is always
handed an httpx client whose transport is a local handler.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Callable
from urllib.parse import parse_qs

import httpx
import pytest
from fastapi import FastAPI

from detector.v1.otp import (
    MockOtp,
    TwilioVerifyOtp,
    UnavailableOtp,
    otp_backend_from_settings,
)
from tests.v1.conftest import (
    FakeClock,
    base_settings,
    build_app,
    client_for,
    create_session,
)

TWILIO = {
    "twilio_account_sid": "AC00000000000000000000000000000000",
    "twilio_auth_token": "twilio-auth-token-not-real",
    "twilio_verify_service_sid": "VA00000000000000000000000000000000",
}

Handler = Callable[[httpx.Request], httpx.Response]


def twilio_backend(handler: Handler) -> TwilioVerifyOtp:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return TwilioVerifyOtp(
        http,
        TWILIO["twilio_account_sid"],
        TWILIO["twilio_auth_token"],
        TWILIO["twilio_verify_service_sid"],
        owns_http=True,
    )


async def send(client: httpx.AsyncClient, session_id: str) -> httpx.Response:
    return await client.post(f"/v1/sessions/{session_id}/otp/send", json={})


async def check(
    client: httpx.AsyncClient, session_id: str, code: str
) -> httpx.Response:
    return await client.post(
        f"/v1/sessions/{session_id}/otp/check", json={"code": code}
    )


def code_of(response: httpx.Response) -> str:
    return response.json()["error"]["code"]


# ----------------------------------------------------------------- mock


async def test_the_mock_code_passes_and_a_wrong_code_does_not(
    v1: httpx.AsyncClient, v1_app: FastAPI
) -> None:
    session_id = await create_session(v1)
    sent = await send(v1, session_id)
    assert sent.status_code == 200
    assert sent.json() == {"sent_to_last_four": "0100"}
    wrong = await check(v1, session_id, "123456")
    assert wrong.status_code == 200
    assert wrong.json() == {"verified": False}
    right = await check(v1, session_id, "424242")
    assert right.json() == {"verified": True}
    record = v1_app.state.v1.registry.get(session_id).otp
    assert (record.checks, record.failed_checks, record.verified) == (2, 1, True)


async def test_send_ignores_any_body(v1: httpx.AsyncClient) -> None:
    session_id = await create_session(v1)
    response = await v1.post(
        f"/v1/sessions/{session_id}/otp/send", json={"phone": "+15555550199"}
    )
    assert response.json() == {"sent_to_last_four": "0100"}
    session_b = await create_session(v1, room="b")
    assert (await v1.post(f"/v1/sessions/{session_b}/otp/send")).status_code == 200


async def test_a_non_ascii_code_is_just_wrong(v1: httpx.AsyncClient) -> None:
    session_id = await create_session(v1)
    await send(v1, session_id)
    fullwidth = "\uff14\uff12\uff14\uff12\uff14\uff12"  # 424242 in fullwidth digits
    assert (await check(v1, session_id, fullwidth)).json() == {"verified": False}


async def test_a_check_before_any_send_is_409(v1: httpx.AsyncClient) -> None:
    session_id = await create_session(v1)
    response = await check(v1, session_id, "424242")
    assert response.status_code == 409
    assert code_of(response) == "otp_not_sent"


async def test_one_send_per_30s(v1: httpx.AsyncClient, clock: FakeClock) -> None:
    session_id = await create_session(v1)
    assert (await send(v1, session_id)).status_code == 200
    clock.advance(29.9)
    response = await send(v1, session_id)
    assert response.status_code == 429
    assert code_of(response) == "otp_rate_limited"
    clock.advance(0.1)
    assert (await send(v1, session_id)).status_code == 200


async def test_three_sends_in_total(v1: httpx.AsyncClient, clock: FakeClock) -> None:
    session_id = await create_session(v1)
    for _ in range(3):
        assert (await send(v1, session_id)).status_code == 200
        clock.advance(31)
    response = await send(v1, session_id)
    assert response.status_code == 429
    assert code_of(response) == "otp_rate_limited"


async def test_five_checks_then_429(v1: httpx.AsyncClient) -> None:
    session_id = await create_session(v1)
    await send(v1, session_id)
    for _ in range(5):
        assert (await check(v1, session_id, "000000")).status_code == 200
    response = await check(v1, session_id, "424242")
    assert response.status_code == 429
    assert code_of(response) == "otp_rate_limited"


async def test_a_locked_session_refuses_both(
    v1: httpx.AsyncClient, v1_app: FastAPI
) -> None:
    session_id = await create_session(v1)
    await send(v1, session_id)
    v1_app.state.v1.registry.get(session_id).locked_code = "speaker_mismatch"
    for response in (await send(v1, session_id), await check(v1, session_id, "424242")):
        assert response.status_code == 409
        assert code_of(response) == "session_locked"


async def test_ended_sessions_are_409(v1: httpx.AsyncClient) -> None:
    session_id = await create_session(v1)
    await v1.post(f"/v1/sessions/{session_id}/end")
    assert code_of(await send(v1, session_id)) == "session_ended"


# ---------------------------------------------------------- selection


def test_mock_mode_uses_the_mock_code_by_default() -> None:
    assert isinstance(otp_backend_from_settings(base_settings(**TWILIO)), MockOtp)


def test_mock_mode_with_mock_sms_twilio_uses_twilio() -> None:
    settings = base_settings(mock_sms="twilio", **TWILIO)
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(500))
    )
    assert isinstance(otp_backend_from_settings(settings, http=http), TwilioVerifyOtp)


def test_mock_sms_twilio_without_credentials_keeps_the_mock_code() -> None:
    assert isinstance(
        otp_backend_from_settings(base_settings(mock_sms="twilio")), MockOtp
    )


def test_real_mode_with_credentials_uses_twilio() -> None:
    settings = base_settings(mock_mode=False, **TWILIO)
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(500))
    )
    assert isinstance(otp_backend_from_settings(settings, http=http), TwilioVerifyOtp)


def test_real_mode_without_credentials_is_unavailable() -> None:
    settings = base_settings(mock_mode=False)
    assert isinstance(otp_backend_from_settings(settings), UnavailableOtp)


async def test_real_mode_without_credentials_is_503(clock: FakeClock) -> None:
    app = build_app(base_settings(mock_mode=False), clock, io.StringIO())
    async with client_for(app) as client:
        session_id = await create_session(client)
        for response in (
            await send(client, session_id),
            await check(client, session_id, "1"),
        ):
            assert response.status_code == 503
            assert code_of(response) == "otp_unavailable"


# -------------------------------------------------------------- Twilio


class FakeTwilio:
    """Records each request and answers from a script."""

    def __init__(self, check_answer: httpx.Response | Exception) -> None:
        self.requests: list[httpx.Request] = []
        self.check_answer = check_answer
        self.send_answer: httpx.Response | Exception = httpx.Response(
            201, json={"status": "pending"}
        )

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = (
            self.send_answer
            if request.url.path.endswith("/Verifications")
            else self.check_answer
        )
        if isinstance(answer, Exception):
            raise answer
        return answer


async def twilio_app(
    clock: FakeClock, fake: FakeTwilio, stream: io.StringIO
) -> FastAPI:
    app = build_app(base_settings(mock_mode=False, **TWILIO), clock, stream)
    await app.state.v1.otp.aclose()
    app.state.v1.otp = twilio_backend(fake)
    return app


async def test_twilio_send_and_approved_check(clock: FakeClock) -> None:
    fake = FakeTwilio(httpx.Response(200, json={"status": "approved", "valid": True}))
    app = await twilio_app(clock, fake, io.StringIO())
    async with client_for(app) as client:
        session_id = await create_session(client)
        sent = await send(client, session_id)
        assert sent.json() == {"sent_to_last_four": "0100"}
        checked = await check(client, session_id, "918273")
        assert checked.json() == {"verified": True}

    send_request, check_request = fake.requests
    service = TWILIO["twilio_verify_service_sid"]
    assert send_request.method == "POST"
    assert str(send_request.url) == (
        f"https://verify.twilio.com/v2/Services/{service}/Verifications"
    )
    assert parse_qs(send_request.content.decode()) == {
        "To": ["+15555550100"],
        "Channel": ["sms"],
    }
    assert str(check_request.url) == (
        f"https://verify.twilio.com/v2/Services/{service}/VerificationCheck"
    )
    assert parse_qs(check_request.content.decode()) == {
        "To": ["+15555550100"],
        "Code": ["918273"],
    }
    expected_auth = base64.b64encode(
        f"{TWILIO['twilio_account_sid']}:{TWILIO['twilio_auth_token']}".encode()
    ).decode()
    assert send_request.headers["authorization"] == f"Basic {expected_auth}"


@pytest.mark.parametrize(
    ("answer", "verified"),
    [
        (httpx.Response(200, json={"status": "pending"}), False),
        (httpx.Response(200, json={"status": "canceled"}), False),
        (httpx.Response(404, json={"code": 20404}), False),
    ],
)
async def test_twilio_not_approved_is_not_verified(
    clock: FakeClock, answer: httpx.Response, verified: bool
) -> None:
    app = await twilio_app(clock, FakeTwilio(answer), io.StringIO())
    async with client_for(app) as client:
        session_id = await create_session(client)
        await send(client, session_id)
        response = await check(client, session_id, "000000")
    assert response.status_code == 200
    assert response.json() == {"verified": verified}
    assert app.state.v1.registry.get(session_id).otp.failed_checks == 1


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(500, json={"message": "+15555550100 down"}),
        httpx.Response(429, json={}),
        httpx.Response(200, content=b"not json"),
        httpx.ConnectTimeout("timed out"),
    ],
)
async def test_twilio_errors_on_check_are_502(
    clock: FakeClock, answer: httpx.Response | Exception
) -> None:
    stream = io.StringIO()
    app = await twilio_app(clock, FakeTwilio(answer), stream)
    async with client_for(app) as client:
        session_id = await create_session(client)
        await send(client, session_id)
        response = await check(client, session_id, "918273")
    assert response.status_code == 502
    assert code_of(response) == "sms_failed"
    assert "+15555550100" not in response.text
    assert "918273" not in stream.getvalue()
    assert "5555550100" not in stream.getvalue()


@pytest.mark.parametrize(
    "answer",
    [httpx.Response(400, json={"message": "bad To"}), httpx.ConnectError("refused")],
)
async def test_twilio_errors_on_send_are_502_and_do_not_count(
    clock: FakeClock, answer: httpx.Response | Exception
) -> None:
    fake = FakeTwilio(httpx.Response(200, json={"status": "approved"}))
    fake.send_answer = answer
    app = await twilio_app(clock, fake, io.StringIO())
    async with client_for(app) as client:
        session_id = await create_session(client)
        response = await send(client, session_id)
        assert response.status_code == 502
        assert code_of(response) == "sms_failed"
        assert app.state.v1.registry.get(session_id).otp.sent is False
        # Nothing went out, so a check is still "not sent".
        assert code_of(await check(client, session_id, "1")) == "otp_not_sent"


# ----------------------------------------------------------------- logs


async def test_neither_the_code_nor_the_number_reaches_the_log(
    v1: httpx.AsyncClient, log_stream: io.StringIO
) -> None:
    session_id = await create_session(v1)
    await send(v1, session_id)
    await check(v1, session_id, "135792")
    await check(v1, session_id, "424242")
    text = log_stream.getvalue()
    assert "v1_otp_checked" in text
    for needle in ("135792", "424242", "+15555550100", "5555550100"):
        assert needle not in text
