"""SMS codes for /v1 (ruling section 2.8): the detector sends and checks.

The agent never sees the number the code goes to, nor a generated code. It
relays only the digits the caller speaks, to `/otp/check`.

Backends:
- `MockOtp` (mock mode, the default): sends nothing, accepts only
  MOCK_OTP_CODE, compared in constant time.
- `TwilioVerifyOtp`: Twilio Verify v2 over httpx, basic auth, 4.0 s timeout.
  Twilio generates the code; this side never generates, stores or logs it.
  Used in real mode, and in mock mode when CALLBACK_MOCK_SMS=twilio and the
  three TWILIO_* values are set (for the recording).
- `UnavailableOtp`: real mode without Twilio. Both calls answer 503
  `otp_unavailable`.

Nothing here logs. A Twilio failure is raised as 502 `sms_failed` carrying
only the HTTP status, never the response body, which can echo the number.
"""

from __future__ import annotations

import secrets
from typing import Protocol

import httpx
from callback_shared.contract import MOCK_OTP_CODE

from detector.config import Settings
from detector.v1.errors import OtpUnavailable, SmsFailed

TWILIO_VERIFY_BASE = "https://verify.twilio.com/v2/Services"
TWILIO_TIMEOUT_S = 4.0


class OtpBackend(Protocol):
    #: False when no code can ever be sent, so /otp/check answers 503 too.
    available: bool

    async def send(self, phone_e164: str) -> None: ...

    async def check(self, phone_e164: str, code: str) -> bool: ...

    async def aclose(self) -> None: ...


class MockOtp:
    available = True

    async def send(self, phone_e164: str) -> None:
        return None  # mock mode texts nothing

    async def check(self, phone_e164: str, code: str) -> bool:
        # Bytes, not str: compare_digest raises TypeError on non-ASCII str.
        return secrets.compare_digest(code.encode("utf-8"), MOCK_OTP_CODE.encode())

    async def aclose(self) -> None:
        return None


class UnavailableOtp:
    available = False

    async def send(self, phone_e164: str) -> None:
        raise OtpUnavailable()

    async def check(self, phone_e164: str, code: str) -> bool:
        raise OtpUnavailable()

    async def aclose(self) -> None:
        return None


class TwilioVerifyOtp:
    available = True

    def __init__(
        self,
        http: httpx.AsyncClient,
        account_sid: str,
        auth_token: str,
        service_sid: str,
        *,
        owns_http: bool = False,
    ) -> None:
        self._http = http
        self._auth = httpx.BasicAuth(account_sid, auth_token)
        self._service_url = f"{TWILIO_VERIFY_BASE}/{service_sid}"
        self._owns_http = owns_http

    async def _post(self, path: str, form: dict[str, str]) -> httpx.Response:
        try:
            return await self._http.post(
                f"{self._service_url}/{path}",
                data=form,
                auth=self._auth,
                timeout=TWILIO_TIMEOUT_S,
            )
        except httpx.HTTPError as exc:
            raise SmsFailed(detail=f"transport error ({type(exc).__name__})") from None

    async def send(self, phone_e164: str) -> None:
        response = await self._post(
            "Verifications", {"To": phone_e164, "Channel": "sms"}
        )
        if not response.is_success:
            raise SmsFailed(detail=f"provider answered HTTP {response.status_code}")

    async def check(self, phone_e164: str, code: str) -> bool:
        response = await self._post(
            "VerificationCheck", {"To": phone_e164, "Code": code}
        )
        if response.status_code == 404:
            # No pending verification: expired, used up or never sent.
            return False
        if not response.is_success:
            raise SmsFailed(detail=f"provider answered HTTP {response.status_code}")
        try:
            status = response.json().get("status")
        except (ValueError, AttributeError):
            raise SmsFailed(detail="provider answer was not JSON") from None
        return status == "approved"

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()


def twilio_configured(settings: Settings) -> bool:
    return bool(
        settings.twilio_account_sid
        and settings.twilio_auth_token
        and settings.twilio_verify_service_sid
    )


def otp_backend_from_settings(
    settings: Settings, *, http: httpx.AsyncClient | None = None
) -> OtpBackend:
    """Pick the backend by the rules in the module docstring."""

    sid = settings.twilio_account_sid
    token = settings.twilio_auth_token
    service = settings.twilio_verify_service_sid
    wanted = not settings.mock_mode or settings.mock_sms == "twilio"
    if sid and token and service and wanted:
        return TwilioVerifyOtp(
            http or httpx.AsyncClient(timeout=TWILIO_TIMEOUT_S),
            sid,
            token,
            service,
            owns_http=http is None,
        )
    if settings.mock_mode:
        return MockOtp()
    return UnavailableOtp()
