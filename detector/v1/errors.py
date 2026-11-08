"""Errors the /v1 routes raise, one class per code in the contract's error table.

Each one is a `DetectorError`, so the app's existing handler renders it. Under
/v1 that handler renders callback_shared's `ErrorResponse` instead of the
internal one (see detector/app.py). The default messages never repeat a value
from the request: a 404 for a user id does not say which id, and nothing here
can carry a name, a phone number, a code or a challenge phrase.
"""

from __future__ import annotations

from callback_shared.contract import ErrorCode as WireErrorCode

from detector.errors import DetectorError


class V1Error(DetectorError):
    """Base class. `http_status` and `code` drive the response."""

    http_status: int = 500
    code: str = WireErrorCode.INTERNAL_ERROR.value  # type: ignore[assignment]
    default_message: str = "internal error"

    def __init__(self, message: str | None = None, detail: str | None = None) -> None:
        super().__init__(message or self.default_message, detail)


# --------------------------------------------------------------------- 400


class InvalidScenario(V1Error):
    http_status = 400
    code = WireErrorCode.INVALID_SCENARIO.value
    default_message = "X-Callback-Scenario names no known scenario"


class ContractVersionMismatch(V1Error):
    http_status = 400
    code = WireErrorCode.CONTRACT_VERSION_MISMATCH.value
    default_message = "X-Contract-Version is missing or has another major version"


# --------------------------------------------------------------------- 401


class Unauthorized(V1Error):
    http_status = 401
    code = WireErrorCode.UNAUTHORIZED.value
    default_message = "missing or invalid bearer token"


# --------------------------------------------------------------------- 404


class SessionNotFound(V1Error):
    http_status = 404
    code = WireErrorCode.SESSION_NOT_FOUND.value
    default_message = "no session with that id"


class UserNotFound(V1Error):
    http_status = 404
    code = WireErrorCode.USER_NOT_FOUND.value
    default_message = "no user on file with that id"


class ChallengeNotFound(V1Error):
    http_status = 404
    code = WireErrorCode.CHALLENGE_NOT_FOUND.value
    default_message = "no challenge with that id in this session"


class EnrollmentNotFound(V1Error):
    http_status = 404
    code = WireErrorCode.ENROLLMENT_NOT_FOUND.value
    default_message = "no enrollment with that id"


# --------------------------------------------------------------------- 409


class SessionConflict(V1Error):
    http_status = 409
    code = WireErrorCode.SESSION_CONFLICT.value
    default_message = "this room already has a live session for another claim"


class SessionEnded(V1Error):
    http_status = 409
    code = WireErrorCode.SESSION_ENDED.value
    default_message = "the session has ended"


class SessionLocked(V1Error):
    http_status = 409
    code = WireErrorCode.SESSION_LOCKED.value
    default_message = "the session is locked by a deny; only holds are served"


class IdempotencyConflict(V1Error):
    http_status = 409
    code = WireErrorCode.IDEMPOTENCY_CONFLICT.value
    default_message = "idempotency_key was already used for a different request"


class OtpNotSent(V1Error):
    http_status = 409
    code = WireErrorCode.OTP_NOT_SENT.value
    default_message = "no code has been sent in this session"


class OtpNotVerified(V1Error):
    http_status = 409
    code = WireErrorCode.OTP_NOT_VERIFIED.value
    default_message = "enrollment needs a verified code in this session first"


class ClaimMismatch(V1Error):
    http_status = 409
    code = WireErrorCode.CLAIM_MISMATCH.value
    default_message = "user_id or channel differs from the session"


class NoOpenEnrollment(V1Error):
    http_status = 409
    code = WireErrorCode.NO_OPEN_ENROLLMENT.value
    default_message = "the session has no open enrollment prompt"


# --------------------------------------------------------------------- 422


class V1ValidationError(V1Error):
    http_status = 422
    code = WireErrorCode.VALIDATION_ERROR.value
    default_message = "request failed contract validation"


# --------------------------------------------------------------------- 429


class OtpRateLimited(V1Error):
    http_status = 429
    code = WireErrorCode.OTP_RATE_LIMITED.value
    default_message = "too many code sends or checks in this session"


# --------------------------------------------------------------------- 5xx


class InternalError(V1Error):
    http_status = 500
    code = WireErrorCode.INTERNAL_ERROR.value
    default_message = "internal error"


class SmsFailed(V1Error):
    http_status = 502
    code = WireErrorCode.SMS_FAILED.value
    default_message = "the SMS provider failed"


class TooManySessions(V1Error):
    http_status = 503
    code = WireErrorCode.TOO_MANY_SESSIONS.value
    default_message = "the detector is at its live session limit"


class OtpUnavailable(V1Error):
    http_status = 503
    code = WireErrorCode.OTP_UNAVAILABLE.value
    default_message = "SMS codes are not configured on this detector"
