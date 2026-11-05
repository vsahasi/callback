"""Domain errors that map onto the contract's ErrorResponse."""

from __future__ import annotations

from shared.contract import ErrorCode


class DetectorError(Exception):
    """Base class. http_status and code drive the HTTP response."""

    http_status: int = 400
    code: ErrorCode = ErrorCode.VALIDATION_ERROR

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class Unauthorized(DetectorError):
    http_status = 401
    code = ErrorCode.UNAUTHORIZED


class SessionNotFound(DetectorError):
    http_status = 404
    code = ErrorCode.SESSION_NOT_FOUND


class SessionTerminal(DetectorError):
    http_status = 409
    code = ErrorCode.SESSION_TERMINAL


class SequenceConflict(DetectorError):
    http_status = 409
    code = ErrorCode.SEQUENCE_CONFLICT


class OtpExhausted(DetectorError):
    http_status = 429
    code = ErrorCode.OTP_EXHAUSTED


class OtpExpired(DetectorError):
    http_status = 410
    code = ErrorCode.OTP_EXPIRED


class InvalidScenario(DetectorError):
    http_status = 400
    code = ErrorCode.INVALID_SCENARIO
