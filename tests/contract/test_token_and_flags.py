"""The token names and the bank-verify flag (final contract ruling, C23 and C30).

The agent reads DETECTOR_TOKEN, so the detector reads it too, and keeps the
old CALLBACK_API_TOKEN as a fallback. Two different values would turn every
call into a silent 401, so that is refused at startup. The bank-details
endpoint receives a full account number, so it is unmounted unless the flag
is on.
"""

from __future__ import annotations

import dataclasses

import httpx
import pytest

from detector.app import create_app
from detector.config import Settings
from tests.contract.conftest import TEST_TOKEN


@pytest.fixture(autouse=True)
def _clean_token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DETECTOR_TOKEN", raising=False)
    monkeypatch.delenv("CALLBACK_API_TOKEN", raising=False)


def test_detector_token_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DETECTOR_TOKEN", "primary-token-value")
    settings = Settings.from_env()
    assert settings.api_token == "primary-token-value"
    assert settings.api_token_generated is False


def test_detector_token_wins_when_both_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DETECTOR_TOKEN", "same-token-value")
    monkeypatch.setenv("CALLBACK_API_TOKEN", "same-token-value")
    settings = Settings.from_env()
    assert settings.api_token == "same-token-value"
    assert settings.api_token_generated is False


def test_old_name_is_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CALLBACK_API_TOKEN", "legacy-token-value")
    settings = Settings.from_env()
    assert settings.api_token == "legacy-token-value"
    assert settings.api_token_generated is False


def test_a_blank_primary_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DETECTOR_TOKEN", "   ")
    monkeypatch.setenv("CALLBACK_API_TOKEN", "legacy-token-value")
    assert Settings.from_env().api_token == "legacy-token-value"


def test_neither_set_generates_a_token() -> None:
    settings = Settings.from_env()
    assert settings.api_token_generated is True
    assert len(settings.api_token) >= 32


def test_two_different_tokens_refuse_to_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DETECTOR_TOKEN", "token-one")
    monkeypatch.setenv("CALLBACK_API_TOKEN", "token-two")
    with pytest.raises(
        ValueError, match="DETECTOR_TOKEN and CALLBACK_API_TOKEN differ"
    ):
        Settings.from_env()


def test_new_settings_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CALLBACK_BANK_VERIFY_ENABLED",
        "USERS_ON_FILE_PATH",
        "USERS_ON_FILE_JSON",
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "TWILIO_VERIFY_SERVICE_SID",
        "CALLBACK_MOCK_SMS",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = Settings.from_env()
    assert settings.bank_verify_enabled is False
    assert settings.users_on_file_path is None
    assert settings.users_on_file_json is None
    assert settings.twilio_account_sid is None
    assert settings.twilio_auth_token is None
    assert settings.twilio_verify_service_sid is None
    assert settings.mock_sms == ""


def test_new_settings_are_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CALLBACK_BANK_VERIFY_ENABLED", "1")
    monkeypatch.setenv("USERS_ON_FILE_PATH", "users_on_file.example.json")
    monkeypatch.setenv("USERS_ON_FILE_JSON", '{"users": []}')
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC-test")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "twilio-secret-value")
    monkeypatch.setenv("TWILIO_VERIFY_SERVICE_SID", "VA-test")
    monkeypatch.setenv("CALLBACK_MOCK_SMS", "Twilio")
    settings = Settings.from_env()
    assert settings.bank_verify_enabled is True
    assert settings.users_on_file_path == "users_on_file.example.json"
    assert settings.users_on_file_json == '{"users": []}'
    assert settings.twilio_account_sid == "AC-test"
    assert settings.twilio_verify_service_sid == "VA-test"
    assert settings.mock_sms == "twilio"
    # The secret-bearing fields stay out of the repr, so printing the settings
    # cannot leak the Twilio token or the phone numbers in the directory.
    assert "twilio-secret-value" not in repr(settings)
    assert '{"users": []}' not in repr(settings)


def test_unknown_mock_sms_value_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CALLBACK_MOCK_SMS", "carrier-pigeon")
    with pytest.raises(ValueError, match="CALLBACK_MOCK_SMS"):
        Settings.from_env()


def _settings(**overrides: object) -> Settings:
    base = Settings(
        api_token=TEST_TOKEN,
        mock_mode=True,
        session_ttl_seconds=300,
        otp_max_attempts=3,
        otp_ttl_seconds=120,
        mock_jitter_min_ms=0,
        mock_jitter_max_ms=0,
        speech_segments_before_decision=2,
    )
    return dataclasses.replace(base, **overrides)


async def _post_bank_details(settings: Settings) -> httpx.Response:
    transport = httpx.ASGITransport(app=create_app(settings))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://detector.test",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    ) as client:
        return await client.post("/verify/bank-details", json={})


async def test_bank_details_is_404_when_the_flag_is_off() -> None:
    response = await _post_bank_details(_settings())
    assert response.status_code == 404


async def test_bank_details_is_mounted_when_the_flag_is_on() -> None:
    response = await _post_bank_details(_settings(bank_verify_enabled=True))
    # Mounted: the empty body is refused by validation, not by routing.
    assert response.status_code == 422
