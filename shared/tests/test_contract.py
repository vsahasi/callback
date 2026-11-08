import json
import pathlib

import pytest
from pydantic import ValidationError

from callback_shared.contract import (
    ACTION_TIER,
    AUTHORIZE_SERVER_DEADLINE_S,
    AUTHORIZE_TIMEOUT_S,
    CONTRACT_VERSION,
    CONTRACT_VERSION_HEADER,
    MOCK_OTP_CODE,
    RISK_TOPIC,
    SCENARIO_HEADER,
    Action,
    Decision,
    DecisionKind,
    DenyReason,
    ErrorBody,
    ErrorCode,
    ErrorResponse,
    Factor,
    OtpState,
    Signals,
    Turn,
    contract_json_schema,
)

SPEC_DECISION = {  # the Decision example from the spec's Contract tab, with concrete values
    "session_id": "s1",
    "action": "update_vendor_bank_account",
    "decision": "step_up",
    "risk_score": 0.0,
    "required_factors": ["voice_challenge", "sms_otp", "callback_on_file"],
    "signals": {
        "voiceprint": {
            "score": 0.71,
            "threshold": 0.62,
            "passed": True,
            "phrase_match": True,
        },
        "deepfake": {"spoof_prob": 0.12, "windows": 9},
        "text": {"risk": 0.8, "flags": [{"type": "payment_change", "evidence": "..."}]},
        "agent_likelihood": 0.3,
        "otp": "pending",
    },
    "reasons": ["Payment-detail change requested", "Voiceprint passed"],
    "agent_say": "For security, I'll text a code to the number we have on file.",
    "ttl_s": 120,
}


def test_spec_decision_example_parses():
    d = Decision.model_validate(SPEC_DECISION)
    assert d.decision is DecisionKind.STEP_UP
    assert d.required_factors == [
        Factor.VOICE_CHALLENGE,
        Factor.SMS_OTP,
        Factor.CALLBACK_ON_FILE,
    ]
    assert d.signals.deepfake.spoof_prob == 0.12


def test_unknown_response_fields_are_ignored():
    d = Decision.model_validate({**SPEC_DECISION, "added_later": 1})
    assert d.decision is DecisionKind.STEP_UP
    assert "added_later" not in d.model_dump()


def test_otp_accepts_n_a():
    assert Signals(otp="n/a").otp is OtpState.NOT_APPLICABLE


def test_action_tiers_match_spec():
    by_tier = {t: {a for a, at in ACTION_TIER.items() if at == t} for t in (0, 1, 2)}
    assert by_tier[0] == {Action.HOLD_VENDOR, Action.HOLD_BILL}
    assert by_tier[1] == {
        Action.LIST_VENDORS,
        Action.LIST_VENDOR_ACCOUNTS,
        Action.LIST_BILLS,
    }
    assert by_tier[2] == {
        Action.UPDATE_VENDOR_BANK_ACCOUNT,
        Action.RELEASE_VENDOR_HOLD,
        Action.RELEASE_BILL_HOLD,
    }
    assert set(ACTION_TIER) == set(Action)


def test_turn_times_are_whole_ms():
    with pytest.raises(ValidationError):
        Turn(role="user", text="hi", start_ms=1.5, end_ms=2)


def test_agent_turn_may_have_no_words():
    assert Turn(role="agent", text="Hi", start_ms=1, end_ms=2).words == []


def test_constants():
    assert (
        CONTRACT_VERSION_HEADER,
        SCENARIO_HEADER,
        RISK_TOPIC,
        AUTHORIZE_TIMEOUT_S,
    ) == ("X-Contract-Version", "X-Callback-Scenario", "risk", 3.0)


def test_version_and_new_constants():
    assert CONTRACT_VERSION == "1.1.0"
    assert MOCK_OTP_CODE == "424242"
    assert AUTHORIZE_SERVER_DEADLINE_S == 2.2


def test_error_response_parses():
    r = ErrorResponse.model_validate(
        {
            "contract_version": "1.1.0",
            "error": {"code": "session_not_found", "message": "x"},
        }
    )
    assert r.error.code == "session_not_found"
    assert r.error.detail is None


def test_error_body_accepts_unknown_code():
    assert ErrorBody(code="not_in_the_enum", message="x").code == "not_in_the_enum"


def test_deny_reason_values():
    assert {r.value for r in DenyReason} == {
        "synthetic_speech",
        "automated_caller",
        "challenge_failed_replay",
        "speaker_mismatch",
        "otp_failed",
        "challenges_refused",
        "callback_denied",
        "callback_no_answer",
        "no_audio",
        "internal_timeout",
        "session_locked",
    }


def test_error_code_values():
    assert {c.value for c in ErrorCode} == {
        "invalid_scenario",
        "contract_version_mismatch",
        "unauthorized",
        "session_not_found",
        "user_not_found",
        "challenge_not_found",
        "enrollment_not_found",
        "session_conflict",
        "session_ended",
        "session_locked",
        "idempotency_conflict",
        "otp_not_sent",
        "otp_not_verified",
        "claim_mismatch",
        "no_open_enrollment",
        "validation_error",
        "otp_rate_limited",
        "internal_error",
        "sms_failed",
        "too_many_sessions",
        "otp_unavailable",
    }


def test_schema_has_error_models_but_not_constant_enums():
    models = contract_json_schema()["models"]
    assert "ErrorBody" in models and "ErrorResponse" in models
    assert "DenyReason" not in models and "ErrorCode" not in models


def test_schema_matches_snapshot():
    snap = pathlib.Path(__file__).parent / "snapshots" / "contract_schema.json"
    assert json.loads(snap.read_text()) == contract_json_schema()
