"""The /v1 authorize policy as a table of rule cases (ruling section 2.10).

`evaluate` is pure, so every case builds a SessionView by hand.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from callback_shared.contract import (
    AuthorizeRequest,
    CallbackResult,
    ChallengeKind,
    ChallengeState,
    DecisionKind,
    Factor,
    OtpState,
)

from detector.v1.policy import (
    ChallengeView,
    SessionView,
    binding_key,
    decide,
    evaluate,
    internal_timeout_verdict,
    risk_score,
    score_lines,
)

NOW = 1_800_000_100.0
BANK_PARAMS = {
    "vendor_id": "v1",
    "vendor_name": "Acme Supplies",
    "new_account_last4": "6789",
    "new_routing_last4": "0021",
    "account_fingerprint": "0123456789abcdef",
}
KEY = binding_key("update_vendor_bank_account", BANK_PARAMS)

TIER0 = AuthorizeRequest(
    action="hold_vendor",
    params={"vendor_id": "v1", "vendor_name": "Acme Supplies", "reason": "deny"},
    idempotency_key="k0",
)
TIER1 = AuthorizeRequest(
    action="list_vendor_accounts",
    params={"vendor_id": "v1", "vendor_name": "Acme Supplies"},
    idempotency_key="k1",
)
TIER2 = AuthorizeRequest(
    action="update_vendor_bank_account", params=BANK_PARAMS, idempotency_key="k2"
)


def view(**overrides: Any) -> SessionView:
    """real_cfo's signals, nothing done yet."""

    base = SessionView(
        session_id="ses_test",
        mock_mode=True,
        listening=False,
        locked_code=None,
        voiceprint=0.83,
        spoof=0.04,
        agent=0.05,
        text=0.10,
        text_flags=(),
        deepfake_windows=8,
        challenges=(),
        caller_changed_at=None,
        otp_sent=False,
        otp_verified=False,
        otp_failed_checks=0,
        callback=None,
    )
    return dataclasses.replace(base, **overrides)


def challenge(
    status: ChallengeState = ChallengeState.PASSED,
    *,
    kind: ChallengeKind = ChallengeKind.PHRASE,
    phrase_match: bool | None = True,
    resolved_at: float | None = NOW - 10,
    bind: str | None = KEY,
    consumed: bool = False,
    answered: bool = True,
    cid: str = "chl_1",
    reason: str | None = None,
) -> ChallengeView:
    return ChallengeView(
        challenge_id=cid,
        kind=kind,
        status=status,
        phrase_match=phrase_match,
        speaker_score=0.83,
        reason=reason,
        issued_at=(resolved_at or NOW) - 5,
        resolved_at=resolved_at,
        bind=bind,
        consumed=consumed,
        answered=answered,
    )


def ready_for_tier2(**overrides: Any) -> SessionView:
    fields: dict[str, Any] = {
        "challenges": (challenge(),),
        "otp_sent": True,
        "otp_verified": True,
        "callback": CallbackResult.CONFIRMED,
        "caller_changed_at": NOW - 30,
    }
    fields.update(overrides)
    return view(**fields)


FAILED_SPEAKER = challenge(
    ChallengeState.FAILED, phrase_match=True, cid="chl_s", reason="speaker_mismatch"
)
FAILED_REPLAY = challenge(
    ChallengeState.FAILED, phrase_match=False, cid="chl_r", reason="phrase_mismatch"
)
NO_AUDIO = challenge(
    ChallengeState.FAILED, phrase_match=None, cid="chl_n", reason="no_audio"
)


def refused(cid: str) -> ChallengeView:
    return challenge(
        ChallengeState.EXPIRED,
        phrase_match=None,
        answered=False,
        cid=cid,
        reason="no_answer",
    )


# ---------------------------------------------------------------- tier 0


@pytest.mark.parametrize(
    "state",
    [view(), view(locked_code="synthetic_speech"), view(mock_mode=False)],
)
def test_tier0_is_always_allowed(state: SessionView) -> None:
    decision = decide(state, TIER0, NOW)
    assert decision.decision is DecisionKind.ALLOW
    assert decision.ttl_s == 0
    assert decision.reasons == ["tier0_hold"]
    assert decision.required_factors == []


def test_tier0_does_not_lock_or_bind() -> None:
    verdict = evaluate(view(spoof=0.92), TIER0, NOW)
    assert verdict.lock_code is None
    assert verdict.set_pending_bind is False


# ------------------------------------------------------------ the lock


@pytest.mark.parametrize("request_", [TIER1, TIER2])
def test_a_locked_session_denies_with_the_original_code(request_) -> None:
    verdict = evaluate(view(locked_code="synthetic_speech"), request_, NOW)
    assert verdict.decision.decision is DecisionKind.DENY
    assert verdict.decision.reasons[:2] == ["session_locked", "synthetic_speech"]
    assert verdict.decision.ttl_s == 0
    assert verdict.lock_code is None  # already locked


def test_a_lock_beats_a_perfect_tier2() -> None:
    decision = decide(ready_for_tier2(locked_code="otp_failed"), TIER2, NOW)
    assert decision.reasons[:2] == ["session_locked", "otp_failed"]


# ------------------------------------------------------- deny conditions


DENY_CASES: list[tuple[str, SessionView, str]] = [
    ("the engine's confirmed cross-leg mismatch", view(fatal="speaker_mismatch"), "speaker_mismatch"),
    (
        "the mismatch beats a perfect tier 2 and a passing voiceprint",
        ready_for_tier2(fatal="speaker_mismatch", mock_mode=False, listening=True),
        "speaker_mismatch",
    ),
    (
        "the mismatch beats a high spoof",
        view(fatal="speaker_mismatch", mock_mode=False, listening=True, spoof=0.92),
        "speaker_mismatch",
    ),
    ("no audio in real mode", view(mock_mode=False, spoof=0.92), "no_audio"),
    ("high spoof, human agent score", view(spoof=0.92, agent=0.10), "synthetic_speech"),
    ("high spoof, agent score high", view(spoof=0.85, agent=0.90), "automated_caller"),
    ("spoof at the high edge", view(spoof=0.6, agent=0.75), "automated_caller"),
    ("high spoof, null agent", view(spoof=0.7, agent=None), "synthetic_speech"),
    (
        "spoof beats a replay failure",
        view(spoof=0.92, challenges=(FAILED_REPLAY,)),
        "synthetic_speech",
    ),
    ("replay failure", view(spoof=0.45, challenges=(FAILED_REPLAY,)), "challenge_failed_replay"),
    (
        "replay beats speaker mismatch",
        view(voiceprint=0.29, challenges=(FAILED_SPEAKER, FAILED_REPLAY)),
        "challenge_failed_replay",
    ),
    ("passive fail plus a speaker fail", view(voiceprint=0.29, challenges=(FAILED_SPEAKER,)), "speaker_mismatch"),
    (
        "two speaker fails",
        view(challenges=(FAILED_SPEAKER, dataclasses.replace(FAILED_SPEAKER, challenge_id="b"))),
        "speaker_mismatch",
    ),
    (
        "speaker beats otp",
        view(voiceprint=0.29, challenges=(FAILED_SPEAKER,), otp_failed_checks=3),
        "speaker_mismatch",
    ),
    ("three failed codes", view(otp_sent=True, otp_failed_checks=3), "otp_failed"),
    (
        "otp beats callback",
        view(otp_failed_checks=3, callback=CallbackResult.DENIED),
        "otp_failed",
    ),
    ("callback denied", view(callback=CallbackResult.DENIED), "callback_denied"),
    ("callback no answer", view(callback=CallbackResult.NO_ANSWER), "callback_no_answer"),
    (
        "callback beats refusals",
        view(callback=CallbackResult.DENIED, challenges=(refused("a"), refused("b"))),
        "callback_denied",
    ),
    ("two refused challenges", view(challenges=(refused("a"), refused("b"))), "challenges_refused"),
]  # fmt: skip


@pytest.mark.parametrize(
    ("name", "state", "code"), DENY_CASES, ids=[c[0] for c in DENY_CASES]
)
@pytest.mark.parametrize("request_", [TIER1, TIER2], ids=["tier1", "tier2"])
def test_deny_conditions_in_order(
    name: str, state: SessionView, code: str, request_
) -> None:
    verdict = evaluate(state, request_, NOW)
    assert verdict.decision.decision is DecisionKind.DENY, name
    assert verdict.decision.reasons[0] == code
    assert verdict.lock_code == code
    assert verdict.decision.ttl_s == 0
    assert verdict.decision.required_factors == []


NOT_DENIED: list[tuple[str, SessionView]] = [
    ("one passive voice fail", view(voiceprint=0.29)),
    ("one speaker fail", view(challenges=(FAILED_SPEAKER,))),
    ("a no_audio failure is not a voice fail", view(voiceprint=0.29, challenges=(NO_AUDIO,))),
    ("two failed codes", view(otp_failed_checks=2)),
    ("one refusal", view(challenges=(refused("a"),))),
    (
        "a superseded challenge that was answered is not refused",
        view(challenges=(refused("a"), dataclasses.replace(refused("b"), answered=True))),
    ),
    ("mid spoof", view(spoof=0.59)),
    ("high agent, low spoof", view(agent=0.95)),
]  # fmt: skip


@pytest.mark.parametrize(("name", "state"), NOT_DENIED, ids=[c[0] for c in NOT_DENIED])
def test_not_denied(name: str, state: SessionView) -> None:
    assert decide(state, TIER2, NOW).decision is DecisionKind.STEP_UP, name


# ---------------------------------------------------------------- tier 1


def test_tier1_allow() -> None:
    state = view(challenges=(challenge(kind=ChallengeKind.DIGITS, bind=None),))
    decision = decide(state, TIER1, NOW)
    assert decision.decision is DecisionKind.ALLOW
    assert decision.ttl_s == 300
    assert decision.required_factors == []


@pytest.mark.parametrize("kind", [ChallengeKind.PHRASE, ChallengeKind.LIVENESS])
def test_tier1_accepts_phrase_and_liveness(kind: ChallengeKind) -> None:
    state = view(challenges=(challenge(kind=kind),))
    assert decide(state, TIER1, NOW).decision is DecisionKind.ALLOW


def test_tier1_accepts_a_consumed_pass() -> None:
    state = view(challenges=(challenge(consumed=True),))
    assert decide(state, TIER1, NOW).decision is DecisionKind.ALLOW


@pytest.mark.parametrize(
    ("name", "state"),
    [
        ("no pass", view()),
        ("a pass 301 s old", view(challenges=(challenge(resolved_at=NOW - 301),))),
        (
            "a pass before the latest caller",
            view(challenges=(challenge(resolved_at=NOW - 20),), caller_changed_at=NOW - 10),
        ),
        (
            "a pass at the same instant as the caller",
            view(challenges=(challenge(resolved_at=NOW - 10),), caller_changed_at=NOW - 10),
        ),
        ("mid spoof", view(spoof=0.45, challenges=(challenge(),))),
        ("null spoof", view(spoof=None, challenges=(challenge(),))),
        ("agent at 0.75", view(agent=0.75, challenges=(challenge(),))),
        ("an enroll pass", view(challenges=(challenge(kind=ChallengeKind.ENROLL),))),
    ],
)  # fmt: skip
def test_tier1_steps_up_for_a_voice_challenge(name: str, state: SessionView) -> None:
    verdict = evaluate(state, TIER1, NOW)
    assert verdict.decision.decision is DecisionKind.STEP_UP, name
    assert verdict.decision.required_factors == [Factor.VOICE_CHALLENGE]
    assert verdict.decision.ttl_s == 120
    assert verdict.set_pending_bind is True
    assert verdict.pending_bind is None


def test_tier1_pass_exactly_300s_old_still_counts() -> None:
    state = view(challenges=(challenge(resolved_at=NOW - 300),))
    assert decide(state, TIER1, NOW).decision is DecisionKind.ALLOW


# ---------------------------------------------------------------- tier 2


def test_tier2_allow_consumes_the_pass() -> None:
    verdict = evaluate(ready_for_tier2(), TIER2, NOW)
    assert verdict.decision.decision is DecisionKind.ALLOW
    assert verdict.decision.ttl_s == 60
    assert verdict.consume == "chl_1"
    assert verdict.set_pending_bind is False


def test_tier2_first_round_asks_for_everything() -> None:
    verdict = evaluate(view(), TIER2, NOW)
    assert verdict.decision.decision is DecisionKind.STEP_UP
    assert verdict.decision.required_factors == [
        Factor.VOICE_CHALLENGE,
        Factor.SMS_OTP,
        Factor.CALLBACK_ON_FILE,
    ]
    assert verdict.decision.ttl_s == 120
    assert verdict.set_pending_bind is True
    assert verdict.pending_bind == KEY


@pytest.mark.parametrize(
    ("name", "state", "factors"),
    [
        ("stale after /caller", ready_for_tier2(caller_changed_at=NOW - 5), [Factor.VOICE_CHALLENGE]),
        ("bound to another request", ready_for_tier2(challenges=(challenge(bind="other"),)), [Factor.VOICE_CHALLENGE]),
        ("unbound", ready_for_tier2(challenges=(challenge(bind=None),)), [Factor.VOICE_CHALLENGE]),
        ("consumed", ready_for_tier2(challenges=(challenge(consumed=True),)), [Factor.VOICE_CHALLENGE]),
        ("older than 60 s", ready_for_tier2(challenges=(challenge(resolved_at=NOW - 61),), caller_changed_at=NOW - 90), [Factor.VOICE_CHALLENGE]),
        ("a digits pass is not enough", ready_for_tier2(challenges=(challenge(kind=ChallengeKind.DIGITS),)), [Factor.VOICE_CHALLENGE]),
        ("agent at 0.4", ready_for_tier2(agent=0.4), [Factor.VOICE_CHALLENGE]),
        ("no code yet", ready_for_tier2(otp_verified=False), [Factor.SMS_OTP]),
        ("no callback yet", ready_for_tier2(callback=None), [Factor.CALLBACK_ON_FILE]),
        ("code and callback missing", ready_for_tier2(otp_verified=False, callback=None), [Factor.SMS_OTP, Factor.CALLBACK_ON_FILE]),
        ("mid spoof: never empty", ready_for_tier2(spoof=0.45), [Factor.VOICE_CHALLENGE]),
        ("mid text: never empty", ready_for_tier2(text=0.40), [Factor.VOICE_CHALLENGE]),
        ("null text never allows tier 2", ready_for_tier2(text=None), [Factor.VOICE_CHALLENGE]),
    ],
)  # fmt: skip
def test_tier2_steps_up_for_what_is_missing(
    name: str, state: SessionView, factors: list[Factor]
) -> None:
    verdict = evaluate(state, TIER2, NOW)
    assert verdict.decision.decision is DecisionKind.STEP_UP, name
    assert verdict.decision.required_factors == factors
    wants_voice = Factor.VOICE_CHALLENGE in factors
    assert verdict.set_pending_bind is wants_voice
    assert verdict.pending_bind == (KEY if wants_voice else None)


def test_the_binding_key_is_canonical() -> None:
    reordered = dict(reversed(list(BANK_PARAMS.items())))
    assert binding_key("update_vendor_bank_account", reordered) == KEY
    assert (
        binding_key("update_vendor_bank_account", {**BANK_PARAMS, "vendor_id": "v2"})
        != KEY
    )
    assert binding_key("release_vendor_hold", BANK_PARAMS) != KEY
    assert len(KEY) == 64


def test_a_step_up_is_never_empty() -> None:
    states = [
        view(),
        ready_for_tier2(spoof=0.45),
        ready_for_tier2(text=None),
        ready_for_tier2(agent=0.5),
        ready_for_tier2(otp_verified=False),
        view(spoof=None, voiceprint=None, text=None, agent=None),
    ]
    for state in states:
        for request_ in (TIER1, TIER2):
            decision = decide(state, request_, NOW)
            if decision.decision is DecisionKind.STEP_UP:
                assert decision.required_factors


# ---------------------------------------------------------- the fields


def test_reasons_and_score_lines() -> None:
    assert decide(ready_for_tier2(), TIER2, NOW).reasons == [
        "voiceprint_pass 0.83≥0.62",
        "spoof_low 0.04",
        "text_low 0.10",
    ]
    impostor = view(
        voiceprint=0.29, text=0.80, text_flags=("urgency", "verification_refusal")
    )
    assert score_lines(impostor) == [
        "voiceprint_fail 0.29<0.45",
        "spoof_low 0.04",
        "text_high 0.80",
        "text_flags: urgency, verification_refusal",
    ]
    assert score_lines(view(voiceprint=0.50, spoof=0.45, text=0.40, agent=0.90)) == [
        "voiceprint_grey 0.50",
        "spoof_mid 0.45",
        "text_mid 0.40",
        "agent_likelihood 0.90",
    ]
    replay = decide(
        view(spoof=0.45, voiceprint=0.80, text=0.20, challenges=(FAILED_REPLAY,)),
        TIER2,
        NOW,
    )
    assert replay.reasons[:3] == [
        "challenge_failed_replay",
        "phrase_match false",
        "voiceprint_pass 0.80≥0.62",
    ]
    assert "spoof_mid 0.45" in replay.reasons


def test_risk_score_is_the_max_of_the_signals() -> None:
    assert risk_score(view()) == pytest.approx(0.17)
    assert risk_score(view(spoof=0.92)) == pytest.approx(0.92)
    assert risk_score(view(agent=0.90, spoof=0.1)) == pytest.approx(0.72)
    assert risk_score(view(voiceprint=None, spoof=None, text=None, agent=None)) == 0.0


def test_signals() -> None:
    decision = decide(
        ready_for_tier2(text_flags=("urgency",), otp_failed_checks=1), TIER2, NOW
    )
    signals = decision.signals
    assert signals.voiceprint is not None
    assert (signals.voiceprint.score, signals.voiceprint.threshold) == (0.83, 0.62)
    assert signals.voiceprint.passed is True
    assert signals.voiceprint.phrase_match is True
    assert signals.deepfake is not None
    assert (signals.deepfake.spoof_prob, signals.deepfake.windows) == (0.04, 8)
    assert signals.text is not None
    assert [(f.type, f.evidence) for f in signals.text.flags] == [("urgency", "")]
    assert signals.agent_likelihood == 0.05
    assert signals.otp is OtpState.VERIFIED
    assert decision.agent_say == ""
    assert decision.session_id == "ses_test"
    assert decision.action == "update_vendor_bank_account"


@pytest.mark.parametrize(
    ("state", "otp"),
    [
        (view(), OtpState.NOT_APPLICABLE),
        (view(otp_sent=True), OtpState.PENDING),
        (view(otp_sent=True, otp_verified=True), OtpState.VERIFIED),
        (view(otp_sent=True, otp_failed_checks=3), OtpState.FAILED),
    ],
)
def test_signals_otp(state: SessionView, otp: OtpState) -> None:
    assert decide(state, TIER0, NOW).signals.otp is otp


def test_real_mode_has_no_signals_and_denies_no_audio() -> None:
    state = view(mock_mode=False, voiceprint=None, spoof=None, agent=None, text=None)
    decision = decide(state, TIER1, NOW)
    assert decision.reasons == ["no_audio"]
    assert decision.signals.voiceprint is None
    assert decision.signals.deepfake is None
    assert decision.risk_score == 0.0


def test_the_internal_timeout_verdict() -> None:
    verdict = internal_timeout_verdict("ses_x", TIER2.action)
    assert verdict.decision.decision is DecisionKind.DENY
    assert verdict.decision.reasons == ["internal_timeout"]
    assert verdict.lock_code == "internal_timeout"


def test_a_hold_is_never_turned_into_an_internal_timeout_deny() -> None:
    verdict = internal_timeout_verdict("ses_x", TIER0.action)
    assert verdict.decision.decision is DecisionKind.ALLOW
    assert verdict.decision.reasons == ["tier0_hold"]
    assert verdict.lock_code is None
