"""The /v1 authorize policy (ruling section 2.10): a pure function.

`evaluate(view, request, now)` reads a frozen `SessionView` and returns a
`Verdict`: the Decision plus what the caller must write back (the lock, the
pending bind key, the consumed pass). `decide` is the same thing returning
only the Decision. Nothing here touches the session, the clock or a log.

Rules, first match wins:
1. Tier 0 is always allowed, even on a locked session.
2. A locked session denies, reasons ["session_locked", <locking code>, ...].
3. Deny conditions, in order; the first that holds gives reasons[0] and locks
   the session: no_audio, synthetic_speech / automated_caller,
   challenge_failed_replay, speaker_mismatch, otp_failed, callback_denied /
   callback_no_answer, challenges_refused.
4. Tier 1 allows on a recent digits or phrase pass after the latest /caller,
   with low spoof and agent < 0.75.
5. Tier 1 otherwise steps up for a voice challenge.
6. Tier 2 allows on a fresh, bound, unconsumed phrase pass after the latest
   /caller, low spoof, a verified OTP, low text risk, a confirmed callback and
   agent < 0.4, and consumes the pass.
7. Tier 2 otherwise steps up for what is missing, never for nothing.

Reasons carry codes and score lines only: never caller words, phrases, codes,
or phone or bank numbers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from callback_shared.contract import (
    ACTION_TIER,
    Action,
    AuthorizeRequest,
    CallbackResult,
    ChallengeKind,
    ChallengeState,
    Decision,
    DecisionKind,
    DeepfakeSignal,
    DenyReason,
    Factor,
    OtpState,
    Signals,
    TextFlag,
    TextSignal,
    VoiceprintSignal,
)

# Bands (N:730-735).
VOICEPRINT_PASS = 0.62
VOICEPRINT_FAIL = 0.45
SPOOF_LOW = 0.2
SPOOF_HIGH = 0.6
TEXT_LOW = 0.3
TEXT_HIGH = 0.7
AGENT_AUTOMATED = 0.75  # with high spoof: automated_caller; tier 1 needs less
AGENT_TIER2 = 0.4  # tier 2 needs less; at or above, a voice challenge is asked
OTP_FAILED_CHECKS = 3
REFUSED_CHALLENGES = 2
VOICE_FAILS = 2

TIER1_PASS_WINDOW_S = 300.0
TIER2_PASS_WINDOW_S = 60.0
TTL_TIER1_ALLOW = 300
TTL_TIER2_ALLOW = 60
TTL_STEP_UP = 120

TIER0_REASON = "tier0_hold"

_TIER1_PASS_KINDS = frozenset(
    {ChallengeKind.DIGITS, ChallengeKind.PHRASE, ChallengeKind.LIVENESS}
)
_FACTOR_ORDER = (Factor.VOICE_CHALLENGE, Factor.SMS_OTP, Factor.CALLBACK_ON_FILE)


@dataclass(frozen=True)
class ChallengeView:
    """One digits, phrase or liveness challenge, resolved at `now`."""

    challenge_id: str
    kind: ChallengeKind
    status: ChallengeState
    phrase_match: bool | None
    speaker_score: float | None
    reason: str | None
    issued_at: float
    resolved_at: float | None
    bind: str | None
    consumed: bool
    answered: bool  # a user turn arrived after issue


@dataclass(frozen=True)
class SessionView:
    session_id: str
    mock_mode: bool
    listening: bool  # real mode: the hidden listener is in the room
    locked_code: str | None
    voiceprint: float | None  # the passive score
    spoof: float | None
    agent: float | None
    text: float | None
    text_flags: tuple[str, ...]
    deepfake_windows: int
    challenges: tuple[ChallengeView, ...]  # verification kinds, in issue order
    caller_changed_at: float | None
    otp_sent: bool
    otp_verified: bool
    otp_failed_checks: int
    callback: CallbackResult | None


@dataclass(frozen=True)
class Verdict:
    decision: Decision
    # The DenyReason this decision locks the session with, if any.
    lock_code: str | None = None
    # Whether to write pending_bind, and the value to write.
    set_pending_bind: bool = False
    pending_bind: str | None = None
    # The phrase pass a tier 2 allow consumed.
    consume: str | None = None


# --------------------------------------------------------------- helpers


def binding_key(action: Action | str, params: dict[str, Any]) -> str:
    """sha256 of the canonical {action, params}. Also the idempotency fingerprint."""

    canonical = json.dumps(
        {"action": str(action), "params": params},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _agent(view: SessionView) -> float:
    return view.agent if view.agent is not None else 0.0  # null counts as 0


def _spoof_low(view: SessionView) -> bool:
    return view.spoof is not None and view.spoof < SPOOF_LOW


def _text_low(view: SessionView) -> bool:
    return view.text is not None and view.text < TEXT_LOW  # null is not low


def _after_caller(view: SessionView, challenge: ChallengeView) -> bool:
    if view.caller_changed_at is None:
        return True
    return challenge.resolved_at is not None and (
        challenge.resolved_at > view.caller_changed_at
    )


def voice_fails(view: SessionView) -> int:
    fails = 0
    if view.voiceprint is not None and view.voiceprint < VOICEPRINT_FAIL:
        fails += 1
    # A failure with the words right is a speaker failure. A no_audio failure
    # has phrase_match None and never counts.
    fails += sum(
        1
        for c in view.challenges
        if c.status is ChallengeState.FAILED and c.phrase_match is True
    )
    return fails


def refused_challenges(view: SessionView) -> int:
    return sum(
        1
        for c in view.challenges
        if c.status is ChallengeState.EXPIRED and not c.answered
    )


def otp_state(view: SessionView) -> OtpState:
    if view.otp_failed_checks >= OTP_FAILED_CHECKS:
        return OtpState.FAILED
    if view.otp_verified:
        return OtpState.VERIFIED
    if view.otp_sent:
        return OtpState.PENDING
    return OtpState.NOT_APPLICABLE


def deny_code(view: SessionView) -> str | None:
    """Rule 3: the first deny condition that holds, or None."""

    if not view.mock_mode and not view.listening:
        return DenyReason.NO_AUDIO.value
    if view.spoof is not None and view.spoof >= SPOOF_HIGH:
        if _agent(view) >= AGENT_AUTOMATED:
            return DenyReason.AUTOMATED_CALLER.value
        return DenyReason.SYNTHETIC_SPEECH.value
    if any(
        c.status is ChallengeState.FAILED and c.phrase_match is False
        for c in view.challenges
    ):
        return DenyReason.CHALLENGE_FAILED_REPLAY.value
    if voice_fails(view) >= VOICE_FAILS:
        return DenyReason.SPEAKER_MISMATCH.value
    if view.otp_failed_checks >= OTP_FAILED_CHECKS:
        return DenyReason.OTP_FAILED.value
    if view.callback is CallbackResult.DENIED:
        return DenyReason.CALLBACK_DENIED.value
    if view.callback is CallbackResult.NO_ANSWER:
        return DenyReason.CALLBACK_NO_ANSWER.value
    if refused_challenges(view) >= REFUSED_CHALLENGES:
        return DenyReason.CHALLENGES_REFUSED.value
    return None


def score_lines(view: SessionView, *, phrase_mismatch: bool = False) -> list[str]:
    lines: list[str] = []
    if phrase_mismatch:
        lines.append("phrase_match false")
    vp = view.voiceprint
    if vp is not None:
        if vp >= VOICEPRINT_PASS:
            lines.append(f"voiceprint_pass {vp:.2f}≥{VOICEPRINT_PASS:.2f}")
        elif vp < VOICEPRINT_FAIL:
            lines.append(f"voiceprint_fail {vp:.2f}<{VOICEPRINT_FAIL:.2f}")
        else:
            lines.append(f"voiceprint_grey {vp:.2f}")
    if view.spoof is not None:
        band = (
            "low"
            if view.spoof < SPOOF_LOW
            else "mid"
            if view.spoof < SPOOF_HIGH
            else "high"
        )
        lines.append(f"spoof_{band} {view.spoof:.2f}")
    if view.text is not None:
        band = (
            "low"
            if view.text < TEXT_LOW
            else "high"
            if view.text >= TEXT_HIGH
            else "mid"
        )
        lines.append(f"text_{band} {view.text:.2f}")
    if view.text_flags:
        lines.append("text_flags: " + ", ".join(view.text_flags))
    if view.agent is not None and view.agent >= AGENT_TIER2:
        lines.append(f"agent_likelihood {view.agent:.2f}")
    return lines


def risk_score(view: SessionView) -> float:
    parts = [
        value
        for value in (
            view.spoof,
            None if view.voiceprint is None else 1.0 - view.voiceprint,
            view.text,
            None if view.agent is None else 0.8 * view.agent,
        )
        if value is not None
    ]
    return round(max(parts), 4) if parts else 0.0


def signals(view: SessionView) -> Signals:
    latest = next(
        (
            c
            for c in reversed(view.challenges)
            if c.status in (ChallengeState.PASSED, ChallengeState.FAILED)
        ),
        None,
    )
    return Signals(
        voiceprint=(
            None
            if view.voiceprint is None
            else VoiceprintSignal(
                score=view.voiceprint,
                threshold=VOICEPRINT_PASS,
                passed=view.voiceprint >= VOICEPRINT_PASS,
                phrase_match=latest.phrase_match if latest else None,
            )
        ),
        deepfake=(
            None
            if view.spoof is None
            else DeepfakeSignal(spoof_prob=view.spoof, windows=view.deepfake_windows)
        ),
        text=(
            None
            if view.text is None
            else TextSignal(
                risk=view.text,
                # evidence is always "": it would be the caller's words.
                flags=[TextFlag(type=flag, evidence="") for flag in view.text_flags],
            )
        ),
        agent_likelihood=view.agent,
        otp=otp_state(view),
    )


def _tier1_pass(view: SessionView, now: float) -> ChallengeView | None:
    for c in reversed(view.challenges):
        if (
            c.kind in _TIER1_PASS_KINDS
            and c.status is ChallengeState.PASSED
            and c.resolved_at is not None
            and now - c.resolved_at <= TIER1_PASS_WINDOW_S
            and _after_caller(view, c)
        ):
            return c
    return None


def _tier2_pass(view: SessionView, key: str, now: float) -> ChallengeView | None:
    for c in reversed(view.challenges):
        if (
            c.kind is ChallengeKind.PHRASE
            and c.status is ChallengeState.PASSED
            and c.resolved_at is not None
            and now - c.resolved_at <= TIER2_PASS_WINDOW_S
            and _after_caller(view, c)
            and c.bind == key
            and not c.consumed
        ):
            return c
    return None


# ---------------------------------------------------------------- policy


def evaluate(view: SessionView, request: AuthorizeRequest, now: float) -> Verdict:
    action = request.action
    tier = ACTION_TIER[action]
    key = binding_key(action, request.params)

    def make(
        kind: DecisionKind,
        reasons: list[str],
        *,
        ttl: int,
        factors: list[Factor] | None = None,
    ) -> Decision:
        return Decision(
            session_id=view.session_id,
            action=action,
            decision=kind,
            risk_score=risk_score(view),
            required_factors=factors or [],
            signals=signals(view),
            reasons=reasons,
            agent_say="",
            ttl_s=ttl,
        )

    # 1. Tier 0: protective, always allowed.
    if tier == 0:
        return Verdict(make(DecisionKind.ALLOW, [TIER0_REASON], ttl=0))

    # 2. A locked session: the first deny is sticky.
    if view.locked_code is not None:
        lines = score_lines(
            view,
            phrase_mismatch=view.locked_code
            == DenyReason.CHALLENGE_FAILED_REPLAY.value,
        )
        return Verdict(
            make(
                DecisionKind.DENY,
                [DenyReason.SESSION_LOCKED.value, view.locked_code, *lines],
                ttl=0,
            )
        )

    # 3. Deny conditions.
    code = deny_code(view)
    if code is not None:
        lines = score_lines(
            view, phrase_mismatch=code == DenyReason.CHALLENGE_FAILED_REPLAY.value
        )
        return Verdict(make(DecisionKind.DENY, [code, *lines], ttl=0), lock_code=code)

    lines = score_lines(view)

    # 4, 5. Tier 1.
    if tier == 1:
        if (
            _tier1_pass(view, now) is not None
            and _spoof_low(view)
            and _agent(view) < AGENT_AUTOMATED
        ):
            return Verdict(make(DecisionKind.ALLOW, lines, ttl=TTL_TIER1_ALLOW))
        return Verdict(
            make(
                DecisionKind.STEP_UP,
                lines,
                ttl=TTL_STEP_UP,
                factors=[Factor.VOICE_CHALLENGE],
            ),
            set_pending_bind=True,
            pending_bind=None,
        )

    # 6, 7. Tier 2.
    voice_pass = _tier2_pass(view, key, now)
    callback_confirmed = view.callback is CallbackResult.CONFIRMED
    if (
        voice_pass is not None
        and _spoof_low(view)
        and view.otp_verified
        and _text_low(view)
        and callback_confirmed
        and _agent(view) < AGENT_TIER2
    ):
        return Verdict(
            make(DecisionKind.ALLOW, lines, ttl=TTL_TIER2_ALLOW),
            consume=voice_pass.challenge_id,
        )
    missing: set[Factor] = set()
    if voice_pass is None or _agent(view) >= AGENT_TIER2:
        missing.add(Factor.VOICE_CHALLENGE)
    if not view.otp_verified:
        missing.add(Factor.SMS_OTP)
    if not callback_confirmed:
        missing.add(Factor.CALLBACK_ON_FILE)
    # Nothing missing but no allow (spoof mid, or text not low): a step_up is
    # never empty, so ask for the voice again. The agent's round cap then denies.
    factors = [f for f in _FACTOR_ORDER if f in missing] or [Factor.VOICE_CHALLENGE]
    wants_voice = Factor.VOICE_CHALLENGE in factors
    return Verdict(
        make(DecisionKind.STEP_UP, lines, ttl=TTL_STEP_UP, factors=factors),
        set_pending_bind=wants_voice,
        pending_bind=key if wants_voice else None,
    )


def decide(view: SessionView, request: AuthorizeRequest, now: float) -> Decision:
    return evaluate(view, request, now).decision


def tier0_verdict(session_id: str, action: Action) -> Verdict:
    """Rule 1 with nothing else read: holding is always allowed, never locks."""

    return Verdict(
        Decision(
            session_id=session_id,
            action=action,
            decision=DecisionKind.ALLOW,
            reasons=[TIER0_REASON],
            agent_say="",
            ttl_s=0,
        )
    )


def internal_timeout_verdict(session_id: str, action: Action) -> Verdict:
    """Past the deadline, or on any internal exception: deny, and lock.

    Except for a hold: rule 1 outranks every failure, so a tier 0 action gets
    the plain tier 0 allow instead.
    """

    if ACTION_TIER[action] == 0:
        return tier0_verdict(session_id, action)
    return Verdict(
        Decision(
            session_id=session_id,
            action=action,
            decision=DecisionKind.DENY,
            reasons=[DenyReason.INTERNAL_TIMEOUT.value],
            agent_say="",
            ttl_s=0,
        ),
        lock_code=DenyReason.INTERNAL_TIMEOUT.value,
    )
