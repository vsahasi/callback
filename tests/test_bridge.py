"""The /events to /v1 seam.

Real mode used to reach the policy with every signal unset, so tiers 1 and 2
always denied `no_audio`. `detector.bridge` carries what POST /events scored
back to `view()`. The risk in it is polarity: a contract Factor scores 1.0
for "consistent with the claimed identity", while the policy's `spoof` and
`agent` are risks. Get that backwards and a clean caller is refused while a
synthetic one is allowed, so it is worth pinning down.
"""

from __future__ import annotations

import pytest

from detector.bridge import (
    NOT_HEARD,
    SPEAKER_MISMATCH,
    EngineSignals,
    signals_from_engine,
    signals_from_factors,
)
from detector.errors import DetectorError
from shared.contract import Factor, FactorName, FactorVerdict


def factor(
    name: FactorName,
    score: float,
    verdict: FactorVerdict = FactorVerdict.PASS,
    weight: float = 0.2,
) -> Factor:
    return Factor(name=name, score=score, weight=weight, verdict=verdict)


class StubCheck:
    """A `legs.CrossLegCheck` as the bridge sees it: one flag."""

    def __init__(self, confirmed: bool) -> None:
        self.confirmed = confirmed


class StubEngine:
    """Enough of DetectorEngine for the bridge: factors_for and cross_leg_for."""

    def __init__(
        self,
        factors: list[Factor] | None = None,
        *,
        raises: bool = False,
        cross_leg: StubCheck | None = None,
    ):
        self._factors = factors or []
        self._raises = raises
        self._cross_leg = cross_leg

    def factors_for(self, session_id: str) -> object:
        if self._raises:
            raise DetectorError("session_not_found")
        return type("Answer", (), {"factors": self._factors})()

    def cross_leg_for(self, session_id: str) -> object | None:
        return self._cross_leg


# --------------------------------------------------------------- polarity


def test_voiceprint_passes_through_unchanged() -> None:
    """The policy's voiceprint is high-good, same as a Factor score."""

    signals = signals_from_factors([factor(FactorName.VOICE_MATCH, 0.83)])
    assert signals.voiceprint == pytest.approx(0.83)


def test_spoof_is_inverted() -> None:
    """A synthetic_speech score of 0.08 is a spoof risk of 0.92."""

    signals = signals_from_factors([factor(FactorName.SYNTHETIC_SPEECH, 0.08)])
    assert signals.spoof == pytest.approx(0.92)


def test_agent_is_inverted() -> None:
    signals = signals_from_factors([factor(FactorName.AGENT_BEHAVIOUR, 0.10)])
    assert signals.agent == pytest.approx(0.90)


def test_a_clean_caller_reads_as_low_risk_end_to_end() -> None:
    """The real_cfo shape: voiceprint high, spoof and agent low."""

    signals = signals_from_factors(
        [
            factor(FactorName.VOICE_MATCH, 0.83),
            factor(FactorName.SYNTHETIC_SPEECH, 0.96),
            factor(FactorName.AGENT_BEHAVIOUR, 0.95),
        ]
    )
    assert signals.voiceprint == pytest.approx(0.83)  # above VOICEPRINT_FAIL 0.45
    assert signals.spoof == pytest.approx(0.04)  # below SPOOF_LOW 0.2
    assert signals.agent == pytest.approx(0.05)


# ------------------------------------------------------- the worst signal


def test_spoof_takes_the_worst_of_the_three() -> None:
    """A clean synthesis check must not average away a failed replay."""

    signals = signals_from_factors(
        [
            factor(FactorName.SYNTHETIC_SPEECH, 0.95),
            factor(FactorName.REPLAY, 0.12, FactorVerdict.FAIL),
            factor(FactorName.LIVENESS, 0.80),
        ]
    )
    assert signals.spoof == pytest.approx(0.88)  # from replay, not synthesis


def test_liveness_alone_can_drive_spoof() -> None:
    signals = signals_from_factors([factor(FactorName.LIVENESS, 0.30)])
    assert signals.spoof == pytest.approx(0.70)


# ------------------------------------------------------------- not heard


def test_no_factors_is_not_heard() -> None:
    assert signals_from_factors([]) is NOT_HEARD


def test_not_run_factors_are_ignored() -> None:
    """A signal that could not be measured is absent, never a zero."""

    signals = signals_from_factors(
        [
            factor(FactorName.VOICE_MATCH, 0.0, FactorVerdict.NOT_RUN),
            factor(FactorName.SYNTHETIC_SPEECH, 0.90),
        ]
    )
    assert signals.heard is True
    assert signals.voiceprint is None  # not 0.0, which would fail the gate
    assert signals.spoof == pytest.approx(0.10)


def test_everything_not_run_is_not_heard() -> None:
    """Nothing measurable keeps the no_audio deny rather than inventing one."""

    signals = signals_from_factors(
        [factor(FactorName.VOICE_MATCH, 0.0, FactorVerdict.NOT_RUN)]
    )
    assert signals is NOT_HEARD
    assert signals.heard is False


def test_heard_is_true_once_anything_scored() -> None:
    signals = signals_from_factors([factor(FactorName.LIVENESS, 0.5)])
    assert signals.heard is True


# ------------------------------------------------------ reading an engine


def test_no_engine_is_not_heard() -> None:
    """Every /v1 test that builds a service by hand passes None."""

    assert signals_from_engine(None, "sess-1") is NOT_HEARD


def test_unknown_session_is_not_heard() -> None:
    """Nothing posted under this id yet: fail closed, do not raise."""

    assert signals_from_engine(StubEngine(raises=True), "sess-1") is NOT_HEARD


def test_engine_factors_reach_the_signals() -> None:
    engine = StubEngine([factor(FactorName.VOICE_MATCH, 0.29, FactorVerdict.FAIL)])
    signals = signals_from_engine(engine, "sess-1")
    assert signals.heard is True
    assert signals.voiceprint == pytest.approx(0.29)  # below VOICEPRINT_FAIL


def test_not_heard_is_a_shared_empty_value() -> None:
    assert EngineSignals() == NOT_HEARD
    assert NOT_HEARD.voiceprint is None
    assert NOT_HEARD.spoof is None
    assert NOT_HEARD.agent is None
    assert NOT_HEARD.fatal is None


# ------------------------------------------------- the confirmed mismatch


def test_a_confirmed_cross_leg_mismatch_is_a_fatal_signal() -> None:
    """The engine's own deny reaches /v1 by name, not as a low voiceprint.

    The policy used to read the mismatch only as `voiceprint < 0.45`, one
    voice fail of the two it wants, and stepped up for the very factors the
    cross-leg check says cannot rescue the call.
    """

    engine = StubEngine(
        [factor(FactorName.VOICE_MATCH, 0.16, FactorVerdict.FAIL)],
        cross_leg=StubCheck(confirmed=True),
    )
    signals = signals_from_engine(engine, "sess-1")
    assert signals.heard is True
    assert signals.fatal == SPEAKER_MISMATCH == "speaker_mismatch"
    assert signals.voiceprint == pytest.approx(0.16)


def test_an_unconfirmed_cross_leg_check_is_not_fatal() -> None:
    """Inconclusive, or a FAIL on thin legs: recorded, never a deny by itself."""

    engine = StubEngine(
        [factor(FactorName.SYNTHETIC_SPEECH, 0.9)],
        cross_leg=StubCheck(confirmed=False),
    )
    signals = signals_from_engine(engine, "sess-1")
    assert signals.heard is True
    assert signals.fatal is None
    assert signals.voiceprint is None


def test_no_cross_leg_check_is_not_fatal() -> None:
    engine = StubEngine([factor(FactorName.SYNTHETIC_SPEECH, 0.9)])
    assert signals_from_engine(engine, "sess-1").fatal is None


def test_the_fatal_signal_alone_counts_as_heard() -> None:
    """A confirmed mismatch means both legs were measured, whatever the list says."""

    signals = signals_from_factors([], speaker_mismatch=True)
    assert signals.heard is True
    assert signals.fatal == SPEAKER_MISMATCH
    assert signals_from_factors([], speaker_mismatch=False) is NOT_HEARD
