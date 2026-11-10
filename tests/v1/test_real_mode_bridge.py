"""Real mode reaches the policy with signals instead of no_audio.

Before the bridge, `view()` sent every signal as None whenever the mock
script was absent, and `listening` was hard-coded False, so real mode denied
every tier 1 and 2 authorize with `no_audio`. These tests pin the two halves
of the change: that nothing heard still denies, and that something heard
arrives at the policy with the right polarity.
"""

from __future__ import annotations

import pytest
from callback_shared.contract import Channel, SessionCreate

from detector.enrolment import InMemoryVoicePrintStore
from detector.errors import DetectorError
from detector.v1.service import V1Service
from shared.contract import Factor, FactorName, FactorVerdict
from tests.v1.conftest import FakeClock, base_settings

USER = "u_eric"


def factor(name: FactorName, score: float, verdict: FactorVerdict) -> Factor:
    return Factor(name=name, score=score, weight=0.2, verdict=verdict)


class StubEngine:
    def __init__(self, factors: list[Factor] | None = None, *, known: bool = True):
        self._factors = factors or []
        self._known = known

    def factors_for(self, session_id: str) -> object:
        if not self._known:
            raise DetectorError("session_not_found")
        return type("Answer", (), {"factors": self._factors})()


def real_mode_service(engine: object | None) -> tuple[V1Service, FakeClock]:
    clock = FakeClock()
    service = V1Service(
        base_settings(mock_mode=False),
        voiceprints=InMemoryVoicePrintStore(),
        clock=clock,
        engine=engine,
    )
    return service, clock


def a_session(service: V1Service, clock: FakeClock):
    created = service.create_session(
        SessionCreate(
            room=f"room-{clock.now}",
            caller_identity="caller-0",
            claimed_user_id=USER,
            channel=Channel.PHONE,
        ),
        None,  # scenario_header: real mode ignores it anyway
    )
    return service.registry.get(created.session_id)


def test_real_mode_without_an_engine_still_hears_nothing() -> None:
    """The pre-bridge behaviour, which is what keeps no_audio fail closed."""

    service, clock = real_mode_service(None)
    session = a_session(service, clock)
    view = service.view(session, clock.now)

    assert view.listening is False
    assert view.voiceprint is None
    assert view.spoof is None


def test_an_unknown_session_hears_nothing() -> None:
    """The agent has not posted to /events under this id yet."""

    service, clock = real_mode_service(StubEngine(known=False))
    session = a_session(service, clock)

    assert service.view(session, clock.now).listening is False


def test_scored_segments_make_the_detector_listening() -> None:
    """One measured signal is enough to stop denying no_audio."""

    engine = StubEngine([factor(FactorName.SYNTHETIC_SPEECH, 0.90, FactorVerdict.PASS)])
    service, clock = real_mode_service(engine)
    session = a_session(service, clock)
    view = service.view(session, clock.now)

    assert view.listening is True
    assert view.spoof == pytest.approx(0.10)


def test_the_impostor_shape_reaches_the_policy() -> None:
    """A different human: voiceprint fails, nothing else does."""

    engine = StubEngine(
        [
            factor(FactorName.VOICE_MATCH, 0.29, FactorVerdict.FAIL),
            factor(FactorName.SYNTHETIC_SPEECH, 0.95, FactorVerdict.PASS),
        ]
    )
    service, clock = real_mode_service(engine)
    session = a_session(service, clock)
    view = service.view(session, clock.now)

    assert view.listening is True
    assert view.voiceprint == pytest.approx(0.29)  # under VOICEPRINT_FAIL 0.45
    assert view.spoof == pytest.approx(0.05)  # under SPOOF_LOW 0.2


def test_the_synthetic_caller_shape_reaches_the_policy() -> None:
    """A TTS caller: the voice may match, the synthesis check does not."""

    engine = StubEngine(
        [
            factor(FactorName.VOICE_MATCH, 0.78, FactorVerdict.PASS),
            factor(FactorName.SYNTHETIC_SPEECH, 0.08, FactorVerdict.FAIL),
        ]
    )
    service, clock = real_mode_service(engine)
    session = a_session(service, clock)
    view = service.view(session, clock.now)

    assert view.voiceprint == pytest.approx(0.78)  # the voiceprint passes
    assert view.spoof == pytest.approx(0.92)  # at or over SPOOF_HIGH 0.6


def test_mock_mode_never_consults_the_engine() -> None:
    """A scripted session behaves exactly as it did before the bridge."""

    class Exploding:
        def factors_for(self, session_id: str) -> object:
            raise AssertionError("mock mode must not read the engine")

    clock = FakeClock()
    service = V1Service(
        base_settings(mock_mode=True),
        voiceprints=InMemoryVoicePrintStore(),
        clock=clock,
        engine=Exploding(),
    )
    session = a_session(service, clock)
    view = service.view(session, clock.now)

    assert view.listening is False
    assert view.mock_mode is True
