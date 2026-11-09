"""A /v1 enrollment that writes a real voice print (ruling section 2.15).

"Real-mode prints go into P3's VoicePrintStore under the identity
'{user_id}:{channel}'. They are prints, never audio." And "Mock mode always
enrolls and writes no print."

Why these tests force two things
--------------------------------
Two inputs to a real enrollment do not exist in a unit test, and one of them
does not exist in this deployment at all:

* **A verified code.** Real mode sends it through Twilio Verify. The tests
  set `session.verified` instead.
* **A passed enroll prompt.** `_outcome_for` returns `no_audio` for every
  challenge in real mode, because the hidden listener that would hear the
  caller is stretch work and is not in Round 2. So no prompt can pass, no
  enrollment can reach the five passes it needs, and the write path below is
  unreachable through the API today. The tests set each challenge's outcome
  to `passed`, which is exactly the decision the listener will make.

Everything between those two inputs and the stored print is the real thing:
the consent log, the enrollment state machine, prompt advance and retry
counting, the five-of-seven rule, combining one print per prompt, the quality
gate, and the key the print lands under.

An `enroll` prompt passing needs no transcription, which is worth knowing
before reading the forcing above as a cheat. `detector/v1/challenges.py` says
it: "`enroll` is none of these: it records a voice, it does not verify one."
A listener that hears the caller has everything an enroll prompt needs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from callback_shared.contract import (
    ChallengeKind,
    ChallengeState,
    Channel,
    Consent,
    EnrollmentStart,
    SessionCreate,
)

from detector.analysis.synthetic import SpeechStyle, live_speech
from detector.enrolment import (
    EnrolmentStoreFull,
    InMemoryVoicePrintStore,
    VoicePrintStore,
)
from detector.identity import candidate_identities, store_identity
from detector.v1.enrollments import PASSES_TO_ENROLL, print_identity
from detector.v1.scenarios import ChallengeOutcome
from detector.v1.service import V1Service
from detector.v1.sessions import V1Session
from shared.enrolment import EnrolmentQuality
from tests.v1.conftest import FakeClock, base_settings

RATE = 8000
USER = "u_eric"
PASSED = ChallengeOutcome(ChallengeState.PASSED, None, None, None)

CONSENT = Consent(
    version="2026-11-04", verbatim="yes I agree, sentinel-consent-5f3a", ts_ms=1
)


class Clips:
    """A `ClipSource` over canned audio. What the listener will be.

    One clip per challenge, each a different realisation of the same talker,
    which is what seven prompts from one person actually sound like.
    """

    def __init__(self, *, rate: int = RATE, style: SpeechStyle | None = None) -> None:
        self.rate = rate
        self.style = style or SpeechStyle()
        self.asked: list[str] = []
        self.silent: set[str] = set()

    def clip_for(self, challenge_id: str) -> tuple[Any, int] | None:
        self.asked.append(challenge_id)
        if challenge_id in self.silent:
            return None
        seed = len(self.asked) * 7
        return live_speech(3.0, self.rate, seed=seed, style=self.style), self.rate


class NoClips:
    """Heard nothing at all, for every challenge."""

    def clip_for(self, challenge_id: str) -> tuple[Any, int] | None:
        return None


def service_for(
    *,
    clips: Any = None,
    store: VoicePrintStore | None = None,
    mock_mode: bool = False,
    clock: FakeClock | None = None,
) -> tuple[V1Service, FakeClock, VoicePrintStore]:
    ticker = clock or FakeClock()
    # `store if store is not None`, never `store or ...`: a VoicePrintStore
    # defines __len__, so an empty one is falsy and `or` would quietly hand
    # the service a different store from the one the test asserts on. The
    # same trap is documented on DetectorEngine for the session store.
    prints = store if store is not None else InMemoryVoicePrintStore()
    service = V1Service(
        base_settings(mock_mode=mock_mode),
        voiceprints=prints,
        clock=ticker,
        clips=clips,
    )
    return service, ticker, prints


def a_session(service: V1Service, clock: FakeClock) -> V1Session:
    created = service.create_session(
        SessionCreate(
            room=f"room-{clock.now}",
            caller_identity="caller-0",
            claimed_user_id=USER,
            channel=Channel.PHONE,
        ),
        None,
    )
    session = service.registry.lookup(created.session_id)
    # Twilio Verify is not in a unit test. The gate itself is covered by
    # tests/v1/test_enrollments.py.
    session.otp.verified = True
    return session


def drive(
    service: V1Service,
    clock: FakeClock,
    session: V1Session,
    *,
    passes: int = PASSES_TO_ENROLL,
    failures: int = 0,
) -> str:
    """Run an enrollment, passing `passes` prompts and failing `failures`."""

    started = service.start_enrollment(
        EnrollmentStart(
            session_id=session.session_id,
            user_id=USER,
            channel=Channel.PHONE,
            consent=CONSENT,
        )
    )
    for index in range(passes + failures):
        issued = service.create_challenge(session.session_id, ChallengeKind.ENROLL)
        record = session.challenges[issued.challenge_id]
        if index < passes:
            record.outcome = PASSED
        record.answered_at = clock.now
        clock.advance(1.0)
    return started.enrollment_id


def stored(prints: VoicePrintStore, channel: str = "phone"):
    return prints.get(store_identity(USER, channel))


# ------------------------------------------------------- it writes a print


def test_an_enrolled_caller_gets_a_print_under_the_ruled_key() -> None:
    service, clock, prints = service_for(clips=Clips())
    session = a_session(service, clock)
    enrollment_id = drive(service, clock, session)

    result = service.complete_enrollment(enrollment_id)

    assert result.status == "enrolled"
    assert result.prompts_passed >= PASSES_TO_ENROLL

    record = stored(prints)
    assert record is not None
    assert record.channel == "phone"
    assert record.source == "listener"
    assert record.voice.has_pitch
    assert record.voice.has_spectrum
    # The key the ruling names, and the typed shorthand agreeing with it.
    assert print_identity(USER, Channel.PHONE) == f"{USER}:phone"
    assert prints.get(print_identity(USER, Channel.PHONE)) is not None


def test_the_print_is_combined_from_every_passed_prompt() -> None:
    """Five three-second clips beat one. More voiced speech is the only thing
    that makes this comparison less noisy."""

    service, clock, prints = service_for(clips=Clips())
    session = a_session(service, clock)
    service.complete_enrollment(drive(service, clock, session, passes=5))

    record = stored(prints)
    assert record is not None
    assert record.voice.duration_s == pytest.approx(15.0, abs=0.5)
    assert record.voice.voiced_seconds > 4.0
    assert record.quality is EnrolmentQuality.GOOD
    assert "5 of 7 prompts" in (record.label or "")


def test_a_failed_prompt_contributes_nothing() -> None:
    service, clock, prints = service_for(clips=Clips())
    session = a_session(service, clock)
    enrollment_id = drive(service, clock, session, passes=5, failures=2)

    assert service.complete_enrollment(enrollment_id).prompts_passed == 5
    record = stored(prints)
    assert record is not None
    assert "5 of 7 prompts" in (record.label or "")


def test_the_print_the_pipeline_looks_for_is_the_one_that_was_written() -> None:
    """The integration this exists to make real.

    `detector.pipeline` reads a phone-channel print first. Before the key
    language was shared, it read the bare identity only, so a print written
    here was invisible to the only code that reads one.
    """

    service, clock, prints = service_for(clips=Clips())
    session = a_session(service, clock)
    service.complete_enrollment(drive(service, clock, session))

    found = next(
        (prints.get(key) for key in candidate_identities(USER) if prints.get(key)),
        None,
    )
    assert found is not None
    assert found.channel == "phone"


# --------------------------------------------------- and when it does not


def test_mock_mode_enrolls_and_writes_no_print() -> None:
    """Ruled, and not an optimisation: a scripted pass heard no voice."""

    service, clock, prints = service_for(clips=Clips(), mock_mode=True)
    session = a_session(service, clock)
    enrollment_id = drive(service, clock, session)

    assert service.complete_enrollment(enrollment_id).status == "enrolled"
    assert len(prints) == 0


def test_no_listener_writes_no_print_and_says_so() -> None:
    """Every deployment today. The write path is unreachable, not broken."""

    service, clock, prints = service_for(clips=None)
    session = a_session(service, clock)
    enrollment_id = drive(service, clock, session)

    assert service.complete_enrollment(enrollment_id).status == "enrolled"
    assert len(prints) == 0


def test_a_listener_that_heard_nothing_writes_no_print() -> None:
    service, clock, prints = service_for(clips=NoClips())
    session = a_session(service, clock)
    enrollment_id = drive(service, clock, session)

    assert service.complete_enrollment(enrollment_id).status == "enrolled"
    assert len(prints) == 0


def test_a_failed_enrollment_writes_no_print() -> None:
    """Four passes is not five, so there is nothing to enrol."""

    service, clock, prints = service_for(clips=Clips())
    session = a_session(service, clock)
    enrollment_id = drive(service, clock, session, passes=4, failures=3)

    assert service.complete_enrollment(enrollment_id).status == "failed"
    assert len(prints) == 0


def test_silence_on_every_prompt_writes_no_print() -> None:
    """A print with nothing measurable in it would make every later
    comparison inconclusive while the vendor looked enrolled."""

    class Silence:
        def clip_for(self, challenge_id: str) -> tuple[Any, int] | None:
            return np.zeros(3 * RATE), RATE

    service, clock, prints = service_for(clips=Silence())
    session = a_session(service, clock)
    enrollment_id = drive(service, clock, session)

    assert service.complete_enrollment(enrollment_id).status == "enrolled"
    assert len(prints) == 0


def test_one_unreadable_clip_does_not_lose_the_enrolment() -> None:
    clips = Clips()
    service, clock, prints = service_for(clips=clips)
    session = a_session(service, clock)
    enrollment_id = drive(service, clock, session)
    # Silence the first challenge the listener is asked about.
    clips.silent.add(next(iter(session.challenges)))

    assert service.complete_enrollment(enrollment_id).status == "enrolled"
    assert stored(prints) is not None


def test_completing_twice_does_not_write_twice() -> None:
    service, clock, prints = service_for(clips=Clips())
    session = a_session(service, clock)
    enrollment_id = drive(service, clock, session)

    first = service.complete_enrollment(enrollment_id)
    before = stored(prints)
    second = service.complete_enrollment(enrollment_id)

    assert first == second
    after = stored(prints)
    assert before is not None
    assert after is not None
    assert after.enrolled_at == before.enrolled_at


def test_a_store_that_refuses_does_not_break_the_answer() -> None:
    """The ruled answer is decided by prompts passed, not by the store.

    So a refusal cannot change what the agent is told, which is exactly why
    it has to be logged. `test_every_path_without_a_print_says_why` holds the
    log line.
    """

    class Full(InMemoryVoicePrintStore):
        def put(self, enrolment: Any, *, replace: bool) -> None:
            raise EnrolmentStoreFull("the store is full", "raise the cap")

    service, clock, prints = service_for(clips=Clips(), store=Full())
    session = a_session(service, clock)
    enrollment_id = drive(service, clock, session)

    assert service.complete_enrollment(enrollment_id).status == "enrolled"
    assert len(prints) == 0


# ----------------------------------------------------------------- erasure


def test_erasing_removes_the_print_the_enrollment_wrote() -> None:
    service, clock, prints = service_for(clips=Clips())
    session = a_session(service, clock)
    service.complete_enrollment(drive(service, clock, session))
    assert stored(prints) is not None

    service.delete_enrollment(USER)

    assert stored(prints) is None
    assert stored(prints, "browser") is None


# --------------------------------------------------- channels stay separate


def test_a_browser_enrolment_does_not_answer_for_the_phone() -> None:
    """Each channel enrolls separately, and for a reason: a handset and a
    browser microphone put a different response in front of one larynx."""

    service, clock, prints = service_for(clips=Clips())
    created = service.create_session(
        SessionCreate(
            room="room-browser",
            caller_identity="caller-0",
            claimed_user_id=USER,
            channel=Channel.BROWSER,
        ),
        None,
    )
    session = service.registry.lookup(created.session_id)
    session.otp.verified = True
    started = service.start_enrollment(
        EnrollmentStart(
            session_id=session.session_id,
            user_id=USER,
            channel=Channel.BROWSER,
            consent=CONSENT,
        )
    )
    for _ in range(PASSES_TO_ENROLL):
        issued = service.create_challenge(session.session_id, ChallengeKind.ENROLL)
        record = session.challenges[issued.challenge_id]
        record.outcome = PASSED
        record.answered_at = clock.now
        clock.advance(1.0)

    assert service.complete_enrollment(started.enrollment_id).status == "enrolled"

    assert stored(prints, "browser") is not None
    # The phone key is untouched, so a phone call still finds no reference
    # and voice_match reports not_run rather than judging a handset by a
    # browser microphone.
    assert stored(prints, "phone") is None


# ------------------------------------------------------------ what it logs


class SpyLogger:
    """The slice of `StructuredLogger` the service calls."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str, dict[str, Any]]] = []

    def log(self, event: str, *, level: str = "info", **fields: Any) -> None:
        self.lines.append((event, level, fields))

    def reasons(self) -> list[str]:
        return [
            str(fields.get("reason"))
            for event, _, fields in self.lines
            if event == "v1_enrollment_print_not_stored"
        ]


def test_a_stored_print_is_logged_without_naming_the_voice() -> None:
    service, clock, _ = service_for(clips=Clips())
    spy = SpyLogger()
    service.bind_logger(spy)
    session = a_session(service, clock)
    service.complete_enrollment(drive(service, clock, session))

    stored_lines = [
        line for line in spy.lines if line[0] == "v1_enrollment_print_stored"
    ]
    assert len(stored_lines) == 1
    _, level, fields = stored_lines[0]
    assert level == "info"
    assert fields["channel"] == "phone"
    assert fields["clips"] == PASSES_TO_ENROLL
    assert fields["quality"] == "good"
    # Durations and counts only. No pitch, no spectrum, no identity.
    assert "median_f0_hz" not in fields
    assert "ltas_db" not in fields


def test_every_path_without_a_print_says_why() -> None:
    """A missing print cannot change the ruled answer, so the log line is the
    only thing that distinguishes "enrolled, print stored" from "enrolled,
    nothing stored". Each reason is distinct so it can be acted on."""

    class Full(InMemoryVoicePrintStore):
        def put(self, enrolment: Any, *, replace: bool) -> None:
            raise EnrolmentStoreFull("the store is full", "raise the cap")

    class Silence:
        def clip_for(self, challenge_id: str) -> tuple[Any, int] | None:
            return np.zeros(3 * RATE), RATE

    cases = {
        # Neither a listener nor an audio policy: nothing could have arrived.
        "no_audio_source": {"clips": None},
        "no_clips_for_passed_prompts": {"clips": NoClips()},
        "combined_print_not_comparable": {"clips": Silence()},
        "store_refused": {"clips": Clips(), "store": Full()},
    }

    for expected, kwargs in cases.items():
        service, clock, _ = service_for(**kwargs)
        spy = SpyLogger()
        service.bind_logger(spy)
        session = a_session(service, clock)
        service.complete_enrollment(drive(service, clock, session))

        reasons = spy.reasons()
        assert len(reasons) == 1, f"{expected}: {reasons}"
        assert reasons[0].startswith(expected), f"{expected}: got {reasons[0]}"


def test_an_absent_listener_is_not_logged_as_a_warning() -> None:
    """It is the documented state of every deployment today, not a fault."""

    service, clock, _ = service_for(clips=None)
    spy = SpyLogger()
    service.bind_logger(spy)
    session = a_session(service, clock)
    service.complete_enrollment(drive(service, clock, session))

    levels = {
        level
        for event, level, _ in spy.lines
        if event == "v1_enrollment_print_not_stored"
    }
    assert levels == {"info"}


def test_a_listener_that_heard_nothing_is_a_warning() -> None:
    """Here a print was expected and did not arrive, which someone should see."""

    service, clock, _ = service_for(clips=NoClips())
    spy = SpyLogger()
    service.bind_logger(spy)
    session = a_session(service, clock)
    service.complete_enrollment(drive(service, clock, session))

    levels = {
        level
        for event, level, _ in spy.lines
        if event == "v1_enrollment_print_not_stored"
    }
    assert levels == {"warning"}
