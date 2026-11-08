"""Tests for the enrollment flow, with a fake detector, a fake capture and fake step-ups.

The fake capture records every line the agent says or asks, in order, as `lines`, and answers each
ask from a script. The fake step-ups answer the SMS code and the challenge polls from scripts. The
fake detector records its calls, and it has no get_challenge on purpose: polling goes through the
step-ups, so a flow that polled the detector itself would fail here. All three also write to one
journal, so a test can check what happened in what order across them. The real parser from
spoken.py reads the consent.
"""

import inspect
import itertools
import logging
import typing
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import pytest
from callback_shared.contract import (
    ChallengeIssued,
    ChallengeKind,
    ChallengeState,
    ChallengeStatus,
    Channel,
    EnrollmentResult,
    EnrollmentStart,
    EnrollmentStarted,
    UserSummary,
)

import callback_agent.enrollment as enrollment_module
from callback_agent import prompts
from callback_agent.detector import DetectorClient, DetectorError
from callback_agent.enrollment import Enrollment, EnrollmentOutcome
from callback_agent.state import CallState
from callback_agent.stepups import StepUps, TurnCapture

SESSION = "s1"
ENROLLMENT = "e1"
ERIC = UserSummary(user_id="eric", display_name="Eric", role="CFO", phone_last_four="0142")
YES = "Yes, I agree"
CALLER_WORDS = "amber harbor four seven two"  # what the caller reads back: never judged here

PASSED = ChallengeStatus(status=ChallengeState.PASSED, phrase_match=True, speaker_score=0.9)
FAILED = ChallengeStatus(status=ChallengeState.FAILED, phrase_match=False, reason="too_short")
EXPIRED = ChallengeStatus(status=ChallengeState.EXPIRED)
PENDING = ChallengeStatus(status=ChallengeState.PENDING)
ENROLLED = EnrollmentResult(status="enrolled", prompts_passed=7)
NOT_ENROLLED = EnrollmentResult(status="failed", prompts_passed=4)


def words(n: int) -> str:
    """What the fake detector issues as the `say` of its nth challenge."""
    return f"enroll words {n}"


def one_try(n: int) -> list[tuple]:
    """The journal of one try on the nth challenge: it is made, spoken and then polled."""
    return [("create_challenge",), ("ask", words(n)), ("poll", f"c{n}")]


class ScriptExhausted(Exception):
    """A fake was asked for an answer its test never scripted."""


def _next[T](script: Iterator[T], what: str) -> T:
    try:
        return next(script)
    except StopIteration:
        raise ScriptExhausted(f"nothing is scripted for the next {what}") from None


@dataclass(frozen=True)
class Ask:
    prompt: str
    timeout_s: float
    secret: bool


class FakeCapture:
    """Records each line said or asked, in order, as `lines`, and answers each ask from `answers`.

    `answers` is an iterator, so a test can script an endless caller with itertools.repeat. An
    answer is a string, or None for no answer in the whole wait. By default the caller says yes to
    the first ask, which is the consent question, and reads back something to every ask after it.
    An ask nobody scripted raises ScriptExhausted: it must not read as a caller who said nothing.
    `asks` keeps what each ask was given.
    """

    def __init__(self, journal: list[tuple]) -> None:
        self.lines: list[str] = []
        self.asks: list[Ask] = []
        self.answers: Iterator[str | None] = itertools.chain([YES], itertools.repeat(CALLER_WORDS))
        self._journal = journal

    async def say(self, text: str) -> None:
        self.lines.append(text)
        self._journal.append(("say", text))

    async def ask(self, prompt: str, *, timeout_s: float, secret: bool = False) -> str | None:
        self.lines.append(prompt)
        self.asks.append(Ask(prompt, timeout_s, secret))
        self._journal.append(("ask", prompt))
        return _next(self.answers, f"ask number {len(self.asks)}")


class FakeStepUps:
    """Answers sms_otp and poll_challenge from scripts.

    `otp_passes` is the verdict on the code, and `otp_error` is raised instead of it. `statuses`
    is an iterator of what each poll gives back. `polled` is the challenge id of each poll.
    """

    def __init__(self, journal: list[tuple]) -> None:
        self.otp_passes = True
        self.otp_error: Exception | None = None
        self.statuses: Iterator[ChallengeStatus] = itertools.repeat(PASSED)
        self.polled: list[str] = []
        self._journal = journal

    async def sms_otp(self) -> bool:
        self._journal.append(("sms_otp",))
        if self.otp_error is not None:
            raise self.otp_error
        return self.otp_passes

    async def poll_challenge(self, challenge_id: str) -> ChallengeStatus:
        self.polled.append(challenge_id)
        self._journal.append(("poll", challenge_id))
        return _next(self.statuses, "poll_challenge")


class FakeDetector:
    """Answers the three enrollment calls and records each in `calls`, as (method, *arguments).

    A call is recorded before it answers, so it is there even when `fail` makes it raise: `fail`
    maps a method name to the exception that call raises. Every challenge it issues has the id
    c0, c1 and so on, and `say_for(n)` for its words. `issued` keeps them. `started` keeps each
    EnrollmentStart. `prompts_required` and `result` are what it answers with.
    """

    def __init__(self, journal: list[tuple]) -> None:
        self.calls: list[tuple] = []
        self.started: list[EnrollmentStart] = []
        self.issued: list[ChallengeIssued] = []
        self.prompts_required = 7
        self.result = ENROLLED
        self.say_for: Callable[[int], str] = words
        self.fail: dict[str, Exception] = {}
        self._journal = journal

    def named(self, method: str) -> list[tuple]:
        return [call for call in self.calls if call[0] == method]

    async def start_enrollment(self, req: EnrollmentStart) -> EnrollmentStarted:
        self._record(("start_enrollment",), req)
        self.started.append(req)
        return EnrollmentStarted(enrollment_id=ENROLLMENT, prompts_required=self.prompts_required)

    async def create_challenge(self, session_id: str, kind: ChallengeKind) -> ChallengeIssued:
        self._record(("create_challenge",), session_id, kind)
        n = len(self.issued)
        issued = ChallengeIssued(challenge_id=f"c{n}", say=self.say_for(n), expires_at=1_000)
        self.issued.append(issued)
        return issued

    async def complete_enrollment(self, enrollment_id: str) -> EnrollmentResult:
        self._record(("complete_enrollment", enrollment_id), enrollment_id)
        return self.result

    def _record(self, entry: tuple, *args) -> None:
        """Note a call in `calls`, with its arguments, and in the journal as `entry`."""
        method = entry[0]
        self.calls.append((method, *args))
        self._journal.append(entry)
        if method in self.fail:
            raise self.fail[method]


@pytest.fixture
def journal() -> list[tuple]:
    return []


@pytest.fixture
def state() -> CallState:
    return CallState(
        room_name="r1",
        caller_identity="caller-1",
        channel=Channel.PHONE,
        session_id=SESSION,
        claimed_user=ERIC,
    )


@pytest.fixture
def detector(journal) -> FakeDetector:
    return FakeDetector(journal)


@pytest.fixture
def capture(journal) -> FakeCapture:
    return FakeCapture(journal)


@pytest.fixture
def stepups(journal) -> FakeStepUps:
    return FakeStepUps(journal)


@pytest.fixture
def make_enrollment(detector, capture, stepups, state):
    """Builds an Enrollment on the fakes. Options override its settings."""

    def make(**options) -> Enrollment:
        return Enrollment(detector, capture, stepups, state, **options)

    return make


@pytest.fixture
def enrollment(make_enrollment) -> Enrollment:
    return make_enrollment()


# --- who may enroll, and what comes first ---


@pytest.mark.parametrize("missing", ["session_id", "claimed_user"])
async def test_requires_an_identified_caller(enrollment, state, detector, journal, missing):
    # A session is made once someone is identified, so none means nobody is. A session without the
    # claimed user has no user_id to enroll, which comes to the same thing.
    setattr(state, missing, None)
    assert await enrollment.run() == "not_identified"
    # Nothing is said or sent: the Toolbox answers the LLM with IDENTIFY_FIRST.
    assert journal == [] and detector.calls == []


async def test_requires_a_verified_code_before_consent(enrollment, stepups, detector, journal):
    stepups.otp_passes = False
    assert await enrollment.run() == "failed"
    # No consent prompt and no start_enrollment: the code comes first (V4).
    assert journal == [("sms_otp",), ("say", prompts.ENROLL_OTP_FAILED)]
    assert detector.calls == []


# --- consent ---

DECLINES = {
    "no": "No",
    "no_and_more": "No thanks, I don't agree",
    "unsure": "I'm not sure",
    "question": "Is it safe?",
    "question_without_mark": "What is a voiceprint",
    "not_a_yes": "Maybe later",
    "yes_and_a_question": "Yes, is that safe?",
    "empty": "",
}


@pytest.mark.parametrize("answer", DECLINES.values(), ids=DECLINES.keys())
async def test_decline_creates_nothing(enrollment, detector, capture, stepups, answer):
    # Only a plain yes is consent: a no, a question and an unclear answer all end it here.
    capture.answers = iter([answer])
    assert await enrollment.run() == "declined"
    assert capture.lines == [prompts.CONSENT_TEXT, prompts.ENROLL_DECLINED]
    assert detector.calls == [] and stepups.polled == []


async def test_silence_is_not_consent(enrollment, detector, capture, stepups):
    # No answer in the whole wait, or a caller who left: ask() gives None, which parse_yes_no
    # cannot read.
    capture.answers = iter([None])
    assert await enrollment.run() == "declined"
    assert capture.lines == [prompts.CONSENT_TEXT, prompts.ENROLL_DECLINED]
    assert detector.calls == [] and stepups.polled == []


YES_ANSWERS = {
    "plain": YES,
    "padded_and_shouted": "  YES, I agree.  ",
    "informal": "yeah, sure",
    "whitespace": "Yes\tI agree\n",
}


@pytest.mark.parametrize("answer", YES_ANSWERS.values(), ids=YES_ANSWERS.keys())
async def test_consent_verbatim_and_version_are_stored(
    enrollment, detector, capture, journal, monkeypatch, answer
):
    # The clock reads one second per event so far, so the stamp is the moment the caller answered:
    # after the code and the consent question, and before the detector hears of it.
    monkeypatch.setattr(enrollment_module, "wall_clock", lambda: 1_800_000_000.1236 + len(journal))
    capture.answers = itertools.chain([answer], itertools.repeat(CALLER_WORDS))
    assert await enrollment.run() == "enrolled"
    [start] = detector.started
    assert start.consent.verbatim == answer  # the caller's words, byte for byte
    assert start.consent.version == prompts.CONSENT_VERSION
    assert start.consent.ts_ms == 1_800_000_002_124  # epoch ms, rounded
    assert (start.session_id, start.user_id, start.channel) == (SESSION, "eric", Channel.PHONE)


@pytest.mark.parametrize("channel", list(Channel))
async def test_the_channel_comes_from_the_call(enrollment, state, detector, channel):
    # Phone and browser enroll separately, so each call enrolls the channel it is on.
    state.channel = channel
    assert await enrollment.run() == "enrolled"
    assert detector.started[0].channel is channel


def test_the_consent_text_is_the_specs_and_a_new_text_needs_a_new_version():
    # Change CONSENT_TEXT only together with CONSENT_VERSION: the version is stored with each
    # answer, and says which words the caller agreed to.
    assert prompts.CONSENT_VERSION == "2026-11-04"
    assert prompts.CONSENT_TEXT == (
        "To protect your company's account, Callback creates a voiceprint, a set of numbers "
        "derived from your voice, and analyzes call audio and transcripts for fraud. Raw audio "
        "is not kept. Your data is deleted by November 30, 2026, or sooner if you ask. "
        "Do you agree?"
    )


# --- the prompts ---


async def test_seven_passes_then_complete(enrollment, detector, journal):
    assert await enrollment.run() == "enrolled"
    expected = [("sms_otp",), ("ask", prompts.CONSENT_TEXT), ("start_enrollment",)]
    for n in range(7):
        expected += one_try(n)
    expected += [("complete_enrollment", ENROLLMENT), ("say", prompts.ENROLL_DONE)]
    assert journal == expected
    assert detector.named("complete_enrollment") == [("complete_enrollment", ENROLLMENT)]


async def test_the_detector_sets_how_many_prompts(enrollment, detector):
    detector.prompts_required = 3
    assert await enrollment.run() == "enrolled"
    assert len(detector.issued) == 3


async def test_a_prompt_is_retried_twice(enrollment, detector, capture, stepups):
    # Failed, failed, passed: the third try passes, so the prompt counts as passed.
    stepups.statuses = itertools.chain([FAILED, FAILED, PASSED], itertools.repeat(PASSED))
    assert await enrollment.run() == "enrolled"
    # A retry is a new challenge with its own words, and comes after "try that again". Moving on
    # to the next prompt says nothing between.
    assert capture.lines[:7] == [
        prompts.CONSENT_TEXT,
        words(0),
        prompts.ENROLL_RETRY,
        words(1),
        prompts.ENROLL_RETRY,
        words(2),
        words(3),
    ]
    assert len(detector.issued) == 3 + 6
    assert stepups.polled[:3] == ["c0", "c1", "c2"]


async def test_retries_do_not_count_as_failed_prompts(enrollment, detector, stepups):
    # Every prompt needs all three tries, but each one passes in the end: nothing aborts.
    stepups.statuses = itertools.cycle([FAILED, FAILED, PASSED])
    assert await enrollment.run() == "enrolled"
    assert len(detector.issued) == 21


async def test_three_failed_prompts_abort_without_completing(
    enrollment, detector, capture, stepups
):
    stepups.statuses = itertools.repeat(FAILED)
    assert await enrollment.run() == "failed"
    # Three prompts took three tries each. A fourth was never started.
    assert len(detector.issued) == 9
    assert capture.lines[-1] == prompts.ENROLL_ABORTED
    assert prompts.ENROLL_DONE not in capture.lines
    assert detector.named("complete_enrollment") == []


async def test_two_failed_prompts_do_not_abort(enrollment, detector, stepups):
    # Prompts 1 and 2 fail all three tries and the other five pass at once. The detector then
    # says whether that is enough.
    stepups.statuses = itertools.chain(itertools.repeat(FAILED, 6), itertools.repeat(PASSED))
    assert await enrollment.run() == "enrolled"
    assert len(detector.issued) == 6 + 5
    assert detector.named("complete_enrollment") == [("complete_enrollment", ENROLLMENT)]


@pytest.mark.parametrize("status", [FAILED, EXPIRED, PENDING], ids=["failed", "expired", "pending"])
async def test_anything_but_passed_is_a_failed_try(enrollment, detector, capture, stepups, status):
    # PENDING is where StepUps.poll_challenge ends when the detector never settles, as it does
    # after a total outage. That is not a pass.
    stepups.statuses = itertools.repeat(status)
    assert await enrollment.run() == "failed"
    assert len(detector.issued) == 9
    assert capture.lines[-1] == prompts.ENROLL_ABORTED
    assert detector.named("complete_enrollment") == []


async def test_a_try_that_is_pending_can_pass_on_the_retry(enrollment, detector, capture, stepups):
    stepups.statuses = itertools.chain([PENDING, PASSED], itertools.repeat(PASSED))
    assert await enrollment.run() == "enrolled"
    assert capture.lines.count(prompts.ENROLL_RETRY) == 1
    assert len(detector.issued) == 8


@pytest.mark.parametrize(
    "say",
    [" Amber  harbor, 4 7 2. ", "AMBER HARBOR", "4 7 2 9 1", "pay\tthe\nbill", ""],
    ids=["padded_punctuated", "capitals", "digits", "whitespace", "empty"],
)
async def test_enroll_say_is_spoken_as_is(enrollment, detector, capture, say):
    detector.say_for = lambda n: say
    assert await enrollment.run() == "enrolled"
    # For an enroll challenge `say` is the whole prompt: no lead-in, nothing changed, and nothing
    # else said between the prompts.
    assert capture.lines == [prompts.CONSENT_TEXT, *[say] * 7, prompts.ENROLL_DONE]
    assert [ask.prompt for ask in capture.asks[1:]] == [say] * 7


async def test_the_consent_waits_30_seconds_and_each_prompt_20_in_secret(enrollment, capture):
    assert await enrollment.run() == "enrolled"
    assert capture.asks[0] == Ask(prompts.CONSENT_TEXT, timeout_s=30.0, secret=False)
    # A challenge phrase is asked in secret, so the LLM never sees it.
    assert capture.asks[1:] == [Ask(words(n), timeout_s=20.0, secret=True) for n in range(7)]


async def test_each_try_is_an_enroll_challenge_polled_through_the_stepups(
    enrollment, detector, stepups
):
    stepups.statuses = itertools.chain([FAILED], itertools.repeat(PASSED))
    assert await enrollment.run() == "enrolled"
    assert (
        detector.named("create_challenge")
        == [("create_challenge", SESSION, ChallengeKind.ENROLL)] * 8
    )
    # Each poll is for the challenge just made. The fake detector has no get_challenge to poll.
    assert stepups.polled == [f"c{n}" for n in range(8)]


# --- a try nobody answers (R59) ---
#
# When ask() gives None no turn ended within the 20 s, so the detector has no turn to score and a
# poll would only wait out its window. The try is failed without a poll, and the retry follows as
# usual.


async def test_a_try_with_no_answer_is_not_polled(enrollment, detector, capture, stepups, journal):
    capture.answers = itertools.chain([YES, None], itertools.repeat(CALLER_WORDS))
    assert await enrollment.run() == "enrolled"
    # The first try is made and asked, and there is no poll. "Try that again" comes next, and the
    # retry is a new challenge that is polled.
    first = journal.index(("create_challenge",))
    assert journal[first : first + 6] == [
        ("create_challenge",),
        ("ask", words(0)),
        ("say", prompts.ENROLL_RETRY),
        *one_try(1),
    ]
    assert "c0" not in stepups.polled
    assert stepups.polled == [f"c{n}" for n in range(1, 8)]  # 8 challenges, 7 polls


async def test_none_on_every_try_of_a_prompt_counts_that_prompt_as_failed(
    make_enrollment, detector, capture, stepups
):
    # With no failed prompt allowed, one prompt that nobody answers is enough to abort.
    capture.answers = itertools.chain([YES, None, None, None], itertools.repeat(CALLER_WORDS))
    assert await make_enrollment(max_failed_prompts=0).run() == "failed"
    assert len(detector.issued) == 3  # its three tries; a second prompt was never started
    assert stepups.polled == []
    assert capture.lines.count(prompts.ENROLL_RETRY) == 2
    assert capture.lines[-1] == prompts.ENROLL_ABORTED
    assert detector.named("complete_enrollment") == []


async def test_one_prompt_nobody_answers_is_one_failed_prompt_and_no_more(
    enrollment, detector, capture, stepups
):
    # It uses one of the two failed prompts that are allowed, so the enrollment goes on.
    capture.answers = itertools.chain([YES, None, None, None], itertools.repeat(CALLER_WORDS))
    assert await enrollment.run() == "enrolled"
    assert len(detector.issued) == 3 + 6
    assert stepups.polled == [f"c{n}" for n in range(3, 9)]  # the other six prompts
    assert detector.named("complete_enrollment") == [("complete_enrollment", ENROLLMENT)]


async def test_no_answer_then_an_answer_that_passes_counts_the_prompt_as_passed(
    make_enrollment, detector, capture, stepups
):
    # With no failed prompt allowed, a prompt that passes on a retry after a silent try must not
    # count as failed.
    capture.answers = itertools.chain([YES, None], itertools.repeat(CALLER_WORDS))
    assert await make_enrollment(max_failed_prompts=0).run() == "enrolled"
    assert len(detector.issued) == 8  # one extra try
    assert capture.lines.count(prompts.ENROLL_RETRY) == 1
    assert stepups.polled == [f"c{n}" for n in range(1, 8)]
    assert detector.named("complete_enrollment") == [("complete_enrollment", ENROLLMENT)]


async def test_a_caller_who_never_answers_aborts_without_a_single_poll(
    enrollment, detector, capture, stepups
):
    capture.answers = itertools.chain([YES], itertools.repeat(None))
    assert await enrollment.run() == "failed"
    assert len(detector.issued) == 9  # three prompts of three tries, and a fourth never started
    assert stepups.polled == []
    assert capture.lines[-1] == prompts.ENROLL_ABORTED
    assert detector.named("complete_enrollment") == []


async def test_an_empty_turn_is_still_polled(enrollment, capture, stepups):
    # Only None means nothing was said. A turn that ended with no words (a cough, noise) is a turn
    # the detector can score, and LiveKitVoice.ask gives "" for one.
    capture.answers = itertools.chain([YES], itertools.repeat(""))
    assert await enrollment.run() == "enrolled"
    assert len(stepups.polled) == 7


# --- a caller who hangs up (final review, issue 5) ---
#
# A hang-up sets state.closing, and from then on ask() gives None at once. The enrollment must not
# go on through every try after that: each one would make a challenge at the detector for a call
# that is over, say "try that again" into a closing session, and end with the abort line.


def hangs_up_at(
    state: CallState, ask_number: int, *, saying: str | None = None
) -> Iterator[str | None]:
    """The answers of a caller who hangs up as the ask_number-th ask is put to them (the first ask
    is the consent). That ask gets `saying`, which is None unless a test has the caller get one
    last answer out. Asks before it are answered as by default, and every one after it with None,
    as a session that is closing does."""
    for n in itertools.count(1):
        if n < ask_number:
            yield YES if n == 1 else CALLER_WORDS
        elif n == ask_number:
            state.closing = True
            yield saying
        else:
            yield None


async def test_a_caller_who_hangs_up_during_a_prompt_gets_no_further_try(
    enrollment, state, detector, capture, stepups
):
    capture.answers = hangs_up_at(state, 2)  # while the first prompt is read
    assert await enrollment.run() == "failed"
    assert len(detector.issued) == 1  # no challenge was made for the retry, nor for a next prompt
    # Nothing more was said: no "try that again", and no abort line to an empty line.
    assert capture.lines == [prompts.CONSENT_TEXT, words(0)]
    assert stepups.polled == []
    assert detector.named("complete_enrollment") == []


async def test_a_call_that_is_closing_before_the_first_prompt_makes_no_challenge(
    enrollment, state, detector, capture
):
    capture.answers = hangs_up_at(state, 1, saying=YES)  # they consent, and hang up on it
    assert await enrollment.run() == "failed"
    assert detector.issued == []
    assert capture.lines == [prompts.CONSENT_TEXT]
    assert detector.named("complete_enrollment") == []


async def test_a_prompt_that_fails_as_the_caller_leaves_does_not_get_the_abort_line(
    make_enrollment, state, detector, capture, stepups
):
    # With one try per prompt and none to spare, the first failure would abort the enrollment, and
    # say so. The caller read it out and hung up, and the detector scored it as failed.
    capture.answers = hangs_up_at(state, 2, saying=CALLER_WORDS)
    stepups.statuses = itertools.repeat(FAILED)
    enrollment = make_enrollment(retries_per_prompt=0, max_failed_prompts=0)
    assert await enrollment.run() == "failed"
    assert capture.lines == [prompts.CONSENT_TEXT, words(0)]
    assert len(detector.issued) == 1 and stepups.polled == ["c0"]
    assert detector.named("complete_enrollment") == []


async def test_a_caller_who_hangs_up_during_try_that_again_gets_no_new_challenge(
    enrollment, state, detector, capture
):
    # Nobody answers the first try, so a retry follows it.
    capture.answers = itertools.chain([YES], itertools.repeat(None))
    said = capture.say

    async def say_and_lose_the_caller(text: str) -> None:
        await said(text)
        if text == prompts.ENROLL_RETRY:
            state.closing = True  # they hung up while the line played

    capture.say = say_and_lose_the_caller
    assert await enrollment.run() == "failed"
    assert len(detector.issued) == 1  # the retry's challenge was never made
    assert capture.lines == [prompts.CONSENT_TEXT, words(0), prompts.ENROLL_RETRY]


async def test_a_call_that_is_not_closing_goes_on_through_every_try(enrollment, detector, capture):
    # The control for the tests above: only the caller leaving stops the tries.
    capture.answers = itertools.chain([YES], itertools.repeat(None))
    assert await enrollment.run() == "failed"
    assert len(detector.issued) == 9  # three prompts of three tries
    assert capture.lines[-1] == prompts.ENROLL_ABORTED


# --- completing ---


async def test_a_completion_that_is_not_enrolled_is_aborted(enrollment, detector, capture):
    detector.result = NOT_ENROLLED
    assert await enrollment.run() == "failed"
    assert capture.lines[-1] == prompts.ENROLL_ABORTED
    assert prompts.ENROLL_DONE not in capture.lines
    assert len(detector.named("complete_enrollment")) == 1


@pytest.mark.parametrize("retries, max_failed", [(0, 0), (0, 2), (1, 1), (2, 2), (3, 0)], ids=str)
async def test_the_counts_are_settings(
    make_enrollment, detector, capture, stepups, retries, max_failed
):
    stepups.statuses = itertools.repeat(FAILED)
    enrollment = make_enrollment(retries_per_prompt=retries, max_failed_prompts=max_failed)
    assert await enrollment.run() == "failed"
    failed_prompts = max_failed + 1  # one more than allowed, and it stops
    assert len(detector.issued) == failed_prompts * (1 + retries)
    assert capture.lines.count(prompts.ENROLL_RETRY) == failed_prompts * retries


# --- the real poll loop ---


class FakeClock:
    """Stands in for time.monotonic and asyncio.sleep. A sleep moves time on and returns at once."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


class DownDetector(FakeDetector):
    """A detector that makes challenges but can no longer say how they went: an outage."""

    async def get_challenge(self, session_id: str, challenge_id: str) -> ChallengeStatus:
        raise DetectorError(
            f"GET /v1/sessions/{session_id}/challenges/{challenge_id}: ConnectError"
        )


class CodeAlreadyVerified(StepUps):
    """The real StepUps, poll loop and all, with the SMS code step passed."""

    async def sms_otp(self) -> bool:
        return True


async def test_a_detector_that_goes_down_while_polling_ends_in_failed(capture, state, journal):
    # StepUps.poll_challenge counts a failed poll as still pending and ends pending, which is a
    # failed try: so an outage costs the enrollment, and never reads as enrolled.
    clock = FakeClock()
    detector = DownDetector(journal)
    stepups = CodeAlreadyVerified(detector, capture, state, clock=clock, sleep=clock.sleep)
    assert await Enrollment(detector, capture, stepups, state).run() == "failed"
    assert len(detector.issued) == 9
    assert capture.lines[-1] == prompts.ENROLL_ABORTED
    assert detector.named("complete_enrollment") == []


# --- logs and errors ---


# What the caller says and what the detector issues in these scenarios, lowercased: none of it
# may reach a log.
SECRETS = ("absolutely definitely", "definitely not", "quokka", "saffron", "mango", "falcon")


def _caller_agrees(capture, stepups, detector):
    capture.answers = itertools.chain(["Absolutely definitely"], itertools.repeat("quokka saffron"))


def _caller_declines(capture, stepups, detector):
    capture.answers = iter(["Definitely not, thanks"])


def _caller_goes_quiet(capture, stepups, detector):
    capture.answers = itertools.chain(["Absolutely definitely"], itertools.repeat(None))


def _every_prompt_fails(capture, stepups, detector):
    _caller_agrees(capture, stepups, detector)
    stepups.statuses = itertools.repeat(FAILED)


def _the_code_fails(capture, stepups, detector):
    stepups.otp_passes = False


@pytest.mark.parametrize(
    "scenario",
    [_caller_agrees, _caller_declines, _caller_goes_quiet, _every_prompt_fails, _the_code_fails],
    ids=lambda f: f.__name__.strip("_"),
)
async def test_logs_never_hold_the_consent_challenge_phrases_or_caller_words(
    enrollment, detector, capture, stepups, caplog, scenario
):
    detector.say_for = lambda n: f"mango{n} falcon"
    scenario(capture, stepups, detector)
    caplog.set_level(logging.DEBUG)
    await enrollment.run()
    text = "\n".join(record.getMessage() for record in caplog.records).lower()
    assert text, "the flow logs nothing, so this test would check nothing"
    for secret in SECRETS:
        assert secret not in text


@pytest.mark.parametrize("method", ["start_enrollment", "create_challenge", "complete_enrollment"])
async def test_a_detector_error_goes_up_and_never_reads_as_enrolled(
    enrollment, detector, capture, method
):
    # Nothing is caught here, as in StepUps: the caller of run() decides what the LLM is told.
    detector.fail[method] = DetectorError(f"POST /v1/{method}: HTTP 503")
    with pytest.raises(DetectorError):
        await enrollment.run()
    assert prompts.ENROLL_DONE not in capture.lines


async def test_an_error_from_the_code_step_goes_up_before_any_consent(
    enrollment, stepups, detector, capture
):
    stepups.otp_error = DetectorError("POST /v1/sessions/s1/otp/send: HTTP 502")
    with pytest.raises(DetectorError):
        await enrollment.run()
    assert capture.lines == [] and detector.calls == []


# --- the interface other code is built against ---


def test_the_constructor_takes_the_documented_settings():
    params = inspect.signature(Enrollment).parameters
    assert list(params) == [
        "detector",
        "capture",
        "stepups",
        "state",
        "retries_per_prompt",
        "max_failed_prompts",
    ]
    assert [n for n, p in params.items() if p.kind is inspect.Parameter.KEYWORD_ONLY] == [
        "retries_per_prompt",
        "max_failed_prompts",
    ]
    assert {
        n: p.default for n, p in params.items() if p.default is not inspect.Parameter.empty
    } == {"retries_per_prompt": 2, "max_failed_prompts": 2}


def test_run_takes_nothing_and_has_four_outcomes():
    assert inspect.iscoroutinefunction(Enrollment.run)
    assert list(inspect.signature(Enrollment.run).parameters) == ["self"]
    assert inspect.signature(Enrollment.run).return_annotation is EnrollmentOutcome
    # The three results the Toolbox maps for the LLM, and the one it answers with IDENTIFY_FIRST.
    assert typing.get_args(EnrollmentOutcome.__value__) == (
        "enrolled",
        "declined",
        "failed",
        "not_identified",
    )


def _shape(fn) -> list[tuple]:
    return [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]


@pytest.mark.parametrize(
    "fake, real",
    [
        (FakeCapture.say, TurnCapture.say),
        (FakeCapture.ask, TurnCapture.ask),
        (FakeStepUps.sms_otp, StepUps.sms_otp),
        (FakeStepUps.poll_challenge, StepUps.poll_challenge),
        (FakeDetector.start_enrollment, DetectorClient.start_enrollment),
        (FakeDetector.create_challenge, DetectorClient.create_challenge),
        (FakeDetector.complete_enrollment, DetectorClient.complete_enrollment),
    ],
    ids=lambda fn: fn.__qualname__,
)
def test_the_fakes_have_the_shape_of_the_real_thing(fake, real):
    # What passes here would pass against the real classes.
    assert _shape(fake) == _shape(real)


async def test_an_enroll_prompt_answer_is_exempt_only_while_it_is_asked(enrollment, capture, state):
    seen: list[tuple[str, bool]] = []
    original = capture.ask

    async def spy(prompt, *, timeout_s, secret=False):
        seen.append((prompt, state.answer_exempt))
        return await original(prompt, timeout_s=timeout_s, secret=secret)

    capture.ask = spy
    assert await enrollment.run() == "enrolled"
    assert [exempt for prompt, exempt in seen if prompt != prompts.CONSENT_TEXT] == [True] * 7
    assert [exempt for prompt, exempt in seen if prompt == prompts.CONSENT_TEXT] == [False]
    assert state.answer_exempt is False
