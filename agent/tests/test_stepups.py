"""Tests for the step-ups, with a fake detector, a fake capture and a fake clock.

The fake capture records every line the agent says or asks, in order, as `lines`, and answers each
ask from a script. The fake clock lets the poll loop run in no time. The real parser from spoken.py
reads the spoken codes.
"""

import asyncio
import inspect
import itertools
import logging
import time
import typing
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from callback_shared.contract import (
    ChallengeIssued,
    ChallengeKind,
    ChallengeState,
    ChallengeStatus,
    Channel,
    Decision,
    DecisionKind,
    Factor,
    OtpCheckResult,
    OtpSent,
)
from fake_ramp import FakeRamp

from callback_agent import prompts
from callback_agent.actions import ActionRequest
from callback_agent.detector import DetectorError
from callback_agent.gate import Gate, GateResult
from callback_agent.spoken import parse_spoken_digits
from callback_agent.state import CallState
from callback_agent.stepups import StepUps, TurnCapture

SESSION = "s1"
ROUTING = "021000021"
ACCOUNT = "000123456789"
# A request of each tier: a bank change and a release are tier 2, a read is tier 1, a hold tier 0.
BANK_CHANGE = ActionRequest.update_vendor_bank_account("v1", "Acme Supplies", ROUTING, ACCOUNT)
RELEASE = ActionRequest.release_vendor_hold("v1", "Acme Supplies")
READ_ACCOUNTS = ActionRequest.list_vendor_accounts("v1", "Acme Supplies")
HOLD = ActionRequest.hold_vendor("v1", "Acme Supplies", "caller asked")
VOICE, OTP, CALLBACK, PROBE = (
    Factor.VOICE_CHALLENGE,
    Factor.SMS_OTP,
    Factor.CALLBACK_ON_FILE,
    Factor.PROBE,
)

ISSUED = ChallengeIssued(challenge_id="c1", say="amber harbor 4 7 2", expires_at=1_000)
PENDING = ChallengeStatus(status=ChallengeState.PENDING)
PASSED = ChallengeStatus(status=ChallengeState.PASSED, phrase_match=True, speaker_score=0.9)
FAILED = ChallengeStatus(status=ChallengeState.FAILED, phrase_match=False, reason="phrase_mismatch")
EXPIRED = ChallengeStatus(status=ChallengeState.EXPIRED)

CHALLENGE_PROMPT = f"{prompts.CHALLENGE_LEAD_IN_CHANGE} {ISSUED.say}"
OTP_PROMPT = prompts.OTP_PROMPT.format(last2="42")  # the fake detector texts a number ending 0142

# What the caller says when it is not a six-digit code, and the codes it reads out.
NOT_A_CODE = "hang on, I'm looking for it"
CODE = "four two four two four two"


class ScriptExhausted(Exception):
    """A fake was asked for an answer its test never scripted."""


class FakeClock:
    """Stands in for time.monotonic and asyncio.sleep. A sleep moves time on and returns at once."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@dataclass(frozen=True)
class Ask:
    prompt: str
    timeout_s: float
    secret: bool


class FakeCapture:
    """Records each line said or asked, in order, as `lines`, and answers each ask from `answers`.

    An answer is a string, None (no answer in the whole wait, or the caller left), an exception to
    raise, or a function to call whose result is the answer (see `leaves`). An ask nobody scripted
    raises ScriptExhausted: it must not read as a caller who left. `asks` keeps what each ask was
    given, and `said` the lines that were only said. `answer_delay_s` is how long the caller takes
    to answer, on the fake clock.
    """

    def __init__(self, clock: FakeClock) -> None:
        self.lines: list[str] = []
        self.said: list[str] = []
        self.asks: list[Ask] = []
        self.answers: list[str | None | BaseException | Callable[[], str | None]] = []
        self.answer_delay_s = 0.0
        self._clock = clock

    async def say(self, text: str) -> None:
        self.lines.append(text)
        self.said.append(text)

    async def ask(self, prompt: str, *, timeout_s: float, secret: bool = False) -> str | None:
        self.lines.append(prompt)
        self.asks.append(Ask(prompt, timeout_s, secret))
        if not self.answers:
            raise ScriptExhausted(f"nobody scripted an answer to ask number {len(self.asks)}")
        self._clock.now += self.answer_delay_s
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        if callable(answer):
            return answer()
        return answer


class FakeDetector:
    """Answers the challenge and OTP calls from scripts and records each call in `calls`.

    A call is recorded as (method, session_id, *arguments) before it answers, so it is there even
    when `fail` makes it raise: `fail` maps a method name to the exception that call raises.
    `statuses` and `verdicts` are iterators, so a test can script an endless wait with
    itertools.repeat. A status can also be an exception, which that one poll raises. `poll_times`
    is when each get_challenge arrived, on the fake clock.
    """

    def __init__(self, clock: FakeClock) -> None:
        self.calls: list[tuple] = []
        self.poll_times: list[float] = []
        self.issued = ISSUED
        self.statuses: Iterator[ChallengeStatus | BaseException] = iter(())
        self.sent = OtpSent(sent_to_last_four="0142")
        self.verdicts: Iterator[bool] = iter(())
        self.fail: dict[str, Exception] = {}
        self._clock = clock

    def named(self, method: str) -> list[tuple]:
        return [call for call in self.calls if call[0] == method]

    @property
    def polls(self) -> list[str]:
        """The challenge id of each get_challenge."""
        return [call[2] for call in self.named("get_challenge")]

    @property
    def otp_checks(self) -> list[str]:
        """The code of each check_otp."""
        return [call[2] for call in self.named("check_otp")]

    async def create_challenge(self, session_id: str, kind: ChallengeKind) -> ChallengeIssued:
        self._record("create_challenge", session_id, kind)
        return self.issued

    async def get_challenge(self, session_id: str, challenge_id: str) -> ChallengeStatus:
        self.poll_times.append(self._clock.now)
        self._record("get_challenge", session_id, challenge_id)
        status = _next(self.statuses, "get_challenge")
        if isinstance(status, BaseException):
            raise status
        return status

    async def send_otp(self, session_id: str) -> OtpSent:
        # Only a session: the real client has no phone number to pass either.
        self._record("send_otp", session_id)
        return self.sent

    async def check_otp(self, session_id: str, code: str) -> OtpCheckResult:
        self._record("check_otp", session_id, code)
        return OtpCheckResult(verified=_next(self.verdicts, "check_otp"))

    def _record(self, method: str, *args) -> None:
        self.calls.append((method, *args))
        if method in self.fail:
            raise self.fail[method]


def _next[T](script: Iterator[T], what: str) -> T:
    try:
        return next(script)
    except StopIteration:
        raise ScriptExhausted(f"nothing is scripted for the next {what}") from None


@pytest.fixture
def state() -> CallState:
    return CallState(
        room_name="r1", caller_identity="caller-1", channel=Channel.PHONE, session_id=SESSION
    )


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def detector(clock) -> FakeDetector:
    return FakeDetector(clock)


@pytest.fixture
def capture(clock) -> FakeCapture:
    return FakeCapture(clock)


@pytest.fixture
def make_stepups(detector, capture, state, clock):
    """Builds a StepUps on the fakes. Options override its settings, such as `callback_leg=`."""

    def make(**options) -> StepUps:
        return StepUps(
            detector, capture, state, **{"clock": clock, "sleep": clock.sleep, **options}
        )

    return make


@pytest.fixture
def stepups(make_stepups) -> StepUps:
    return make_stepups()


async def run(stepups: StepUps, factor: Factor, req: ActionRequest = BANK_CHANGE) -> bool:
    """Run a factor the way the gate does."""
    decision = Decision(
        session_id=SESSION,
        action=req.action,
        decision=DecisionKind.STEP_UP,
        required_factors=[factor],
    )
    return await stepups.run(factor, req, decision)


def leaves(state: CallState) -> Callable[[], None]:
    """A scripted answer for a caller who hangs up during the ask: the call is then closing, and
    the ask comes back with None."""

    def hang_up() -> None:
        state.closing = True

    return hang_up


# --- the interface other code is built against ---


def test_turn_capture_has_exactly_say_and_ask():
    assert typing.get_protocol_members(TurnCapture) == {"say", "ask"}
    assert list(inspect.signature(TurnCapture.say).parameters) == ["self", "text"]
    ask = inspect.signature(TurnCapture.ask).parameters
    assert list(ask) == ["self", "prompt", "timeout_s", "secret"]
    assert ask["timeout_s"].kind is inspect.Parameter.KEYWORD_ONLY
    assert ask["timeout_s"].default is inspect.Parameter.empty
    assert ask["secret"].kind is inspect.Parameter.KEYWORD_ONLY and ask["secret"].default is False
    # The fake in these tests is the same shape, so what passes here would pass with the real voice.
    for name in ("say", "ask"):
        assert inspect.signature(getattr(FakeCapture, name)) == inspect.signature(
            getattr(TurnCapture, name)
        )


def test_the_constructor_takes_the_documented_settings():
    params = inspect.signature(StepUps).parameters
    assert list(params) == [
        "detector",
        "capture",
        "state",
        "callback_leg",
        "poll_interval_s",
        "poll_timeout_s",
        "clock",
        "sleep",
    ]
    assert [n for n, p in params.items() if p.kind is inspect.Parameter.KEYWORD_ONLY] == [
        "callback_leg",
        "poll_interval_s",
        "poll_timeout_s",
        "clock",
        "sleep",
    ]
    assert {
        n: p.default for n, p in params.items() if p.default is not inspect.Parameter.empty
    } == {
        "callback_leg": None,
        "poll_interval_s": 0.25,
        "poll_timeout_s": 8.0,
        "clock": time.monotonic,
        "sleep": asyncio.sleep,
    }


@pytest.mark.parametrize("factor", [VOICE, OTP, PROBE])
async def test_every_prompt_is_asked_so_the_answer_is_captured(stepups, detector, capture, factor):
    # A line that is only said leaves a gap before the capture starts, where the caller's answer
    # would go to the LLM.
    detector.statuses = iter([PASSED])
    detector.verdicts = iter([True])
    capture.answers = [CODE if factor is OTP else "answer"]
    await run(stepups, factor)
    assert capture.said == [] and capture.lines == [ask.prompt for ask in capture.asks]
    assert len(capture.lines) == 1


# --- the voice challenge ---


async def test_challenge_prompt_ends_with_say_verbatim(stepups, detector, capture):
    detector.statuses = iter([PASSED])
    capture.answers = ["amber harbor 4 7 2"]
    assert await run(stepups, VOICE) is True
    assert capture.lines == [f"{prompts.CHALLENGE_LEAD_IN_CHANGE} {ISSUED.say}"]


@pytest.mark.parametrize(
    "say",
    [" Amber  harbor, 4 7 2. ", "AMBER HARBOR", "4 7 2 9 1", "pay\tthe\nbill", ""],
    ids=["padded_punctuated", "capitals", "digits", "whitespace", "empty"],
)
async def test_challenge_phrase_is_spoken_exactly_as_issued(stepups, detector, capture, say):
    detector.issued = ChallengeIssued(challenge_id="c1", say=say, expires_at=1_000)
    detector.statuses = iter([PASSED])
    capture.answers = ["whatever"]
    await run(stepups, VOICE)
    assert capture.lines == [f"{prompts.CHALLENGE_LEAD_IN_CHANGE} {say}"]


async def test_challenge_is_asked_in_secret_and_waits_20_seconds(stepups, detector, capture):
    detector.statuses = iter([PASSED])
    capture.answers = ["amber harbor 4 7 2"]
    await run(stepups, VOICE)
    assert capture.asks == [Ask(CHALLENGE_PROMPT, timeout_s=20.0, secret=True)]


@pytest.mark.parametrize(
    "req, kind, lead_in",
    [
        (READ_ACCOUNTS, ChallengeKind.DIGITS, prompts.CHALLENGE_LEAD_IN_READ),
        (BANK_CHANGE, ChallengeKind.PHRASE, prompts.CHALLENGE_LEAD_IN_CHANGE),
        (RELEASE, ChallengeKind.PHRASE, prompts.CHALLENGE_LEAD_IN_CHANGE),
    ],
    ids=["tier1_digits", "tier2_phrase", "tier2_release_phrase"],
)
async def test_challenge_kind_follows_tier(stepups, detector, capture, req, kind, lead_in):
    detector.statuses = iter([PASSED])
    capture.answers = ["go"]
    assert await run(stepups, VOICE, req) is True
    assert detector.calls[0] == ("create_challenge", SESSION, kind)
    assert capture.lines == [f"{lead_in} {ISSUED.say}"]


async def test_a_tier_0_action_takes_no_voice_challenge(stepups, detector, capture):
    # The gate never steps up a hold. If it did, there is no challenge to give it.
    with pytest.raises(ValueError):
        await run(stepups, VOICE, HOLD)
    assert detector.calls == [] and capture.lines == []


async def test_challenge_polls_until_passed(stepups, detector, capture, clock):
    detector.statuses = iter([PENDING, PENDING, PASSED])
    capture.answers = ["amber harbor 4 7 2"]
    assert await run(stepups, VOICE) is True
    assert detector.polls == ["c1", "c1", "c1"]
    assert clock.sleeps == [0.25, 0.25]
    assert {call[1] for call in detector.calls} == {
        SESSION
    }  # every call is for this call's session


async def test_a_challenge_that_has_passed_is_not_waited_for(stepups, detector, capture, clock):
    detector.statuses = iter([PASSED])
    capture.answers = ["amber harbor 4 7 2"]
    assert await run(stepups, VOICE) is True
    assert detector.polls == ["c1"] and clock.sleeps == []


async def test_challenge_still_pending_after_8s_fails(stepups, detector, capture, clock):
    detector.statuses = itertools.repeat(PENDING)
    capture.answers = ["amber harbor 4 7 2"]
    start = clock.now
    assert await run(stepups, VOICE) is False
    # A poll at once, one every 0.25 s, and the last as the 8 s end.
    assert detector.poll_times == [start + 0.25 * i for i in range(33)]
    assert clock.now == start + 8.0


@pytest.mark.parametrize(
    "pending_polls, passed", [(32, True), (33, False)], ids=["at_8s", "just_after_8s"]
)
async def test_a_pass_that_arrives_at_8s_counts_but_a_later_one_does_not(
    stepups, detector, capture, pending_polls, passed
):
    detector.statuses = itertools.chain([PENDING] * pending_polls, [PASSED])
    capture.answers = ["amber harbor 4 7 2"]
    assert await run(stepups, VOICE) is passed
    assert len(detector.polls) == 33


async def test_the_poll_window_starts_when_the_caller_has_answered(
    stepups, detector, capture, clock
):
    capture.answer_delay_s = 19.0  # the caller takes nearly all of the 20 s
    detector.statuses = iter([PENDING, PENDING, PASSED])
    capture.answers = ["amber harbor 4 7 2"]
    assert await run(stepups, VOICE) is True
    assert len(detector.polls) == 3


@pytest.mark.parametrize("final", [FAILED, EXPIRED], ids=["failed", "expired"])
async def test_a_challenge_that_failed_or_expired_fails_without_more_polls(
    stepups, detector, capture, final
):
    detector.statuses = iter([PENDING, final])  # a third poll would find nothing scripted
    capture.answers = ["amber harbor 4 7 2"]
    assert await run(stepups, VOICE) is False
    assert detector.polls == ["c1", "c1"]


async def test_no_answer_fails_without_polling(stepups, detector, capture, clock):
    capture.answers = [None]
    assert await run(stepups, VOICE) is False
    assert [call[0] for call in detector.calls] == ["create_challenge"]
    assert clock.sleeps == []


@pytest.mark.parametrize(
    "answer, status, passed",
    [("", PASSED, True), ("banana", PASSED, True), (ISSUED.say, FAILED, False)],
    ids=["empty_answer_passes", "wrong_words_pass", "right_words_fail"],
)
async def test_the_detector_judges_the_answer_not_the_agent(
    stepups, detector, capture, answer, status, passed
):
    detector.statuses = iter([status])
    capture.answers = [answer]
    assert await run(stepups, VOICE) is passed
    assert detector.polls == ["c1"]


# --- poll_challenge, which enrollment uses on its own ---


async def test_poll_challenge_returns_the_status_it_settled_on(stepups, detector):
    detector.statuses = iter([PENDING, FAILED])
    status = await stepups.poll_challenge("c7")
    assert status == FAILED and status.reason == "phrase_mismatch"
    assert detector.calls == [("get_challenge", SESSION, "c7")] * 2


async def test_poll_challenge_hands_back_pending_when_time_runs_out(stepups, detector):
    detector.statuses = itertools.repeat(PENDING)
    status = await stepups.poll_challenge("c7")
    assert status.status is ChallengeState.PENDING


async def test_poll_interval_and_timeout_are_settings(make_stepups, detector, clock):
    stepups = make_stepups(poll_interval_s=1.0, poll_timeout_s=2.5)
    detector.statuses = itertools.repeat(PENDING)
    start = clock.now
    await stepups.poll_challenge("c7")
    assert detector.poll_times == [start, start + 1.0, start + 2.0, start + 2.5]
    assert clock.sleeps == [1.0, 1.0, 0.5]  # the last wait is cut short so it ends on the timeout


# --- a poll that fails ---


async def test_a_failed_poll_counts_as_pending_and_polling_goes_on(
    stepups, detector, capture, clock
):
    detector.statuses = iter([DetectorError("GET /v1/challenges/c1: HTTP 502"), PASSED])
    capture.answers = ["amber harbor 4 7 2"]
    assert await run(stepups, VOICE) is True
    assert len(detector.polls) == 2 and clock.sleeps == [0.25]


async def test_failed_polls_between_pending_ones_do_not_end_the_wait(stepups, detector, capture):
    error = DetectorError("GET /v1/challenges/c1: HTTP 502")
    detector.statuses = iter([PENDING, error, error, PENDING, PASSED])
    capture.answers = ["amber harbor 4 7 2"]
    assert await run(stepups, VOICE) is True
    assert len(detector.polls) == 5


async def test_polls_that_fail_until_the_deadline_fail_the_challenge(
    stepups, detector, capture, clock
):
    detector.fail["get_challenge"] = DetectorError("GET /v1/challenges/c1: HTTP 502")
    capture.answers = ["amber harbor 4 7 2"]
    start = clock.now
    assert await run(stepups, VOICE) is False  # and nothing is raised
    assert detector.poll_times == [start + 0.25 * i for i in range(33)]  # the whole 8 s
    assert clock.now == start + 8.0


async def test_poll_challenge_hands_back_pending_when_every_poll_fails(stepups, detector):
    detector.fail["get_challenge"] = DetectorError("down")
    status = await stepups.poll_challenge("c7")
    assert status.status is ChallengeState.PENDING
    assert len(detector.polls) == 33


@pytest.mark.parametrize(
    "error",
    [RuntimeError("bug"), ValueError("bad value"), OSError("reset"), asyncio.CancelledError()],
    ids=["runtime", "value", "os", "cancelled"],
)
async def test_any_other_error_in_a_poll_still_propagates(stepups, detector, capture, error):
    detector.statuses = iter([PENDING, error])
    capture.answers = ["amber harbor 4 7 2"]
    with pytest.raises(type(error)):
        await run(stepups, VOICE)
    assert len(detector.polls) == 2  # it did not go on polling


async def test_a_failed_poll_is_logged_by_its_type_only(stepups, detector, capture, caplog):
    caplog.set_level(logging.DEBUG, logger="callback_agent.stepups")
    error = DetectorError("GET /v1/sessions/s1/challenges/c1: HTTP 502 oops")
    detector.statuses = iter([error, PASSED])
    capture.answers = ["amber harbor 4 7 2"]
    assert await run(stepups, VOICE) is True
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "DetectorError" in warnings[0].getMessage()
    for detail in ("HTTP 502", "oops", "/v1/"):
        assert detail not in caplog.text
    assert all(r.exc_info is None for r in caplog.records)  # no traceback, which has the message


# --- the SMS code ---


@pytest.mark.parametrize("last_four, last2", [("0142", "42"), ("9917", "17"), ("0000", "00")])
async def test_otp_names_last_two_digits_and_checks_the_parsed_code(
    stepups, detector, capture, last_four, last2
):
    detector.sent = OtpSent(sent_to_last_four=last_four)
    capture.answers = [CODE]
    detector.verdicts = iter([True])
    assert await run(stepups, OTP) is True
    assert f"ending {last2}" in capture.lines[0]
    assert capture.lines == [prompts.OTP_PROMPT.format(last2=last2)]
    assert detector.otp_checks == ["424242"]  # answer "four two four two four two"


async def test_otp_is_sent_once_before_the_prompt_and_each_try_waits_30_seconds(
    stepups, detector, capture
):
    capture.answers = ["111111", "222222", "333333"]
    detector.verdicts = iter([False, False, False])
    await run(stepups, OTP)
    assert detector.calls[0] == ("send_otp", SESSION)
    assert detector.named("send_otp") == [("send_otp", SESSION)]
    assert [ask.timeout_s for ask in capture.asks] == [30.0, 30.0, 30.0]
    assert not any(ask.secret for ask in capture.asks)
    assert {call[1] for call in detector.calls} == {SESSION}


async def test_otp_gives_three_tries_then_fails(stepups, detector, capture):
    capture.answers = ["111111", "222222", "333333"]
    detector.verdicts = iter([False, False, False])
    assert await run(stepups, OTP) is False
    assert detector.otp_checks == ["111111", "222222", "333333"]
    # Nothing is said after the last try: the gate asks the detector what a failure means.
    assert capture.lines == [OTP_PROMPT, prompts.OTP_RETRY, prompts.OTP_RETRY]


async def test_otp_a_correct_code_on_a_later_try_passes_and_stops_asking(
    stepups, detector, capture
):
    capture.answers = ["111111", "four two four two four two"]
    detector.verdicts = iter([False, True])
    assert await run(stepups, OTP) is True
    assert detector.otp_checks == ["111111", "424242"]
    assert capture.lines == [OTP_PROMPT, prompts.OTP_RETRY]


async def test_otp_asks_for_six_digits_on_a_non_code_answer(stepups, detector, capture):
    capture.answers = [NOT_A_CODE, CODE]
    detector.verdicts = iter([True])
    assert await run(stepups, OTP) is True
    assert capture.lines == [OTP_PROMPT, prompts.OTP_SIX_DIGITS]
    assert detector.otp_checks == ["424242"]  # the non-code answer was never checked


@pytest.mark.parametrize(
    "spoken",
    ["", NOT_A_CODE, "four two four two four", "4 2 4 2 4 2 4", "415 555 0100"],
    ids=["silence", "no_digits", "five_digits", "seven_digits", "a_phone_number"],
)
async def test_otp_needs_exactly_six_digits_and_checks_nothing_else(
    stepups, detector, capture, spoken
):
    capture.answers = [spoken, spoken, spoken]
    assert await run(stepups, OTP) is False
    assert detector.otp_checks == []
    assert capture.lines == [OTP_PROMPT, prompts.OTP_SIX_DIGITS, prompts.OTP_SIX_DIGITS]


@pytest.mark.parametrize(
    "answers, verdicts, passed, lines",
    [
        (
            [NOT_A_CODE, "111111", CODE],
            [False, True],
            True,
            [OTP_PROMPT, prompts.OTP_SIX_DIGITS, prompts.OTP_RETRY],
        ),
        (
            ["111111", NOT_A_CODE, "222222"],
            [False, False],
            False,
            [OTP_PROMPT, prompts.OTP_RETRY, prompts.OTP_SIX_DIGITS],
        ),
        (
            [NOT_A_CODE, NOT_A_CODE, CODE],
            [True],
            True,
            [OTP_PROMPT, prompts.OTP_SIX_DIGITS, prompts.OTP_SIX_DIGITS],
        ),
        (
            [None, "111111", CODE],
            [False, True],
            True,
            [OTP_PROMPT, prompts.OTP_SIX_DIGITS, prompts.OTP_RETRY],
        ),
        (
            ["111111", None, NOT_A_CODE],
            [False],
            False,
            [OTP_PROMPT, prompts.OTP_RETRY, prompts.OTP_SIX_DIGITS],
        ),
    ],
    ids=[
        "non_code_wrong_right",
        "wrong_non_code_wrong",
        "non_code_non_code_right",
        "silence_wrong_right",
        "wrong_silence_non_code",
    ],
)
async def test_otp_silences_and_non_codes_use_up_tries_like_wrong_codes(
    stepups, detector, capture, answers, verdicts, passed, lines
):
    capture.answers = list(answers)
    detector.verdicts = iter(verdicts)
    assert await run(stepups, OTP) is passed
    assert capture.lines == lines
    assert len(detector.named("send_otp")) == 1


@pytest.mark.parametrize(
    "spoken, code",
    [
        ("four two four two four two", "424242"),
        ("424 242", "424242"),
        ("4-2-4, 2-4-2", "424242"),
        ("um the code is 4 2 4 2 4 2", "424242"),
        ("Four. Two. Four. Two. Four. Two.", "424242"),
        ("oh one two three four five", "012345"),
        ("zero zero zero zero zero seven", "000007"),
        ("double four two one three one", "442131"),
        ("triple seven one two three", "777123"),
        ("forty two forty two forty two", "424242"),
    ],
)
async def test_otp_checks_what_the_spoken_digit_parser_hears(
    stepups, detector, capture, spoken, code
):
    capture.answers = [spoken]
    detector.verdicts = iter([True])
    assert await run(stepups, OTP) is True
    assert detector.otp_checks == [parse_spoken_digits(spoken)] == [code]


async def test_otp_goes_only_to_the_number_on_file(stepups, detector, capture):
    # A caller who asks for the code to go to another number gets nothing sent: the one text went
    # to the number on file before they spoke, and what they say is only ever read for digits.
    capture.answers = ["send it to 415 555 0100 instead", CODE]
    detector.verdicts = iter([True])
    assert await run(stepups, OTP) is True
    assert detector.calls == [
        ("send_otp", SESSION),
        ("check_otp", SESSION, "424242"),
    ]
    assert capture.lines == [OTP_PROMPT, prompts.OTP_SIX_DIGITS]


async def test_otp_silence_uses_a_try_and_asks_for_six_digits(stepups, detector, capture):
    # No answer in the 30 s is a try used, like an answer that is not a code.
    capture.answers = [None, None, CODE]
    detector.verdicts = iter([True])
    assert await run(stepups, OTP) is True
    assert len(capture.asks) == 3
    assert capture.lines == [OTP_PROMPT, prompts.OTP_SIX_DIGITS, prompts.OTP_SIX_DIGITS]
    assert detector.otp_checks == ["424242"]  # nothing was checked for the silences
    assert len(detector.named("send_otp")) == 1


async def test_otp_three_silences_fail_without_a_check(stepups, detector, capture):
    capture.answers = [None, None, None]
    assert await run(stepups, OTP) is False
    assert len(capture.asks) == 3
    assert capture.lines == [OTP_PROMPT, prompts.OTP_SIX_DIGITS, prompts.OTP_SIX_DIGITS]
    assert detector.otp_checks == []


async def test_otp_silence_from_a_caller_who_has_gone_fails_after_a_single_ask(
    stepups, detector, capture, state
):
    state.closing = True
    capture.answers = [None]
    assert await run(stepups, OTP) is False
    assert capture.lines == [OTP_PROMPT]
    assert detector.otp_checks == []


@pytest.mark.parametrize(
    "before, verdicts",
    [([], []), (["111111"], [False]), ([NOT_A_CODE], []), ([None], [])],
    ids=["first_ask", "after_a_wrong_code", "after_a_non_code", "after_a_silence"],
)
async def test_caller_leaving_mid_otp_fails_the_factor_at_once(
    stepups, detector, capture, state, before, verdicts
):
    capture.answers = [*before, leaves(state)]
    detector.verdicts = iter(verdicts)
    assert await run(stepups, OTP) is False
    assert state.closing
    assert len(capture.asks) == len(before) + 1  # no further try is asked of nobody


async def test_caller_leaving_mid_capture_fails_the_factor(stepups, detector, capture):
    capture.answers = [None]
    assert await run(stepups, VOICE) is False
    assert detector.polls == []


async def test_a_challenge_never_passes_when_the_caller_has_gone(stepups, detector, capture, state):
    # Only the SMS code treats a caller who has gone differently from one who is silent. A
    # challenge with no answer fails and polls nothing, closing or not.
    capture.answers = [leaves(state)]
    assert await run(stepups, VOICE) is False
    assert state.closing and detector.polls == []


# --- the probe ---


@pytest.mark.parametrize(
    "answer",
    ["Sunny, why do you ask?", "It's raining.", "", "purple monkey dishwasher", None],
    ids=["on_topic", "plain", "empty", "off_topic", "silence_or_caller_left"],
)
async def test_probe_always_returns_true(stepups, detector, capture, answer):
    capture.answers = [answer]
    assert await run(stepups, PROBE) is True
    assert capture.asks == [Ask(prompts.PROBE_QUESTION, timeout_s=15.0, secret=False)]
    assert detector.calls == []  # the detector reads the timing from the call itself


# --- the callback ---


async def test_callback_without_a_callback_fails_closed(stepups, detector, capture):
    assert await run(stepups, CALLBACK) is False
    assert detector.calls == [] and capture.lines == []


# StepUps.callback itself is tested in test_stepups_callback.py. These check that run() hands the
# factor to it and passes back what it returns or raises.


@pytest.mark.parametrize("result", [True, False])
async def test_callback_runs_the_callback_method_with_the_request(
    stepups, detector, capture, monkeypatch, result
):
    seen: list[ActionRequest] = []

    async def callback(req: ActionRequest) -> bool:
        seen.append(req)
        return result

    monkeypatch.setattr(stepups, "callback", callback)
    assert await run(stepups, CALLBACK, RELEASE) is result
    assert len(seen) == 1 and seen[0] is RELEASE
    assert detector.calls == [] and capture.lines == []


async def test_a_callback_that_raises_propagates(stepups, monkeypatch):
    async def callback(req: ActionRequest) -> bool:
        raise RuntimeError("the leg broke")

    monkeypatch.setattr(stepups, "callback", callback)
    with pytest.raises(RuntimeError, match="the leg broke"):
        await run(stepups, CALLBACK)


# --- errors ---


@pytest.mark.parametrize(
    "factor, method, answers, lines",
    [
        (VOICE, "create_challenge", [], []),
        (OTP, "send_otp", [], []),  # no "I've texted a code" for a text that never went
        (OTP, "check_otp", [CODE], [OTP_PROMPT]),  # and no retry line for a check that never ran
    ],
    ids=["create_challenge", "send_otp", "check_otp"],
)
async def test_a_detector_error_propagates_for_the_gate_to_deny(
    stepups, detector, capture, factor, method, answers, lines
):
    detector.fail[method] = DetectorError(f"POST /v1/{method}: HTTP 502")
    capture.answers = list(answers)
    with pytest.raises(DetectorError):
        await run(stepups, factor)
    assert capture.lines == lines


@pytest.mark.parametrize("factor", [VOICE, OTP], ids=["voice_challenge", "sms_otp"])
async def test_a_step_up_without_a_session_raises_and_does_nothing(
    stepups, detector, capture, state, factor
):
    state.session_id = None
    with pytest.raises(RuntimeError):
        await run(stepups, factor)
    with pytest.raises(RuntimeError):
        await stepups.poll_challenge("c1")
    assert detector.calls == [] and capture.lines == []


# --- what is logged ---


async def test_logs_never_hold_the_phrase_the_code_or_what_the_caller_said(
    stepups, detector, capture, caplog
):
    caplog.set_level(logging.DEBUG, logger="callback_agent.stepups")
    detector.issued = ChallengeIssued(challenge_id="c1", say="quokka nebula 8675309", expires_at=1)
    detector.statuses = iter([PENDING, PASSED])
    detector.verdicts = iter([False, True])
    capture.answers = [
        "quokka nebula eight six seven five three oh nine",
        "one three five seven nine two",
        "banana smoothie",
        "one three five seven nine three",
    ]
    assert await run(stepups, VOICE) is True
    assert await run(stepups, OTP) is True
    assert caplog.records  # something is logged, or this proves nothing
    for secret in ("quokka", "nebula", "8675309", "135792", "135793", "banana", "seven"):
        assert secret not in caplog.text


# --- with the real gate ---


class GateDetector(FakeDetector):
    """A FakeDetector that also answers the gate's authorize calls, from `decisions`."""

    def __init__(self, clock: FakeClock) -> None:
        super().__init__(clock)
        self.decisions: list[tuple[DecisionKind, list[Factor]]] = []

    async def authorize(self, session_id: str, req) -> Decision:
        kind, factors = self.decisions.pop(0)
        return Decision(
            session_id=session_id,
            action=req.action,
            decision=kind,
            required_factors=factors,
        )


class Hangup:
    def __init__(self) -> None:
        self.ended = False

    async def end_call(self) -> None:
        self.ended = True


@pytest.fixture
async def rig(clock, capture, state):
    """A real gate on a real StepUps, with the fakes under them. The capture also speaks the gate's
    lines, as the voice does on a call. Has `gate`, `detector`, `ramp` and `hangup`."""
    detector, ramp, hangup = GateDetector(clock), FakeRamp(), Hangup()
    stepups = StepUps(detector, capture, state, clock=clock, sleep=clock.sleep)
    gate = Gate(
        detector=detector,
        ramp=ramp,
        speaker=capture,
        stepups=stepups,
        call=hangup,
        state=state,
    )
    yield SimpleNamespace(gate=gate, detector=detector, ramp=ramp, hangup=hangup)
    await gate.aclose()


async def test_passed_step_ups_let_the_gate_allow_the_change(rig, capture):
    rig.detector.decisions = [(DecisionKind.STEP_UP, [VOICE, OTP]), (DecisionKind.ALLOW, [])]
    rig.detector.statuses = iter([PENDING, PASSED])
    rig.detector.verdicts = iter([True])
    capture.answers = ["amber harbor 4 7 2", CODE]
    result = await rig.gate.run(BANK_CHANGE)
    assert result == GateResult("done", "Acme Supplies now pays to the account ending 6789.")
    assert rig.ramp.calls == [("add_default_bank_account", "v1", ROUTING, ACCOUNT)]
    assert capture.lines == [prompts.VERIFYING, CHALLENGE_PROMPT, OTP_PROMPT]


async def test_a_detector_error_inside_a_step_up_is_a_deny_at_the_gate(rig, state):
    rig.detector.decisions = [(DecisionKind.STEP_UP, [VOICE, OTP])]
    rig.detector.fail["create_challenge"] = DetectorError("POST /v1/challenges: HTTP 502")
    assert await rig.gate.run(BANK_CHANGE) == GateResult("denied", prompts.DENY_HELD)
    assert rig.ramp.calls == [("hold_vendor", "v1")]
    assert rig.hangup.ended and state.locked
    assert rig.detector.named("send_otp") == []  # the round ended where the step-up failed


async def test_polls_that_fail_are_a_failed_challenge_and_the_gate_asks_again(rig, capture, state):
    rig.detector.decisions = [(DecisionKind.STEP_UP, [VOICE]), (DecisionKind.DENY, [])]
    rig.detector.fail["get_challenge"] = DetectorError("down")
    capture.answers = ["amber harbor 4 7 2"]
    assert await rig.gate.run(BANK_CHANGE) == GateResult("denied", prompts.DENY_HELD)
    assert rig.detector.decisions == []  # the gate asked a second time: no exception ended round 1
    assert rig.ramp.calls == [("hold_vendor", "v1")]


async def test_a_callback_with_no_callback_never_lets_the_change_through(rig, state):
    rig.detector.decisions = [(DecisionKind.STEP_UP, [CALLBACK])] * 6  # the detector keeps asking
    assert await rig.gate.run(BANK_CHANGE) == GateResult("denied", prompts.DENY_HELD)
    assert rig.ramp.calls == [("hold_vendor", "v1")]  # held, and never changed
    assert rig.hangup.ended and state.locked


# --- the answer window of a secret ask (redaction) ---


async def test_a_challenge_answer_is_exempt_only_while_it_is_asked(
    stepups, detector, capture, state
):
    detector.statuses = iter([PASSED])
    seen: list[bool] = []

    def answer() -> str:
        seen.append(state.answer_exempt)
        return "amber harbor 4 7 2"

    capture.answers = [answer]
    assert state.answer_exempt is False
    await run(stepups, VOICE)
    assert seen == [True]
    assert state.answer_exempt is False


async def test_the_exemption_is_cleared_when_the_ask_fails(stepups, detector, capture, state):
    capture.answers = [RuntimeError("boom")]
    with pytest.raises(RuntimeError):
        await run(stepups, VOICE)
    assert state.answer_exempt is False


async def test_an_otp_answer_is_never_exempt(stepups, detector, capture, state):
    detector.verdicts = iter([True])
    seen: list[bool] = []

    def answer() -> str:
        seen.append(state.answer_exempt)
        return "four two four two four two"

    capture.answers = [answer]
    await run(stepups, OTP)
    assert seen == [False]
