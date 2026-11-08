"""Tests for the callback to the number on file (StepUps.callback), with a fake leg, a fake
detector and a fake capture.

The three fakes write to one `events` list, so a test can check the order of everything the
callback does. The fake leg records each number it is asked to dial in `dialed`, and answers with
the identity of whoever picked up, or None. The fake detector serves the user directory and keeps
each caller update and factor report. The fake capture records every line said or asked, in
order, as `lines`, and answers each ask from a script. The leg and the capture also note the value
of state.callback_in_progress each time they are called.
"""

import asyncio
import inspect
import logging
import typing
from contextlib import suppress
from dataclasses import dataclass

import pytest
from callback_shared.contract import (
    CallbackResult,
    CallerUpdate,
    Channel,
    Decision,
    DecisionKind,
    Factor,
    FactorReport,
    UserOnFile,
    UserSummary,
)

from callback_agent import prompts
from callback_agent import stepups as stepups_module
from callback_agent.actions import ActionRequest
from callback_agent.detector import DetectorError
from callback_agent.spoken import parse_yes_no
from callback_agent.state import CallState
from callback_agent.stepups import CallbackLeg, StepUps

SESSION = "s1"
NUMBER_ON_FILE = "+15555550100"
CALLEE = "callback-1a2b3c4d"
CLAIMED = UserSummary(user_id="u1", display_name="Dana Reyes", role="CFO", phone_last_four="0100")
ON_FILE = UserOnFile(**CLAIMED.model_dump(), phone_e164=NUMBER_ON_FILE)
# Someone else in the directory, so a test can tell whose number was looked up.
SOMEONE_ELSE = UserOnFile(
    user_id="u2",
    display_name="Sam Ortiz",
    role="AP clerk",
    phone_last_four="0199",
    phone_e164="+15555550199",
)
BANK_CHANGE = ActionRequest.update_vendor_bank_account(
    "v1", "Acme Supplies", "021000021", "000123456789"
)
RELEASE = ActionRequest.release_vendor_hold("v1", "Acme Supplies")
CONFIRM = prompts.CALLBACK_CONFIRM.format(
    pending_action="change Acme Supplies' bank account to one ending 6789"
)
CONFIRMED = FactorReport(result=CallbackResult.CONFIRMED)
DENIED = FactorReport(result=CallbackResult.DENIED)


class ScriptExhausted(Exception):
    """A fake was asked for an answer its test never scripted."""


@dataclass(frozen=True)
class Ask:
    prompt: str
    timeout_s: float
    secret: bool


class FakeLeg:
    """Stands in for LiveKitCallbackLeg. Records each number it is asked to dial in `dialed`, and
    answers with `answer`: who picked up, None for nobody, or an exception to raise. `flags` keeps
    the value state.callback_in_progress had at each dial. Unlike the LiveKit leg, it never sets
    or clears that flag itself. `removed_the_caller` makes it act like the LiveKit leg after it has
    taken the caller off the line: when nobody answers, it marks the call as ending."""

    def __init__(self, events: list[tuple], state: CallState) -> None:
        self.dialed: list[str] = []
        self.flags: list[bool] = []
        self.answer: str | None | BaseException = CALLEE
        self.removed_the_caller = False
        self._events = events
        self._state = state

    async def dial_and_link(self, phone_e164: str) -> str | None:
        self.dialed.append(phone_e164)
        self.flags.append(self._state.callback_in_progress)
        self._events.append(("dial", phone_e164))
        if isinstance(self.answer, BaseException):
            raise self.answer
        if self.answer is None and self.removed_the_caller:
            self._state.closing = True
        return self.answer


class FakeDetector:
    """Serves the user directory from `users`, and records each call in `events` as (method,
    *arguments) before it answers. `fail` maps a method name to the exception that call raises.
    `caller_updates` and `factors` keep what update_caller and report_factor were sent."""

    def __init__(self, events: list[tuple]) -> None:
        self.users = {user.user_id: user for user in (ON_FILE, SOMEONE_ELSE)}
        self.caller_updates: list[CallerUpdate] = []
        self.factors: list[FactorReport] = []
        self.fail: dict[str, Exception] = {}
        self._events = events

    async def get_user(self, user_id: str) -> UserOnFile:
        self._record("get_user", user_id)
        return self.users[user_id]

    async def update_caller(self, session_id: str, req: CallerUpdate) -> None:
        self._record("update_caller", session_id, req)
        self.caller_updates.append(req)

    async def report_factor(self, session_id: str, report: FactorReport) -> None:
        self._record("report_factor", session_id, report)
        self.factors.append(report)

    def _record(self, method: str, *args) -> None:
        self._events.append((method, *args))
        if method in self.fail:
            raise self.fail[method]


class FakeCapture:
    """Records each line said or asked, in order, as `lines`, and answers each ask from `answers`:
    a string, or None for no answer in the whole wait. `asks` keeps what each ask was given, and
    `flags` each line with the value state.callback_in_progress had as it was said or asked.
    `callee_hangs_up` makes whoever answers hang up on the question: state.closing is up when the
    answer comes back, as the worker's hang-up handler leaves it."""

    def __init__(self, events: list[tuple], state: CallState) -> None:
        self.lines: list[str] = []
        self.flags: list[tuple[str, bool]] = []
        self.asks: list[Ask] = []
        self.answers: list[str | None] = []
        self.callee_hangs_up = False
        self._events = events
        self._state = state

    async def say(self, text: str) -> None:
        self.lines.append(text)
        self.flags.append((text, self._state.callback_in_progress))
        self._events.append(("say", text))

    async def ask(self, prompt: str, *, timeout_s: float, secret: bool = False) -> str | None:
        self.lines.append(prompt)
        self.flags.append((prompt, self._state.callback_in_progress))
        self.asks.append(Ask(prompt, timeout_s, secret))
        self._events.append(("ask", prompt))
        if not self.answers:
            raise ScriptExhausted(f"nobody scripted an answer to ask number {len(self.asks)}")
        if self.callee_hangs_up:
            self._state.closing = True
        return self.answers.pop(0)


@pytest.fixture
def events() -> list[tuple]:
    return []


@pytest.fixture
def state() -> CallState:
    return CallState(
        room_name="r1",
        caller_identity="caller-1",
        channel=Channel.PHONE,
        session_id=SESSION,
        claimed_user=CLAIMED,
    )


@pytest.fixture
def leg(events, state) -> FakeLeg:
    return FakeLeg(events, state)


@pytest.fixture
def detector(events) -> FakeDetector:
    return FakeDetector(events)


@pytest.fixture
def capture(events, state) -> FakeCapture:
    return FakeCapture(events, state)


@pytest.fixture
def stepups(detector, capture, state, leg) -> StepUps:
    return StepUps(detector, capture, state, callback_leg=leg)


def heard_after_the_dial(events: list[tuple]) -> list[str]:
    """Every line said or asked after the dial: what the person who answered heard."""
    dialed = next(i for i, event in enumerate(events) if event[0] == "dial")
    return [event[1] for event in events[dialed + 1 :] if event[0] in ("say", "ask")]


# --- the interface other code is built against ---


def test_callback_leg_has_exactly_dial_and_link():
    assert typing.get_protocol_members(CallbackLeg) == {"dial_and_link"}
    assert list(inspect.signature(CallbackLeg.dial_and_link).parameters) == ["self", "phone_e164"]
    # The fake is the same shape, so what passes here would pass with the LiveKit leg.
    assert inspect.signature(FakeLeg.dial_and_link) == inspect.signature(CallbackLeg.dial_and_link)


# --- who is called, and what they are asked ---


async def test_browser_session_never_dials(stepups, leg, state, events):
    state.channel = Channel.BROWSER
    assert await stepups.callback(BANK_CHANGE) is False
    assert leg.dialed == []
    assert events == []  # nothing looked up, said or reported


async def test_dials_only_the_number_on_file(stepups, leg, capture, events):
    capture.answers = ["yes"]
    assert await stepups.callback(BANK_CHANGE) is True
    assert leg.dialed == ["+15555550100"]  # from detector.get_user, never from the transcript
    assert [event for event in events if event[0] == "get_user"] == [("get_user", "u1")]


async def test_no_answer_is_reported(stepups, leg, detector, capture):
    leg.answer = None
    assert await stepups.callback(BANK_CHANGE) is False
    assert detector.factors == [FactorReport(result=CallbackResult.NO_ANSWER)]
    assert detector.caller_updates == []
    assert capture.lines == [prompts.CALLBACK_LINE]  # nothing is asked of nobody


async def test_no_answer_once_the_caller_is_gone_is_reported_and_then_raises(
    stepups, leg, detector, capture, state, events
):
    # The leg took the caller off the line and nobody took their place, so there is no one to ask
    # again. A False would have the gate ask the detector again, and the factor it named next (a
    # code to the number on file, another call) would reach the real user for nothing. Raising
    # makes the gate deny at once, which holds the vendor and hangs up.
    leg.answer = None
    leg.removed_the_caller = True
    with pytest.raises(stepups_module.CallbackNoAnswer) as raised:
        await stepups.callback(BANK_CHANGE)
    assert detector.factors == [FactorReport(result=CallbackResult.NO_ANSWER)]  # told first
    assert events[-1] == ("report_factor", SESSION, FactorReport(result=CallbackResult.NO_ANSWER))
    assert detector.caller_updates == []
    assert capture.lines == [prompts.CALLBACK_LINE]  # and nothing said to whoever is not there
    assert state.callback_in_progress is False and state.closing is True
    # The gate logs the traceback of a step-up that raises, so the message holds no number.
    assert "5555550100" not in str(raised.value)


async def test_run_lets_the_no_answer_error_through_to_the_gate(stepups, leg):
    leg.answer = None
    leg.removed_the_caller = True
    decision = Decision(
        session_id=SESSION,
        action=BANK_CHANGE.action,
        decision=DecisionKind.STEP_UP,
        required_factors=[Factor.CALLBACK_ON_FILE],
    )
    with pytest.raises(stepups_module.CallbackNoAnswer):
        await stepups.run(Factor.CALLBACK_ON_FILE, BANK_CHANGE, decision)


async def test_yes_points_detector_at_new_caller_then_confirms(stepups, detector, capture):
    capture.answers = ["yes"]
    assert await stepups.callback(BANK_CHANGE) is True
    assert detector.caller_updates == [
        CallerUpdate(caller_identity="callback-1a2b3c4d", channel=Channel.PHONE)
    ]
    assert detector.factors == [FactorReport(result=CallbackResult.CONFIRMED)]


@pytest.mark.parametrize(
    "answer",
    ["no", "No, that wasn't me.", "", "Sorry, who is this?", "I'm sure it's a scam", "yes?"],
    ids=["no", "negated", "empty_turn", "question", "agreement_word_inside", "yes_as_a_question"],
)
async def test_no_or_silence_is_denied(stepups, detector, capture, answer):
    assert parse_yes_no(answer) is not True  # what makes each of these not a yes
    capture.answers = [answer]
    assert await stepups.callback(BANK_CHANGE) is False
    assert detector.factors == [FactorReport(result=CallbackResult.DENIED)]


@pytest.mark.parametrize(
    "req, question",
    [
        (
            BANK_CHANGE,
            "I'm calling you back about your request. Did you just ask me to change Acme "
            "Supplies' bank account to one ending 6789?",
        ),
        (
            RELEASE,
            "I'm calling you back about your request. Did you just ask me to release the hold on "
            "Acme Supplies?",
        ),
    ],
    ids=["bank_change", "release"],
)
async def test_confirm_question_names_the_pending_action(stepups, capture, req, question):
    capture.answers = ["yes"]
    await stepups.callback(req)
    assert capture.asks == [Ask(question, timeout_s=20.0, secret=False)]


async def test_disclosure_repeats_before_the_confirm_question(stepups, capture, events):
    capture.answers = ["yes"]
    await stepups.callback(BANK_CHANGE)
    # CALLBACK_LINE went to the caller before the dial. The new call opens with the disclosure.
    assert heard_after_the_dial(events)[:2] == [
        prompts.DISCLOSURE,
        prompts.CALLBACK_CONFIRM.format(
            pending_action="change Acme Supplies' bank account to one ending 6789"
        ),
    ]


async def test_no_answer_to_the_confirm_question_is_denied(stepups, detector, capture):
    capture.answers = [None]  # the 20 s passed, or whoever answered hung up
    assert await stepups.callback(BANK_CHANGE) is False
    assert detector.factors == [FactorReport(result=CallbackResult.DENIED)]


# --- a callee who hangs up on the confirm question (R79) ---
#
# The caller was taken off the line to be called back, so nobody is left to ask again, and the
# gate would otherwise stop at its next round without holding the vendor. The step-up reports the
# callee's silence as a no, as it always did, and then raises, so that the gate denies on the spot.


async def test_a_callee_who_hangs_up_during_the_confirm_question_is_denied_and_then_raises(
    stepups, detector, capture, state, events
):
    capture.answers = [None]
    capture.callee_hangs_up = True
    with pytest.raises(stepups_module.CallbackCalleeHungUp) as raised:
        await stepups.callback(BANK_CHANGE)
    assert isinstance(raised.value, stepups_module.CallbackNoAnswer)  # the gate's one kind of exit
    assert detector.factors == [DENIED]  # told first, as for any silence
    assert events[-1] == ("report_factor", SESSION, DENIED)
    assert state.callback_in_progress is False
    assert "5555550100" not in str(raised.value)  # the gate logs the traceback of the error


@pytest.mark.parametrize("callee_hangs_up", [False, True], ids=["silent_callee", "callee_left"])
async def test_the_error_for_a_callee_who_left_is_raised_only_when_the_call_is_ending(
    stepups, detector, capture, callee_hangs_up
):
    capture.answers = [None]  # nothing heard in the whole wait
    capture.callee_hangs_up = callee_hangs_up
    if callee_hangs_up:
        with pytest.raises(stepups_module.CallbackCalleeHungUp):
            await stepups.callback(BANK_CHANGE)
    else:
        assert await stepups.callback(BANK_CHANGE) is False  # a callee who stays silent
    assert detector.factors == [DENIED]


async def test_a_callee_who_said_no_before_hanging_up_is_denied_without_the_error(
    stepups, detector, capture
):
    # Only silence is turned into the error. A no is an answer, and the detector has it.
    capture.answers = ["no"]
    capture.callee_hangs_up = True
    assert await stepups.callback(BANK_CHANGE) is False
    assert detector.factors == [DENIED]


async def test_run_lets_the_callee_hung_up_error_through_to_the_gate(stepups, capture):
    capture.answers = [None]
    capture.callee_hangs_up = True
    decision = Decision(
        session_id=SESSION,
        action=BANK_CHANGE.action,
        decision=DecisionKind.STEP_UP,
        required_factors=[Factor.CALLBACK_ON_FILE],
    )
    with pytest.raises(stepups_module.CallbackNoAnswer):
        await stepups.run(Factor.CALLBACK_ON_FILE, BANK_CHANGE, decision)


async def test_the_callback_runs_in_order(stepups, capture, events):
    capture.answers = ["yes"]
    assert await stepups.callback(BANK_CHANGE) is True
    assert events == [
        ("get_user", "u1"),
        ("say", prompts.CALLBACK_LINE),
        ("dial", NUMBER_ON_FILE),
        ("update_caller", SESSION, CallerUpdate(caller_identity=CALLEE, channel=Channel.PHONE)),
        ("say", prompts.DISCLOSURE),
        ("ask", CONFIRM),
        ("report_factor", SESSION, CONFIRMED),
    ]


# --- failing closed ---


async def test_without_a_leg_it_fails_closed_and_calls_no_one(detector, capture, state, events):
    stepups = StepUps(detector, capture, state)
    assert await stepups.callback(BANK_CHANGE) is False
    assert events == []  # no detector call, and nothing said


async def test_a_call_that_is_already_ending_is_not_called_back(stepups, state, events):
    state.closing = True
    assert await stepups.callback(BANK_CHANGE) is False
    assert events == []  # nothing looked up, said, dialed or reported
    assert state.callback_in_progress is False


async def test_a_caller_who_hangs_up_while_the_number_is_looked_up_is_not_called_back(
    stepups, detector, capture, leg, state
):
    look_up = detector.get_user

    async def hang_up_during(user_id: str) -> UserOnFile:
        state.closing = True  # an ordinary hangup: the flag isn't up yet
        return await look_up(user_id)

    detector.get_user = hang_up_during
    assert await stepups.callback(BANK_CHANGE) is False
    assert capture.flags == [] and leg.dialed == []  # nothing said, nobody dialed
    assert detector.factors == []
    assert state.callback_in_progress is False


@pytest.mark.parametrize("missing", ["session_id", "claimed_user"])
async def test_a_callback_before_the_caller_is_identified_raises_and_calls_no_one(
    stepups, state, events, missing
):
    setattr(state, missing, None)
    with pytest.raises(RuntimeError):
        await stepups.callback(BANK_CHANGE)
    assert events == []


async def test_a_directory_answer_for_someone_else_is_never_dialed(stepups, detector, leg, events):
    detector.users["u1"] = SOMEONE_ELSE
    with pytest.raises(DetectorError):
        await stepups.callback(BANK_CHANGE)
    assert leg.dialed == []
    assert events == [("get_user", "u1")]  # and nothing said


@pytest.mark.parametrize(
    "method, answers, lines",
    [
        ("get_user", [], []),  # nothing is said before the number is known
        ("update_caller", [], [prompts.CALLBACK_LINE]),  # nothing is asked of whoever answered
        # A yes the detector never heard about is not a pass.
        ("report_factor", ["yes"], [prompts.CALLBACK_LINE, prompts.DISCLOSURE, CONFIRM]),
    ],
    ids=["get_user", "update_caller", "report_factor"],
)
async def test_a_detector_error_propagates_for_the_gate_to_deny(
    stepups, detector, capture, method, answers, lines
):
    detector.fail[method] = DetectorError(f"{method}: HTTP 502")
    capture.answers = list(answers)
    with pytest.raises(DetectorError):
        await stepups.callback(BANK_CHANGE)
    assert capture.lines == lines


@pytest.mark.parametrize(
    "error", [RuntimeError("the leg broke"), asyncio.CancelledError()], ids=["error", "cancelled"]
)
async def test_an_error_from_the_leg_propagates_unreported(stepups, leg, detector, capture, error):
    leg.answer = error
    with pytest.raises(type(error)):
        await stepups.callback(BANK_CHANGE)
    assert detector.factors == [] and detector.caller_updates == []
    assert capture.lines == [prompts.CALLBACK_LINE]


# --- callback_in_progress ---


async def test_the_flag_is_up_while_the_callback_line_is_said(stepups, leg, capture, state):
    capture.answers = ["yes"]
    assert state.callback_in_progress is False
    assert await stepups.callback(BANK_CHANGE) is True
    # Up as the caller hears they'll be hung up on, so hanging up then can't end the call before
    # the dial, and still up when the leg dials.
    assert capture.flags[0] == (prompts.CALLBACK_LINE, True)
    assert leg.flags == [True]
    assert state.callback_in_progress is False


def confirmed(leg, detector, capture, state) -> None:
    capture.answers = ["yes"]


def denied(leg, detector, capture, state) -> None:
    capture.answers = ["no"]


def nobody_answered(leg, detector, capture, state) -> None:
    leg.answer = None


def nobody_answered_and_the_caller_is_gone(leg, detector, capture, state) -> None:
    leg.answer = None
    leg.removed_the_caller = True


def callee_hung_up_on_the_confirm_question(leg, detector, capture, state) -> None:
    capture.answers = [None]
    capture.callee_hangs_up = True


def leg_raised(leg, detector, capture, state) -> None:
    leg.answer = RuntimeError("the leg broke")


def leg_cancelled(leg, detector, capture, state) -> None:
    leg.answer = asyncio.CancelledError()


def get_user_raised(leg, detector, capture, state) -> None:
    detector.fail["get_user"] = DetectorError("GET /v1/users/u1: HTTP 502")


def report_raised(leg, detector, capture, state) -> None:
    capture.answers = ["yes"]
    detector.fail["report_factor"] = DetectorError("POST /v1/sessions/s1/factors: HTTP 502")


def browser_session(leg, detector, capture, state) -> None:
    state.channel = Channel.BROWSER


def no_leg(leg, detector, capture, state) -> None:
    pass  # the test builds the StepUps without the leg


def call_ending(leg, detector, capture, state) -> None:
    state.closing = True


@pytest.mark.parametrize("before", [False, True], ids=["down_before", "up_before"])
@pytest.mark.parametrize(
    "way_out",
    [
        confirmed,
        denied,
        nobody_answered,
        nobody_answered_and_the_caller_is_gone,
        callee_hung_up_on_the_confirm_question,
        leg_raised,
        leg_cancelled,
        get_user_raised,
        report_raised,
        browser_session,
        no_leg,
        call_ending,
    ],
    ids=lambda way_out: way_out.__name__,
)
async def test_the_flag_is_down_after_every_way_out(detector, capture, state, leg, way_out, before):
    stepups = StepUps(detector, capture, state, callback_leg=None if way_out is no_leg else leg)
    way_out(leg, detector, capture, state)
    # Up before is a flag left up by something else: a way out that never raises the flag, such
    # as a browser session, still has to lower it.
    state.callback_in_progress = before
    # Some ways out raise: this test is about the flag.
    with suppress(Exception, asyncio.CancelledError):
        await stepups.callback(BANK_CHANGE)
    assert state.callback_in_progress is False


# --- run() ---


async def test_run_hands_the_callback_factor_to_the_method(
    stepups, leg, detector, capture, monkeypatch
):
    seen: list[ActionRequest] = []
    method = stepups.callback

    async def spy(req: ActionRequest) -> bool:
        seen.append(req)
        return await method(req)

    monkeypatch.setattr(stepups, "callback", spy)
    capture.answers = ["yes"]
    decision = Decision(
        session_id=SESSION,
        action=BANK_CHANGE.action,
        decision=DecisionKind.STEP_UP,
        required_factors=[Factor.CALLBACK_ON_FILE],
    )
    assert await stepups.run(Factor.CALLBACK_ON_FILE, BANK_CHANGE, decision) is True
    assert len(seen) == 1 and seen[0] is BANK_CHANGE
    assert leg.dialed == [NUMBER_ON_FILE] and detector.factors == [CONFIRMED]


# --- what is logged ---


async def test_logs_never_hold_the_number_the_answer_or_the_request(stepups, capture, caplog):
    caplog.set_level(logging.DEBUG, logger="callback_agent.stepups")
    capture.answers = ["absolutely, go ahead", "nope, never asked for that, pal"]
    assert await stepups.callback(BANK_CHANGE) is True
    assert await stepups.callback(BANK_CHANGE) is False
    assert caplog.records  # something is logged, or this proves nothing
    for secret in ("5555550100", "absolutely", "ahead", "nope", "pal", "Acme", "6789"):
        assert secret not in caplog.text
