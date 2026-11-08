"""The per-call pieces the worker puts together (Tasks 6 and 14): the Ramp client, the hang-up, the
gate and toolbox with the step-ups, the callback leg and the enrollment under them, the room
options, the hang-up the worker owns when the caller leaves, the risk packets, and the end of the
call with the gate and Ramp in it.

worker.py reads its settings when it is imported, so a test cannot import it. These pieces live in
session_wiring.py, next to the rest of the wiring, for that reason. tests/test_structure.py checks
that the worker gets the gate, the toolbox and the room options from here and builds none by hand,
and the tests that use worker_call below read from its source what else it calls and in what
order. A few tests start a real AgentSession over a room that is not connected, to see what
LiveKit does with the options and with a hang-up.
"""

import ast
import asyncio
import contextlib
import dataclasses
import gc
import inspect
import logging
import typing
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest
from callback_shared.contract import (
    RISK_TOPIC,
    CallbackResult,
    CallerUpdate,
    Channel,
    Decision,
    DecisionKind,
    Factor,
    FactorReport,
    OtpSent,
    RiskUpdate,
    Role,
    Turn,
    UserOnFile,
    UserSummary,
)
from fake_llm import FakeLLM
from fake_ramp import FakeRamp
from livekit import api, rtc
from livekit.agents import Agent, AgentSession, JobContext, room_io
from livekit.rtc.room import EventTypes

import callback_agent.session_wiring as wiring
from callback_agent import prompts
from callback_agent.actions import ActionRequest
from callback_agent.callback_leg import LiveKitCallbackLeg
from callback_agent.config import Settings
from callback_agent.gate import GateResult
from callback_agent.ramp import RampClient
from callback_agent.session_wiring import (
    TOOL_HANDLING,
    RoomHangup,
    build_ramp,
    build_room_options,
    build_tools,
    finish_call,
    wire_risk_updates,
)
from callback_agent.state import CallState
from callback_agent.turns import Forwarder

TURN = Turn(role=Role.USER, text="my code is 482913", start_ms=1_000, end_ms=2_000)


def new_state(**fields) -> CallState:
    return CallState(room_name="room", caller_identity="caller", channel=Channel.PHONE, **fields)


# --- the state of a call ---


def test_each_call_has_its_own_bills_and_no_risk_update_yet():
    first, second = new_state(), new_state()
    first.bills_seen["b1"] = "v1"
    assert second.bills_seen == {} and first.latest_risk is None and second.latest_risk is None


# --- what the LLM is told about a call it should not repeat ---


def test_the_reject_text_is_the_one_livekit_will_render_for_a_turned_away_call():
    # LiveKit renders it with str.format over function_name, fnc_calls_json and fnc_calls_text
    options = TOOL_HANDLING["async_options"]
    template = options["duplicate_reject_template"]
    said = template.format(
        function_name="release_vendor_hold", fnc_calls_json=["{...}"], fnc_calls_text="{...}"
    )
    assert "`release_vendor_hold` is already running" in said and "{...}" not in said


def test_nothing_else_about_the_async_tool_handling_is_changed():
    assert set(TOOL_HANDLING) == {"async_options"}
    assert set(TOOL_HANDLING["async_options"]) == {"duplicate_reject_template"}


# --- the Ramp client of a call ---


def test_the_ramp_client_is_built_from_the_settings():
    settings = Settings(
        agent_name="callback",
        cartesia_voice_id="voice",
        ramp_base_url="https://ramp.test",
        ramp_client_id="id",
        ramp_client_secret="secret",
    )
    ramp = build_ramp(settings)
    assert isinstance(ramp, RampClient)
    assert ramp._base_url == "https://ramp.test"  # the one private read: nothing public says it


@pytest.mark.parametrize("missing", ["ramp_client_id", "ramp_client_secret"])
def test_a_ramp_client_needs_both_credentials(missing):
    fields = {"ramp_client_id": "id", "ramp_client_secret": "secret", missing: None}
    with pytest.raises(ValueError, match="RAMP_CLIENT_ID and RAMP_CLIENT_SECRET"):
        build_ramp(Settings(agent_name="callback", cartesia_voice_id="voice", **fields))


# --- the hang-up ---


class FakeJob:
    """The part of a JobContext that the hang-up uses."""

    def __init__(self) -> None:
        self.deleted = 0
        self.answer = asyncio.Event()
        self.answer.set()
        self.error: Exception | None = None

    def delete_room(self) -> "asyncio.Future[None]":
        self.deleted += 1

        async def request() -> None:
            await self.answer.wait()
            if self.error is not None:
                raise self.error

        return asyncio.ensure_future(request())


async def test_hanging_up_deletes_the_room_and_marks_the_call_as_ending():
    job, state = FakeJob(), new_state()
    await RoomHangup(job, state).end_call()
    assert job.deleted == 1 and state.closing


async def test_the_call_is_marked_as_ending_before_the_room_answers():
    job, state = FakeJob(), new_state()
    job.answer.clear()
    hanging_up = asyncio.create_task(RoomHangup(job, state).end_call())
    for _ in range(5):
        await asyncio.sleep(0)
    assert state.closing and not hanging_up.done()  # a queued gated request already stops
    job.answer.set()
    await asyncio.wait_for(hanging_up, timeout=5)


async def test_a_room_that_never_answers_cannot_hold_the_caller_of_end_call_forever(monkeypatch):
    # The gate calls this with its lock held, so a request that never answered would queue every
    # gated request behind it. The hang-up gives up on its own, long before the test's own limit.
    monkeypatch.setattr(wiring, "HANGUP_S", 0.05)
    job = FakeJob()
    job.answer.clear()
    started = asyncio.get_running_loop().time()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(RoomHangup(job, new_state()).end_call(), timeout=5)
    assert asyncio.get_running_loop().time() - started < 2


async def test_a_failed_request_reaches_the_caller_of_end_call():
    job = FakeJob()
    job.error = RuntimeError("the API is down")
    with pytest.raises(RuntimeError, match="the API is down"):
        await RoomHangup(job, new_state()).end_call()


def test_the_hangup_calls_delete_room_the_way_livekits_job_context_takes_it():
    # JobContext can't be built in a test without leaking its temporary directory, so check the
    # shape of the call instead: no required argument, and a future back (not a coroutine function).
    parameters = inspect.signature(JobContext.delete_room).parameters
    assert [n for n, p in parameters.items() if n != "self" and p.default is p.empty] == []
    assert not inspect.iscoroutinefunction(JobContext.delete_room)


# --- the gate, the step-ups and the enrollment of a call ---

BANK = ActionRequest.update_vendor_bank_account("v1", "Acme", "021000021", "000123456789")
ERIC = UserSummary(user_id="eric", display_name="Eric Catalano", role="CFO", phone_last_four="0100")
NUMBER_ON_FILE = "+15555550100"  # the detector's directory has this for ERIC
CALLEE = "callback-1a2b3c4d"  # who picks up when the callback is answered


class RecordingVoice:
    """Stands in for LiveKitVoice, which needs a live session. It keeps the session it was built
    over, each line said and each question asked, and answers the questions from `answers` (with
    silence once they run out)."""

    def __init__(self, session) -> None:
        self.session = session
        self.said: list[str] = []
        self.asked: list[str] = []
        self.answers: list[str | None] = []

    async def say(self, text: str) -> None:
        self.said.append(text)

    async def ask(self, prompt: str, *, timeout_s: float, secret: bool = False) -> str | None:
        self.asked.append(prompt)
        return self.answers.pop(0) if self.answers else None


class ScriptedDetector:
    """What the gate, the step-ups and the enrollment ask of the detector. `decisions` are the
    answers to authorize, in order, as (kind, factors)."""

    def __init__(self) -> None:
        self.decisions: list[tuple[DecisionKind, list[Factor]]] = []
        self.otps_sent: list[str] = []
        self.caller_updates: list[CallerUpdate] = []
        self.factors: list[FactorReport] = []

    async def authorize(self, session_id: str, req) -> Decision:
        kind, factors = self.decisions.pop(0)
        return Decision(
            session_id=session_id, action=req.action, decision=kind, required_factors=factors
        )

    async def send_otp(self, session_id: str) -> OtpSent:
        self.otps_sent.append(session_id)
        return OtpSent(sent_to_last_four="0100")

    async def get_user(self, user_id: str) -> UserOnFile:
        return UserOnFile(**ERIC.model_dump(), phone_e164=NUMBER_ON_FILE)

    async def update_caller(self, session_id: str, update: CallerUpdate) -> None:
        self.caller_updates.append(update)

    async def report_factor(self, session_id: str, report: FactorReport) -> None:
        self.factors.append(report)


class FakeHangup:
    def __init__(self) -> None:
        self.ended = False

    async def end_call(self) -> None:
        self.ended = True


@pytest.fixture
def voices(monkeypatch) -> list[RecordingVoice]:
    """The voices the wiring builds: a call is built over a LiveKitVoice, so it is swapped for one
    that records, and the test sees how many there were."""
    built: list[RecordingVoice] = []

    def make(session) -> RecordingVoice:
        built.append(RecordingVoice(session))
        return built[-1]

    monkeypatch.setattr(wiring, "LiveKitVoice", make)
    return built


class FakeLeg:
    """Stands in for LiveKitCallbackLeg, which needs a live job. It keeps what it was built over
    (when the wiring builds it) and each number it is asked to dial, and answers with `answer`:
    the identity of whoever picked up, or None for nobody."""

    def __init__(self, ctx=None, session=None, state=None, settings=None) -> None:
        self.built_over = (ctx, session, state, settings)
        self.dialed: list[str] = []
        self.answer: str | None = None

    async def dial_and_link(self, phone_e164: str) -> str | None:
        self.dialed.append(phone_e164)
        return self.answer


def build_wired(voices, callback_leg, state: CallState | None = None) -> SimpleNamespace:
    """A call built by build_tools, on an identified caller (`state`, if the test made it), over
    fakes."""
    session = object()
    state = state or new_state(session_id="s1", claimed_user=ERIC)
    detector, ramp, hangup = ScriptedDetector(), FakeRamp(), FakeHangup()
    tools = build_tools(
        session,
        state,
        detector=detector,
        forwarder=None,
        ramp=ramp,
        call=hangup,
        callback_leg=callback_leg,
    )
    return SimpleNamespace(
        tools=tools,
        session=session,
        state=state,
        detector=detector,
        ramp=ramp,
        hangup=hangup,
        voice=voices[0],
        leg=callback_leg,
    )


@pytest.fixture
async def wired(voices):
    """A call with no callback leg: the trunk and the Telnyx number are not set up."""
    call = build_wired(voices, None)
    yield call
    await call.tools.gate.aclose()


@pytest.fixture
async def wired_with_leg(voices):
    """A call with a callback leg, which nobody answers until a test says so."""
    call = build_wired(voices, FakeLeg())
    yield call
    await call.tools.gate.aclose()


async def test_the_gate_speaks_and_the_step_ups_ask_through_the_one_voice_of_the_call(
    wired, voices
):
    # A probe is a factor that the real StepUps passes. A stand-in that passed no factor, as the
    # one this replaced, would have ended this request in a deny.
    wired.detector.decisions = [
        (DecisionKind.STEP_UP, [Factor.PROBE]),
        (DecisionKind.ALLOW, []),
    ]
    result = await wired.tools.gate.run(BANK)
    assert result.status == "done"
    assert ("add_default_bank_account", "v1", "021000021", "000123456789") in wired.ramp.calls
    assert len(voices) == 1 and voices[0].session is wired.session
    assert voices[0].said == [prompts.VERIFYING]  # the gate's line
    assert voices[0].asked == [prompts.PROBE_QUESTION]  # the step-up's question, on the same voice


async def test_a_callback_the_detector_asks_for_fails_closed_when_no_leg_is_set_up(wired, caplog):
    wired.detector.decisions = [
        (DecisionKind.STEP_UP, [Factor.CALLBACK_ON_FILE]),
        (DecisionKind.DENY, []),
    ]
    with caplog.at_level(logging.WARNING, logger="callback_agent.stepups"):
        result = await wired.tools.gate.run(BANK)
    assert "none is set up" in caplog.text  # the factor ran, and it was the missing leg that failed
    assert wired.detector.decisions == []  # so the gate asked again, and the detector denied
    assert result == GateResult("denied", prompts.DENY_HELD)
    assert wired.ramp.calls == [("hold_vendor", "v1")]  # held, and never changed
    assert wired.hangup.ended and wired.state.locked
    # It failed without speaking: the caller was never told they would be called back.
    assert wired.voice.said == [prompts.VERIFYING, prompts.DENY_HELD]
    assert wired.detector.factors == [] and wired.detector.caller_updates == []


# --- the callback leg of a call ---

CONFIGURED = Settings(
    agent_name="callback",
    cartesia_voice_id="voice",
    sip_outbound_trunk_id="ST_outbound",
    telnyx_number="+15555550199",
)


@pytest.fixture
def legs(monkeypatch) -> list[FakeLeg]:
    """The legs the wiring builds: LiveKitCallbackLeg is swapped for a recorder, and the test sees
    how many there were."""
    built: list[FakeLeg] = []

    def make(ctx, session, state, settings) -> FakeLeg:
        built.append(FakeLeg(ctx, session, state, settings))
        return built[-1]

    monkeypatch.setattr(wiring, "LiveKitCallbackLeg", make)
    return built


def test_a_leg_is_built_over_the_job_the_session_the_state_and_the_settings(legs):
    ctx, session, state = object(), object(), new_state()
    leg = wiring.build_callback_leg(ctx, session, state, CONFIGURED)
    assert legs == [leg] and leg.built_over == (ctx, session, state, CONFIGURED)


def test_the_leg_of_a_configured_call_is_the_livekit_one():
    leg = wiring.build_callback_leg(object(), object(), new_state(), CONFIGURED)
    assert isinstance(leg, LiveKitCallbackLeg)


@pytest.mark.parametrize(
    "missing",
    [
        {"sip_outbound_trunk_id": None},
        {"telnyx_number": None},
        {"sip_outbound_trunk_id": None, "telnyx_number": None},
        {"sip_outbound_trunk_id": ""},
        {"telnyx_number": ""},
    ],
    ids=["no trunk", "no number", "neither", "blank trunk", "blank number"],
)
def test_no_leg_is_built_unless_the_trunk_and_the_telnyx_number_are_both_set(legs, missing):
    # A leg built without them would say the callback line, find it could not dial, and report
    # no answer. With none, the step-up fails closed before it says a word.
    settings = dataclasses.replace(CONFIGURED, **missing)
    assert wiring.build_callback_leg(object(), object(), new_state(), settings) is None
    assert legs == []


async def test_the_step_ups_dial_the_number_on_file_through_the_leg_they_are_given(wired_with_leg):
    call = wired_with_leg
    call.detector.decisions = [
        (DecisionKind.STEP_UP, [Factor.CALLBACK_ON_FILE]),
        (DecisionKind.DENY, []),
    ]
    result = await call.tools.gate.run(BANK)
    assert call.leg.dialed == [NUMBER_ON_FILE]  # the detector's directory, never what was heard
    assert call.detector.factors == [FactorReport(result=CallbackResult.NO_ANSWER)]
    assert call.voice.said == [prompts.VERIFYING, prompts.CALLBACK_LINE, prompts.DENY_HELD]
    assert result == GateResult("denied", prompts.DENY_HELD)  # nobody answered
    assert call.ramp.calls == [("hold_vendor", "v1")]  # held, and never changed


async def test_a_callback_that_is_answered_and_confirmed_lets_the_change_through(wired_with_leg):
    call = wired_with_leg
    call.leg.answer = CALLEE
    call.voice.answers = ["yes"]
    call.detector.decisions = [
        (DecisionKind.STEP_UP, [Factor.CALLBACK_ON_FILE]),
        (DecisionKind.ALLOW, []),
    ]
    result = await call.tools.gate.run(BANK)
    assert result.status == "done"
    assert ("add_default_bank_account", "v1", "021000021", "000123456789") in call.ramp.calls
    assert call.leg.dialed == [NUMBER_ON_FILE]
    # The detector is pointed at whoever answered, and the new call hears the disclosure again.
    assert call.detector.caller_updates == [
        CallerUpdate(caller_identity=CALLEE, channel=Channel.PHONE)
    ]
    assert call.voice.said == [prompts.VERIFYING, prompts.CALLBACK_LINE, prompts.DISCLOSURE]
    assert call.voice.asked == [
        prompts.CALLBACK_CONFIRM.format(pending_action=BANK.context().pending_action)
    ]
    assert call.detector.factors == [FactorReport(result=CallbackResult.CONFIRMED)]


async def test_a_callee_who_hangs_up_on_the_confirm_question_is_a_silent_deny_with_a_hold(
    wired_with_leg,
):
    # R79. The callee picks up and hangs up during the question: the worker's hang-up handler marks
    # the call as ending, and the capture gives None. The step-up reports it and raises, so the
    # gate takes its deny path on the spot: it holds the vendor, says nothing to a line nobody is
    # on, and hangs up. It does not ask the detector again, whose next factor would text the real
    # user a code, and it does not stop at the start of that round without holding anything.
    call = wired_with_leg
    call.leg.answer = CALLEE

    async def ask_and_lose_the_callee(prompt: str, *, timeout_s: float, secret: bool = False):
        call.voice.asked.append(prompt)
        call.state.closing = True  # and the answer is None, as there is nobody to give one

    call.voice.ask = ask_and_lose_the_callee
    call.detector.decisions = [
        (DecisionKind.STEP_UP, [Factor.CALLBACK_ON_FILE]),
        (DecisionKind.STEP_UP, [Factor.SMS_OTP]),  # what it would have asked for next
    ]
    result = await call.tools.gate.run(BANK)
    assert result == GateResult("denied", prompts.DENY_HELD)
    assert call.ramp.calls == [("hold_vendor", "v1")]  # held, and never changed
    # The callee heard the disclosure and the question, and not a word after.
    assert call.voice.said == [prompts.VERIFYING, prompts.CALLBACK_LINE, prompts.DISCLOSURE]
    assert call.voice.asked == [
        prompts.CALLBACK_CONFIRM.format(pending_action=BANK.context().pending_action)
    ]
    assert call.detector.factors == [FactorReport(result=CallbackResult.DENIED)]
    assert call.detector.decisions == [(DecisionKind.STEP_UP, [Factor.SMS_OTP])]  # asked once
    assert call.detector.otps_sent == []
    assert call.hangup.ended and call.state.locked and call.state.closing


class NobodyAnswers:
    """Stands in for LiveKit's job and session under a real LiveKitCallbackLeg, on a call whose
    callee never picks up: removing the caller takes them out of the room, and the dial ends as
    a SIP 480. `dialed` keeps each number dialed."""

    def __init__(self, caller: str) -> None:
        self.room = SimpleNamespace(
            name="room", remote_participants={caller: SimpleNamespace(identity=caller)}
        )
        self.dialed: list[str] = []
        self.ctx = SimpleNamespace(
            room=self.room,
            api=SimpleNamespace(
                room=SimpleNamespace(remove_participant=self._remove),
                sip=SimpleNamespace(create_sip_participant=self._dial),
            ),
        )
        self.session = SimpleNamespace(
            room_io=SimpleNamespace(set_participant=lambda identity: None),
            clear_user_turn=lambda: None,
        )

    async def _remove(self, remove) -> None:
        self.room.remote_participants.pop(remove.identity, None)

    async def _dial(self, create, *, timeout=None) -> None:
        self.dialed.append(create.sip_call_to)
        raise self._dial_failure(create)

    def _dial_failure(self, create) -> BaseException:
        return api.SipCallError(
            "unavailable",
            "no answer",
            status=503,
            metadata={"sip_status_code": "480", "sip_status": "Temporarily Unavailable"},
        )


async def test_a_callback_nobody_answers_once_the_caller_is_gone_is_a_silent_deny_at_once(voices):
    # The real leg over a fake LiveKit: it takes the caller off the line, nobody picks up, and it
    # marks the call as ending. The step-up then raises, so the gate takes its deny path on the
    # spot: it holds the vendor, says nothing to an empty line, and hangs up. It does not ask the
    # detector again, whose next factor would text the real user a code for a call nobody is on.
    state = new_state(session_id="s1", claimed_user=ERIC)
    livekit = NobodyAnswers(state.caller_identity)
    leg = LiveKitCallbackLeg(livekit.ctx, livekit.session, state, CONFIGURED)
    call = build_wired(voices, leg, state=state)
    try:
        call.detector.decisions = [
            (DecisionKind.STEP_UP, [Factor.CALLBACK_ON_FILE]),
            (DecisionKind.STEP_UP, [Factor.SMS_OTP]),  # what it would have asked for next
        ]
        result = await call.tools.gate.run(BANK)
    finally:
        await call.tools.gate.aclose()
    assert livekit.dialed == [NUMBER_ON_FILE]
    assert call.detector.factors == [FactorReport(result=CallbackResult.NO_ANSWER)]
    assert call.detector.decisions == [(DecisionKind.STEP_UP, [Factor.SMS_OTP])]  # asked once
    assert call.detector.otps_sent == []
    assert result == GateResult("denied", prompts.DENY_HELD)
    assert call.ramp.calls == [("hold_vendor", "v1")]  # held, and never changed
    assert call.voice.said == [prompts.VERIFYING, prompts.CALLBACK_LINE]  # not a word after
    assert call.hangup.ended and state.locked and state.closing


class DialDiesOnTheWire(NobodyAnswers):
    """The same fake LiveKit, but the dial request fails on the way to the API instead of ending
    as a SIP answer: the connection is reset, and nobody can say whether the phone rang."""

    def _dial_failure(self, create) -> BaseException:
        return aiohttp.ClientConnectionError(f"reset while dialing {create.sip_call_to}")


async def test_a_dial_that_dies_on_the_wire_once_the_caller_is_gone_is_a_silent_deny_at_once(
    voices,
):
    # As for a dial the server refuses (R74): the caller was taken off the line, so the gate must
    # not say its deny line to an empty room. It holds the vendor, says nothing and hangs up, and
    # does not ask the detector again, whose next factor would text the real user a code.
    state = new_state(session_id="s1", claimed_user=ERIC)
    livekit = DialDiesOnTheWire(state.caller_identity)
    leg = LiveKitCallbackLeg(livekit.ctx, livekit.session, state, CONFIGURED)
    call = build_wired(voices, leg, state=state)
    try:
        call.detector.decisions = [
            (DecisionKind.STEP_UP, [Factor.CALLBACK_ON_FILE]),
            (DecisionKind.STEP_UP, [Factor.SMS_OTP]),  # what it would have asked for next
        ]
        result = await call.tools.gate.run(BANK)
    finally:
        await call.tools.gate.aclose()
    assert livekit.dialed == [NUMBER_ON_FILE]
    assert call.detector.decisions == [(DecisionKind.STEP_UP, [Factor.SMS_OTP])]  # asked once
    assert call.detector.otps_sent == []
    assert result == GateResult("denied", prompts.DENY_HELD)
    assert call.ramp.calls == [("hold_vendor", "v1")]  # held, and never changed
    assert call.voice.said == [prompts.VERIFYING, prompts.CALLBACK_LINE]  # not a word after
    assert call.hangup.ended and state.locked and state.closing


async def test_the_toolbox_enrolls_through_the_gate_the_step_ups_and_the_voice_of_the_call(wired):
    # No one answers the code, so the enrollment ends where the real one would: the code was
    # texted by the detector to the number on file, asked for three times, and the enrollment says
    # why it stopped. A toolbox with no enrollment would answer ENROLL_NOT_NOW instead.
    reply = await asyncio.wait_for(wired.tools.toolbox.start_voice_enrollment(), timeout=5)
    assert reply == prompts.ENROLL_TOOL_FAILED
    assert wired.detector.otps_sent == ["s1"]
    assert wired.voice.asked == [
        prompts.OTP_PROMPT.format(last2="00"),
        prompts.OTP_SIX_DIGITS,
        prompts.OTP_SIX_DIGITS,
    ]
    assert wired.voice.said == [prompts.ENROLL_OTP_FAILED]  # on the voice the gate speaks with


async def test_a_second_enrollment_on_the_call_sends_no_code_and_asks_nothing(wired):
    # One enrollment per call (R81), here with the real Enrollment: a second start must not text
    # the number on file again, nor ask the caller anything, whatever the first came to.
    first = await asyncio.wait_for(wired.tools.toolbox.start_voice_enrollment(), timeout=5)
    assert first == prompts.ENROLL_TOOL_FAILED and wired.detector.otps_sent == ["s1"]
    asked, said = list(wired.voice.asked), list(wired.voice.said)
    second = await asyncio.wait_for(wired.tools.toolbox.start_voice_enrollment(), timeout=5)
    assert second == prompts.ENROLL_TOOL_ALREADY_RAN
    assert wired.detector.otps_sent == ["s1"]  # no second code
    assert (wired.voice.asked, wired.voice.said) == (asked, said)  # and nothing more was said


# --- the room options of a call ---


def test_the_chat_topic_takes_no_typed_input():
    # A browser caller could otherwise type a turn, or the answer to a challenge, into the chat
    # (R53). With text_input off, RoomIO registers no handler for the chat topic at all. That is
    # one of two ways in; the session host is the other (see the tests below).
    options = build_room_options("caller-1")
    assert options.text_input is False
    assert options.get_text_input_options() is None


def test_the_call_stays_linked_to_the_caller_who_joined():
    assert build_room_options("sip_+15555550100").participant_identity == "sip_+15555550100"


def test_livekit_is_told_not_to_close_the_session_when_the_caller_disconnects():
    # A hang-up is CLIENT_INITIATED, which closes the session unless this is off. The worker ends
    # the call itself (wire_hangups), so that a callback can hang up on the caller and dial the
    # number on file inside the same session.
    assert build_room_options("caller-1").close_on_disconnect is False


def test_the_room_options_are_the_callers_identity_with_typed_input_off_and_nothing_else():
    assert build_room_options("caller-1") == room_io.RoomOptions(
        participant_identity="caller-1", text_input=False, close_on_disconnect=False
    )


# The two ways in for typed input, by the topic each one listens on. RoomIO registers a handler on
# the chat topic unless text_input is False. AgentSession.start hosts the session on the other
# unless session_host is False, and a run_input request there is a user turn from whoever sent it.
CHAT_TOPIC = "lk.chat"
SESSION_TOPIC = "lk.agent.session"


def participant(identity: str) -> SimpleNamespace:
    """A participant as the room hands one to its handlers, and as RoomIO reads the one it links
    to: who they are, what kind, and that they left of their own accord, as a hang-up does."""
    return SimpleNamespace(
        identity=identity,
        sid="PA_test",
        name="",
        metadata="",
        kind=rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
        attributes={},
        disconnect_reason=rtc.DisconnectReason.CLIENT_INITIATED,
    )


async def settle() -> None:
    """Let everything that is ready to run, run."""
    for _ in range(20):
        await asyncio.sleep(0)


@contextlib.asynccontextmanager
async def started_session(
    room: rtc.Room, options: room_io.RoomOptions, *, agent: Agent | None = None, **start_kwargs
):
    """A real AgentSession started on `room` with `options`, closed again when the block ends.

    The room is not connected, so the options are given with audio and transcription turned off:
    what is real is the session, RoomIO, and what the room's events and topics do to them.
    """
    options = dataclasses.replace(
        options, audio_input=False, audio_output=False, text_output=False
    )
    session = AgentSession(llm=FakeLLM(fake_responses=[]))
    try:
        await asyncio.wait_for(
            session.start(
                agent or Agent(instructions="test"),
                room=room,
                room_options=options,
                record=False,
                **start_kwargs,
            ),
            timeout=10,
        )
        yield session
    finally:
        await asyncio.wait_for(session.aclose(), timeout=10)


async def topics_taken_after_start(room: rtc.Room, options: room_io.RoomOptions, **start_kwargs):
    """Start a real session and name the topics that something now handles.

    A topic is taken when registering a handler of our own on it raises, as LiveKit allows one
    handler per topic.
    """
    async with started_session(room, options, **start_kwargs):
        taken: set[str] = set()
        for topic, register, unregister in (
            (CHAT_TOPIC, room.register_text_stream_handler, room.unregister_text_stream_handler),
            (SESSION_TOPIC, room.register_byte_stream_handler, room.unregister_byte_stream_handler),
        ):
            try:
                register(topic, lambda reader, identity: None)
            except ValueError:
                taken.add(topic)
            else:
                unregister(topic)
        return taken


async def test_left_to_its_defaults_a_session_takes_typed_input_on_both_topics(room):
    # The control for the next test: without it, the probe could be looking at nothing.
    assert await topics_taken_after_start(room, room_io.RoomOptions()) == {
        CHAT_TOPIC,
        SESSION_TOPIC,
    }


async def test_started_as_the_worker_starts_it_a_session_takes_typed_input_on_neither_topic(room):
    # The room options come from build_room_options and session_host is False, as in worker.py,
    # which tests/test_structure.py pins by its source because no test can import it.
    taken = await topics_taken_after_start(
        room, build_room_options("caller-1"), session_host=False
    )
    assert taken == set()


async def closes_when_its_linked_caller_hangs_up(
    room: rtc.Room, options: room_io.RoomOptions, *, wait_s: float
) -> bool:
    """Whether a real session on `options` closes by itself, within `wait_s`, when the caller it
    is linked to leaves of their own accord: a hang-up, as LiveKit sees one."""
    caller = participant("caller-1")
    async with started_session(room, options) as session:
        closed = asyncio.Event()
        session.on("close", lambda ev: closed.set())
        room.emit("participant_connected", caller)  # RoomIO links the participant the options name
        await settle()
        assert session.room_io.linked_participant is not None
        room.emit("participant_disconnected", caller)
        try:
            await asyncio.wait_for(closed.wait(), timeout=wait_s)
        except TimeoutError:
            return False
        return True


async def test_left_to_its_defaults_a_session_closes_itself_when_its_caller_hangs_up(room):
    # The control for the next test, and the reason for close_on_disconnect=False: during a
    # callback the caller is hung up on, and a session that closed with them would end the call.
    options = room_io.RoomOptions(participant_identity="caller-1")
    assert await closes_when_its_linked_caller_hangs_up(room, options, wait_s=5) is True


async def test_started_with_the_worker_options_a_session_stays_open_when_its_caller_hangs_up(room):
    options = build_room_options("caller-1")
    assert await closes_when_its_linked_caller_hangs_up(room, options, wait_s=0.3) is False


# --- the hang-up the worker owns ---
#
# The session stays open when the caller leaves, so ending the call is left to wire_hangups. It
# does so for the caller, and never during a callback, which hangs up on the caller on purpose.

SPOKEN = "my code is 482913, from sip_+15555550142"  # nothing of this may reach a log


class ClosableSession:
    """The part of an AgentSession that a hang-up uses: `shutdown`, which starts the close in a
    task of the session's own, and `on`, for the close event.

    `shutdowns` keeps each call as (drain, whether the call was already marked as ending). The
    close waits for `finish`, which is set from the start and which a test clears to make the close
    slow. Then it notes in `events` that the session closed, with the same flag, and fires the
    close event. `error` makes shutdown raise. `closing_task` is the close, and `cut_short` says
    whether something cancelled it.
    """

    def __init__(self, state: CallState, events: list[tuple]) -> None:
        self.finish = asyncio.Event()
        self.finish.set()
        self.error: BaseException | None = None
        self.shutdowns: list[tuple[bool, bool]] = []
        self.closing_task: asyncio.Task[None] | None = None
        self.cut_short = False
        self._state, self._events = state, events
        self._listeners: list[typing.Callable[[object], None]] = []

    def on(self, event: str, callback) -> None:
        assert event == "close"
        self._listeners.append(callback)

    def shutdown(self, *, drain: bool = True) -> None:
        self.shutdowns.append((drain, self._state.closing))
        if self.error is not None:
            raise self.error
        self.closing_task = asyncio.create_task(self._close())

    async def _close(self) -> None:
        try:
            await self.finish.wait()
        except asyncio.CancelledError:
            self.cut_short = True
            raise
        self.fire_close()

    def fire_close(self) -> None:
        """The close event, as LiveKit fires it when the close is done."""
        self._events.append(("session closed", self._state.closing))
        for callback in self._listeners:
            callback(None)


class DeletableRoom:
    """The call's CallControl: end_call deletes the room. `deleted` is set once it has run, however
    it ended. `hang` makes it never finish, and `error` makes it raise."""

    def __init__(self, state: CallState, events: list[tuple]) -> None:
        self.deleted = asyncio.Event()
        self.hang = False
        self.error: BaseException | None = None
        self._state, self._events = state, events

    async def end_call(self) -> None:
        self._events.append(("room deleted", self._state.closing))
        try:
            if self.hang:
                await asyncio.Event().wait()
            if self.error is not None:
                raise self.error
        finally:
            self.deleted.set()


def wire_live_call(room: rtc.Room) -> SimpleNamespace:
    """A call whose participants can leave `room`: the hang-up wired on a real Room, over fakes."""
    events: list[tuple] = []
    state = new_state()  # its caller is "caller"
    session, call = ClosableSession(state, events), DeletableRoom(state, events)
    wiring.wire_hangups(room, session, state, call)
    return SimpleNamespace(
        room=room,
        state=state,
        session=session,
        call=call,
        events=events,
        leave=lambda identity: room.emit("participant_disconnected", participant(identity)),
    )


@pytest.fixture
async def live_call(room) -> SimpleNamespace:
    return wire_live_call(room)


@pytest.fixture
async def loop_errors() -> list[dict]:
    """What the event loop reports about something no code asked about, such as a task that failed
    and was never awaited."""
    reported: list[dict] = []
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: reported.append(context))
    yield reported
    loop.set_exception_handler(previous)


ENDED = [("session closed", True), ("room deleted", True)]  # with the call already marked as ending


def test_the_event_the_hangup_listens_for_is_one_the_room_emits():
    # EventEmitter.on takes any name, so a misspelled one would never fire and nothing would say so.
    assert "participant_disconnected" in typing.get_args(EventTypes)


async def test_the_caller_hanging_up_marks_the_call_as_ending_before_anything_awaits(live_call):
    live_call.leave("caller")
    assert live_call.state.closing  # no await since the event: nothing else has run yet
    assert live_call.events == [] and live_call.session.shutdowns == []
    await asyncio.wait_for(live_call.call.deleted.wait(), timeout=5)


async def test_the_session_is_shut_down_and_then_the_room_deleted_on_a_call_already_ending(
    live_call, monkeypatch
):
    monkeypatch.setattr(wiring, "SESSION_CLOSE_S", 30.0)  # a missed close event would show
    live_call.leave("caller")
    await asyncio.wait_for(live_call.call.deleted.wait(), timeout=5)
    assert live_call.events == ENDED
    # Closed without draining, as LiveKit closed a session whose participant left: nobody is
    # listening to what is still queued. And the call was already marked as ending when it began.
    assert live_call.session.shutdowns == [(False, True)]


@pytest.mark.parametrize("identity", ["detector", CALLEE, "caller-2", ""])
async def test_anyone_else_leaving_is_ignored(live_call, identity):
    live_call.leave(identity)
    await settle()
    assert not live_call.state.closing and live_call.events == []


async def test_the_caller_leaving_during_a_callback_is_ignored(live_call):
    # StepUps.callback raises the flag before the caller hears they will be hung up on, and the
    # leg removes them: both are planned, and the call goes on with whoever answers.
    live_call.state.callback_in_progress = True
    live_call.leave("caller")
    await settle()
    assert not live_call.state.closing and live_call.events == []


async def test_after_a_callback_is_answered_the_callee_hanging_up_ends_the_call(live_call):
    state = live_call.state
    state.callback_in_progress = True
    live_call.leave("caller")  # the leg removes the caller
    state.caller_identity, state.callback_in_progress = CALLEE, False  # the callee answers
    live_call.leave("caller")  # the removed caller's event can arrive late
    await settle()
    assert not state.closing and live_call.events == []  # neither ends the call
    live_call.leave(CALLEE)
    assert state.closing
    await asyncio.wait_for(live_call.call.deleted.wait(), timeout=5)
    assert live_call.events == ENDED


async def test_a_second_hang_up_while_the_call_is_ending_does_not_end_it_again(live_call):
    live_call.leave("caller")
    live_call.leave("caller")
    await asyncio.wait_for(live_call.call.deleted.wait(), timeout=5)
    await settle()
    assert live_call.events == ENDED
    assert live_call.session.shutdowns == [(False, True)]


async def test_a_shutdown_that_raises_is_logged_by_type_and_swallowed(
    live_call, caplog, loop_errors
):
    live_call.session.error = RuntimeError(SPOKEN)
    with caplog.at_level(logging.INFO, logger="callback_agent.session_wiring"):
        live_call.leave("caller")
        await asyncio.wait_for(live_call.call.deleted.wait(), timeout=5)
        await settle()
    assert live_call.events == [("room deleted", True)]  # the room is deleted all the same
    assert "could not shut the session down: RuntimeError" in caplog.text
    assert "482913" not in caplog.text and "5550142" not in caplog.text
    gc.collect()
    assert loop_errors == []  # no task was left holding an error nobody read


async def test_a_session_slow_to_close_is_not_cut_short_and_the_room_goes_without_waiting_for_it(
    live_call, caplog, monkeypatch
):
    # Only the wait is bounded. A close cancelled partway leaves the session half closed: no close
    # event, the last turn never committed, the activity and RoomIO still open.
    monkeypatch.setattr(wiring, "SESSION_CLOSE_S", 0.05)
    session = live_call.session
    session.finish.clear()  # the close takes longer than the wait
    with caplog.at_level(logging.WARNING, logger="callback_agent.session_wiring"):
        live_call.leave("caller")
        await asyncio.wait_for(live_call.call.deleted.wait(), timeout=5)
        await settle()
    assert live_call.events == [("room deleted", True)]  # it went at the bound, close still running
    assert "the session was still closing after 0.05 s" in caplog.text
    assert not session.cut_short
    session.finish.set()
    await asyncio.wait_for(session.closing_task, timeout=5)
    assert live_call.events == ENDED[::-1] and not session.cut_short  # and it did end


async def test_a_session_that_has_already_closed_is_neither_closed_again_nor_waited_for(
    live_call, monkeypatch
):
    monkeypatch.setattr(wiring, "SESSION_CLOSE_S", 30.0)  # a wait for a close would hit the limit
    live_call.session.fire_close()  # an unrecoverable error closed it earlier, say
    live_call.events.clear()
    live_call.leave("caller")
    await asyncio.wait_for(live_call.call.deleted.wait(), timeout=5)
    assert live_call.session.shutdowns == []
    assert live_call.events == [("room deleted", True)]


async def test_a_hangup_that_itself_fails_is_logged_by_type_and_goes_unnoticed_by_the_loop(
    live_call, caplog, loop_errors, monkeypatch
):
    async def broken(session, call, closed):
        raise RuntimeError(SPOKEN)

    monkeypatch.setattr(wiring, "hang_up", broken)
    with caplog.at_level(logging.ERROR, logger="callback_agent.session_wiring"):
        live_call.leave("caller")
        await settle()
    assert "could not end the call after a hang-up: RuntimeError" in caplog.text
    assert "482913" not in caplog.text and "5550142" not in caplog.text
    gc.collect()
    assert loop_errors == []


async def test_a_handler_that_fails_logs_by_type_and_raises_nothing_into_the_room(
    live_call, caplog
):
    with caplog.at_level(logging.ERROR):
        live_call.room.emit("participant_disconnected", object())  # nobody to name
    assert "could not end the call after a hang-up: AttributeError" in caplog.text
    assert "failed to emit event" not in caplog.text  # LiveKit's own log of it, with the traceback
    assert not live_call.state.closing


class HeldAtExit(Agent):
    """An agent whose on_exit waits for `release`: a tool or a line still finishing when the caller
    hangs up, and so a close that outlasts the wait. `exiting` is set once the close gets there."""

    def __init__(self) -> None:
        super().__init__(instructions="test")
        self.exiting = asyncio.Event()
        self.release = asyncio.Event()

    async def on_exit(self) -> None:
        self.exiting.set()
        await self.release.wait()


async def test_a_real_session_is_closed_and_then_the_room_deleted_when_the_caller_hangs_up(
    room, monkeypatch
):
    monkeypatch.setattr(wiring, "SESSION_CLOSE_S", 30.0)  # a missed close event would show
    state, events = new_state(), []
    call = DeletableRoom(state, events)
    caller = participant("caller")
    async with started_session(room, build_room_options("caller")) as session:
        session.on("close", lambda ev: events.append(("session closed", state.closing)))
        wiring.wire_hangups(room, session, state, call)
        room.emit("participant_connected", caller)  # RoomIO links them
        await settle()
        assert events == []  # the session is open: LiveKit was told not to close it
        room.emit("participant_disconnected", caller)
        await asyncio.wait_for(call.deleted.wait(), timeout=5)
    assert events == ENDED


async def test_a_real_session_whose_close_outlasts_the_wait_still_closes_and_the_room_is_deleted(
    room, monkeypatch
):
    # What the wait being bounded must not do: cut LiveKit's close off. A close cancelled partway
    # left the session marked as not started with its close event never fired, so the job's own
    # close at shutdown returned at once and nothing closed the activity or RoomIO.
    monkeypatch.setattr(wiring, "SESSION_CLOSE_S", 0.1)
    state, events = new_state(), []
    call = DeletableRoom(state, events)
    caller = participant("caller")
    agent = HeldAtExit()
    closed = asyncio.Event()
    async with started_session(room, build_room_options("caller"), agent=agent) as session:
        session.on("close", lambda ev: events.append(("session closed", state.closing)))
        session.on("close", lambda ev: closed.set())
        wiring.wire_hangups(room, session, state, call)
        room.emit("participant_connected", caller)
        await settle()
        room.emit("participant_disconnected", caller)
        await asyncio.wait_for(call.deleted.wait(), timeout=5)
        assert agent.exiting.is_set() and not closed.is_set()  # the close is under way, held
        assert events == [("room deleted", True)]  # and the room went at the bound without it
        agent.release.set()
        await asyncio.wait_for(closed.wait(), timeout=5)  # the close still ended
    assert events == [("room deleted", True), ("session closed", True)]
    with pytest.raises(RuntimeError):  # and went all the way: RoomIO was closed, not left open
        _ = session.room_io


# hang_up, the steps of it, each bounded and none able to stop the other or to raise


async def hang_up_with(session_fault=None, room_fault=None) -> SimpleNamespace:
    """Run hang_up over fakes, with the given faults set, and hand back what they recorded."""
    events: list[tuple] = []
    state = new_state()
    session, call = ClosableSession(state, events), DeletableRoom(state, events)
    closed = asyncio.Event()
    session.on("close", lambda ev: closed.set())
    if session_fault == "slow":
        session.finish.clear()
    elif session_fault is not None:
        session.error = session_fault
    if room_fault == "hang":
        call.hang = True
    elif room_fault is not None:
        call.error = room_fault
    await asyncio.wait_for(wiring.hang_up(session, call, closed), timeout=5)
    return SimpleNamespace(events=events, session=session, call=call)


async def test_hang_up_starts_the_close_of_the_session_and_then_deletes_the_room():
    run = await hang_up_with()
    assert run.events == [("session closed", False), ("room deleted", False)]
    assert run.session.shutdowns == [(False, False)]


@pytest.mark.parametrize(
    ("faulty", "said", "events"),
    [
        ("session", "could not shut the session down", [("room deleted", False)]),
        ("room", "could not delete the room", [("session closed", False), ("room deleted", False)]),
    ],
)
async def test_hang_up_logs_a_step_that_raises_by_type_and_goes_on(caplog, faulty, said, events):
    with caplog.at_level(logging.INFO, logger="callback_agent.session_wiring"):
        run = await hang_up_with(**{f"{faulty}_fault": RuntimeError(SPOKEN)})
    assert run.events == events
    assert f"{said}: RuntimeError" in caplog.text
    assert "482913" not in caplog.text and "5550142" not in caplog.text


async def test_hang_up_gives_up_on_a_room_that_will_not_be_deleted(caplog, monkeypatch):
    monkeypatch.setattr(wiring, "HANGUP_S", 0.05)
    with caplog.at_level(logging.INFO, logger="callback_agent.session_wiring"):
        run = await hang_up_with(room_fault="hang")
    assert run.events == [("session closed", False), ("room deleted", False)]
    assert "could not delete the room: timed out" in caplog.text


async def test_hang_up_stops_waiting_for_a_slow_session_and_leaves_its_close_running(
    caplog, monkeypatch
):
    monkeypatch.setattr(wiring, "SESSION_CLOSE_S", 0.05)
    with caplog.at_level(logging.INFO, logger="callback_agent.session_wiring"):
        run = await hang_up_with(session_fault="slow")
    assert run.events == [("room deleted", False)]  # the room did not wait for the close
    assert "the session was still closing after 0.05 s" in caplog.text
    assert not run.session.cut_short and not run.session.closing_task.done()
    run.session.finish.set()
    await asyncio.wait_for(run.session.closing_task, timeout=5)
    assert run.events == [("room deleted", False), ("session closed", False)]


def test_the_session_is_given_a_few_seconds_to_close_and_not_more():
    assert 0 < wiring.SESSION_CLOSE_S <= 5.0


# worker.py reads its settings when it is imported, so no test can import it. What it calls, and
# in what order, is read from its source instead: tests/test_structure.py does the same.
WORKER = Path(wiring.__file__).with_name("worker.py")


def worker_calls() -> list[ast.Call]:
    return [node for node in ast.walk(ast.parse(WORKER.read_text())) if isinstance(node, ast.Call)]


def worker_call(name: str) -> ast.Call:
    """The one call of `name` (a function, or an attribute such as session.start) in worker.py."""
    calls = [call for call in worker_calls() if ast.unparse(call.func) == name]
    assert len(calls) == 1, f"worker.py should call {name} exactly once"
    return calls[0]


def test_the_worker_wires_the_hangup_on_the_room_before_it_starts_the_session():
    # Without it a caller who hangs up would leave the call open: LiveKit is told not to close the
    # session, and nothing else ends the call. Before the start, so that a caller who hangs up
    # while the session starts is not missed.
    hangups = worker_call("wire_hangups")
    assert [ast.unparse(arg) for arg in hangups.args] == ["ctx.room", "session", "state", "call"]
    assert hangups.lineno < worker_call("session.start").lineno


def test_the_worker_hands_the_tools_a_leg_built_by_the_wiring_and_never_one_by_hand():
    # Only build_callback_leg knows that a leg exists only when the trunk and the Telnyx number
    # are both set, and the tests of it can't see a leg built here by hand, or none handed over.
    leg = worker_call("build_callback_leg")
    assert [ast.unparse(arg) for arg in leg.args] == ["ctx", "session", "state", "settings"]
    given = {kw.arg: ast.unparse(kw.value) for kw in worker_call("build_tools").keywords}
    assert given["callback_leg"] == "callback_leg"
    assert "LiveKitCallbackLeg" not in {ast.unparse(call.func) for call in worker_calls()}


# --- the detector's risk packets ---


def risk_packet(update: RiskUpdate | bytes, *, topic: str | None = RISK_TOPIC, sender=None):
    payload = update if isinstance(update, bytes) else update.model_dump_json().encode()
    return rtc.DataPacket(data=payload, kind=0, participant=sender, topic=topic)


@pytest.fixture
async def room() -> rtc.Room:  # async, because a Room needs the running event loop
    return rtc.Room()


async def test_a_risk_packet_from_the_detector_is_kept_on_the_state(room):
    state = new_state(session_id="s1")
    wire_risk_updates(room, state)
    update = RiskUpdate(session_id="s1", spoof_prob=0.2, text_risk=0.7, flags=["payment_change"])
    room.emit("data_received", risk_packet(update))
    assert state.latest_risk == update


async def test_the_newest_risk_packet_replaces_the_one_before(room):
    state = new_state(session_id="s1")
    wire_risk_updates(room, state)
    room.emit("data_received", risk_packet(RiskUpdate(session_id="s1", spoof_prob=0.2)))
    newest = RiskUpdate(session_id="s1", spoof_prob=0.9)
    room.emit("data_received", risk_packet(newest))
    assert state.latest_risk == newest


@pytest.mark.parametrize("topic", ["turn", "chat", "", None])
async def test_packets_on_other_topics_are_ignored(room, topic):
    state = new_state(session_id="s1")
    wire_risk_updates(room, state)
    room.emit("data_received", risk_packet(RiskUpdate(session_id="s1"), topic=topic))
    assert state.latest_risk is None


async def test_a_packet_for_another_session_is_ignored(room):
    state = new_state(session_id="s1")
    wire_risk_updates(room, state)
    room.emit("data_received", risk_packet(RiskUpdate(session_id="s2", spoof_prob=0.9)))
    assert state.latest_risk is None


async def test_a_packet_that_arrives_before_the_caller_is_identified_is_ignored(room):
    state = new_state()
    wire_risk_updates(room, state)
    room.emit("data_received", risk_packet(RiskUpdate(session_id="s1")))
    assert state.latest_risk is None


async def test_a_risk_packet_from_anyone_livekit_can_name_is_not_the_detectors(room, caplog):
    # The detector is a hidden participant, so its packets have no sender. The caller is not hidden:
    # a "risk" packet they publish names them, and it is dropped, however it is filled in.
    state = new_state(session_id="s1")
    wire_risk_updates(room, state)
    caller = SimpleNamespace(identity="caller")
    with caplog.at_level(logging.WARNING, logger="callback_agent.session_wiring"):
        for _ in range(3):
            room.emit(
                "data_received",
                risk_packet(RiskUpdate(session_id="s1", spoof_prob=0.0), sender=caller),
            )
    assert state.latest_risk is None
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1  # not a flood


async def test_a_garbled_packet_is_logged_by_type_and_changes_nothing(room, caplog):
    state = new_state(session_id="s1")
    wire_risk_updates(room, state)
    before = RiskUpdate(session_id="s1", spoof_prob=0.1)
    room.emit("data_received", risk_packet(before))
    with caplog.at_level(logging.ERROR, logger="callback_agent.session_wiring"):
        room.emit("data_received", risk_packet(b'{"session_id": "s1", "spoof_prob": "482913x"}'))
        room.emit("data_received", risk_packet(b"not json 482913"))
    assert state.latest_risk == before
    assert "ValidationError" in caplog.text
    assert "482913" not in caplog.text


# --- the end of the call, with the gate and Ramp ---


class Detector:
    """Records the order of the end-of-call steps that touch it."""

    def __init__(self, log: list[str]) -> None:
        self._log = log

    async def post_turn(self, session_id, turn) -> None:
        self._log.append("post_turn")

    async def post_event(self, session_id, event) -> None:
        self._log.append("post_event")

    async def end_session(self, session_id) -> None:
        self._log.append("end_session")

    async def aclose(self) -> None:
        self._log.append("detector.aclose")


class Closable:
    def __init__(self, name: str, log: list[str]) -> None:
        self._name, self._log = name, log
        self.hang = False
        self.error: Exception | None = None

    async def aclose(self) -> None:
        self._log.append(f"{self._name}.aclose")
        if self.hang:
            await asyncio.Event().wait()
        if self.error is not None:
            raise self.error


@pytest.fixture
async def ending():
    log: list[str] = []
    detector = Detector(log)
    forwarder = Forwarder(detector)
    consumer = asyncio.create_task(forwarder.run())
    forwarder.attach("s1")
    yield SimpleNamespace(
        log=log,
        detector=detector,
        forwarder=forwarder,
        consumer=consumer,
        state=new_state(session_id="s1"),
        gate=Closable("gate", log),
        ramp=Closable("ramp", log),
    )
    await forwarder.aclose()
    await asyncio.gather(consumer, return_exceptions=True)


async def finish(ending, **kwargs) -> None:
    await asyncio.wait_for(
        finish_call(
            ending.state,
            ending.forwarder,
            ending.detector,
            consumer=ending.consumer,
            gate=ending.gate,
            ramp=ending.ramp,
            **kwargs,
        ),
        timeout=10,
    )


async def test_the_end_of_a_call_stops_the_gate_before_it_closes_the_clients(ending):
    ending.forwarder.submit(TURN)
    await finish(ending)
    assert ending.log == [
        "post_turn",
        "end_session",
        "gate.aclose",  # the gate's hold reports use the detector's client
        "detector.aclose",
        "ramp.aclose",
    ]


async def test_a_call_that_never_identified_anyone_still_stops_the_gate_and_closes_both_clients(
    ending,
):
    ending.state.session_id = None
    await finish(ending)
    assert ending.log == ["gate.aclose", "detector.aclose", "ramp.aclose"]


async def test_a_gate_that_will_not_stop_is_given_up_on(ending, monkeypatch):
    monkeypatch.setattr(wiring, "GATE_CLOSE_S", 0.05)
    ending.gate.hang = True
    await finish(ending)
    assert ending.log[-3:] == ["gate.aclose", "detector.aclose", "ramp.aclose"]


@pytest.mark.parametrize("failing", ["gate", "ramp"])
async def test_a_part_that_fails_to_close_does_not_stop_the_rest(ending, caplog, failing):
    getattr(ending, failing).error = RuntimeError("already closed")
    with caplog.at_level(logging.WARNING, logger="callback_agent.session_wiring"):
        await finish(ending)
    assert ending.log[-3:] == ["gate.aclose", "detector.aclose", "ramp.aclose"]
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


async def test_the_gate_and_ramp_are_optional_for_a_call_without_them(ending):
    await asyncio.wait_for(
        finish_call(ending.state, ending.forwarder, ending.detector, consumer=ending.consumer),
        timeout=10,
    )
    assert ending.log == ["end_session", "detector.aclose"]


def test_the_new_limit_is_short_enough_for_the_whole_end_to_fit_in_ten_seconds():
    # LiveKit gives a job process 10 s to shut down; the other steps are bounded as well.
    assert wiring.GATE_CLOSE_S <= 3.0 and wiring.HANGUP_S <= 5.0
