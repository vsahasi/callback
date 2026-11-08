"""The tools of CallbackAgent, driven by a scripted LLM in a roomless text session.

Each tool is a thin delegate: it hands its arguments to the Toolbox and the Toolbox's reply back to
the LLM. None of them can be interrupted, and none of them raises. One test runs a denied bank
change through the real gate, toolbox and agent, with a hang-up that closes the session the way a
deleted room does, to show the hang-up cannot deadlock against the gate's lock. It runs twice once
capture.py is here: with a speaker that says nothing, and with the real LiveKitVoice, which says
each line in a task. The last tests hang up while the denied change's vendor hold is still in
flight, to show that the hold is carried through to the end (the comment above them says why that
needs a test).
"""

import asyncio
import json
import logging
from types import SimpleNamespace

import pytest
from callback_shared.contract import (
    ACTION_TIER,
    Channel,
    Decision,
    DecisionKind,
    UserSummary,
)
from fake_llm import FakeLLM, FakeLLMResponse
from fake_ramp import FakeRamp
from fake_toolbox import FakeToolbox
from livekit import rtc
from livekit.agents import AgentSession, RunContext
from livekit.agents.llm import ChatMessage, FunctionToolCall
from livekit.agents.voice import speech_handle
from session_helpers import started

import callback_agent.session_wiring as wiring
from callback_agent import prompts
from callback_agent.actions import ActionRequest
from callback_agent.agents import CallbackAgent
from callback_agent.gate import Gate
from callback_agent.ramp import HoldOutcome, RampClient, Vendor
from callback_agent.session_wiring import TOOL_HANDLING
from callback_agent.state import CallState
from callback_agent.toolbox import Toolbox

try:  # the gate's real speaker is built on another branch, and joins this test once merged
    from callback_agent.capture import LiveKitVoice
except ImportError:
    LiveKitVoice = None

ROUTING = "021000021"
ACCOUNT = "000123456789"


def tool_call(name: str, call_id: str = "c1", **arguments) -> FunctionToolCall:
    return FunctionToolCall(name=name, arguments=json.dumps(arguments), call_id=call_id)


def step(input: str, content: str = "", *calls: FunctionToolCall) -> FakeLLMResponse:
    """What the scripted LLM answers when its last input is `input`."""
    return FakeLLMResponse(
        input=input, content=content, ttft=0, duration=0, tool_calls=list(calls)
    )


def scripted(*responses: FakeLLMResponse) -> FakeLLM:
    """A scripted LLM that also answers the agent's opening question."""
    ask = step(prompts.ASK_WHO, "Who am I speaking with?")
    return FakeLLM(fake_responses=[ask, *responses])


TOOL_CALLS = [
    ("identify_caller", {"full_name": "Eric Catalano"}, ("identify_caller", "Eric Catalano")),
    ("list_vendors", {}, ("list_vendors",)),
    ("get_vendor_bank_account", {"vendor_name": "Acme"}, ("get_vendor_bank_account", "Acme")),
    ("list_open_bills", {"vendor_name": "Acme"}, ("list_open_bills", "Acme")),
    (
        "hold_vendor",
        {"vendor_name": "Acme", "reason": "caller asked"},
        ("hold_vendor", "Acme", "caller asked"),
    ),
    (
        "hold_bill",
        {"vendor_name": "Acme", "invoice_number": "INV-1001", "reason": "caller asked"},
        ("hold_bill", "Acme", "INV-1001", "caller asked"),
    ),
    (
        "change_vendor_bank_account",
        {"vendor_name": "Acme", "routing_number": ROUTING, "account_number": ACCOUNT},
        ("change_vendor_bank_account", "Acme", ROUTING, ACCOUNT),
    ),
    ("release_vendor_hold", {"vendor_name": "Acme"}, ("release_vendor_hold", "Acme")),
    ("release_bill_hold", {"bill_id": "b1"}, ("release_bill_hold", "b1")),
]


@pytest.mark.parametrize(("name", "arguments", "call"), TOOL_CALLS, ids=[t[0] for t in TOOL_CALLS])
async def test_a_tool_hands_its_arguments_to_the_toolbox_and_the_reply_back(name, arguments, call):
    toolbox = FakeToolbox({name: f"reply of {name}"})
    responses = [step("go", "", tool_call(name, **arguments)), step(f"reply of {name}", "ok")]
    async with started(CallbackAgent(toolbox=toolbox), scripted(*responses)) as session:
        result = await asyncio.wait_for(session.run(user_input="go"), timeout=5)
    assert toolbox.calls == [call]
    result.expect.next_event().is_function_call(name=name, arguments=arguments)
    result.expect.next_event().is_function_call_output(output=f"reply of {name}", is_error=False)
    result.expect.next_event().is_message(role="assistant")


@pytest.mark.parametrize(("name", "arguments", "call"), TOOL_CALLS, ids=[t[0] for t in TOOL_CALLS])
async def test_no_tool_can_be_interrupted_while_it_runs(name, arguments, call):
    toolbox = FakeToolbox()
    responses = [step("go", "", tool_call(name, **arguments)), step(f"{name} was done", "ok")]
    allowed: list[bool] = []
    async with started(CallbackAgent(toolbox=toolbox), scripted(*responses)) as session:
        toolbox.during = lambda: allowed.append(session.current_speech.allow_interruptions)
        await asyncio.wait_for(session.run(user_input="go"), timeout=5)
    assert allowed == [False]  # the tool ran, and a barge-in could not have pulled the turn off it


async def test_a_reply_that_was_cut_off_already_runs_nothing(monkeypatch):
    def refuse(self):
        raise RuntimeError("the SpeechHandle is already interrupted")

    monkeypatch.setattr(RunContext, "disallow_interruptions", refuse)
    toolbox = FakeToolbox()
    responses = [
        step("go", "", tool_call("hold_vendor", vendor_name="Acme", reason="caller asked")),
        step(prompts.INTERRUPTED, "ok"),
    ]
    async with started(CallbackAgent(toolbox=toolbox), scripted(*responses)) as session:
        result = await asyncio.wait_for(session.run(user_input="go"), timeout=5)
    assert toolbox.calls == []
    result.expect.next_event().is_function_call(name="hold_vendor")
    result.expect.next_event().is_function_call_output(output=prompts.INTERRUPTED, is_error=False)


async def test_a_toolbox_that_raises_still_gives_the_llm_a_reply(caplog):
    toolbox = FakeToolbox()
    toolbox.fail["list_vendors"] = RuntimeError("secret detail 000123456789")
    responses = [step("go", "", tool_call("list_vendors")), step(prompts.TOOL_FAILED, "ok")]
    with caplog.at_level(logging.DEBUG, logger="callback_agent"):
        async with started(CallbackAgent(toolbox=toolbox), scripted(*responses)) as session:
            result = await asyncio.wait_for(session.run(user_input="go"), timeout=5)
    result.expect.next_event().is_function_call(name="list_vendors")
    result.expect.next_event().is_function_call_output(output=prompts.TOOL_FAILED, is_error=False)
    assert "000123456789" not in caplog.text


# --- a second identical call in one reply (R57) ---

REJECTING = [
    "identify_caller",
    "change_vendor_bank_account",
    "release_vendor_hold",
    "release_bill_hold",
]


async def two_calls_in_one_reply(toolbox, first, second):
    """Run one LLM reply that makes both tool calls, keeping the first answer back until the second
    call has had every chance to start. Returns what each call was answered."""
    toolbox.hold = asyncio.Event()
    responses = [step("go", "", first, second)]
    agent = CallbackAgent(toolbox=toolbox)
    async with started(agent, scripted(*responses), tool_handling=TOOL_HANDLING) as session:

        async def run():
            return await session.run(user_input="go")

        running = asyncio.create_task(run())
        while not toolbox.calls:  # the first call is in
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.3)  # and the second has been dispatched, or turned away
        toolbox.hold.set()
        result = await asyncio.wait_for(running, timeout=10)
    return [e.item.output for e in result.events if e.type == "function_call_output"]


@pytest.mark.parametrize(
    ("name", "arguments", "call"),
    [t for t in TOOL_CALLS if t[0] in REJECTING],
    ids=REJECTING,
)
async def test_a_second_identical_call_in_one_reply_is_turned_away_while_the_first_runs(
    name, arguments, call
):
    toolbox = FakeToolbox({name: f"reply of {name}"})
    outputs = await two_calls_in_one_reply(
        toolbox, tool_call(name, "c1", **arguments), tool_call(name, "c2", **arguments)
    )
    assert toolbox.calls == [call]  # asked once, so there is one request to be allowed and written
    turned_away = prompts.DUPLICATE_CALL.format(function_name=name)
    assert sorted(outputs) == sorted([f"reply of {name}", turned_away])


@pytest.mark.parametrize(
    ("name", "arguments"),
    [(t[0], t[1]) for t in TOOL_CALLS if t[0] in REJECTING],
    ids=REJECTING,
)
async def test_the_call_turned_away_is_not_answered_with_the_arguments_of_the_one_running(
    name, arguments
):
    # LiveKit's own text for it quotes the running call, bank numbers included, and points at a
    # cancel tool this agent doesn't have
    toolbox = FakeToolbox({name: "ok"})
    outputs = await two_calls_in_one_reply(
        toolbox, tool_call(name, "c1", **arguments), tool_call(name, "c2", **arguments)
    )
    turned_away = next(o for o in outputs if o != "ok")
    assert ROUTING not in turned_away and ACCOUNT not in turned_away
    assert "lk_agents" not in turned_away and "{" not in turned_away


async def test_two_holds_in_one_reply_both_run():
    toolbox = FakeToolbox()
    acme = tool_call("hold_vendor", "c1", vendor_name="Acme", reason="caller asked")
    globex = tool_call("hold_vendor", "c2", vendor_name="Globex", reason="caller asked")
    outputs = await two_calls_in_one_reply(toolbox, acme, globex)
    assert sorted(toolbox.calls) == [
        ("hold_vendor", "Acme", "caller asked"),
        ("hold_vendor", "Globex", "caller asked"),
    ]
    assert outputs == ["hold_vendor was done"] * 2


# --- end_call ---


class FakeCall:
    def __init__(self) -> None:
        self.ended = False
        self.heard: list[str] = []  # what had been said when the line dropped
        self.messages: list[ChatMessage] = []

    async def end_call(self) -> None:
        self.ended = True
        self.heard = [m.text_content for m in self.messages if m.role == "assistant"]


async def test_end_call_hangs_up_after_the_goodbye_has_been_said():
    call = FakeCall()
    responses = [
        step("bye", "Goodbye, take care.", tool_call("end_call")),
        step(prompts.CALL_ENDING, ""),
    ]
    agent = CallbackAgent(toolbox=FakeToolbox(), call=call)
    async with started(agent, scripted(*responses)) as session:
        session.on(
            "conversation_item_added",
            lambda ev: call.messages.append(ev.item) if isinstance(ev.item, ChatMessage) else None,
        )
        result = await asyncio.wait_for(session.run(user_input="bye"), timeout=5)
    assert call.ended
    assert "Goodbye, take care." in call.heard  # it had been said before the line dropped
    result.expect.contains_function_call(name="end_call")


async def test_end_call_without_a_hangup_shuts_the_session_down():
    closed = asyncio.Event()
    responses = [step("bye", "Goodbye.", tool_call("end_call")), step(prompts.CALL_ENDING, "")]
    async with started(CallbackAgent(toolbox=FakeToolbox()), scripted(*responses)) as session:
        session.on("close", lambda ev: closed.set())
        session.generate_reply(user_input="bye")
        await asyncio.wait_for(closed.wait(), timeout=5)


async def test_a_hangup_that_fails_is_a_reply_and_not_an_exception():
    class BrokenCall:
        async def end_call(self) -> None:
            raise RuntimeError("the API is down")

    responses = [
        step("bye", "Goodbye.", tool_call("end_call")),
        step(prompts.TOOL_FAILED, "ok"),
    ]
    async with started(
        CallbackAgent(toolbox=FakeToolbox(), call=BrokenCall()), scripted(*responses)
    ) as s:
        result = await asyncio.wait_for(s.run(user_input="bye"), timeout=5)
    result.expect.contains_function_call_output(output=prompts.TOOL_FAILED, is_error=False)


# --- the hang-up under the gate's lock (Task 5 review) ---


ERIC = UserSummary(user_id="eric", display_name="Eric Catalano", role="CFO", phone_last_four="0100")


class DenyingDetector:
    """The detector for the hang-up test: it knows Eric, and denies everything it is asked."""

    async def find_users(self, name):
        return [ERIC]

    async def create_session(self, req):
        from callback_shared.contract import SessionCreated

        return SessionCreated(session_id="s1")

    async def authorize(self, session_id, req):
        kind = DecisionKind.ALLOW if ACTION_TIER[req.action] == 0 else DecisionKind.DENY
        return Decision(session_id=session_id, action=req.action, decision=kind)


class ClosingFromTheNetwork:
    """A hang-up that behaves like deleting the room: the job's shutdown closes the session at
    once, without draining it, and the HTTP answer reaches the tool a moment later.
    """

    def __init__(self, session_of) -> None:
        self._session_of = session_of
        self.ended = False

    async def end_call(self) -> None:
        self.ended = True
        # The close comes from the job's shutdown once the room is deleted, and it does not drain.
        # (RoomIO no longer closes the session when the caller leaves: close_on_disconnect is off.)
        self._session_of().shutdown(drain=False)
        await asyncio.sleep(0.05)  # the DeleteRoom reply is still on its way


class SilentSpeaker:
    def __init__(self) -> None:
        self.lines: list[str] = []

    async def say(self, text: str) -> None:
        self.lines.append(text)


class SessionVoice(SilentSpeaker):
    """The real LiveKitVoice, built once the session exists. It says each line in an AgentTask."""

    def __init__(self, holder: dict) -> None:
        super().__init__()
        self._holder = holder

    async def say(self, text: str) -> None:
        await super().say(text)
        await LiveKitVoice(self._holder["session"]).say(text)


class NoStepUps:
    async def run(self, factor, req, decision) -> bool:
        return False


@pytest.mark.parametrize(
    "voice",
    [
        "silent",
        pytest.param(
            "livekit",
            marks=pytest.mark.skipif(LiveKitVoice is None, reason="capture.py is not here yet"),
        ),
    ],
)
async def test_a_deny_that_hangs_up_while_the_tool_holds_the_gate_lock_cannot_deadlock(voice):
    state = CallState(room_name="r1", caller_identity="c", channel=Channel.PHONE)
    ramp = FakeRamp()
    ramp.vendors = [Vendor(id="v1", name="Acme Supplies Inc", is_active=True)]
    ramp.find_vendor = lambda name: RampClient.find_vendor(ramp, name)
    holder: dict[str, AgentSession] = {}
    call = ClosingFromTheNetwork(lambda: holder["session"])
    speaker = SilentSpeaker() if voice == "silent" else SessionVoice(holder)
    detector = DenyingDetector()
    gate = Gate(
        detector=detector, ramp=ramp, speaker=speaker, stepups=NoStepUps(), call=call, state=state
    )
    toolbox = Toolbox(state=state, gate=gate, ramp=ramp, detector=detector, forwarder=None)
    bank_change = tool_call(
        "change_vendor_bank_account",
        vendor_name="Acme",
        routing_number=ROUTING,
        account_number=ACCOUNT,
    )
    responses = [step("go", "", bank_change), step(prompts.DENY_HELD, "")]
    closed = asyncio.Event()
    async with started(CallbackAgent(toolbox=toolbox, call=call), scripted(*responses)) as session:
        holder["session"] = session
        session.on("close", lambda ev: closed.set())
        await toolbox.identify_caller("Eric Catalano")
        session.generate_reply(user_input="go")
        # If closing the session waited on a tool that waits on the close, this would time out.
        await asyncio.wait_for(closed.wait(), timeout=5)

        assert call.ended and state.locked
        assert speaker.lines == [prompts.VERIFYING, prompts.DENY_HELD]
        assert ramp.calls[-1] == ("hold_vendor", "v1")  # the deny held the vendor first
        assert ("add_default_bank_account", "v1", ROUTING, ACCOUNT) not in ramp.calls
        # The tool let go of the lock: a later gated request is answered, not queued behind it.
        later = await asyncio.wait_for(gate.run(ActionRequest.list_vendors()), timeout=2)
        assert later.status == "denied"
    await gate.aclose()


# --- a gated tool through a hang-up (final review, issue 3) ---
#
# A deny's hold is the one write that must not be lost, and it is often still in flight when the
# caller hangs up: a deny can follow a long verification. Nothing in this code keeps it alive.
# Three things in livekit-agents 1.8.10 do, and `~=1.8` allows any 1.8.x:
# - LiveKit's forced interrupt (SpeechHandle._cancel, INTERRUPTION_TIMEOUT, 5 s) cancels the tasks
#   of the reply and not the tool, whose body runs in a task of its own;
# - the session's close then drains the activity (AgentActivity.drain, _pause_scheduling_task),
#   which waits for every speech task to run its function tools to the end, before aclose() sweeps
#   what is left (its ToolExecutor.drain() awaits a tool not flagged CANCELLABLE, as none of ours
#   is, and its aclose() cancels the rest);
# - the job's shutdown awaits session.aclose() (up to 60 s) before it runs any shutdown callback,
#   and that call waits on the session lock the first close holds, so the Ramp client is closed
#   only after the close, which waited for the tool.
# The tests below run the real wire_hangups, hang_up, gate, toolbox and agent over a real
# AgentSession, hang up while the hold is in flight, and watch it through LiveKit's interrupt timer
# and the room's deletion. The timers are shortened, and each test checks that they went off while
# the hold was still running, so that it cannot pass without having met them.

APPLIED_HOLD_S = 0.6  # a vendor hold that takes at once: longer than the shortened timers below
PENDING_HOLD_S = 0.4  # one that waits for approval, and so is followed by a hold on each bill
BILL_HOLD_S = 0.2
HANG_UP_AFTER_S = 0.1  # how far into the vendor hold the caller hangs up


class HoldsInFlight(FakeRamp):
    """A Ramp whose holds take their time, and which refuses a request once it is closed.

    A vendor hold takes APPLIED_HOLD_S, or PENDING_HOLD_S if `pending`, and then the vendor's two
    open bills are held in BILL_HOLD_S each. `events` gets each request as "<call> sent" and then
    "<call> answered", or "<call> cancelled" if something cut it off, or "<call> refused" if the
    client was closed before it was sent or under it, as httpx's is. Whenever a request is
    answered it asks `reply_is_cut_off` and keeps the answer in `cut_off_at_each_answer`.
    """

    def __init__(self, events: list[str], *, pending: bool) -> None:
        super().__init__()
        self.events = events
        self.closed = False
        self.holding = asyncio.Event()  # set once the vendor hold is in flight
        self.reply_is_cut_off = lambda: False  # a test points it at the speech of the tool's reply
        self.cut_off_at_each_answer: list[bool] = []
        self.vendors = [Vendor(id="v1", name="Acme Supplies Inc", is_active=True)]
        self._vendor_hold_s = PENDING_HOLD_S if pending else APPLIED_HOLD_S
        if pending:
            self.vendor_hold = HoldOutcome.PENDING_APPROVAL

    async def find_vendor(self, name: str):
        return await RampClient.find_vendor(self, name)

    async def hold_vendor(self, vendor_id: str, reason: str) -> HoldOutcome:
        self.holding.set()
        await self._takes("hold_vendor", self._vendor_hold_s)
        return await super().hold_vendor(vendor_id, reason)

    async def hold_bill(self, bill_id: str, reason: str):
        await self._takes(f"hold_bill {bill_id}", BILL_HOLD_S)
        return await super().hold_bill(bill_id, reason)

    async def aclose(self) -> None:
        self.closed = True
        self.events.append("ramp.aclose()")

    async def _takes(self, call: str, seconds: float) -> None:
        if self.closed:
            self.events.append(f"{call} refused")
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        self.events.append(f"{call} sent")
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            self.events.append(f"{call} cancelled")
            raise
        if self.closed:  # closed under it
            self.events.append(f"{call} refused")
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        self.events.append(f"{call} answered")
        self.cut_off_at_each_answer.append(self.reply_is_cut_off())


class JobEndedByTheRoom:
    """The call's CallControl, as the job sees it: deleting the room ends the job.

    The first deletion starts what the 1.8.10 job does next (job_proc_lazy_main._shutdown_job): it
    waits for session.aclose(), which waits for the session's close, and then runs the shutdown
    callbacks, whose last step in finish_call is closing the Ramp client. The deny path deletes the
    room a second time once its hold is done, which is harmless for the job too, as
    JobContext.delete_room swallows not_found. `deleted` is set by the first deletion, and
    `shutdown` is the task of the job's shutdown.
    """

    def __init__(self, state: CallState, session_of, ramp: HoldsInFlight, events: list[str]):
        self._state = state
        self._session_of = session_of
        self._ramp = ramp
        self._events = events
        self.deleted = asyncio.Event()
        self.shutdown: asyncio.Task[None] | None = None

    async def end_call(self) -> None:
        self._state.closing = True
        self._events.append("room deleted")
        self.deleted.set()
        if self.shutdown is None:
            self.shutdown = asyncio.create_task(self._shut_the_job_down())

    async def _shut_the_job_down(self) -> None:
        # 1.8.10 allows session.aclose() 60 s. A test whose tool got stuck would wait that long, so
        # it gives up after far less and fails.
        await asyncio.wait_for(self._session_of().aclose(), timeout=5)
        self._events.append("session.aclose() returned")
        await self._ramp.aclose()


@pytest.mark.parametrize(
    "pending", [False, True], ids=["applied_vendor_hold", "pending_vendor_hold_then_bill_holds"]
)
async def test_the_hold_of_a_deny_is_carried_through_when_the_caller_hangs_up_during_it(
    pending, monkeypatch
):
    # Shortened from LiveKit's 5 s and the wiring's 3 s. Counted from the start of the hold, the
    # caller hangs up at 0.1 s, the room is deleted at about 0.2 s and LiveKit's forced interrupt
    # fires at about 0.3 s, all inside a hold that ends at 0.6 s (or, pending, at 0.8 s).
    monkeypatch.setattr(speech_handle, "INTERRUPTION_TIMEOUT", 0.2)
    monkeypatch.setattr(wiring, "SESSION_CLOSE_S", 0.1)
    events: list[str] = []
    state = CallState(room_name="r1", caller_identity="caller", channel=Channel.PHONE)
    ramp = HoldsInFlight(events, pending=pending)
    holder: dict[str, AgentSession] = {}
    call = JobEndedByTheRoom(state, lambda: holder["session"], ramp, events)
    speaker = SessionVoice(holder)  # the real LiveKitVoice says the VERIFYING line
    detector = DenyingDetector()
    gate = Gate(
        detector=detector, ramp=ramp, speaker=speaker, stepups=NoStepUps(), call=call, state=state
    )
    toolbox = Toolbox(state=state, gate=gate, ramp=ramp, detector=detector, forwarder=None)
    replies: list[str] = []
    real_change = toolbox.change_vendor_bank_account

    async def change(**arguments) -> str:
        ramp.reply_is_cut_off = holder["session"].current_speech.done  # this tool's own reply
        replies.append(await real_change(**arguments))
        return replies[-1]

    toolbox.change_vendor_bank_account = change
    bank_change = tool_call(
        "change_vendor_bank_account",
        vendor_name="Acme",
        routing_number=ROUTING,
        account_number=ACCOUNT,
    )
    responses = [step("go", "", bank_change), step(prompts.DENY_HELD, "")]
    room = rtc.Room()
    agent = CallbackAgent(toolbox=toolbox, call=call)
    async with started(agent, scripted(*responses), tool_handling=TOOL_HANDLING) as session:
        holder["session"] = session
        wiring.wire_hangups(room, session, state, call)
        await toolbox.identify_caller("Eric Catalano")
        session.generate_reply(user_input="go")
        await asyncio.wait_for(ramp.holding.wait(), timeout=5)
        await asyncio.sleep(HANG_UP_AFTER_S)
        events.append("caller hangs up")
        room.emit("participant_disconnected", SimpleNamespace(identity="caller"))
        await asyncio.wait_for(call.deleted.wait(), timeout=5)
        await asyncio.wait_for(call.shutdown, timeout=5)
    await gate.aclose()

    # What the review asked for: the vendor hold answered, and in the pending case so did each
    # bill hold, before the job's shutdown closed the Ramp client. None was cancelled or refused.
    held = [("hold_vendor", "v1")]
    if pending:
        held += [("list_bills", "v1"), ("hold_bill", "b1"), ("hold_bill", "b2")]
    assert ramp.calls == [("list_vendors",), *held], events
    assert [e for e in events if e.endswith(("cancelled", "refused"))] == [], events
    assert events[-1] == "ramp.aclose()", events
    assert replies == [prompts.DENY_HELD]  # the tool reported what it did
    assert speaker.lines == [prompts.VERIFYING]  # and said nothing to a caller who had gone
    assert state.locked

    # The test is worth something only if the hazards went off while the tool was still running:
    # the caller hung up during the hold, the room was deleted before the last request was
    # answered, and LiveKit's forced interrupt had cut the reply off by then.
    last_answer = max(i for i, event in enumerate(events) if event.endswith("answered"))
    assert events.index("hold_vendor sent") < events.index("caller hangs up"), events
    assert events.index("caller hangs up") < events.index("room deleted") < last_answer, events
    assert ramp.cut_off_at_each_answer[-1] is True, events
