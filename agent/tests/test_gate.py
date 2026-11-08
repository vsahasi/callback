import asyncio
import logging
import re
from collections.abc import Awaitable, Callable, Iterator
from types import SimpleNamespace

import pytest
from callback_shared.contract import (
    Action,
    AuthorizeRequest,
    Channel,
    Decision,
    DecisionKind,
    Factor,
)
from fake_ramp import FakeRamp

from callback_agent import prompts
from callback_agent.actions import ActionRequest
from callback_agent.gate import Gate, GateResult
from callback_agent.ramp import HoldOutcome, RampError
from callback_agent.state import CallState

ROUTING = "021000021"
ACCOUNT = "000123456789"
BANK = ActionRequest.update_vendor_bank_account("v1", "Acme Supplies", ROUTING, ACCOUNT)
RELEASE = ActionRequest.release_vendor_hold("v1", "Acme Supplies")
ACCOUNTS = ActionRequest.list_vendor_accounts("v1", "Acme Supplies")
BILLS = ActionRequest.list_bills("v1", "Acme Supplies")
HOLD = ActionRequest.hold_vendor("v1", "Acme Supplies", "caller asked")
HOLD_BILL = ActionRequest.hold_bill("b1", "v1", "caller asked")
VOICE, OTP, CALLBACK = Factor.VOICE_CHALLENGE, Factor.SMS_OTP, Factor.CALLBACK_ON_FILE

# One scripted answer from the detector: a decision kind and its required factors.
ALLOW = (DecisionKind.ALLOW, [])
DENY = (DecisionKind.DENY, [])


def step_up(*factors: Factor) -> tuple[DecisionKind, list[Factor]]:
    return (DecisionKind.STEP_UP, list(factors))


async def hang() -> None:
    await asyncio.Event().wait()


def after_hangup(state: CallState, answer):
    """A detector script step: the caller hangs up while the answer is on its way."""

    async def step():
        state.closing = True
        if isinstance(answer, BaseException):
            raise answer
        return answer

    return step


class ScriptExhausted(Exception):
    """The fake detector was asked for a decision its test never scripted."""


class FakeSpeaker:
    """Records each line said. `fail` maps a line to the exception that saying it raises."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.fail: dict[str, BaseException] = {}

    async def say(self, text: str) -> None:
        if text in self.fail:
            raise self.fail[text]
        self.lines.append(text)


class FakeCall:
    """`ended` is set as soon as end_call is called, even when `fail` then makes it raise."""

    def __init__(self) -> None:
        self.ended = False
        self.fail: BaseException | None = None

    async def end_call(self) -> None:
        self.ended = True
        if self.fail is not None:
            raise self.fail


class FakeDetector:
    """Answers authorize from `script`, in order, and records each request.

    A script step is a (kind, factors) pair, an exception to raise, or an async function whose
    result is used as the step. `heard` keeps what the caller had heard when each request arrived.
    A request with nothing left in the script raises ScriptExhausted and is kept in `exhausted`:
    the gate turns any exception into a deny, so the detector fixture fails the test on a record.
    """

    def __init__(self, speaker: FakeSpeaker) -> None:
        self.script: list = []
        self.requests: list[AuthorizeRequest] = []
        self.sessions: list[str] = []
        self.heard: list[list[str]] = []
        self.called = asyncio.Event()
        self.cancelled = 0
        self.exhausted: list[ScriptExhausted] = []
        self._speaker = speaker

    async def authorize(self, session_id: str, req: AuthorizeRequest) -> Decision:
        self.sessions.append(session_id)
        self.requests.append(req)
        self.heard.append(list(self._speaker.lines))
        self.called.set()
        if not self.script:
            error = ScriptExhausted(f"nothing is scripted for the authorize of {req.action}")
            self.exhausted.append(error)
            raise error
        step = self.script.pop(0)
        if callable(step):
            try:
                step = await step()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        if isinstance(step, BaseException):
            raise step
        kind, factors = step
        return Decision(
            session_id=session_id,
            action=req.action,
            decision=kind,
            required_factors=factors,
            reasons=[f"scripted {kind}"],
        )


class FakeStepUps:
    """Answers each factor from `script` (True when unscripted) and records each run.

    An answer is a bool, an exception to raise, or an async function returning the bool.
    """

    def __init__(self) -> None:
        self.script: dict[Factor, bool | BaseException | Callable[[], Awaitable[bool]]] = {}
        self.calls: list[Factor] = []
        self.runs: list[tuple[ActionRequest, Decision]] = []
        self.started = asyncio.Event()

    async def run(self, factor: Factor, req: ActionRequest, decision: Decision) -> bool:
        self.calls.append(factor)
        self.runs.append((req, decision))
        self.started.set()
        answer = self.script.get(factor, True)
        if isinstance(answer, BaseException):
            raise answer
        if callable(answer):
            return await answer()
        return answer


@pytest.fixture
def state() -> CallState:
    return CallState(
        room_name="r1", caller_identity="caller-1", channel=Channel.PHONE, session_id="s1"
    )


@pytest.fixture
def speaker() -> FakeSpeaker:
    return FakeSpeaker()


@pytest.fixture
def detector(speaker) -> Iterator[FakeDetector]:
    detector = FakeDetector(speaker)
    yield detector
    assert not detector.exhausted, f"the detector's script ran out: {detector.exhausted}"


@pytest.fixture
def ramp() -> FakeRamp:
    return FakeRamp()


@pytest.fixture
def stepups() -> FakeStepUps:
    return FakeStepUps()


@pytest.fixture
def call() -> FakeCall:
    return FakeCall()


@pytest.fixture
async def make_gate(detector, ramp, speaker, stepups, call, state):
    """Builds a gate. Options override its parts or settings, such as `detector=` or
    `max_rounds=`. Every gate it builds is closed after the test."""
    built: list[Gate] = []

    def make(**options) -> Gate:
        parts = dict(
            detector=detector,
            ramp=ramp,
            speaker=speaker,
            stepups=stepups,
            call=call,
            state=state,
        )
        built.append(Gate(**{**parts, **options}))
        return built[-1]

    yield make
    for gate in built:
        await asyncio.wait_for(gate.aclose(), timeout=2)  # a broken aclose fails instead of hanging


@pytest.fixture
def gate(make_gate) -> Gate:
    return make_gate()


def assert_denied_with_hold(
    result, ramp, speaker, call, state, *, first_lines=(prompts.VERIFYING,)
):
    assert result == GateResult("denied", prompts.DENY_HELD)
    assert ramp.calls == [("hold_vendor", "v1")]
    assert speaker.lines == [*first_lines, prompts.DENY_HELD]
    assert call.ended and state.locked


# --- allow, step-up and deny ---


async def test_allow_runs_exactly_the_authorized_request(gate, detector, ramp):
    detector.script = [ALLOW]
    assert (await gate.run(BANK)).status == "done"
    assert ramp.calls == [("add_default_bank_account", "v1", ROUTING, ACCOUNT)]
    assert "000123456789" not in detector.requests[0].model_dump_json()
    assert ROUTING not in detector.requests[0].model_dump_json()
    assert detector.sessions == ["s1"]
    assert detector.requests[0] == AuthorizeRequest(
        action=Action.UPDATE_VENDOR_BANK_ACCOUNT,
        params=BANK.detector_params(),
        idempotency_key=BANK.idempotency_key("s1", 0),
        context=BANK.context(),
    )


async def test_step_up_runs_factors_in_order_then_reauthorizes_with_a_new_key(
    gate, detector, stepups, ramp
):
    detector.script = [step_up(VOICE, OTP, CALLBACK), ALLOW]
    assert (await gate.run(BANK)) == GateResult(
        "done", "Acme Supplies now pays to the account ending 6789."
    )
    assert stepups.calls == [Factor.VOICE_CHALLENGE, Factor.SMS_OTP, Factor.CALLBACK_ON_FILE]
    assert detector.requests[0].idempotency_key != detector.requests[1].idempotency_key
    assert [r.idempotency_key for r in detector.requests] == [
        BANK.idempotency_key("s1", 0),
        BANK.idempotency_key("s1", 1),
    ]
    # Each factor gets the very request being authorized and the decision that asked for it.
    assert all(
        req is BANK and decision.required_factors == [VOICE, OTP, CALLBACK]
        for req, decision in stepups.runs
    )
    assert ramp.calls == [("add_default_bank_account", "v1", ROUTING, ACCOUNT)]


async def test_a_failed_factor_ends_the_round_and_reauthorizes(gate, detector, stepups):
    # No OTP and no callback to the number on file after a failed voice challenge: the detector
    # decides at once what the failure means.
    stepups.script = {VOICE: False}
    detector.script = [step_up(VOICE, OTP, CALLBACK), DENY]
    assert (await gate.run(BANK)).status == "denied"
    assert stepups.calls == [Factor.VOICE_CHALLENGE]
    assert len(detector.requests) == 2


async def test_the_round_stops_at_the_first_failed_factor(gate, detector, stepups):
    stepups.script = {OTP: False}
    detector.script = [step_up(VOICE, OTP, CALLBACK), ALLOW]
    assert (await gate.run(BANK)).status == "done"  # the detector, not the gate, judges a failure
    assert stepups.calls == [VOICE, OTP] and len(detector.requests) == 2


async def test_deny_holds_vendor_speaks_ends_call_and_locks(
    gate, detector, ramp, speaker, call, state
):
    detector.script = [DENY]
    result = await gate.run(BANK)
    assert result.status == "denied" and ramp.calls[0] == ("hold_vendor", "v1")
    assert speaker.lines == [prompts.VERIFYING, prompts.DENY_HELD] and call.ended and state.locked
    assert (await gate.run(BANK)).status == "denied" and len(detector.requests) == 1
    assert_denied_with_hold(result, ramp, speaker, call, state)


async def test_detector_exception_is_deny(gate, detector, ramp, speaker, call, state):
    detector.script = [RuntimeError("detector blew up")]
    assert_denied_with_hold(await gate.run(BANK), ramp, speaker, call, state)


async def test_tier2_says_verifying_before_authorizing(gate, detector, speaker):
    detector.script = [ALLOW, ALLOW]
    assert (await gate.run(ACCOUNTS)).status == "done"
    assert detector.heard == [[]] and speaker.lines == []  # tier 1 says nothing
    assert (await gate.run(BANK)).status == "done"
    assert detector.heard[1] == [prompts.VERIFYING] and speaker.lines[0] == prompts.VERIFYING


async def test_step_up_exception_is_deny(gate, detector, stepups, ramp, speaker, call, state):
    stepups.script = {OTP: RuntimeError("twilio is down")}
    detector.script = [step_up(VOICE, OTP, CALLBACK)]
    assert_denied_with_hold(await gate.run(BANK), ramp, speaker, call, state)
    assert stepups.calls == [VOICE, OTP] and len(detector.requests) == 1


async def test_endless_step_up_is_denied_after_max_rounds(
    gate, detector, stepups, ramp, speaker, call, state
):
    # max_rounds is 6. The sixth round's step-up runs no factor: whatever it passed, a deny follows.
    detector.script = [step_up(VOICE)] * 6
    assert_denied_with_hold(await gate.run(BANK), ramp, speaker, call, state)
    assert len(detector.requests) == 6 and stepups.calls == [VOICE] * 5
    assert len({r.idempotency_key for r in detector.requests}) == 6


@pytest.mark.parametrize("max_rounds", [1, 2, 3, 6])
async def test_a_step_up_on_the_last_round_runs_no_factors(
    make_gate, detector, stepups, ramp, speaker, call, state, caplog, max_rounds
):
    # A factor would text a real code or phone the number on file, then the deny would follow.
    gate = make_gate(max_rounds=max_rounds)
    detector.script = [step_up(VOICE, OTP)] * max_rounds
    assert_denied_with_hold(await gate.run(BANK), ramp, speaker, call, state)
    assert len(detector.requests) == max_rounds
    assert stepups.calls == [VOICE, OTP] * (max_rounds - 1)
    assert "too many step-ups" in caplog.text


async def test_an_allow_on_the_last_round_still_runs(make_gate, detector, stepups, ramp, state):
    gate = make_gate(max_rounds=2)
    detector.script = [step_up(VOICE), ALLOW]
    assert (await gate.run(BANK)).status == "done"
    assert stepups.calls == [VOICE] and not state.locked
    assert ramp.calls == [("add_default_bank_account", "v1", ROUTING, ACCOUNT)]


async def test_step_up_without_factors_is_deny(gate, detector, stepups, ramp, speaker, call, state):
    detector.script = [step_up()]
    assert_denied_with_hold(await gate.run(BANK), ramp, speaker, call, state)
    assert stepups.calls == [] and len(detector.requests) == 1


async def test_failed_hold_on_deny_uses_the_no_hold_line(
    gate, detector, ramp, speaker, call, state
):
    detector.script = [DENY]
    ramp.fail["hold_vendor"] = RampError(500)
    assert await gate.run(BANK) == GateResult("denied", prompts.DENY_NOT_HELD)
    assert (
        speaker.lines == [prompts.VERIFYING, prompts.DENY_NOT_HELD] and call.ended and state.locked
    )


@pytest.mark.parametrize(
    "bills_held_as, line",
    [
        ({}, prompts.DENY_HELD),  # the bills held, which take effect at once
        ({"b1": "PAYMENT_SCHEDULED", "b2": "PAYMENT_SCHEDULED"}, prompts.DENY_NOT_HELD),
        (None, prompts.DENY_NOT_HELD),  # no open bills: the vendor hold is only pending
    ],
)
async def test_a_pending_vendor_hold_on_deny_says_held_only_if_a_bill_held(
    gate, detector, ramp, speaker, call, bills_held_as, line
):
    detector.script = [DENY]
    ramp.vendor_hold = HoldOutcome.PENDING_APPROVAL
    if bills_held_as is None:
        ramp.bills = []
    else:
        ramp.held_as = bills_held_as
    assert await gate.run(BANK) == GateResult("denied", line)
    assert speaker.lines == [prompts.VERIFYING, line] and call.ended


async def test_tier1_deny_holds_the_vendor_it_names(gate, detector, ramp, speaker, call, state):
    detector.script = [DENY]
    assert await gate.run(ACCOUNTS) == GateResult("denied", prompts.DENY_READ_HELD)
    assert ramp.calls == [("hold_vendor", "v1")]
    assert speaker.lines == [prompts.DENY_READ_HELD]
    assert call.ended and state.locked


async def test_a_denied_read_whose_hold_failed_does_not_claim_one(
    gate, detector, ramp, speaker, call, state
):
    detector.script = [DENY]
    ramp.fail["hold_vendor"] = RampError(500)
    assert await gate.run(ACCOUNTS) == GateResult("denied", prompts.DENY_NOT_HELD)
    assert speaker.lines == [prompts.DENY_NOT_HELD] and call.ended and state.locked


def test_the_read_line_says_a_hold_and_names_no_reason():
    line = prompts.DENY_READ_HELD.lower()
    assert "hold" in line and "payments" in line and "can't help with that" in line
    assert "fraud" not in line and "change" not in line


async def test_a_denied_vendor_list_holds_nothing(gate, detector, ramp, speaker, call, state):
    # list_vendors names no vendor, so there is nothing to hold.
    detector.script = [DENY]
    assert await gate.run(ActionRequest.list_vendors()) == GateResult(
        "denied", prompts.DENY_NOT_HELD
    )
    assert ramp.calls == [] and speaker.lines == [prompts.DENY_NOT_HELD]
    assert call.ended and state.locked


async def test_detector_down_holds_the_vendor_of_a_tier1_read(
    gate, detector, ramp, speaker, call, state
):
    state.session_id, state.detector_down = None, True
    assert await gate.run(ACCOUNTS) == GateResult("denied", prompts.DENY_READ_HELD)
    assert detector.requests == [] and ramp.calls == [("hold_vendor", "v1")]
    assert speaker.lines == [prompts.DENY_READ_HELD] and call.ended and state.locked


# --- before identification ---


@pytest.mark.parametrize("req", [BANK, ACCOUNTS])
async def test_unidentified_caller_is_told_to_identify_first(
    gate, detector, ramp, speaker, call, state, req
):
    state.session_id = None
    assert await gate.run(req) == GateResult("failed", prompts.IDENTIFY_FIRST)
    assert detector.requests == [] and ramp.calls == [] and speaker.lines == []
    assert not call.ended and not state.locked


async def test_detector_down_before_identification_is_deny(
    gate, detector, ramp, speaker, call, state
):
    state.session_id, state.detector_down = None, True
    assert_denied_with_hold(await gate.run(BANK), ramp, speaker, call, state, first_lines=())
    assert detector.requests == []


# --- the call ending, a lock and cancellation ---


@pytest.mark.parametrize(
    "answer", [DENY, RuntimeError("detector blew up")], ids=["deny", "detector_error"]
)
async def test_a_deny_after_the_caller_hung_up_holds_but_says_nothing(
    gate, detector, ramp, speaker, call, state, answer
):
    # A deny (or a timeout, which is a deny) can arrive for an authorize still in flight when the
    # caller leaves. The hold protects the account either way. There is nobody to tell.
    detector.script = [after_hangup(state, answer)]
    assert await gate.run(BANK) == GateResult("denied", prompts.DENY_HELD)
    assert ramp.calls == [("hold_vendor", "v1")]
    assert speaker.lines == [prompts.VERIFYING]  # the deny line is not said
    assert call.ended and state.locked


async def test_a_read_denied_after_the_caller_hung_up_holds_but_says_nothing(
    gate, detector, ramp, speaker, call, state
):
    detector.script = [after_hangup(state, DENY)]
    assert await gate.run(ACCOUNTS) == GateResult("denied", prompts.DENY_READ_HELD)
    assert ramp.calls == [("hold_vendor", "v1")]
    assert speaker.lines == [] and call.ended and state.locked


async def test_a_step_up_that_raises_after_the_caller_hung_up_holds_but_says_nothing(
    gate, detector, stepups, ramp, speaker, call, state
):
    async def raises_as_the_caller_leaves() -> bool:
        state.closing = True
        raise RuntimeError("the call dropped")

    stepups.script = {VOICE: raises_as_the_caller_leaves}
    detector.script = [step_up(VOICE, OTP)]
    assert await gate.run(BANK) == GateResult("denied", prompts.DENY_HELD)
    assert stepups.calls == [VOICE] and ramp.calls == [("hold_vendor", "v1")]
    assert speaker.lines == [prompts.VERIFYING]
    assert call.ended and state.locked


async def test_a_hang_up_during_the_hold_also_skips_the_deny_line(
    gate, detector, ramp, speaker, call, state, monkeypatch
):
    # The check comes right before the line is said, not when the deny path starts.
    hold = ramp.hold_vendor

    async def hold_as_the_caller_leaves(vendor_id: str, reason: str) -> HoldOutcome:
        state.closing = True
        return await hold(vendor_id, reason)

    monkeypatch.setattr(ramp, "hold_vendor", hold_as_the_caller_leaves)
    detector.script = [DENY]
    assert await gate.run(BANK) == GateResult("denied", prompts.DENY_HELD)
    assert speaker.lines == [prompts.VERIFYING] and call.ended and state.locked


async def test_closing_mid_step_up_stops_without_ramp_or_speech(
    gate, detector, stepups, ramp, speaker, call, state
):
    async def caller_hangs_up() -> bool:
        state.closing = True
        return False

    stepups.script = {VOICE: caller_hangs_up}
    detector.script = [step_up(VOICE, OTP, CALLBACK), ALLOW]
    result = await gate.run(BANK)
    assert result.status == "denied" and ramp.calls == [] and speaker.lines == [prompts.VERIFYING]
    assert stepups.calls == [VOICE] and len(detector.requests) == 1 and not call.ended


@pytest.mark.parametrize("flag", ["closing", "locked"])
async def test_a_stop_set_while_one_factor_runs_keeps_the_next_from_running(
    gate, detector, stepups, ramp, speaker, state, flag
):
    # The stop is checked before each factor, not only before each round: the next factor would
    # text a code or phone the number on file for a call that is over or already denied.
    async def passes_as_the_call_stops() -> bool:
        setattr(state, flag, True)
        return True

    stepups.script = {VOICE: passes_as_the_call_stops}
    detector.script = [step_up(VOICE, OTP)]
    assert await gate.run(BANK) == GateResult("denied", prompts.DENY_NOT_HELD)
    assert stepups.calls == [Factor.VOICE_CHALLENGE]
    assert ramp.calls == [] and len(detector.requests) == 1
    assert speaker.lines == [prompts.VERIFYING]


async def test_closing_before_the_request_says_and_sends_nothing(
    gate, detector, ramp, speaker, state
):
    state.closing = True
    assert (await gate.run(BANK)).status == "denied"
    assert speaker.lines == [] and detector.requests == [] and ramp.calls == []


async def test_closing_while_authorizing_never_executes(gate, detector, ramp, speaker, state):
    async def allow_after_hangup():
        state.closing = True
        return ALLOW

    detector.script = [allow_after_hangup]
    assert (await gate.run(BANK)).status == "denied"
    assert ramp.calls == [] and speaker.lines == [prompts.VERIFYING]


async def test_a_lock_set_mid_request_stops_it_before_ramp(
    gate, detector, stepups, ramp, speaker, state
):
    async def denied_elsewhere() -> bool:
        state.locked = True
        return True

    stepups.script = {VOICE: denied_elsewhere}
    detector.script = [step_up(VOICE), ALLOW]
    assert (await gate.run(BANK)).status == "denied"
    assert ramp.calls == [] and len(detector.requests) == 1 and speaker.lines == [prompts.VERIFYING]


async def test_cancel_during_step_up_never_touches_ramp(
    gate, detector, stepups, ramp, speaker, call, state
):
    stepups.script = {VOICE: hang}  # stepups.run awaits forever
    detector.script = [step_up(VOICE), ALLOW]
    task = asyncio.create_task(gate.run(BANK))
    await asyncio.wait_for(stepups.started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ramp.calls == []
    assert speaker.lines == [prompts.VERIFYING] and not call.ended and not state.locked
    assert len(detector.requests) == 1


async def test_cancel_during_authorize_never_touches_ramp(
    gate, detector, stepups, ramp, speaker, call, state
):
    detector.script = [hang]
    task = asyncio.create_task(gate.run(BANK))
    await asyncio.wait_for(detector.called.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ramp.calls == [] and stepups.calls == [] and detector.cancelled == 1
    assert speaker.lines == [prompts.VERIFYING] and not call.ended and not state.locked


async def test_gated_requests_run_one_at_a_time(gate, detector):
    # LiveKit runs a reply's tool calls concurrently; two verifications must not interleave.
    release = asyncio.Event()

    async def slow_allow():
        await release.wait()
        return ALLOW

    detector.script = [slow_allow, ALLOW]
    first = asyncio.create_task(gate.run(ACCOUNTS))
    second = asyncio.create_task(gate.run(BILLS))
    await asyncio.wait_for(detector.called.wait(), 1)
    for _ in range(10):
        await asyncio.sleep(0)
    assert len(detector.requests) == 1
    release.set()
    assert [r.status for r in await asyncio.gather(first, second)] == ["done", "done"]
    assert [r.action for r in detector.requests] == [Action.LIST_VENDOR_ACCOUNTS, Action.LIST_BILLS]


# --- exclusive: the lock that keeps an enrollment from overlapping a gated request ---


async def hold_exclusively(gate: Gate, entered: asyncio.Event, release: asyncio.Event) -> None:
    """Take the gate's exclusive lock, say so, and keep it until `release` is set."""
    async with gate.exclusive():
        entered.set()
        await release.wait()


async def test_a_gated_request_started_while_exclusive_is_held_waits_until_it_is_released(
    gate, detector, ramp
):
    detector.script = [ALLOW]
    entered, release = asyncio.Event(), asyncio.Event()
    holder = asyncio.create_task(hold_exclusively(gate, entered, release))
    await asyncio.wait_for(entered.wait(), 1)
    request = asyncio.create_task(gate.run(ACCOUNTS))
    for _ in range(10):
        await asyncio.sleep(0)
    # it has not so much as asked the detector, let alone reached Ramp
    assert not request.done() and detector.requests == [] and ramp.calls == []
    release.set()
    assert (await asyncio.wait_for(request, 1)).status == "done"
    await holder
    assert len(detector.requests) == 1 and ramp.calls == [("list_vendor_accounts", "v1")]


async def test_exclusive_entered_while_a_gated_request_runs_waits_for_it(gate, detector):
    release = asyncio.Event()

    async def slow_allow():
        await release.wait()
        return ALLOW

    detector.script = [slow_allow]
    request = asyncio.create_task(gate.run(ACCOUNTS))
    await asyncio.wait_for(detector.called.wait(), 1)
    entered, release_exclusive = asyncio.Event(), asyncio.Event()
    release_exclusive.set()  # it lets go the moment it has the lock
    taker = asyncio.create_task(hold_exclusively(gate, entered, release_exclusive))
    for _ in range(10):
        await asyncio.sleep(0)
    assert not entered.is_set() and not request.done()
    release.set()
    assert (await asyncio.wait_for(request, 1)).status == "done"
    await asyncio.wait_for(taker, 1)
    assert entered.is_set()


async def test_two_exclusive_sections_take_turns(gate):
    first_entered, release_first, second_entered = (asyncio.Event() for _ in range(3))
    first = asyncio.create_task(hold_exclusively(gate, first_entered, release_first))
    await asyncio.wait_for(first_entered.wait(), 1)
    release_second = asyncio.Event()
    release_second.set()
    second = asyncio.create_task(hold_exclusively(gate, second_entered, release_second))
    for _ in range(10):
        await asyncio.sleep(0)
    assert not second_entered.is_set()
    release_first.set()
    await asyncio.wait_for(asyncio.gather(first, second), 1)
    assert second_entered.is_set()


async def test_a_hold_does_not_wait_for_exclusive(gate, detector, ramp):
    # Holding is always allowed: an enrollment that takes minutes cannot keep a caller from it.
    detector.script = [ALLOW]
    entered, release = asyncio.Event(), asyncio.Event()
    holder = asyncio.create_task(hold_exclusively(gate, entered, release))
    await asyncio.wait_for(entered.wait(), 1)
    assert (await asyncio.wait_for(gate.run(HOLD), 1)).status == "done"
    assert ramp.calls == [("hold_vendor", "v1")]
    release.set()
    await holder


@pytest.mark.parametrize(
    "error",
    [RuntimeError("enrollment broke"), asyncio.CancelledError()],
    ids=["a failure", "a cancellation"],
)
async def test_exclusive_lets_go_of_the_lock_when_its_body_fails(gate, detector, error):
    with pytest.raises(type(error)):
        async with gate.exclusive():
            raise error
    detector.script = [ALLOW]
    assert (await asyncio.wait_for(gate.run(ACCOUNTS), 1)).status == "done"


async def test_a_task_cancelled_while_it_waits_for_exclusive_does_not_keep_the_lock(
    gate, detector
):
    entered, release = asyncio.Event(), asyncio.Event()
    holder = asyncio.create_task(hold_exclusively(gate, entered, release))
    await asyncio.wait_for(entered.wait(), 1)
    waiting_entered = asyncio.Event()
    waiting = asyncio.create_task(hold_exclusively(gate, waiting_entered, asyncio.Event()))
    for _ in range(10):
        await asyncio.sleep(0)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    release.set()
    await holder
    detector.script = [ALLOW]
    assert (await asyncio.wait_for(gate.run(ACCOUNTS), 1)).status == "done"
    assert not waiting_entered.is_set()


async def test_a_repeated_request_never_reuses_a_key(gate, detector):
    # A reused key would get the detector's cached decision for it, a spent allow included.
    detector.script = [ALLOW, ALLOW]
    await gate.run(ACCOUNTS)
    await gate.run(ACCOUNTS)
    assert detector.requests[0].idempotency_key != detector.requests[1].idempotency_key


# --- what Ramp answers after allow ---


@pytest.mark.parametrize(
    "req, setting, value, summary",
    [
        (
            BANK,
            "bank_status",
            "pending_approval",
            "The change to the account ending 6789 for Acme Supplies is waiting for approval "
            "in Ramp.",
        ),
        (
            RELEASE,
            "vendor_release",
            HoldOutcome.PENDING_APPROVAL,
            "Releasing the hold on Acme Supplies is waiting for approval in Ramp.",
        ),
    ],
)
async def test_pending_approval_is_not_reported_done(
    gate, detector, ramp, req, setting, value, summary
):
    detector.script = [ALLOW]
    setattr(ramp, setting, value)
    assert await gate.run(req) == GateResult("pending_approval", summary)


@pytest.mark.parametrize("status", [400, 409, 422, 429])
async def test_ramp_error_after_allow_is_failed_without_hold_or_hangup(
    gate, detector, ramp, speaker, call, state, status
):
    detector.script = [ALLOW]
    ramp.fail["add_default_bank_account"] = RampError(status, "DEVELOPER_7001")
    assert await gate.run(BANK) == GateResult("failed", prompts.RAMP_FAILED)
    assert ramp.calls == [("add_default_bank_account", "v1", ROUTING, ACCOUNT)]  # never retried
    assert speaker.lines == [prompts.VERIFYING] and not call.ended and not state.locked


@pytest.mark.parametrize("status", [0, 500, 502, 503, 200])  # 200: an answer that didn't parse
async def test_ramp_error_with_no_answer_or_5xx_after_allow_is_unknown(
    gate, detector, ramp, speaker, call, state, status
):
    # Ramp: a timeout doesn't prove a write failed, so the caller mustn't hear "it didn't go
    # through".
    detector.script = [ALLOW]
    ramp.fail["add_default_bank_account"] = RampError(status)
    assert await gate.run(BANK) == GateResult("failed", prompts.RAMP_UNKNOWN)
    assert ramp.calls == [("add_default_bank_account", "v1", ROUTING, ACCOUNT)]  # never retried
    assert speaker.lines == [prompts.VERIFYING] and not call.ended and not state.locked


@pytest.mark.parametrize("status", [0, 503, 422])
async def test_a_failed_read_changed_nothing_so_it_is_never_unknown(gate, detector, ramp, status):
    detector.script = [ALLOW]
    ramp.fail["list_vendor_accounts"] = RampError(status)
    assert await gate.run(ACCOUNTS) == GateResult("failed", prompts.RAMP_FAILED)


async def test_a_ramp_failure_is_logged_without_ramps_message(gate, detector, ramp, caplog):
    # Ramp may echo a submitted number back, reformatted so the client's redaction misses it.
    caplog.set_level(logging.DEBUG)
    detector.script = [ALLOW]
    ramp.fail["add_default_bank_account"] = RampError(
        422, "DEVELOPER_7001", "account 0001-2345-6789 is not valid", "trace-9"
    )
    await gate.run(BANK)
    assert "422" in caplog.text and "DEVELOPER_7001" in caplog.text and "trace-9" in caplog.text
    assert "0001-2345-6789" not in caplog.text and ACCOUNT not in caplog.text


# --- edges that fail: the gate still answers ---

# Each request, the RampClient method that carries it out, and what the LLM is told when that
# method dies with something other than a RampError: a write may have gone through, a read didn't.
WRITES_AND_READS = [
    (BANK, "add_default_bank_account", prompts.RAMP_UNKNOWN),
    (RELEASE, "release_vendor", prompts.RAMP_UNKNOWN),
    (HOLD, "hold_vendor", prompts.RAMP_UNKNOWN),
    (HOLD_BILL, "hold_bill", prompts.RAMP_UNKNOWN),
    (ACCOUNTS, "list_vendor_accounts", prompts.RAMP_FAILED),
    (BILLS, "list_bills", prompts.RAMP_FAILED),
]


@pytest.mark.parametrize("error_type", [RuntimeError, KeyError, TimeoutError])
@pytest.mark.parametrize(
    "req, method, message", WRITES_AND_READS, ids=[method for _, method, _ in WRITES_AND_READS]
)
async def test_an_exception_that_is_not_a_ramp_error_after_allow_is_failed(
    gate, detector, ramp, speaker, call, state, caplog, req, method, message, error_type
):
    caplog.set_level(logging.DEBUG)
    detector.script = [ALLOW]
    ramp.fail[method] = error_type(f"no route to Ramp for account 0001-2345-6789 / {ACCOUNT}")
    assert await gate.run(req) == GateResult("failed", message)
    assert [name for name, *_ in ramp.calls] == [method]  # never retried
    assert speaker.lines == ([prompts.VERIFYING] if req.tier == 2 else [])
    assert not call.ended and not state.locked
    # The exception's type is logged, never its message, which could echo what was sent.
    assert error_type.__name__ in caplog.text
    assert "0001-2345-6789" not in caplog.text and ACCOUNT not in caplog.text


async def test_a_verifying_line_that_cannot_be_said_does_not_stop_the_request(
    gate, detector, ramp, speaker, caplog
):
    speaker.fail[prompts.VERIFYING] = RuntimeError("the TTS session is gone")
    detector.script = [ALLOW]
    assert (await gate.run(BANK)).status == "done"
    assert ramp.calls == [("add_default_bank_account", "v1", ROUTING, ACCOUNT)]
    assert speaker.lines == [] and "could not say" in caplog.text


async def test_a_verifying_line_that_cannot_be_said_still_denies_normally(
    gate, detector, ramp, speaker, call, state
):
    speaker.fail[prompts.VERIFYING] = RuntimeError("the TTS session is gone")
    detector.script = [DENY]
    result = await gate.run(BANK)
    assert_denied_with_hold(result, ramp, speaker, call, state, first_lines=())


async def test_a_hangup_that_fails_still_returns_denied(
    gate, detector, ramp, speaker, call, state, caplog
):
    call.fail = RuntimeError("the room is already gone")
    detector.script = [DENY]
    assert await gate.run(BANK) == GateResult("denied", prompts.DENY_HELD)
    assert call.ended and state.locked and ramp.calls == [("hold_vendor", "v1")]
    assert speaker.lines == [prompts.VERIFYING, prompts.DENY_HELD]
    assert "could not hang up" in caplog.text
    assert (await gate.run(ACCOUNTS)).status == "denied"  # and the call stays locked


async def test_a_deny_line_that_cannot_be_said_still_hangs_up(
    gate, detector, ramp, speaker, call, state, caplog
):
    speaker.fail[prompts.DENY_HELD] = RuntimeError("the TTS session is gone")
    detector.script = [DENY]
    assert await gate.run(BANK) == GateResult("denied", prompts.DENY_HELD)
    assert call.ended and state.locked and ramp.calls == [("hold_vendor", "v1")]
    assert speaker.lines == [prompts.VERIFYING] and "could not say" in caplog.text


@pytest.mark.parametrize("closing", [False, True], ids=["call_open", "caller_gone"])
async def test_a_hold_that_raises_on_deny_counts_as_not_held(
    gate, detector, ramp, speaker, call, state, closing
):
    # Not a RampError: a closed HTTP client, say. Nothing is known to be held, so the caller (if
    # still there) isn't told it is.
    ramp.fail["hold_vendor"] = RuntimeError("the client is closed")
    detector.script = [after_hangup(state, DENY) if closing else DENY]
    assert await gate.run(BANK) == GateResult("denied", prompts.DENY_NOT_HELD)
    assert speaker.lines == [prompts.VERIFYING] + ([] if closing else [prompts.DENY_NOT_HELD])
    assert call.ended and state.locked


async def test_every_edge_failing_at_once_still_ends_denied(
    gate, detector, ramp, speaker, call, state
):
    speaker.fail = {
        prompts.VERIFYING: RuntimeError("tts"),
        prompts.DENY_NOT_HELD: RuntimeError("tts"),
    }
    ramp.fail["hold_vendor"] = RuntimeError("client closed")
    call.fail = RuntimeError("room gone")
    detector.script = [DENY]
    assert await gate.run(BANK) == GateResult("denied", prompts.DENY_NOT_HELD)
    assert speaker.lines == [] and call.ended and state.locked


@pytest.mark.parametrize(
    "answer",
    [
        None,
        "allow",
        SimpleNamespace(decision=DecisionKind.ALLOW, required_factors=[], reasons=[]),
    ],
    ids=["none", "string", "look_alike_that_says_allow"],
)
async def test_an_answer_that_is_not_a_decision_is_a_deny(
    make_gate, ramp, speaker, call, state, answer
):
    # DetectorClient.authorize always returns a Decision. A stand-in that doesn't, not even one
    # that looks like an allow, may open the gate or break it.
    class NotADetector:
        async def authorize(self, session_id: str, req: AuthorizeRequest):
            return answer

    gate = make_gate(detector=NotADetector())
    assert_denied_with_hold(await gate.run(BANK), ramp, speaker, call, state)


@pytest.mark.parametrize("where", ["verifying", "execute", "deny_hold", "deny_line", "hangup"])
async def test_cancellation_still_propagates_from_every_guarded_edge(
    gate, detector, ramp, speaker, call, where
):
    cancelled = asyncio.CancelledError()
    detector.script = [DENY if where.startswith(("deny", "hangup")) else ALLOW]
    match where:
        case "verifying":
            speaker.fail[prompts.VERIFYING] = cancelled
        case "execute":
            ramp.fail["add_default_bank_account"] = cancelled
        case "deny_hold":
            ramp.fail["hold_vendor"] = cancelled
        case "deny_line":
            speaker.fail[prompts.DENY_HELD] = cancelled
        case "hangup":
            call.fail = cancelled
    with pytest.raises(asyncio.CancelledError):
        await gate.run(BANK)


# --- tier 0 ---


async def test_hold_never_waits_for_the_detector(gate, detector, ramp):
    detector.script = [hang]  # authorize awaits an Event that is never set
    async with asyncio.timeout(0.5):
        result = await gate.run(ActionRequest.hold_vendor("v1", "Acme Supplies", "caller asked"))
    assert result.status == "done" and ramp.calls == [("hold_vendor", "v1")]
    await asyncio.wait_for(detector.called.wait(), 1)  # ...but the detector is still told
    assert detector.requests[0].action == Action.HOLD_VENDOR


async def test_aclose_cancels_a_hold_still_being_logged(gate, detector):
    detector.script = [hang]
    async with asyncio.timeout(0.5):
        await gate.run(HOLD)
    await asyncio.wait_for(detector.called.wait(), 1)
    await asyncio.wait_for(gate.aclose(), timeout=2)
    assert detector.cancelled == 1


async def test_hold_by_an_unidentified_caller_runs_without_the_detector(
    gate, detector, ramp, state
):
    state.session_id = None
    assert (await gate.run(HOLD)).status == "done"
    for _ in range(10):
        await asyncio.sleep(0)
    assert ramp.calls == [("hold_vendor", "v1")] and detector.requests == []


async def test_holds_still_run_after_a_deny(gate, detector, ramp, state):
    # A deny locks the call to tier 0: holding is always allowed.
    detector.script = [DENY, ALLOW]
    await gate.run(BANK)
    assert state.locked
    assert (await gate.run(HOLD_BILL)).status == "done"
    assert ramp.calls[-1] == ("hold_bill", "b1")


@pytest.mark.parametrize("locked", [False, True], ids=["unlocked", "locked"])
@pytest.mark.parametrize(
    "req, made",
    [(HOLD, ("hold_vendor", "v1")), (HOLD_BILL, ("hold_bill", "b1"))],
    ids=["hold_vendor", "hold_bill"],
)
async def test_holds_bypass_closing(gate, detector, ramp, state, req, made, locked):
    # A hold protects, so a call that is ending doesn't stop it, with or without a lock.
    state.closing, state.locked = True, locked
    detector.script = [ALLOW]
    assert (await gate.run(req)).status == "done"
    assert ramp.calls == [made]


async def test_a_bill_hold_that_does_not_take_is_failed(gate, detector, ramp):
    detector.script = [ALLOW]
    ramp.held_as = {"b1": "PAYMENT_SCHEDULED"}
    assert await gate.run(HOLD_BILL) == GateResult("failed", prompts.RAMP_FAILED)


@pytest.mark.parametrize("status, message", [(422, prompts.RAMP_FAILED), (0, prompts.RAMP_UNKNOWN)])
async def test_a_hold_ramp_fails_is_failed(gate, detector, ramp, status, message):
    detector.script = [ALLOW]
    ramp.fail["hold_vendor"] = RampError(status)
    assert await gate.run(HOLD) == GateResult("failed", message)


# --- logs ---


async def test_each_authorize_is_logged(gate, detector, caplog):
    # round counts the authorizes of the whole call (it feeds the idempotency key). attempt counts
    # the authorizes of one request, from 1.
    caplog.set_level(logging.INFO, logger="callback_agent.gate")
    detector.script = [ALLOW, step_up(VOICE), ALLOW]
    await gate.run(ACCOUNTS)
    await gate.run(BILLS)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("authorize ")]
    assert len(lines) == 3
    for line, action, round_no, attempt, decision, factors in [
        (lines[0], "list_vendor_accounts", 0, 1, "allow", "-"),
        (lines[1], "list_bills", 1, 1, "step_up", "voice_challenge"),
        (lines[2], "list_bills", 2, 2, "allow", "-"),
    ]:
        assert f"session_id=s1 action={action} round={round_no} attempt={attempt} " in line
        assert f"decision={decision} factors={factors} reasons=['scripted {decision}']" in line
        assert re.search(r"latency_ms=\d+$", line)


async def test_a_holds_authorize_is_logged_as_its_first_attempt(gate, detector, caplog):
    caplog.set_level(logging.INFO, logger="callback_agent.gate")
    detector.script = [ALLOW]
    await gate.run(HOLD)
    await asyncio.wait_for(detector.called.wait(), 1)
    for _ in range(10):
        await asyncio.sleep(0)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("authorize ")]
    assert len(lines) == 1 and "action=hold_vendor round=0 attempt=1 " in lines[0]


# --- the test doubles themselves ---


async def test_an_exhausted_detector_script_is_recorded_not_just_denied(gate, detector):
    # The gate fails closed on any exception from the detector, so a test that scripted too few
    # decisions would see a deny and could pass for the wrong reason. The fake records it, and its
    # fixture fails the test unless the test clears the record, as this one does.
    assert (await gate.run(ACCOUNTS)).status == "denied"
    assert [type(error) for error in detector.exhausted] == [ScriptExhausted]
    detector.exhausted.clear()
