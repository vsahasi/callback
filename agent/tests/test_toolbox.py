"""The Toolbox: what each LLM tool does.

Most tests use a recording gate, so they show exactly which ActionRequest a tool builds and what
it hands back. A few at the end run the same tools through the real Gate, with a scripted detector
and the FakeRamp, to show that the pieces fit.
"""

import asyncio
import contextlib
import logging
import re
from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace

import pytest
from callback_shared.contract import (
    ACTION_TIER,
    AuthorizeRequest,
    Channel,
    Decision,
    DecisionKind,
    SessionCreate,
    SessionCreated,
    UserSummary,
)
from fake_ramp import FakeRamp, bill

from callback_agent import prompts
from callback_agent.actions import ActionRequest
from callback_agent.detector import DetectorError
from callback_agent.gate import Gate, GateResult
from callback_agent.ramp import Bill, BillVendor, RampClient, RampError, Vendor
from callback_agent.state import CallState
from callback_agent.toolbox import DEFAULT_HOLD_REASON, MAX_REASON_CHARS, Toolbox

ROUTING = "021000021"
ACCOUNT = "000123456789"
ERIC = UserSummary(user_id="eric", display_name="Eric Catalano", role="CFO", phone_last_four="0100")
SABRINA = UserSummary(
    user_id="sabrina", display_name="Sabrina Rao", role="Controller", phone_last_four="0199"
)


class FakeGate:
    """Records every request it is given and answers `result`, or raises `fail`.

    exclusive() stands in for the gate's lock: it records "enter" and "exit" in `events`, and runs
    `once_held` (if a test set one) as soon as it has the lock, which is where a call that was
    denied or hung up on while the caller waited for the gate would show.
    """

    def __init__(self) -> None:
        self.requests: list[ActionRequest] = []
        self.result = GateResult("done", "the gate said this")
        self.fail: BaseException | None = None
        self.events: list[str] = []
        self.once_held: Callable[[], None] | None = None

    async def run(self, req: ActionRequest) -> GateResult:
        self.requests.append(req)
        if self.fail is not None:
            raise self.fail
        return self.result

    @contextlib.asynccontextmanager
    async def exclusive(self) -> AsyncIterator[None]:
        self.events.append("enter")
        try:
            if self.once_held is not None:
                self.once_held()
            yield
        finally:
            self.events.append("exit")


class FakeEnrollment:
    """Stands in for Enrollment: run() records itself in `events` and answers `outcome`, or raises
    `fail`. A test can set `hold` to keep run() from finishing; `started` is set once it is in.
    """

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.outcome = "enrolled"
        self.fail: BaseException | None = None
        self.hold: asyncio.Event | None = None
        self.started = asyncio.Event()
        self.runs = 0

    async def run(self) -> str:
        self.events.append("run")
        self.runs += 1
        self.started.set()
        if self.hold is not None:
            await self.hold.wait()
        if self.fail is not None:
            raise self.fail
        return self.outcome


class FakeDetector:
    """find_users and create_session, plus the authorize the real gate tests need."""

    def __init__(self) -> None:
        self.users = {"Eric Catalano": [ERIC], "Sabrina Rao": [SABRINA]}
        self.searched: list[str] = []
        self.created: list[SessionCreate] = []
        self.find_error: Exception | None = None
        self.create_error: Exception | None = None
        self.hold_searches: asyncio.Event | None = None  # set it to keep find_users from answering
        self.searching = asyncio.Event()  # set once a find_users is waiting there
        self.kinds: list[DecisionKind] = []  # the scripted answers to authorize
        self.authorized: list[AuthorizeRequest] = []  # and what it was asked
        self.hold_authorizing: asyncio.Event | None = None  # set it to keep authorize waiting
        self.authorizing = asyncio.Event()  # set once an authorize is waiting there

    async def find_users(self, name: str) -> list[UserSummary]:
        self.searched.append(name)
        self.searching.set()
        if self.hold_searches is not None:
            await self.hold_searches.wait()
        if self.find_error is not None:
            raise self.find_error
        return list(self.users.get(name, []))

    async def create_session(self, req: SessionCreate) -> SessionCreated:
        if self.create_error is not None:
            raise self.create_error
        self.created.append(req)
        return SessionCreated(session_id=f"s{len(self.created)}")

    async def authorize(self, session_id, req):
        self.authorized.append(req)
        if ACTION_TIER[req.action] == 0:  # the gate tells the detector about a hold, and ignores it
            kind = DecisionKind.ALLOW
        else:
            self.authorizing.set()
            if self.hold_authorizing is not None:
                await self.hold_authorizing.wait()
            kind = self.kinds.pop(0)
        return Decision(session_id=session_id, action=req.action, decision=kind)


class FakeForwarder:
    """Records each attach, and the session id the call's state held at that moment."""

    def __init__(self, state: CallState) -> None:
        self._state = state
        self.attached: list[str] = []
        self.state_then: list[str | None] = []

    def attach(self, session_id: str) -> None:
        self.attached.append(session_id)
        self.state_then.append(self._state.session_id)


class ToolboxRamp(FakeRamp):
    """FakeRamp plus find_vendor, which is the real one, over this fake's vendor list."""

    def __init__(self) -> None:
        super().__init__()
        self.vendors = [
            Vendor(id="v1", name="Acme Supplies Inc", display_name=None, is_active=True),
            Vendor(id="v2", name="Globex Corporation", display_name="Globex", is_active=True),
            Vendor(id="v3", name="Initech Legacy LLC", display_name="Initech", is_active=False),
        ]

    async def find_vendor(self, name: str) -> Vendor | None:
        return await RampClient.find_vendor(self, name)  # type: ignore[arg-type]


@pytest.fixture
def state() -> CallState:
    return CallState(room_name="r1", caller_identity="sip_+1555", channel=Channel.PHONE)


@pytest.fixture
def gate() -> FakeGate:
    return FakeGate()


@pytest.fixture
def detector() -> FakeDetector:
    return FakeDetector()


@pytest.fixture
def ramp() -> ToolboxRamp:
    return ToolboxRamp()


@pytest.fixture
def forwarder(state) -> FakeForwarder:
    return FakeForwarder(state)


@pytest.fixture
def enrollment(gate) -> FakeEnrollment:
    return FakeEnrollment(gate.events)  # one list, so a test sees the order the gate and it ran in


def make_toolbox(state, gate, ramp, detector, forwarder, enrollment=None) -> Toolbox:
    return Toolbox(
        state=state,
        gate=gate,
        ramp=ramp,
        detector=detector,
        forwarder=forwarder,
        enrollment=enrollment,
    )


@pytest.fixture
def toolbox(state, gate, ramp, detector, forwarder, enrollment) -> Toolbox:
    return make_toolbox(state, gate, ramp, detector, forwarder, enrollment)


def identify(state: CallState) -> None:
    """The state of a call after identify_caller worked."""
    state.claimed_user, state.session_id = ERIC, "s1"


# --- identify_caller ---


async def test_identify_creates_session_and_flushes_buffer(toolbox, state, detector, forwarder):
    reply = await toolbox.identify_caller("Eric Catalano")
    assert detector.searched == ["Eric Catalano"]
    assert detector.created == [
        SessionCreate(
            room="r1", caller_identity="sip_+1555", claimed_user_id="eric", channel=Channel.PHONE
        )
    ]
    assert forwarder.attached == ["s1"] and state.session_id == "s1"
    assert state.claimed_user == ERIC
    assert reply == prompts.IDENTIFIED.format(name="Eric Catalano")


async def test_the_session_and_the_forwarder_are_set_up_together(toolbox, forwarder):
    await toolbox.identify_caller("Eric Catalano")
    # attach() already saw the session id on the state: nothing between the two steps can run
    assert forwarder.state_then == ["s1"]


async def test_the_reply_does_not_pass_on_what_the_directory_holds(toolbox):
    reply = await toolbox.identify_caller("Eric Catalano")
    assert "CFO" not in reply and "0100" not in reply and "eric" not in reply


async def test_a_call_without_a_forwarder_still_identifies(state, gate, ramp, detector):
    toolbox = make_toolbox(state, gate, ramp, detector, None)
    await toolbox.identify_caller("Eric Catalano")
    assert state.session_id == "s1"


async def test_identity_is_locked_once_set(toolbox, state, detector, forwarder):
    await toolbox.identify_caller("Eric Catalano")
    reply = await toolbox.identify_caller("Sabrina Rao")
    assert reply == prompts.IDENTITY_LOCKED
    assert detector.searched == ["Eric Catalano"]  # the second name wasn't even looked up
    assert len(detector.created) == 1 and forwarder.attached == ["s1"]
    assert state.claimed_user == ERIC and state.session_id == "s1"


async def test_the_same_name_again_is_not_a_second_identification(toolbox, detector):
    await toolbox.identify_caller("Eric Catalano")
    assert await toolbox.identify_caller("Eric Catalano") == prompts.IDENTITY_LOCKED
    assert len(detector.created) == 1


async def test_two_names_at_once_make_one_session(toolbox, state, detector, forwarder):
    detector.hold_searches = asyncio.Event()
    eric = asyncio.create_task(toolbox.identify_caller("Eric Catalano"))
    sabrina = asyncio.create_task(toolbox.identify_caller("Sabrina Rao"))
    await asyncio.wait_for(detector.searching.wait(), timeout=5)
    for _ in range(5):
        await asyncio.sleep(0)  # let the second one run as far as it can
    detector.hold_searches.set()
    replies = await asyncio.wait_for(asyncio.gather(eric, sabrina), timeout=5)
    assert len(detector.created) == 1 and forwarder.attached == ["s1"]
    assert state.claimed_user == ERIC
    assert replies == [prompts.IDENTIFIED.format(name="Eric Catalano"), prompts.IDENTITY_LOCKED]


async def test_no_match_creates_no_session(toolbox, state, detector, forwarder):
    reply = await toolbox.identify_caller("Nobody Here")
    assert reply == prompts.NOT_IDENTIFIED
    assert detector.created == [] and forwarder.attached == []
    assert state.session_id is None and state.claimed_user is None and not state.detector_down


async def test_several_matches_create_no_session_and_answer_like_no_match(
    toolbox, state, detector, forwarder
):
    detector.users["Eric"] = [ERIC, SABRINA]
    reply = await toolbox.identify_caller("Eric")
    assert reply == prompts.NOT_IDENTIFIED == await toolbox.identify_caller("Nobody Here")
    assert detector.created == [] and forwarder.attached == []
    assert state.session_id is None and state.claimed_user is None


async def test_after_a_miss_the_caller_can_try_again(toolbox, state):
    assert await toolbox.identify_caller("Eric") == prompts.NOT_IDENTIFIED
    await toolbox.identify_caller("Eric Catalano")
    assert state.claimed_user == ERIC


@pytest.mark.parametrize("name", ["", "   ", "\n"])
async def test_a_blank_name_asks_again_without_asking_the_detector(toolbox, detector, name):
    assert await toolbox.identify_caller(name) == prompts.NOT_IDENTIFIED
    assert detector.searched == []


@pytest.mark.parametrize("where", ["find_error", "create_error"])
async def test_detector_down_marks_state(toolbox, state, detector, forwarder, where):
    setattr(detector, where, DetectorError("GET /v1/users: HTTP 503"))
    reply = await toolbox.identify_caller("Eric Catalano")
    assert reply == prompts.IDENTIFY_UNAVAILABLE
    assert state.detector_down
    assert state.session_id is None and state.claimed_user is None  # nothing half done
    assert detector.created == [] and forwarder.attached == []


async def test_a_detector_that_comes_back_lets_the_caller_identify(toolbox, state, detector):
    detector.find_error = DetectorError("GET /v1/users: HTTP 503")
    await toolbox.identify_caller("Eric Catalano")
    detector.find_error = None
    await toolbox.identify_caller("Eric Catalano")
    assert state.session_id == "s1" and state.claimed_user == ERIC and not state.detector_down


@pytest.mark.parametrize("flag", ["locked", "closing"])
async def test_a_call_that_is_ending_identifies_no_one(toolbox, state, detector, flag):
    setattr(state, flag, True)
    assert await toolbox.identify_caller("Eric Catalano") == prompts.DENY_NOT_HELD
    assert detector.searched == []


# --- the bank change ---


async def test_bank_change_goes_through_the_gate_with_resolved_vendor(toolbox, gate):
    reply = await toolbox.change_vendor_bank_account("acme supplies", ROUTING, ACCOUNT)
    assert gate.requests == [
        ActionRequest.update_vendor_bank_account("v1", "Acme Supplies Inc", ROUTING, ACCOUNT)
    ]
    assert reply == "the gate said this"


async def test_the_display_name_is_what_the_caller_hears_when_there_is_one(toolbox, gate):
    await toolbox.change_vendor_bank_account("globex corp", ROUTING, ACCOUNT)
    assert gate.requests == [
        ActionRequest.update_vendor_bank_account("v2", "Globex", ROUTING, ACCOUNT)
    ]


@pytest.mark.parametrize(
    "routing, account",
    [
        ("021000022", ACCOUNT),  # fails the ABA checksum
        ("12345", ACCOUNT),
        ("abcdefghi", ACCOUNT),
        ("", ACCOUNT),
        (ROUTING, "12"),
        (ROUTING, "abc"),
        (ROUTING, "1" * 18),
        (ROUTING, ""),
    ],
)
async def test_bad_routing_number_asks_again_without_the_gate(toolbox, gate, routing, account):
    reply = await toolbox.change_vendor_bank_account("Acme", routing, account)
    assert reply == prompts.BANK_NUMBERS_UNCLEAR
    assert gate.requests == []
    assert not (routing and routing in reply) and not (account and account in reply)


async def test_the_numbers_a_caller_gave_are_never_logged(toolbox, caplog):
    with caplog.at_level(logging.DEBUG):
        await toolbox.change_vendor_bank_account("Acme", "021000022", "987654321012")
        await toolbox.change_vendor_bank_account("Acme", ROUTING, ACCOUNT)
    assert "021000022" not in caplog.text and "987654321012" not in caplog.text
    assert ROUTING not in caplog.text and ACCOUNT not in caplog.text


async def test_spaces_and_dashes_in_the_numbers_are_ignored(toolbox, gate):
    await toolbox.change_vendor_bank_account("Acme", "021 000-021", " 0001 2345-6789 ")
    assert gate.requests == [
        ActionRequest.update_vendor_bank_account("v1", "Acme Supplies Inc", ROUTING, ACCOUNT)
    ]


# --- browser calls (V3) ---

TIER_2_TOOLS = {
    "change_vendor_bank_account": lambda t: t.change_vendor_bank_account("Acme", ROUTING, ACCOUNT),
    "release_vendor_hold": lambda t: t.release_vendor_hold("Acme"),
    "release_bill_hold": lambda t: t.release_bill_hold("b1"),
}


@pytest.mark.parametrize("tool", TIER_2_TOOLS)
async def test_tier2_on_browser_says_phone_only(tool, toolbox, state, gate, ramp):
    state.channel = Channel.BROWSER
    assert await TIER_2_TOOLS[tool](toolbox) == prompts.TIER2_PHONE_ONLY
    assert gate.requests == [] and ramp.calls == []  # nothing was asked of Ramp, let alone the gate


def test_the_phone_only_line_is_what_the_brief_says():
    assert prompts.TIER2_PHONE_ONLY == (
        "Payment changes need a phone call. Tell the caller to call this number from their phone "
        "on file."
    )


@pytest.mark.parametrize("tool", TIER_2_TOOLS)
async def test_tier2_on_a_phone_goes_ahead(tool, toolbox, state, gate):
    state.bills_seen["b1"] = "v1"
    assert await TIER_2_TOOLS[tool](toolbox) == "the gate said this"
    assert len(gate.requests) == 1


TIER_0_AND_1_TOOLS = {
    "list_vendors": lambda t: t.list_vendors(),
    "get_vendor_bank_account": lambda t: t.get_vendor_bank_account("Acme"),
    "list_open_bills": lambda t: t.list_open_bills("Acme"),
    "hold_vendor": lambda t: t.hold_vendor("Acme", "caller asked"),
    "hold_bill": lambda t: t.hold_bill("Acme", "INV-1001", "caller asked"),
}


@pytest.mark.parametrize("tool", TIER_0_AND_1_TOOLS)
async def test_holds_and_reads_work_on_a_browser_call(tool, toolbox, state, gate):
    state.channel = Channel.BROWSER
    assert await TIER_0_AND_1_TOOLS[tool](toolbox) == "the gate said this"
    assert len(gate.requests) == 1


# --- what each tool asks of the gate ---


async def test_each_tool_builds_the_request_it_is_named_for(toolbox, state, gate, ramp):
    ramp.bills = [*ramp.bills, globex_bill("b7", "INV-2001")]
    state.bills_seen["b7"] = "v2"  # a Globex bill the caller was told about earlier
    await toolbox.list_vendors()
    await toolbox.get_vendor_bank_account("globex")
    await toolbox.list_open_bills("globex")
    await toolbox.hold_vendor("globex", "suspicious")
    await toolbox.hold_bill("globex", "INV-2001", "wrong amount")
    await toolbox.release_vendor_hold("globex")
    await toolbox.release_bill_hold("b7")
    assert gate.requests == [
        ActionRequest.list_vendors(),
        ActionRequest.list_vendor_accounts("v2", "Globex"),
        ActionRequest.list_bills("v2", "Globex"),
        ActionRequest.hold_vendor("v2", "Globex", "suspicious"),
        ActionRequest.hold_bill("b7", "v2", "wrong amount"),
        ActionRequest.release_vendor_hold("v2", "Globex"),
        ActionRequest.release_bill_hold("b7", "v2"),
    ]


@pytest.mark.parametrize("status", ["done", "pending_approval", "denied", "failed"])
async def test_a_tool_hands_back_whatever_the_gate_says(toolbox, gate, status):
    gate.result = GateResult(status, f"message for {status}")
    assert await toolbox.hold_vendor("Acme", "caller asked") == f"message for {status}"
    assert await toolbox.change_vendor_bank_account("Acme", ROUTING, ACCOUNT) == (
        f"message for {status}"
    )


async def test_a_hold_with_no_reason_still_gets_one(toolbox, state, gate):
    await toolbox.hold_vendor("Acme", "  ")
    await toolbox.hold_bill("Acme", "INV-1001", "")
    assert [r.params["reason"] for r in gate.requests] == [DEFAULT_HOLD_REASON] * 2


# --- the reason a hold is given (R58) ---


@pytest.mark.parametrize(
    ("said", "sent"),
    [
        ("someone tried to switch it to 000123456789", "someone tried to switch it to [number]"),
        ("routing 021000021 and account 000123456789", "routing [number] and account [number]"),
        ("account 0001 2345 6789", "account [number]"),
        ("account 000-123-456-789", "account [number]"),
        ("account 0 0 0 1 2 3", "account [number]"),  # digits read out one at a time
        ("account 0001\n2345\n6789", "account [number]"),
        ("12345", "[number]"),
        ("INV-100234 looks wrong", "INV-[number] looks wrong"),
        # groups split by a dot, a comma and a space, or a spaced dash (final review, issue 4)
        ("routing 021.000.021", "routing [number]"),
        ("account 0001.2345.6789", "account [number]"),
        ("account 0001, 2345, 6789", "account [number]"),
        ("account 0001 - 2345 - 6789", "account [number]"),
        # a punctuated amount has five digits too, and is masked with them: accepted, since the
        # reason is only context for the hold
        ("paid $12,000 twice", "paid $[number] twice"),
        ("paid $1,204.50 twice", "paid $[number] twice"),
        # what is not a long number stays
        ("caller asked", "caller asked"),
        ("the last four were 6789", "the last four were 6789"),
        ("an account ending 4417", "an account ending 4417"),
        ("INV-1001 looks wrong", "INV-1001 looks wrong"),
        ("12 34", "12 34"),
        ("INV-1001, INV-1002 look wrong", "INV-1001, INV-1002 look wrong"),
    ],
)
async def test_a_long_number_in_a_reason_never_reaches_ramp_or_the_detector(
    toolbox, state, gate, said, sent
):
    state.bills_seen["b1"] = "v1"
    await toolbox.hold_vendor("Acme", said)
    await toolbox.hold_bill("Acme", "INV-1001", said)
    assert [r.params["reason"] for r in gate.requests] == [sent, sent]
    assert [r.detector_params()["reason"] for r in gate.requests] == [sent, sent]


async def test_a_reason_is_masked_before_it_is_cut_to_length(toolbox, gate):
    # cut first, a number that starts just inside the limit would leave its first four digits
    await toolbox.hold_vendor("Acme", "x" * (MAX_REASON_CHARS - 5) + " 000123456789")
    reason = gate.requests[0].params["reason"]
    assert len(reason) <= MAX_REASON_CHARS and not re.search(r"\d", reason)


async def test_a_reason_that_is_only_a_number_is_still_a_reason(toolbox, gate):
    await toolbox.hold_vendor("Acme", "000123456789")
    assert gate.requests[0].params["reason"] == "[number]"


async def test_with_the_real_gate_no_full_number_reaches_ramp_or_the_detector(real, detector):
    toolbox, ramp, _, _ = real
    await toolbox.identify_caller("Eric Catalano")  # a session, so the detector is told of the hold
    reason = f"someone tried to switch it to {ACCOUNT} with routing {ROUTING}"
    await toolbox.hold_vendor("Acme", reason)
    await toolbox.hold_bill("Acme", "INV-1001", reason)
    await asyncio.sleep(0.05)  # the gate tells the detector of a hold in the background
    assert ramp.reasons == ["someone tried to switch it to [number] with routing [number]"] * 2
    told = " ".join(request.model_dump_json() for request in detector.authorized)
    assert len(detector.authorized) == 2 and ACCOUNT not in told and ROUTING not in told


async def test_a_hold_needs_no_identification(toolbox, state, gate):
    assert state.session_id is None
    await toolbox.hold_vendor("Acme", "caller asked")
    assert len(gate.requests) == 1  # the toolbox doesn't stop it; holds never need verification


# --- vendors that can't be acted on ---

VENDOR_TOOLS = {
    "get_vendor_bank_account": lambda t, name: t.get_vendor_bank_account(name),
    "list_open_bills": lambda t, name: t.list_open_bills(name),
    "hold_vendor": lambda t, name: t.hold_vendor(name, "caller asked"),
    "hold_bill": lambda t, name: t.hold_bill(name, "INV-1001", "caller asked"),
    "change_vendor_bank_account": lambda t, name: t.change_vendor_bank_account(
        name, ROUTING, ACCOUNT
    ),
    "release_vendor_hold": lambda t, name: t.release_vendor_hold(name),
}


@pytest.mark.parametrize("tool", VENDOR_TOOLS)
async def test_an_unknown_vendor_gets_the_same_reply_whether_or_not_the_caller_is_identified(
    tool, gate, ramp, detector
):
    replies = []
    for identified in (False, True):
        state = CallState(room_name="r1", caller_identity="c", channel=Channel.PHONE)
        if identified:
            identify(state)
        toolbox = make_toolbox(state, gate, ramp, detector, None)
        replies.append(await VENDOR_TOOLS[tool](toolbox, "Umbrella Corporation"))
    assert replies == [prompts.NO_ACTIVE_VENDOR] * 2
    assert gate.requests == []


@pytest.mark.parametrize("tool", VENDOR_TOOLS)
async def test_an_inactive_vendor_is_not_acted_on_and_sounds_like_an_unknown_one(
    tool, toolbox, gate, ramp
):
    reply = await VENDOR_TOOLS[tool](toolbox, "initech")
    assert reply == prompts.NO_ACTIVE_VENDOR
    assert gate.requests == []
    assert ("list_vendors",) in ramp.calls  # it was looked up, and then not used


@pytest.mark.parametrize("tool", VENDOR_TOOLS)
@pytest.mark.parametrize("error", [RampError(0), RampError(503), RuntimeError("bug")])
async def test_a_vendor_lookup_that_fails_does_nothing(tool, toolbox, gate, ramp, error):
    ramp.fail["list_vendors"] = error
    assert await VENDOR_TOOLS[tool](toolbox, "Acme") == prompts.VENDOR_LOOKUP_FAILED
    assert gate.requests == []


async def test_a_blank_vendor_name_finds_no_vendor(toolbox, gate):
    assert await toolbox.hold_vendor("   ", "caller asked") == prompts.NO_ACTIVE_VENDOR
    assert gate.requests == []


# --- open bills ---


def globex_bill(bill_id: str, invoice: str | None) -> Bill:
    """An open Globex bill. FakeRamp's own are Acme's."""
    return bill(bill_id, invoice=invoice).model_copy(
        update={"vendor": BillVendor(id="v2", name="Globex Corporation")}
    )


def a_caller(identified: bool, gate, ramp, detector) -> Toolbox:
    state = CallState(room_name="r1", caller_identity="c", channel=Channel.PHONE)
    if identified:
        identify(state)
    return make_toolbox(state, gate, ramp, detector, None)


async def test_a_bill_is_held_by_its_vendor_and_invoice_number(toolbox, state, gate, ramp):
    reply = await toolbox.hold_bill("acme", "INV-1002", "caller asked")
    assert gate.requests == [ActionRequest.hold_bill("b2", "v1", "caller asked")]
    assert reply == "the gate said this"
    assert ("list_bills", "v1") in ramp.calls  # the vendor's open bills, read before the gate
    assert state.bills_seen == {}  # a hold by invoice is no listing: only a gated list adds bills


@pytest.mark.parametrize("spoken", ["inv-1002", "INV-1002", "  Inv-1002 \n"])
async def test_the_invoice_number_ignores_case_and_the_spaces_around_it(toolbox, gate, spoken):
    await toolbox.hold_bill("Acme", spoken, "caller asked")
    assert gate.requests == [ActionRequest.hold_bill("b2", "v1", "caller asked")]


@pytest.mark.parametrize(
    "spoken", ["INV-100", "INV-10022", "1002", "INV 1002", "INV-1002 please", "", "   "]
)
async def test_an_invoice_number_that_is_not_exactly_a_bills_holds_nothing(toolbox, gate, spoken):
    assert await toolbox.hold_bill("Acme", spoken, "caller asked") == prompts.NO_SUCH_BILL
    assert gate.requests == []


async def test_an_unknown_invoice_gets_the_same_reply_whether_or_not_the_caller_is_identified(
    gate, ramp, detector
):
    replies = [
        await a_caller(identified, gate, ramp, detector).hold_bill("Acme", "INV-9999", "x")
        for identified in (False, True)
    ]
    assert replies == [prompts.NO_SUCH_BILL] * 2
    assert gate.requests == []


async def test_a_bill_of_another_vendor_is_not_held_under_this_vendors_name(toolbox, gate, ramp):
    # FakeRamp answers list_bills with every bill whatever the vendor, so the toolbox has to check
    ramp.bills = [*ramp.bills, globex_bill("b7", "INV-2001")]
    assert await toolbox.hold_bill("Acme", "INV-2001", "caller asked") == prompts.NO_SUCH_BILL
    assert gate.requests == []
    await toolbox.hold_bill("Globex", "INV-2001", "caller asked")
    assert gate.requests == [ActionRequest.hold_bill("b7", "v2", "caller asked")]


async def test_a_bill_with_no_invoice_number_cannot_be_named(toolbox, gate, ramp):
    ramp.bills = [bill("b5", invoice=None)]
    for spoken in ("None", "null", "", "no invoice"):
        assert await toolbox.hold_bill("Acme", spoken, "caller asked") == prompts.NO_SUCH_BILL
    assert gate.requests == []


async def test_every_open_bill_with_the_invoice_number_is_held(toolbox, gate, ramp):
    ramp.bills = [bill("b1"), bill("b8", invoice="inv-1001"), bill("b9", invoice="INV-1002")]
    gate.result = GateResult("done", "held")
    reply = await toolbox.hold_bill("Acme", "INV-1001", "caller asked")
    assert gate.requests == [
        ActionRequest.hold_bill("b1", "v1", "caller asked"),
        ActionRequest.hold_bill("b8", "v1", "caller asked"),
    ]
    assert reply == "held held"  # a resubmitted copy is not left to be paid


@pytest.mark.parametrize("error", [RampError(0), RampError(503), RuntimeError("bug")])
async def test_a_bill_lookup_that_fails_holds_nothing(toolbox, gate, ramp, error):
    ramp.fail["list_bills"] = error
    assert await toolbox.hold_bill("Acme", "INV-1001", "caller asked") == prompts.BILL_LOOKUP_FAILED
    assert gate.requests == []


async def test_a_bill_hold_needs_no_identification(toolbox, state, gate):
    assert state.session_id is None and state.claimed_user is None
    await toolbox.hold_bill("Acme", "INV-1001", "caller asked")
    assert len(gate.requests) == 1


async def test_releasing_a_bill_needs_a_bill_the_caller_was_told_about(toolbox, state, gate):
    assert await toolbox.release_bill_hold("b1") == prompts.BILL_NOT_LISTED
    assert gate.requests == []

    await toolbox.list_open_bills("Acme")
    assert state.bills_seen == {"b1": "v1", "b2": "v1"}
    gate.requests.clear()

    await toolbox.release_bill_hold("b2")
    assert gate.requests == [ActionRequest.release_bill_hold("b2", "v1")]
    assert await toolbox.release_bill_hold("b9") == prompts.BILL_NOT_LISTED
    assert len(gate.requests) == 1


async def test_a_bill_id_is_matched_exactly(toolbox, state, gate):
    state.bills_seen["b1"] = "v1"
    assert await toolbox.release_bill_hold("B1") == prompts.BILL_NOT_LISTED
    assert await toolbox.release_bill_hold("b") == prompts.BILL_NOT_LISTED
    assert gate.requests == []
    await toolbox.release_bill_hold(" b1 ")  # the LLM may pad it
    assert len(gate.requests) == 1


@pytest.mark.parametrize("line", [prompts.NOT_IDENTIFIED, prompts.IDENTIFY_UNAVAILABLE])
def test_the_replies_that_say_only_holds_are_possible_name_both_kinds_of_hold(line):
    assert "put a vendor on hold" in line
    assert "bill on hold by its vendor and invoice number" in line


def test_the_reply_for_an_unlisted_bill_is_about_releasing_one():
    assert "release" in prompts.BILL_NOT_LISTED and "hold_bill" not in prompts.BILL_NOT_LISTED


async def test_the_bills_seen_are_those_of_the_vendors_listed_so_far(toolbox, state, ramp):
    await toolbox.list_open_bills("Acme")
    ramp.bills = [bill("b3").model_copy(update={"vendor": BillVendor(id="v2", name="Globex")})]
    await toolbox.list_open_bills("Globex")
    assert state.bills_seen == {"b1": "v1", "b2": "v1", "b3": "v2"}


@pytest.mark.parametrize("status", ["denied", "failed"])
async def test_a_list_the_gate_did_not_give_adds_no_bills(toolbox, state, gate, ramp, status):
    gate.result = GateResult(status, "no list for you")
    assert await toolbox.list_open_bills("Acme") == "no list for you"
    assert state.bills_seen == {}
    assert ("list_bills", "v1") not in ramp.calls  # and Ramp wasn't asked for bills it won't use


async def test_a_second_read_of_the_bills_that_fails_leaves_the_reply_alone(
    toolbox, state, gate, ramp, caplog
):
    ramp.fail["list_bills"] = RampError(500, message="account 000123456789 is odd")
    with caplog.at_level(logging.DEBUG):
        assert await toolbox.list_open_bills("Acme") == "the gate said this"
    assert state.bills_seen == {}  # so a release of one of them is asked to list the bills again
    assert "000123456789" not in caplog.text


# --- no tool raises ---

EVERY_TOOL = {**TIER_2_TOOLS, **TIER_0_AND_1_TOOLS}


@pytest.mark.parametrize("name", EVERY_TOOL)
async def test_a_bug_inside_a_tool_becomes_a_reply_not_an_exception(
    name, toolbox, state, gate, caplog
):
    state.bills_seen["b1"] = "v1"
    gate.fail = RuntimeError("secret detail 000123456789")
    with caplog.at_level(logging.DEBUG):
        assert await EVERY_TOOL[name](toolbox) == prompts.TOOL_FAILED
    assert "000123456789" not in caplog.text  # the failure is logged by type and place only


async def test_identify_caller_cannot_raise_either(toolbox, detector):
    detector.find_error = RuntimeError("bug")
    assert await toolbox.identify_caller("Eric Catalano") == prompts.TOOL_FAILED


async def test_cancellation_gets_through_a_tool(toolbox, gate):
    gate.fail = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await toolbox.hold_vendor("Acme", "caller asked")


# --- start_voice_enrollment ---


@pytest.mark.parametrize(
    "outcome, reply",
    [
        ("enrolled", prompts.ENROLL_TOOL_DONE),
        ("declined", prompts.ENROLL_TOOL_DECLINED),
        ("failed", prompts.ENROLL_TOOL_FAILED),
    ],
)
async def test_each_enrollment_result_becomes_the_line_the_llm_voices(
    toolbox, enrollment, outcome, reply
):
    enrollment.outcome = outcome
    assert await toolbox.start_voice_enrollment() == reply
    assert enrollment.runs == 1


@pytest.mark.parametrize(
    "detector_down, reply",
    [(False, prompts.IDENTIFY_FIRST), (True, prompts.IDENTIFY_UNAVAILABLE)],
    ids=["detector up", "detector down"],
)
async def test_an_enrollment_that_found_no_one_identified_sends_the_llm_to_identify(
    toolbox, state, enrollment, detector_down, reply
):
    enrollment.outcome = "not_identified"
    state.detector_down = detector_down
    assert await toolbox.start_voice_enrollment() == reply


def test_the_enrollment_lines_are_what_the_brief_says():
    assert prompts.ENROLL_NOT_NOW == (
        "Voice enrollment isn't available on this call. "
        "Tell the caller they can enroll on another call."
    )
    assert prompts.ENROLL_TOOL_DONE == (
        "Enrollment finished and the caller already heard that. Ask if there's anything else."
    )
    assert prompts.ENROLL_TOOL_DECLINED == (
        "The caller declined enrollment and already heard that. Don't offer it again on this call."
    )
    assert prompts.ENROLL_TOOL_FAILED == (
        "Enrollment didn't finish and the caller already heard why. Don't retry on this call."
    )


@pytest.mark.parametrize("flag", ["locked", "closing"])
async def test_a_call_that_is_locked_or_ending_is_not_enrolled(
    toolbox, state, gate, enrollment, flag
):
    setattr(state, flag, True)
    assert await toolbox.start_voice_enrollment() == prompts.ENROLL_NOT_NOW
    # nothing ran, and the gate's lock was not so much as taken
    assert enrollment.runs == 0 and gate.events == []


@pytest.mark.parametrize("flag", ["locked", "closing"])
async def test_a_call_that_became_locked_or_ending_while_enrollment_waited_is_not_enrolled(
    toolbox, state, gate, enrollment, flag
):
    # A gated request that was running when the caller asked can deny (which locks the call and
    # hangs up) before the lock is free. A locked call must not be sent a code, or asked anything.
    gate.once_held = lambda: setattr(state, flag, True)
    assert await toolbox.start_voice_enrollment() == prompts.ENROLL_NOT_NOW
    assert enrollment.runs == 0 and gate.events == ["enter", "exit"]


async def test_the_enrollment_runs_inside_the_gates_exclusive_lock(toolbox, gate, enrollment):
    await toolbox.start_voice_enrollment()
    assert gate.events == ["enter", "run", "exit"]


# One enrollment per call (R81): each run texts the number on file a code, and Twilio Verify caps
# the sends. So a call that has had its enrollment is not offered another, whatever the first came
# to, and the second start is turned away before it takes the gate's lock or touches the detector.


@pytest.mark.parametrize("outcome", ["enrolled", "declined", "failed"])
async def test_a_second_enrollment_on_a_call_is_turned_away_whatever_the_first_came_to(
    toolbox, gate, enrollment, outcome
):
    enrollment.outcome = outcome
    first = await toolbox.start_voice_enrollment()
    events_after_the_first = list(gate.events)
    assert await toolbox.start_voice_enrollment() == prompts.ENROLL_TOOL_ALREADY_RAN
    assert enrollment.runs == 1  # so no code was sent, and the detector was not asked anything
    assert gate.events == events_after_the_first  # nor was the gate's lock taken again
    assert first != prompts.ENROLL_TOOL_ALREADY_RAN  # the first one worked as it always did


@pytest.mark.parametrize(
    "error",
    [DetectorError("POST /v1/enrollments: HTTP 503"), RuntimeError("a bug")],
    ids=["detector_error", "bug"],
)
async def test_an_enrollment_that_blew_up_still_used_up_the_calls_one_enrollment(
    toolbox, enrollment, error
):
    enrollment.fail = error
    assert await toolbox.start_voice_enrollment() == prompts.TOOL_FAILED
    assert await toolbox.start_voice_enrollment() == prompts.ENROLL_TOOL_ALREADY_RAN
    assert enrollment.runs == 1


async def test_an_enrollment_cancelled_partway_used_up_the_calls_one_enrollment(
    toolbox, enrollment
):
    enrollment.fail = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await toolbox.start_voice_enrollment()
    assert await toolbox.start_voice_enrollment() == prompts.ENROLL_TOOL_ALREADY_RAN
    assert enrollment.runs == 1


async def test_a_second_start_that_waited_for_the_lock_is_turned_away_too(
    toolbox, gate, enrollment
):
    # The first start holds the lock while it runs; the second is called meanwhile. When it gets
    # the lock the first has finished, so it must look again.
    enrollment.hold = asyncio.Event()
    first = asyncio.create_task(toolbox.start_voice_enrollment())
    await asyncio.wait_for(enrollment.started.wait(), timeout=5)
    second = asyncio.create_task(toolbox.start_voice_enrollment())
    for _ in range(5):
        await asyncio.sleep(0)
    enrollment.hold.set()
    replies = await asyncio.wait_for(asyncio.gather(first, second), timeout=5)
    assert replies == [prompts.ENROLL_TOOL_DONE, prompts.ENROLL_TOOL_ALREADY_RAN]
    assert enrollment.runs == 1


async def test_a_start_before_the_caller_is_identified_does_not_use_up_the_enrollment(
    toolbox, state, enrollment
):
    # Nothing was sent or asked, and the LLM is sent to identify the caller. It tries again once
    # that is done, and the caller must still have their enrollment.
    enrollment.outcome = "not_identified"
    assert await toolbox.start_voice_enrollment() == prompts.IDENTIFY_FIRST
    enrollment.outcome = "enrolled"
    assert await toolbox.start_voice_enrollment() == prompts.ENROLL_TOOL_DONE
    assert enrollment.runs == 2
    assert await toolbox.start_voice_enrollment() == prompts.ENROLL_TOOL_ALREADY_RAN
    assert enrollment.runs == 2


@pytest.mark.parametrize("flag", ["locked", "closing"])
async def test_a_call_that_is_over_is_told_so_before_it_is_told_it_already_ran(
    toolbox, state, enrollment, flag
):
    await toolbox.start_voice_enrollment()
    setattr(state, flag, True)
    assert await toolbox.start_voice_enrollment() == prompts.ENROLL_NOT_NOW


def test_the_line_for_a_second_enrollment_says_it_already_ran_on_this_call():
    line = prompts.ENROLL_TOOL_ALREADY_RAN
    assert "already ran on this call" in line and "another call" in line


async def test_a_browser_caller_can_enroll_too(toolbox, state, enrollment):
    # Unlike a money move, enrollment needs no callback: the voiceprint is per channel (the spec).
    state.channel = Channel.BROWSER
    assert await toolbox.start_voice_enrollment() == prompts.ENROLL_TOOL_DONE
    assert enrollment.runs == 1


async def test_a_toolbox_with_no_enrollment_says_it_is_not_available(
    state, gate, ramp, detector, forwarder
):
    toolbox = make_toolbox(state, gate, ramp, detector, forwarder)
    assert await toolbox.start_voice_enrollment() == prompts.ENROLL_NOT_NOW
    assert gate.events == []


async def test_a_detector_error_inside_the_enrollment_gives_the_tool_failed_line(
    toolbox, gate, enrollment, caplog
):
    enrollment.fail = DetectorError("POST /v1/enrollments: HTTP 503")
    with caplog.at_level(logging.DEBUG):
        assert await toolbox.start_voice_enrollment() == prompts.TOOL_FAILED
    assert gate.events == ["enter", "run", "exit"]  # and the lock was let go
    assert "DetectorError" in caplog.text


async def test_a_bug_inside_the_enrollment_becomes_a_reply_and_is_logged_by_type_only(
    toolbox, gate, enrollment, caplog
):
    enrollment.fail = RuntimeError("secret detail 000123456789 and the code 482913")
    with caplog.at_level(logging.DEBUG):
        assert await toolbox.start_voice_enrollment() == prompts.TOOL_FAILED
    assert gate.events == ["enter", "run", "exit"]
    assert "000123456789" not in caplog.text and "482913" not in caplog.text


async def test_cancellation_gets_through_the_enrollment_and_frees_the_lock(
    toolbox, gate, enrollment
):
    enrollment.fail = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await toolbox.start_voice_enrollment()
    assert gate.events == ["enter", "run", "exit"]


# --- the same tools with the real gate ---


class FakeSpeaker:
    def __init__(self) -> None:
        self.lines: list[str] = []

    async def say(self, text: str) -> None:
        self.lines.append(text)


class FakeCall:
    def __init__(self) -> None:
        self.ended = False

    async def end_call(self) -> None:
        self.ended = True


class NoStepUps:
    async def run(self, factor, req, decision) -> bool:
        return False


@pytest.fixture
async def real(state, detector):
    """A Toolbox over the real Gate, a scripted detector and the FakeRamp."""
    ramp = ToolboxRamp()
    speaker, call = FakeSpeaker(), FakeCall()
    gate = Gate(
        detector=detector, ramp=ramp, speaker=speaker, stepups=NoStepUps(), call=call, state=state
    )
    forwarder = FakeForwarder(state)
    toolbox = make_toolbox(state, gate, ramp, detector, forwarder)
    yield toolbox, ramp, speaker, call
    await gate.aclose()


async def test_with_the_real_gate_an_allowed_bank_change_is_done(real, state, detector):
    toolbox, ramp, speaker, _ = real
    await toolbox.identify_caller("Eric Catalano")
    detector.kinds = [DecisionKind.ALLOW]
    reply = await toolbox.change_vendor_bank_account("Acme", ROUTING, ACCOUNT)
    assert reply == "Acme Supplies Inc now pays to the account ending 6789."
    assert ("add_default_bank_account", "v1", ROUTING, ACCOUNT) in ramp.calls
    assert speaker.lines == [prompts.VERIFYING]


async def test_with_the_real_gate_a_denied_bank_change_holds_the_resolved_vendor(
    real, state, detector
):
    toolbox, ramp, speaker, call = real
    await toolbox.identify_caller("Eric Catalano")
    detector.kinds = [DecisionKind.DENY]
    reply = await toolbox.change_vendor_bank_account("Acme", ROUTING, ACCOUNT)
    assert reply == prompts.DENY_HELD
    assert ("hold_vendor", "v1") in ramp.calls
    assert ("add_default_bank_account", "v1", ROUTING, ACCOUNT) not in ramp.calls
    assert call.ended and state.locked


async def test_with_the_real_gate_an_unidentified_caller_is_sent_to_identify(real):
    toolbox, ramp, _, _ = real
    assert await toolbox.list_vendors() == prompts.IDENTIFY_FIRST
    assert await toolbox.change_vendor_bank_account("Acme", ROUTING, ACCOUNT) == (
        prompts.IDENTIFY_FIRST
    )
    assert not any(call[0] in ("hold_vendor", "add_default_bank_account") for call in ramp.calls)


async def test_with_the_real_gate_a_hold_needs_no_identification(real):
    toolbox, ramp, _, _ = real
    assert await toolbox.hold_vendor("Acme", "caller asked") == "Acme Supplies Inc is on hold."
    assert ("hold_vendor", "v1") in ramp.calls


async def test_with_the_real_gate_an_unidentified_caller_holds_a_bill_by_vendor_and_invoice(
    real, state
):
    toolbox, ramp, speaker, call = real
    assert state.session_id is None and not state.detector_down
    reply = await toolbox.hold_bill("Acme", "inv-1001", "caller asked")
    assert reply == "Acme Supplies Inc's invoice INV-1001 is on hold."
    assert ("hold_bill", "b1") in ramp.calls
    assert speaker.lines == [] and not call.ended and not state.locked


async def test_with_the_real_gate_a_detector_down_caller_holds_a_bill_and_is_not_hung_up_on(
    real, state
):
    toolbox, ramp, speaker, call = real
    state.detector_down = True  # the detector failed before a session could be made
    reply = await toolbox.hold_bill("Acme", "INV-1002", "caller asked")
    assert reply == "Acme Supplies Inc's invoice INV-1002 is on hold."
    assert ("hold_bill", "b2") in ramp.calls
    assert speaker.lines == [] and not call.ended and not state.locked


@pytest.mark.parametrize("invoice", ["INV-1001", "INV-1002"])
async def test_with_the_real_gate_a_bill_hold_reply_has_no_dollar_sign_and_no_digit_of_the_amount(
    real, invoice
):
    # an unidentified caller may be a stranger who guessed a vendor and an invoice number (R56, R63)
    toolbox, ramp, _, _ = real
    reply = await toolbox.hold_bill("Acme", invoice, "caller asked")
    assert ("hold_bill", "b1" if invoice == "INV-1001" else "b2") in ramp.calls
    assert "$" not in reply
    for amount in ("1,204.50", "1204.50", "120450", "3,980.00", "3980.00", "398000", "204", "980"):
        assert amount not in reply
    assert re.findall(r"\d+", reply) == re.findall(r"\d+", invoice)  # the invoice's digits only


async def test_with_the_real_gate_two_copies_of_an_invoice_are_held_without_either_amount(real):
    toolbox, ramp, _, _ = real
    ramp.bills = [bill("b1"), bill("b8", invoice="inv-1001", cents=777700)]
    reply = await toolbox.hold_bill("Acme", "INV-1001", "caller asked")
    held = [c for c in ramp.calls if c[0] == "hold_bill"]
    assert held == [("hold_bill", "b1"), ("hold_bill", "b8")]
    assert "$" not in reply and "7,777" not in reply and "777" not in reply
    assert set(re.findall(r"\d+", reply)) == {"1001"}


async def test_with_the_real_gate_an_unknown_invoice_holds_nothing_and_leaves_the_call_alone(
    real, state
):
    toolbox, ramp, speaker, call = real
    state.detector_down = True
    assert await toolbox.hold_bill("Acme", "INV-9999", "caller asked") == prompts.NO_SUCH_BILL
    assert not any(c[0] == "hold_bill" for c in ramp.calls)
    assert speaker.lines == [] and not call.ended and not state.locked


async def test_with_the_real_gate_a_listed_bill_can_be_released(real, detector):
    toolbox, ramp, speaker, _ = real
    await toolbox.identify_caller("Eric Catalano")
    detector.kinds = [DecisionKind.ALLOW, DecisionKind.ALLOW]
    listing = await toolbox.list_open_bills("Acme")
    assert "(bill id b1)" in listing and "(bill id b2)" in listing
    reply = await toolbox.release_bill_hold("b1")
    assert reply == "The hold on invoice INV-1001 from Acme Supplies Inc for $1,204.50 is released."
    assert ("release_bill", "b1") in ramp.calls
    assert speaker.lines == [prompts.VERIFYING]


# --- enrollment and the real gate's lock ---


@pytest.fixture
async def enrolling(state, detector):
    """A Toolbox over the real Gate and a FakeEnrollment a test can hold, to show what the gate's
    lock does. Has `toolbox`, `enrollment`, `ramp` and `call`."""
    ramp, call = ToolboxRamp(), FakeCall()
    gate = Gate(
        detector=detector,
        ramp=ramp,
        speaker=FakeSpeaker(),
        stepups=NoStepUps(),
        call=call,
        state=state,
    )
    enrollment = FakeEnrollment([])
    toolbox = make_toolbox(state, gate, ramp, detector, FakeForwarder(state), enrollment)
    yield SimpleNamespace(toolbox=toolbox, enrollment=enrollment, ramp=ramp, call=call)
    await gate.aclose()


async def let_other_tasks_run() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


async def test_with_the_real_gate_an_enrollment_waits_for_a_gated_request_that_is_running(
    enrolling, detector
):
    toolbox, enrollment = enrolling.toolbox, enrolling.enrollment
    await toolbox.identify_caller("Eric Catalano")
    detector.kinds = [DecisionKind.ALLOW]
    detector.hold_authorizing = asyncio.Event()
    reading = asyncio.create_task(toolbox.list_vendors())
    await asyncio.wait_for(detector.authorizing.wait(), timeout=5)
    enrolling_now = asyncio.create_task(toolbox.start_voice_enrollment())
    await let_other_tasks_run()
    assert enrollment.runs == 0 and not enrolling_now.done()  # the request still has the floor
    detector.hold_authorizing.set()
    assert "Acme Supplies Inc" in await asyncio.wait_for(reading, timeout=5)
    assert await asyncio.wait_for(enrolling_now, timeout=5) == prompts.ENROLL_TOOL_DONE
    assert enrollment.runs == 1


async def test_with_the_real_gate_a_gated_request_waits_for_an_enrollment_that_is_running(
    enrolling, detector
):
    toolbox, enrollment = enrolling.toolbox, enrolling.enrollment
    await toolbox.identify_caller("Eric Catalano")
    enrollment.hold = asyncio.Event()
    enrolling_now = asyncio.create_task(toolbox.start_voice_enrollment())
    await asyncio.wait_for(enrollment.started.wait(), timeout=5)
    detector.kinds = [DecisionKind.ALLOW]
    reading = asyncio.create_task(toolbox.list_vendors())
    await let_other_tasks_run()
    # the detector has not been asked, so no verification talks over the enrollment
    assert detector.authorized == [] and not reading.done()
    enrollment.hold.set()
    assert await asyncio.wait_for(enrolling_now, timeout=5) == prompts.ENROLL_TOOL_DONE
    assert "Acme Supplies Inc" in await asyncio.wait_for(reading, timeout=5)
    assert len(detector.authorized) == 1


async def test_with_the_real_gate_a_hold_goes_through_while_an_enrollment_runs(enrolling):
    toolbox, enrollment, ramp = enrolling.toolbox, enrolling.enrollment, enrolling.ramp
    enrollment.hold = asyncio.Event()
    enrolling_now = asyncio.create_task(toolbox.start_voice_enrollment())
    await asyncio.wait_for(enrollment.started.wait(), timeout=5)
    reply = await asyncio.wait_for(toolbox.hold_vendor("Acme", "caller asked"), timeout=5)
    assert reply == "Acme Supplies Inc is on hold." and ("hold_vendor", "v1") in ramp.calls
    enrollment.hold.set()
    await asyncio.wait_for(enrolling_now, timeout=5)


async def test_with_the_real_gate_a_deny_that_comes_first_stops_the_waiting_enrollment(
    enrolling, state, detector
):
    # The bank change is denied while the caller waits to enroll. That locks the call and hangs
    # up, so no code may be texted to the number on file afterwards.
    toolbox, enrollment, call = enrolling.toolbox, enrolling.enrollment, enrolling.call
    await toolbox.identify_caller("Eric Catalano")
    detector.kinds = [DecisionKind.DENY]
    detector.hold_authorizing = asyncio.Event()
    changing = asyncio.create_task(toolbox.change_vendor_bank_account("Acme", ROUTING, ACCOUNT))
    await asyncio.wait_for(detector.authorizing.wait(), timeout=5)
    enrolling_now = asyncio.create_task(toolbox.start_voice_enrollment())
    await let_other_tasks_run()
    assert not enrolling_now.done()
    detector.hold_authorizing.set()
    assert await asyncio.wait_for(changing, timeout=5) == prompts.DENY_HELD
    assert await asyncio.wait_for(enrolling_now, timeout=5) == prompts.ENROLL_NOT_NOW
    assert enrollment.runs == 0 and state.locked and call.ended
