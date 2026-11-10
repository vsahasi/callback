"""Tests for LiveKitCallbackLeg, with fakes for the LiveKit pieces it uses.

The fakes stand in for ctx.api.room (remove_participant), ctx.api.sip (create_sip_participant),
ctx.room (its name and remote_participants) and the session (room_io.set_participant and
clear_user_turn). They write to one `events` list, so a test can check the order of the calls, and
the last item of each event is whether state.callback_in_progress was set at the time.
"""

import asyncio
import dataclasses
import inspect
import logging
import re
import traceback
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import numpy as np
import pytest
from callback_shared.contract import Channel
from google.protobuf import json_format
from livekit import api, rtc

from callback_agent.audio_segments import SegmentCapture
from callback_agent.callback_leg import CallbackLegError, LiveKitCallbackLeg
from callback_agent.config import Settings
from callback_agent.state import CallState
from callback_agent.stepups import CallbackLeg
from callback_agent.turns import FinalSegment

ROOM = "call-1"
CALLER = "sip_+15555550142"  # LiveKit names an inbound SIP caller after their number
NUMBER_ON_FILE = "+15555550100"
TRUNK_ID = "ST_outbound"
TELNYX_NUMBER = "+15555550199"
CALLEE = re.compile(r"callback-[0-9a-f]{8}")


def sip_error(code: str, status: str) -> api.SipCallError:
    # The message and the metadata quote the number, as LiveKit's may, so a log test can catch it.
    return api.SipCallError(
        "unavailable",
        f"could not reach {NUMBER_ON_FILE}",
        status=503,
        metadata={"sip_status_code": code, "sip_status": status, "sip_call_to": NUMBER_ON_FILE},
    )


SIP_ERRORS = {
    "busy": sip_error("486", "Busy Here"),
    "declined": sip_error("603", "Decline"),
    "no_answer": sip_error("480", "Temporarily Unavailable"),
    "trunk_failure": sip_error("503", "Service Unavailable"),
}


class FakeRoom:
    """ctx.room: a name, and the remote participants by identity. The caller starts in it."""

    def __init__(self) -> None:
        self.name = ROOM
        self.remote_participants = {CALLER: SimpleNamespace(identity=CALLER)}


class FakeRoomService:
    """ctx.api.room. remove_participant takes the participant out of the room, or raises
    `error`."""

    def __init__(self, room: FakeRoom, state: CallState, events: list[tuple]) -> None:
        self.error: BaseException | None = None
        self._room, self._state, self._events = room, state, events

    async def remove_participant(
        self, remove: api.RoomParticipantIdentity
    ) -> api.RemoveParticipantResponse:
        self._events.append(
            ("remove_participant", remove.room, remove.identity, self._state.callback_in_progress)
        )
        if self.error is not None:
            raise self.error
        self._room.remote_participants.pop(remove.identity, None)
        return api.RemoveParticipantResponse()


class FakeSipService:
    """ctx.api.sip. create_sip_participant keeps each request and its timeout in `requests`, and
    ends as `outcome` says: "answered" puts the callee in the room, "gone" leaves them out (the
    call ended just before the dial returned), and an exception is raised."""

    def __init__(self, room: FakeRoom, state: CallState, events: list[tuple]) -> None:
        self.requests: list[tuple[api.CreateSIPParticipantRequest, float | None]] = []
        self.outcome: str | BaseException = "answered"
        self._room, self._state, self._events = room, state, events

    async def create_sip_participant(
        self, create: api.CreateSIPParticipantRequest, *, timeout: float | None = None
    ) -> api.SIPParticipantInfo:
        self._events.append(("create_sip_participant", self._state.callback_in_progress))
        self.requests.append((create, timeout))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        identity = create.participant_identity
        if self.outcome == "answered":
            self._room.remote_participants[identity] = SimpleNamespace(identity=identity)
        return api.SIPParticipantInfo(participant_identity=identity, room_name=create.room_name)


class FakeRoomIO:
    """session.room_io. set_participant records who the session is linked to, or raises
    `error`."""

    def __init__(self, state: CallState, events: list[tuple]) -> None:
        self.linked: str | None = CALLER
        self.error: BaseException | None = None
        self._state, self._events = state, events

    def set_participant(self, participant_identity: str | None) -> None:
        self._events.append(
            ("set_participant", participant_identity, self._state.callback_in_progress)
        )
        if self.error is not None:
            raise self.error
        self.linked = participant_identity


class FakeSession:
    """The AgentSession: its room_io, and clear_user_turn, which records who the call state
    named when it ran, or raises `error`."""

    def __init__(self, state: CallState, events: list[tuple]) -> None:
        self.room_io = FakeRoomIO(state, events)
        self.error: BaseException | None = None
        self._state, self._events = state, events

    def clear_user_turn(self) -> None:
        self._events.append(
            ("clear_user_turn", self._state.caller_identity, self._state.callback_in_progress)
        )
        if self.error is not None:
            raise self.error


@pytest.fixture
def events() -> list[tuple]:
    return []


@pytest.fixture
def state() -> CallState:
    return CallState(room_name=ROOM, caller_identity=CALLER, channel=Channel.PHONE, session_id="s1")


@pytest.fixture
def settings() -> Settings:
    return Settings(
        agent_name="callback",
        cartesia_voice_id="voice",
        sip_outbound_trunk_id=TRUNK_ID,
        telnyx_number=TELNYX_NUMBER,
    )


@pytest.fixture
def ctx(state, events) -> SimpleNamespace:
    room = FakeRoom()
    return SimpleNamespace(
        room=room,
        api=SimpleNamespace(
            room=FakeRoomService(room, state, events), sip=FakeSipService(room, state, events)
        ),
    )


@pytest.fixture
def session(state, events) -> FakeSession:
    return FakeSession(state, events)


@pytest.fixture
def leg(ctx, session, state, settings) -> LiveKitCallbackLeg:
    return LiveKitCallbackLeg(ctx, session, state, settings)


def assert_nobody_answered(state: CallState, events: list[tuple]) -> None:
    """The call state still names the caller, and no user turn was cleared."""
    assert state.caller_identity == CALLER and state.channel is Channel.PHONE
    assert [event for event in events if event[0] == "clear_user_turn"] == []


# --- the interface ---


def test_the_leg_takes_the_job_the_session_the_state_and_the_settings():
    params = inspect.signature(LiveKitCallbackLeg).parameters
    assert list(params) == ["ctx", "session", "state", "settings"]
    assert inspect.signature(LiveKitCallbackLeg.dial_and_link) == inspect.signature(
        CallbackLeg.dial_and_link
    )


# --- an answer ---


async def test_links_the_callee_then_removes_the_caller_then_dials(leg, events):
    callee = await leg.dial_and_link(NUMBER_ON_FILE)
    assert CALLEE.fullmatch(callee)
    assert events == [
        ("set_participant", callee, True),
        ("remove_participant", ROOM, CALLER, True),
        ("create_sip_participant", True),
        ("clear_user_turn", CALLER, True),
    ]


async def test_dials_the_number_through_the_outbound_trunk_and_waits_for_the_answer(leg, ctx):
    callee = await leg.dial_and_link(NUMBER_ON_FILE)
    # No other field is set: krisp noise cancellation stays off, so the detector hears the raw line.
    assert ctx.api.sip.requests == [
        (
            api.CreateSIPParticipantRequest(
                room_name=ROOM,
                sip_trunk_id=TRUNK_ID,
                sip_call_to=NUMBER_ON_FILE,
                sip_number=TELNYX_NUMBER,
                participant_identity=callee,
                wait_until_answered=True,
            ),
            45,
        )
    ]


async def test_an_answer_clears_the_user_turn_then_points_the_state_at_the_callee(
    leg, session, state, events
):
    callee = await leg.dial_and_link(NUMBER_ON_FILE)
    # Cleared once, while the state still named the caller: nothing heard before the switch can
    # count as the callee's answer.
    assert [event for event in events if event[0] == "clear_user_turn"] == [
        ("clear_user_turn", CALLER, True)
    ]
    assert state.caller_identity == callee and state.channel is Channel.PHONE
    assert session.room_io.linked == callee


async def test_the_relink_discards_the_segment_of_the_callers_speech_that_was_open(leg, state):
    # The caller was mid-sentence when the leg relinked. Nothing of that stretch goes anywhere,
    # and nothing the callee says lands in a segment that began on the caller.
    heard = []
    state.audio_capture = SegmentCapture(sink=heard.append)
    state.audio_capture.begin(1000.0, track_id="caller")
    for k in range(50):
        pcm = np.zeros(480, dtype=np.int16)
        state.audio_capture.add_frame(rtc.AudioFrame(pcm.tobytes(), 24000, 1, 480), 1000 + k / 50)
    assert state.audio_capture.open
    await leg.dial_and_link(NUMBER_ON_FILE)
    assert not state.audio_capture.open and heard == []


async def test_an_answer_drops_the_callers_leftover_transcript_segments(leg, state):
    segments = state.pending_segments
    segments.append(
        FinalSegment(text="yes", start_s=1.0, end_s=1.4, words=(("yes", 1.0, 1.4),), t0=1_000.0)
    )
    await leg.dial_and_link(NUMBER_ON_FILE)
    # The caller's "yes" can't be claimed into the callee's first turn. Emptied in place, as the
    # session wiring empties it.
    assert state.pending_segments == [] and state.pending_segments is segments


async def test_each_dial_uses_a_fresh_identity(leg):
    callees = [await leg.dial_and_link(NUMBER_ON_FILE) for _ in range(3)]
    assert all(CALLEE.fullmatch(callee) for callee in callees)
    # LiveKit Cloud refuses a removed identity's token for 10 minutes, so none is used twice.
    assert len({CALLER, *callees}) == 4


# --- no answer ---


@pytest.mark.parametrize("error", SIP_ERRORS.values(), ids=SIP_ERRORS.keys())
async def test_a_sip_call_error_is_no_answer(leg, ctx, state, events, error):
    ctx.api.sip.outcome = error
    assert await leg.dial_and_link(NUMBER_ON_FILE) is None
    assert_nobody_answered(state, events)


async def test_a_dial_that_times_out_is_no_answer(leg, ctx, state, events):
    ctx.api.sip.outcome = asyncio.TimeoutError()
    assert await leg.dial_and_link(NUMBER_ON_FILE) is None
    assert_nobody_answered(state, events)


async def test_a_callee_who_never_appears_is_no_answer(leg, ctx, state, events):
    ctx.api.sip.outcome = "gone"
    assert await leg.dial_and_link(NUMBER_ON_FILE) is None
    assert_nobody_answered(state, events)


# --- the caller's removal ---

# Failures of the removal other than not_found. The server error quotes the caller's identity,
# which holds their number, as LiveKit's message may.
REMOVAL_ERRORS = {
    "server_error": api.ServerError("unavailable", f"could not remove {CALLER}", status=503),
    "livekit_unreachable": aiohttp.ClientConnectionError("connection reset"),
    "timeout": asyncio.TimeoutError(),
}


async def test_a_not_found_removal_still_dials(leg, ctx, state):
    # The caller had already left. The room hasn't caught up here, and still lists them.
    ctx.api.room.error = api.ServerError("not_found", f"participant {CALLER} not found", status=404)
    callee = await leg.dial_and_link(NUMBER_ON_FILE)
    assert CALLER in ctx.room.remote_participants
    assert len(ctx.api.sip.requests) == 1 and state.caller_identity == callee


@pytest.mark.parametrize("error", REMOVAL_ERRORS.values(), ids=REMOVAL_ERRORS.keys())
async def test_a_failed_removal_with_the_caller_still_present_raises_without_dialing(
    leg, ctx, state, events, error
):
    ctx.api.room.error = error
    with pytest.raises(CallbackLegError) as raised:
        await leg.dial_and_link(NUMBER_ON_FILE)
    # Nobody is dialed: a caller still in the room could hear or coach the real user.
    assert ctx.api.sip.requests == []
    # The gate logs the traceback, so it names the failure but never quotes LiveKit's message.
    logged = "".join(traceback.format_exception(raised.value))
    assert type(error).__name__ in logged
    for number in ("5555550100", "5555550142"):
        assert number not in logged
    assert_nobody_answered(state, events)
    assert state.callback_in_progress is False


@pytest.mark.parametrize("error", REMOVAL_ERRORS.values(), ids=REMOVAL_ERRORS.keys())
async def test_a_failed_removal_with_the_caller_already_gone_still_dials(
    leg, ctx, state, caplog, error
):
    del ctx.room.remote_participants[CALLER]  # they hung up on hearing the callback line
    ctx.api.room.error = error
    callee = await leg.dial_and_link(NUMBER_ON_FILE)
    assert len(ctx.api.sip.requests) == 1 and state.caller_identity == callee
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1 and type(error).__name__ in warnings[0].getMessage()


async def test_any_other_error_from_the_removal_propagates_without_dialing(leg, ctx):
    ctx.api.room.error = RuntimeError("not one of the LiveKit API's errors")
    with pytest.raises(RuntimeError):
        await leg.dial_and_link(NUMBER_ON_FILE)
    assert ctx.api.sip.requests == []


# --- no answer ends the call ---
#
# By the time the dial gets no answer the caller is off the line: removed, or already gone. There
# is nobody left to ask again, so the leg marks the call as ending (state.closing), and the
# step-up turns that into a deny that holds and hangs up without a word (StepUps.callback).

NO_ANSWER = {
    **{f"sip_{name}": error for name, error in SIP_ERRORS.items()},
    "timeout": TimeoutError(),
    "callee_never_appears": "gone",
}


@pytest.mark.parametrize("outcome", NO_ANSWER.values(), ids=NO_ANSWER.keys())
async def test_no_answer_after_the_caller_was_removed_marks_the_call_as_ending(
    leg, ctx, state, outcome
):
    ctx.api.sip.outcome = outcome
    assert await leg.dial_and_link(NUMBER_ON_FILE) is None
    assert CALLER not in ctx.room.remote_participants  # they are off the line
    assert state.closing is True


async def test_the_call_is_not_marked_as_ending_until_the_dial_has_failed(leg, ctx, state):
    # Not before: a callee who answers finds a call that is open.
    seen: list[bool] = []
    dial = ctx.api.sip.create_sip_participant

    async def spy(create, *, timeout=None):
        seen.append(state.closing)
        return await dial(create, timeout=timeout)

    ctx.api.sip.create_sip_participant = spy
    ctx.api.sip.outcome = SIP_ERRORS["no_answer"]
    assert await leg.dial_and_link(NUMBER_ON_FILE) is None
    assert seen == [False] and state.closing is True


async def test_an_answer_leaves_the_call_open(leg, state):
    assert await leg.dial_and_link(NUMBER_ON_FILE) is not None
    assert state.closing is False


async def test_a_caller_found_already_gone_is_off_the_line_too(leg, ctx, state):
    ctx.api.room.error = api.ServerError("not_found", f"participant {CALLER} not found", status=404)
    ctx.api.sip.outcome = SIP_ERRORS["no_answer"]
    assert await leg.dial_and_link(NUMBER_ON_FILE) is None
    assert state.closing is True


async def test_a_dial_the_server_refuses_after_the_caller_was_removed_ends_the_call_too(
    leg, ctx, state
):
    # It raises and the gate denies at once, but the deny line would be said to an empty line.
    ctx.api.sip.outcome = api.ServerError("permission_denied", "no SIP grant", status=403)
    with pytest.raises(CallbackLegError):
        await leg.dial_and_link(NUMBER_ON_FILE)
    assert CALLER not in ctx.room.remote_participants
    assert state.closing is True


# A dial request that dies on the way to LiveKit's API, or whose answer never comes back whole.
# The message quotes the number, as an aiohttp error can quote the address it was given.
DIAL_TRANSPORT_ERRORS = {
    "connection_reset": aiohttp.ClientConnectionError(f"reset while dialing {NUMBER_ON_FILE}"),
    "server_disconnected": aiohttp.ServerDisconnectedError(f"gone while dialing {NUMBER_ON_FILE}"),
    "os_error": aiohttp.ClientOSError(104, f"connection reset by peer, dialing {NUMBER_ON_FILE}"),
}


@pytest.mark.parametrize("error", DIAL_TRANSPORT_ERRORS.values(), ids=DIAL_TRANSPORT_ERRORS.keys())
async def test_a_dial_that_dies_on_the_wire_after_the_caller_was_removed_ends_the_call_too(
    leg, ctx, state, events, error
):
    # The caller is off the line and the dial's outcome is unknown, so it is handled as a dial the
    # server refused: the call is over, and the error is the leg's own.
    ctx.api.sip.outcome = error
    with pytest.raises(CallbackLegError) as raised:
        await leg.dial_and_link(NUMBER_ON_FILE)
    assert CALLER not in ctx.room.remote_participants
    assert state.closing is True and state.callback_in_progress is False
    # The gate logs the traceback, so it names the failure but never quotes what aiohttp said.
    logged = "".join(traceback.format_exception(raised.value))
    assert type(error).__name__ in logged
    for number in ("5555550100", "5555550142"):
        assert number not in logged


async def test_a_dial_that_dies_on_the_wire_is_not_mistaken_for_nobody_answering(leg, ctx, state):
    # A SIP outcome returns None and reports no answer. A dead connection is an error to raise.
    ctx.api.sip.outcome = aiohttp.ClientConnectionError("reset")
    with pytest.raises(CallbackLegError):
        await leg.dial_and_link(NUMBER_ON_FILE)
    assert state.closing is True


@pytest.mark.parametrize("error", REMOVAL_ERRORS.values(), ids=REMOVAL_ERRORS.keys())
async def test_a_failed_removal_of_a_caller_who_has_left_is_off_the_line_too(
    leg, ctx, state, error
):
    del ctx.room.remote_participants[CALLER]  # they hung up on hearing the callback line
    ctx.api.room.error = error
    ctx.api.sip.outcome = SIP_ERRORS["no_answer"]
    assert await leg.dial_and_link(NUMBER_ON_FILE) is None
    assert state.closing is True


@pytest.mark.parametrize("error", REMOVAL_ERRORS.values(), ids=REMOVAL_ERRORS.keys())
async def test_a_caller_who_could_not_be_removed_and_is_still_there_leaves_the_call_open(
    leg, ctx, state, error
):
    # Nobody is dialed, and the caller is still on the line: this is not a call that is over.
    ctx.api.room.error = error
    with pytest.raises(CallbackLegError):
        await leg.dial_and_link(NUMBER_ON_FILE)
    assert state.closing is False


async def test_a_link_that_fails_before_anyone_is_removed_leaves_the_call_open(
    leg, ctx, session, state
):
    session.room_io.error = RuntimeError("Cannot access room_io")
    with pytest.raises(RuntimeError):
        await leg.dial_and_link(NUMBER_ON_FILE)
    assert CALLER in ctx.room.remote_participants and state.closing is False


# --- settings ---


@pytest.mark.parametrize(
    "missing, name",
    [
        ("sip_outbound_trunk_id", "LIVEKIT_SIP_OUTBOUND_TRUNK_ID"),
        ("telnyx_number", "TELNYX_NUMBER"),
    ],
)
async def test_missing_settings_dial_no_one_and_leave_the_call_alone(
    ctx, session, state, settings, events, caplog, missing, name
):
    leg = LiveKitCallbackLeg(ctx, session, state, dataclasses.replace(settings, **{missing: None}))
    assert await leg.dial_and_link(NUMBER_ON_FILE) is None
    assert events == []  # not linked, not removed, not dialed: the caller is still on the line
    assert state.callback_in_progress is False
    assert state.closing is False  # nobody was taken off the line, so the call is not over
    assert_nobody_answered(state, events)
    assert name in caplog.text


# --- callback_in_progress ---


def answered(ctx, session) -> None:
    pass


def nobody_answered(ctx, session) -> None:
    ctx.api.sip.outcome = SIP_ERRORS["no_answer"]


def timed_out(ctx, session) -> None:
    ctx.api.sip.outcome = asyncio.TimeoutError()


def callee_gone(ctx, session) -> None:
    ctx.api.sip.outcome = "gone"


def removal_failed(ctx, session) -> None:
    ctx.api.room.error = api.ServerError("not_found", "participant not found", status=404)


def removal_failed_caller_present(ctx, session) -> None:
    ctx.api.room.error = api.ServerError("unavailable", "try again", status=503)


def dial_refused(ctx, session) -> None:
    ctx.api.sip.outcome = api.ServerError("permission_denied", "no SIP grant", status=403)


def dial_died_on_the_wire(ctx, session) -> None:
    ctx.api.sip.outcome = aiohttp.ClientConnectionError("connection reset")


def link_failed(ctx, session) -> None:
    session.room_io.error = RuntimeError("Cannot access room_io")


def clear_failed(ctx, session) -> None:
    session.error = RuntimeError("AgentSession isn't running")


def cancelled(ctx, session) -> None:
    ctx.api.sip.outcome = asyncio.CancelledError()


@pytest.mark.parametrize(
    "path",
    [
        answered,
        nobody_answered,
        timed_out,
        callee_gone,
        removal_failed,
        removal_failed_caller_present,
        dial_refused,
        dial_died_on_the_wire,
        link_failed,
        clear_failed,
        cancelled,
    ],
    ids=lambda path: path.__name__,
)
async def test_callback_in_progress_is_set_while_dialing_and_cleared_on_every_path(
    leg, ctx, session, state, events, path
):
    path(ctx, session)
    # Some paths raise: this test is about the flag.
    with suppress(Exception, asyncio.CancelledError):
        await leg.dial_and_link(NUMBER_ON_FILE)
    assert events and all(event[-1] is True for event in events)  # set from the link on
    assert state.callback_in_progress is False


# --- a dial that fails another way ---


async def test_a_refused_dial_raises_without_what_livekit_said(leg, ctx, state, events):
    ctx.api.sip.outcome = api.ServerError(
        "invalid_argument",
        f"cannot dial {NUMBER_ON_FILE}",
        status=400,
        metadata={"sip_call_to": NUMBER_ON_FILE},
    )
    with pytest.raises(CallbackLegError) as raised:
        await leg.dial_and_link(NUMBER_ON_FILE)
    # The gate logs the traceback of a step-up that raises, so it must not quote the number.
    logged = "".join(traceback.format_exception(raised.value))
    assert "invalid_argument" in logged and "5555550100" not in logged
    assert_nobody_answered(state, events)


@pytest.mark.parametrize("path", [link_failed, clear_failed], ids=lambda path: path.__name__)
async def test_any_other_error_propagates(leg, ctx, session, path):
    path(ctx, session)
    with pytest.raises(RuntimeError):
        await leg.dial_and_link(NUMBER_ON_FILE)


# --- the outbound trunk ---


def test_the_outbound_trunk_sends_the_telnyx_username_on_the_invite():
    # `lk sip outbound create` reads the file as a CreateSIPOutboundTrunkRequest. Only `headers`
    # goes out on the INVITE (headers_to_attributes maps headers of the 200 OK), and without this
    # one Telnyx may match an unrelated connection by source IP and skip its 407 challenge.
    raw = (Path(__file__).parents[1] / "outbound-trunk.json").read_text()
    trunk = json_format.Parse(raw, api.CreateSIPOutboundTrunkRequest()).trunk
    assert dict(trunk.headers) == {"X-Telnyx-Username": "<TELNYX_SIP_USER>"}
    # Committed with placeholders: the number and the user are filled in locally.
    assert trunk.address == "sip.telnyx.com" and list(trunk.numbers) == ["<TELNYX_NUMBER>"]


# --- what is logged ---


async def test_logs_hold_at_most_the_last_four_digits(leg, ctx, caplog):
    caplog.set_level(logging.DEBUG, logger="callback_agent.callback_leg")
    ctx.api.room.error = api.ServerError("not_found", f"participant {CALLER} not found", status=404)
    ctx.api.sip.outcome = SIP_ERRORS["busy"]
    assert await leg.dial_and_link(NUMBER_ON_FILE) is None
    ctx.api.sip.outcome = "answered"
    assert await leg.dial_and_link(NUMBER_ON_FILE) is not None
    assert "ending 0100" in caplog.text  # the last four are enough to tell calls apart
    for number in ("5555550100", "5555550142"):  # the number dialed, and the caller's
        assert number not in caplog.text
