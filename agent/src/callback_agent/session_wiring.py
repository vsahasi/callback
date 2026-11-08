"""Connects one call's AgentSession to the forwarder, and ends the forwarding when the call ends.

wire_turn_forwarding turns the session's events into the turns and speech events the detector
wants. finish_call is what the job's shutdown callback runs. The other pieces a call is built from
are here too: the Ramp client, the hang-ups (the agent's own, and the caller's, which the worker
owns because the session is not closed when the caller disconnects), the gate and the toolbox
(with the step-ups and the enrollment under them), the room options and the reader of the
detector's risk packets. They live here and not in worker.py, which reads its settings at import
time and so can't be imported by a test; tests/test_structure.py checks that the worker builds
none of them by hand. Where each time comes from is in the LiveKit notes
(docs/research_notes/Callback agent build research/
livekit_agents_framework.md, Q3 and Q4); the word times of a user turn come from the FinalSegments
that TimedSttMixin collects in CallState.pending_segments. A user turn is interrupted if its first
word starts inside one of the stretches of the agent speaking kept in CallState.agent_speech.

Logs name an item by its kind and time, never by its text: a turn can hold a spoken code. For the
same reason a handler that fails is logged by the exception's type and place, not its message,
which can quote the turn it was working on.
"""

import asyncio
import functools
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, NamedTuple, Protocol

from callback_shared.contract import (
    RISK_TOPIC,
    Channel,
    RiskUpdate,
    Role,
    SpeechEvent,
    SpeechEventType,
    Turn,
)
from livekit import rtc
from livekit.agents import AgentSession, JobContext, room_io
from livekit.agents.llm import ChatMessage

from callback_agent import prompts
from callback_agent.callback_leg import LiveKitCallbackLeg
from callback_agent.capture import LiveKitVoice
from callback_agent.config import Settings
from callback_agent.detector import DetectorClient, DetectorError
from callback_agent.enrollment import Enrollment
from callback_agent.gate import CallControl, Gate
from callback_agent.ramp import RampClient
from callback_agent.redact import redact_turn
from callback_agent.safelog import log_failure
from callback_agent.state import CallState, SpeechInterval
from callback_agent.stepups import CallbackLeg, StepUps
from callback_agent.toolbox import Toolbox
from callback_agent.turns import (
    FinalSegment,
    Forwarder,
    build_agent_turn,
    build_user_turn,
    speech_event,
)

logger = logging.getLogger(__name__)

LOG_TURNS_ENV = "CALLBACK_LOG_TURNS"

# The steps of the end of a call. In livekit-agents 1.8.10 the job's shutdown awaits
# session.aclose() (up to 60 s) before it runs any shutdown callback, finish_call among them. The
# parent's 10 s shutdown_process_timeout for the process to exit starts only once the job reports
# that it is shutting down, which is after that close. That 10 s is all the steps have, so each
# is bounded and none can hold the others up.
CLOSE_WAIT_S = 3.0  # for the session to finish closing: its last turns come out while it does
DRAIN_S = 2.0  # for the forwarder to send what it holds
END_SESSION_S = 3.0  # for the detector to end the session
FORWARDER_CLOSE_S = 3.0  # for the post in flight to finish when the forwarder closes
GATE_CLOSE_S = 2.0  # for the gate to stop the hold reports it still has out to the detector
HANGUP_S = 5.0  # for the server to answer the request that deletes the room
SESSION_CLOSE_S = 3.0  # for hang_up to wait for the session's close event, then delete the room

# How the session words what it tells the LLM about tools it should not repeat (AgentSession's
# tool_handling): only the text for a call turned away because the same tool is still running.
TOOL_HANDLING = {"async_options": {"duplicate_reject_template": prompts.DUPLICATE_CALL}}


class Forwarding(NamedTuple):
    """What a call needs to reach the detector: its client, and the forwarder that feeds it."""

    detector: DetectorClient
    forwarder: Forwarder


def turn_logging_enabled(env: Mapping[str, str]) -> bool:
    """CALLBACK_LOG_TURNS turns the per-item log on. Any value that isn't blank does."""
    return bool((env.get(LOG_TURNS_ENV) or "").strip())


def call_state_for(room_name: str, participant: rtc.RemoteParticipant) -> CallState:
    """The state of a new call: phone if the caller joined through SIP, browser otherwise."""
    phone = participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
    return CallState(
        room_name=room_name,
        caller_identity=participant.identity,
        channel=Channel.PHONE if phone else Channel.BROWSER,
    )


def build_forwarding(settings: Settings) -> Forwarding | None:
    """The detector client and forwarder of a call, or None when no detector is configured."""
    if not (settings.detector_url and settings.detector_token):
        logger.warning(
            "DETECTOR_URL or DETECTOR_TOKEN is not set, so turns and events are not forwarded"
        )
        return None
    detector = DetectorClient(
        settings.detector_url, settings.detector_token, scenario=settings.detector_scenario
    )
    return Forwarding(detector, Forwarder(detector))


def build_ramp(settings: Settings) -> RampClient:
    """The Ramp client of a call. The worker requires both credentials, so a gap is a bug."""
    if not (settings.ramp_client_id and settings.ramp_client_secret):
        raise ValueError("RAMP_CLIENT_ID and RAMP_CLIENT_SECRET are required to act on Ramp")
    return RampClient(settings.ramp_base_url, settings.ramp_client_id, settings.ramp_client_secret)


def build_callback_leg(
    ctx: JobContext, session: AgentSession[CallState], state: CallState, settings: Settings
) -> LiveKitCallbackLeg | None:
    """The callback leg of a call, or None when the call has no way to dial out.

    A leg dials through the outbound trunk, from the Telnyx number, so it is built only when both
    are set. Without either there is none, and a callback_on_file step-up fails closed before it
    says a word. A leg built anyway would tell the caller they will be called back, find that it
    cannot dial, and report that nobody answered.
    """
    if not (settings.sip_outbound_trunk_id and settings.telnyx_number):
        logger.info(
            "no callback leg: LIVEKIT_SIP_OUTBOUND_TRUNK_ID and TELNYX_NUMBER must both be set, "
            "so a callback step-up fails closed"
        )
        return None
    return LiveKitCallbackLeg(ctx, session, state, settings)


class CallTools(NamedTuple):
    """What a call's tools stand on: the gate, which finish_call stops, and the toolbox."""

    gate: Gate
    toolbox: Toolbox


def build_tools(
    session: AgentSession[CallState],
    state: CallState,
    *,
    detector: DetectorClient,
    forwarder: Forwarder | None,
    ramp: RampClient,
    call: CallControl,
    callback_leg: CallbackLeg | None,
) -> CallTools:
    """The gate and the toolbox of a call, over the one LiveKitVoice that talks to its caller.

    The gate says its lines through the voice, the step-ups ask their questions through it, and so
    does the enrollment. The gate runs the step-ups the detector asks for. The toolbox starts the
    enrollment inside the gate's exclusive lock, so it never overlaps a gated request.

    `callback_leg` is what a callback_on_file step-up hangs up and dials with (build_callback_leg
    makes it). None means the call has no way to dial, and that step-up fails closed: it never
    passes, and the gate goes on to a deny. It has no default, so that no call is built without
    having decided.
    """
    voice = LiveKitVoice(session)
    stepups = StepUps(detector, voice, state, callback_leg=callback_leg)
    gate = Gate(
        detector=detector, ramp=ramp, speaker=voice, stepups=stepups, call=call, state=state
    )
    enrollment = Enrollment(detector, voice, stepups, state)
    toolbox = Toolbox(
        state=state,
        gate=gate,
        ramp=ramp,
        detector=detector,
        forwarder=forwarder,
        enrollment=enrollment,
    )
    return CallTools(gate, toolbox)


def build_room_options(participant_identity: str) -> room_io.RoomOptions:
    """The room options of a call: linked to the caller who joined, with the chat topic closed,
    and kept open when the caller disconnects.

    LiveKit's RoomIO hands text sent to a room's chat topic (lk.chat) to the session as a user
    turn unless text_input is False. A browser caller could then type a turn, or even the answer
    to a challenge or the consent. Every answer the gate and the enrollment take is meant to be
    spoken, where the detector can hear the voice.

    That closes one of two ways in for typed input, not both. The other is the session host that
    AgentSession.start sets up unless it is told not to: on the lk.agent.session topic, a
    run_input request from any participant becomes a user turn. worker.py turns it off with
    session_host=False, and tests/test_structure.py pins that, as no test can import the worker.

    close_on_disconnect is off because a caller who hangs up is CLIENT_INITIATED, and LiveKit
    would close the session for that. The callback hangs up on the caller and dials the number on
    file inside this one session, so a session that closed with the caller would end the call.
    Ending the call when the caller hangs up is left to wire_hangups.
    """
    return room_io.RoomOptions(
        participant_identity=participant_identity,
        text_input=False,
        close_on_disconnect=False,
    )


def wire_turn_forwarding(
    session: AgentSession[CallState],
    state: CallState,
    forwarder: Forwarder | None,
    *,
    log_turns: bool = False,
) -> None:
    """Forward the session's turns and speech events to `forwarder`, and keep `state` up to date.

    Without a forwarder nothing is sent, but the state is kept and the log still works, because
    the live timing check runs before a detector exists. With `log_turns` every item that is
    forwarded also gets one log line.
    """

    def emit(item: Turn | SpeechEvent) -> None:
        if isinstance(item, Turn):
            # No full bank number leaves the agent. Only the caller's first turn after a secret
            # ask keeps its 4 to 6 digit runs, and that turn uses the exemption up.
            exempt = item.role is Role.USER and state.answer_exempt
            item = redact_turn(item, exempt_short_runs=exempt)
            if item.role is Role.USER:
                state.answer_exempt = False
        if log_turns:
            logger.info(_describe(item))
        if forwarder is not None:
            forwarder.submit(item)

    @_guarded("forward a conversation item")
    def on_item_added(ev) -> None:
        message = ev.item
        if not isinstance(message, ChatMessage):
            return  # an agent handoff marker
        if message.role == "user":
            turn = _user_turn(state, message)
        elif message.role == "assistant":
            turn = _agent_turn(message, ev.created_at)
        else:
            return
        if turn is not None:
            emit(turn)

    @_guarded("forward an agent state change")
    def on_agent_state(ev) -> None:
        if ev.new_state == "speaking":
            event = speech_event(SpeechEventType.AGENT_SPEECH_START, ev.created_at)
            _agent_began_speaking(state, event.ts_ms)
            emit(event)
        elif ev.old_state == "speaking":
            event = speech_event(SpeechEventType.AGENT_SPEECH_END, ev.created_at)
            _agent_stopped_speaking(state, event.ts_ms)
            emit(event)

    @_guarded("forward a user state change")
    def on_user_state(ev) -> None:
        if ev.new_state == "speaking":
            emit(speech_event(SpeechEventType.USER_SPEECH_START, ev.created_at))

    @_guarded("note that the session closed")
    def on_close(ev) -> None:
        state.closing = True

    session.on("conversation_item_added", on_item_added)
    session.on("agent_state_changed", on_agent_state)
    session.on("user_state_changed", on_user_state)
    session.on("close", on_close)


def wire_risk_updates(room: rtc.Room, state: CallState) -> None:
    """Keep the newest risk update of this call's session on `state`, for the logs.

    The detector publishes it on the RISK_TOPIC data topic as a hidden participant, so LiveKit
    cannot name the sender and the packet's participant is None. A packet from anyone it can name
    (the caller is not hidden and can publish on any topic) is not the detector's and is dropped,
    and so is one for another session, or for none yet. Nothing acts on a risk update: the gate
    asks the detector, and this only records what the detector last said.
    """
    said = False  # one warning per call about a sender that isn't hidden, not one per packet

    @_guarded("store a risk update")
    def on_data(packet: rtc.DataPacket) -> None:
        nonlocal said
        if packet.topic != RISK_TOPIC:
            return
        if packet.participant is not None:
            if not said:
                said = True
                logger.warning(
                    "ignoring a risk packet from a visible participant: only the detector's, "
                    "which has no sender LiveKit can name, is kept"
                )
            return
        update = RiskUpdate.model_validate_json(packet.data)
        if state.session_id is None or update.session_id != state.session_id:
            return
        state.latest_risk = update

    room.on("data_received", on_data)


def wire_hangups(
    room: rtc.Room, session: AgentSession[CallState], state: CallState, call: CallControl
) -> None:
    """End the call when the caller hangs up, which LiveKit no longer does for us.

    build_room_options keeps the session open when its linked participant leaves, so that a
    callback can hang up on the caller and dial the number on file inside the same session. That
    leaves ending the call to this. A participant who leaves ends it if they are the caller the
    call state names at that moment and no callback is in progress. state.caller_identity changes
    when a callback is answered, so the callee hanging up ends the call, and a late event for the
    caller the leg removed does not. The flag is up from before the caller hears they will be hung
    up on until the dial is over, so leaving then is planned. Anyone else leaving is ignored.

    state.closing is set first, before anything awaits, so the gate and the step-ups stop at
    once. The rest is hang_up, in a task: this runs inside LiveKit's event loop, where it can't
    await, and nothing may be raised into it. A call is ended once.
    """
    ending: set[asyncio.Task[None]] = set()  # asyncio holds a task weakly: keep the one running
    closed = asyncio.Event()  # set by the session's close event, whoever started the close
    session.on("close", lambda _: closed.set())

    def settled(task: asyncio.Task[None]) -> None:
        ending.discard(task)
        if not task.cancelled() and (exc := task.exception()) is not None:
            log_failure(logger, "end the call after a hang-up", exc)

    @_guarded("end the call after a hang-up")
    def on_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
        if participant.identity != state.caller_identity or state.callback_in_progress:
            return
        state.closing = True
        if ending:
            return
        task = asyncio.create_task(hang_up(session, call, closed), name="hang up")
        ending.add(task)
        task.add_done_callback(settled)

    room.on("participant_disconnected", on_participant_disconnected)


def _guarded(action: str) -> Callable[[Callable[[Any], None]], Callable[[Any], None]]:
    """Make an event handler safe to give to LiveKit: nothing it raises gets out.

    The handlers run inside LiveKit's event loop. rtc.EventEmitter lets a TypeError out of emit(),
    into whatever LiveKit code emitted the event, and logs any other exception with its message,
    which can quote a turn. So a handler that fails is logged here instead, and the call goes on.
    """

    def decorate(handler: Callable[[Any], None]) -> Callable[[Any], None]:
        @functools.wraps(handler)
        def guarded(ev: Any) -> None:
            try:
                handler(ev)
            except Exception as exc:
                log_failure(logger, action, exc)

        return guarded

    return decorate


def _user_turn(state: CallState, message: ChatMessage) -> Turn | None:
    text = message.text_content or ""
    segments = _claim_segments(state, text)
    if segments:
        turn = build_user_turn(segments, interrupted=False)
        # The turn began over the agent if its first word did. A VAD pause inside the turn, and
        # the agent giving way to it, change nothing: only where the first word starts counts.
        began_ms = turn.words[0].start_ms if turn.words else turn.start_ms
        if any(speech.covers(began_ms) for speech in state.agent_speech):
            turn = turn.model_copy(update={"interrupted": True})
        return turn
    if not text.strip():
        return None
    # Typed input: nobody spoke, so there are no words and nothing to interrupt, and the message
    # is its own clock.
    when = _ms(message.created_at)
    return Turn(role=Role.USER, text=text, start_ms=when, end_ms=when, words=[], interrupted=False)


def _agent_began_speaking(state: CallState, at_ms: int) -> None:
    if state.agent_speech and state.agent_speech[-1].end_ms is None:
        return  # already speaking: keep the start we have
    state.agent_speech.append(SpeechInterval(at_ms))


def _agent_stopped_speaking(state: CallState, at_ms: int) -> None:
    if state.agent_speech and state.agent_speech[-1].end_ms is None:
        speech = state.agent_speech[-1]
        speech.end_ms = max(at_ms, speech.start_ms)


def _claim_segments(state: CallState, text: str) -> list[FinalSegment]:
    """Take the segments of the user message `text` out of state.pending_segments.

    Normally that is all of them. But LiveKit drops a turn that ends while an uninterruptible
    prompt plays, and no message follows its FINAL, so a stale segment can lead the list. The
    framework builds a message from its FINALs by joining them with spaces, so the latest run of
    segments that reads as `text` is the message's own. What comes before it is stale and goes;
    what comes after belongs to a later turn and stays. If nothing matches (something rewrote the
    message), the words are still the best record of it, so all of them are taken.
    """
    pending = state.pending_segments
    wanted = " ".join(text.split())
    for end in range(len(pending), 0, -1):
        joined = ""
        for start in range(end - 1, -1, -1):
            joined = f"{pending[start].text} {joined}"
            if " ".join(joined.split()) == wanted:
                claimed = pending[start:end]
                if start:
                    logger.warning(
                        "dropping %d stale transcript segment(s) that no user message claimed",
                        start,
                    )
                del pending[:end]
                return claimed
    claimed = list(pending)
    pending.clear()
    return claimed


def _agent_turn(message: ChatMessage, event_created_at: float) -> Turn | None:
    text = message.text_content or ""
    if not text.strip():
        return None
    metrics = message.metrics or {}
    started = metrics.get("started_speaking_at", message.created_at)
    stopped = max(metrics.get("stopped_speaking_at", event_created_at), started)
    return build_agent_turn(text, started, stopped, interrupted=message.interrupted)


def _ms(epoch_s: float) -> int:
    return round(epoch_s * 1000)  # the one rule turns.py uses for every time


def _describe(item: Turn | SpeechEvent) -> str:
    """One log line for a forwarded item: its kind and times, never its text."""
    if isinstance(item, Turn):
        return (
            f"forwarding turn: role={item.role.value} start_ms={item.start_ms} "
            f"end_ms={item.end_ms} words={len(item.words)} interrupted={item.interrupted}"
        )
    return f"forwarding event: type={item.type.value} ts_ms={item.ts_ms}"


class _RoomDeleter(Protocol):
    def delete_room(self) -> Awaitable[Any]: ...


class RoomHangup:
    """How the agent hangs up: it deletes the room, and the server drops every participant.

    This is the gate's CallControl, and what the end_call tool uses. The call is marked as ending
    first, so a gated request that is waiting its turn stops instead of running after the line has
    dropped. The wait is bounded: the gate calls this while it holds its lock, and a request that
    never answered would keep every later gated request waiting behind it.

    Hanging up cannot deadlock against the session closing. The tool that hangs up waits for one
    HTTP request, made by the job context, which finishes whatever the session does. The session's
    close waits for a running tool that can't be cancelled, with no limit of its own, so the tool
    has to be able to end: nothing the close does needs the gate's lock, and this wait is bounded.
    """

    def __init__(self, job: _RoomDeleter, state: CallState) -> None:
        self._job = job
        self._state = state

    async def end_call(self) -> None:
        self._state.closing = True
        async with asyncio.timeout(HANGUP_S):
            await self._job.delete_room()


async def hang_up(
    session: AgentSession[CallState], call: CallControl, closed: asyncio.Event
) -> None:
    """End a call whose caller has left: close the session, then delete the room.

    The session goes first, so that it commits and forwards the last turns of the call before the
    room goes and the job shuts down. It is closed by LiveKit's own background close, started with
    shutdown(drain=False), and what is bounded is the wait for its close event (`closed`, set when
    that fires), not the close. A close cut off partway leaves the session half closed: its close
    event never fires, the caller's last turn is never committed, the activity (STT, TTS, audio
    forwarding) and RoomIO stay open, and LiveKit's own close at the job's shutdown then returns
    at once, because the session no longer counts as started. So a close that outlasts
    SESSION_CLOSE_S goes on while the room is deleted, and the job's shutdown close waits for it.
    A session that has already closed is not closed again, and not waited for.

    Each step logs instead of raising, so a session that won't close can't keep the room from
    being deleted, and nothing gets out of the task this runs in. A step that fails is logged by
    its type, never its message, which can quote the caller. Each step is wrapped in a function of
    its own, so that one that raises when it is called is caught as well.
    """
    logger.info("the caller hung up, so the call is ending")

    async def close_session() -> None:
        if closed.is_set():
            return
        session.shutdown(drain=False)
        try:
            await asyncio.wait_for(closed.wait(), SESSION_CLOSE_S)
        except TimeoutError:
            logger.warning(
                "the session was still closing after %g s, so the room is deleted without it",
                SESSION_CLOSE_S,
            )

    async def delete_room() -> None:
        await call.end_call()

    await _step("shut the session down", close_session())
    await _step("delete the room", delete_room(), limit_s=HANGUP_S)


async def finish_call(
    state: CallState,
    forwarder: Forwarder,
    detector: DetectorClient,
    *,
    consumer: asyncio.Task[None] | None = None,
    session_closed: asyncio.Event | None = None,
    gate: Gate | None = None,
    ramp: RampClient | None = None,
) -> None:
    """The end of a call: send what is left, end the detector's session, close everything.

    Never raises, and every wait is bounded, so a slow or dead detector can't hold the job open.
    If the caller was never identified there is no session to send to or end, so those steps are
    skipped. `session_closed` is set when the AgentSession has finished closing. In 1.8.10 the job
    awaits session.aclose() (up to 60 s) before it runs a shutdown callback, so by now the last
    turns of the call have come out and the wait on it returns at once: it is a safety net, for a
    close that outlasted that bound and was left running. `consumer` is the task running
    forwarder.run(); it is cancelled if the forwarder won't stop in time.

    The gate stops before the detector's client closes, because the hold reports it still has out
    use that client. The Ramp client closes last, once nothing can reach it.
    """
    session_id = state.session_id
    if session_id is not None:
        await _wait_for_close(session_closed)
        await _step("drain the forwarder", forwarder.drain(DRAIN_S))
        await _step(
            "end the detector's session", detector.end_session(session_id), limit_s=END_SESSION_S
        )
    await _close_forwarder(forwarder, consumer)
    if gate is not None:
        await _step("stop the gate", gate.aclose(), limit_s=GATE_CLOSE_S)
    await _step("close the detector client", detector.aclose())
    if ramp is not None:
        await _step("close the Ramp client", ramp.aclose())


async def _wait_for_close(session_closed: asyncio.Event | None) -> None:
    if session_closed is None or session_closed.is_set():
        return
    try:
        await asyncio.wait_for(session_closed.wait(), CLOSE_WAIT_S)
    except TimeoutError:
        logger.warning(
            "the session was still closing after %g s, ending the forwarding anyway", CLOSE_WAIT_S
        )


async def _step(action: str, work: Awaitable[None], *, limit_s: float | None = None) -> None:
    """Do one step of the end of a call: for at most `limit_s`, and logging instead of raising."""
    try:
        async with asyncio.timeout(limit_s):
            await work
    except TimeoutError:
        logger.warning("could not %s: timed out", action)
    except DetectorError as exc:
        logger.warning("could not %s: %s", action, exc)  # a DetectorError never quotes a body
    except Exception as exc:
        log_failure(logger, action, exc)


async def _close_forwarder(forwarder: Forwarder, consumer: asyncio.Task[None] | None) -> None:
    try:
        await asyncio.wait_for(forwarder.aclose(), FORWARDER_CLOSE_S)
    except TimeoutError:
        logger.warning(
            "the forwarder was still posting after %g s, cancelling it", FORWARDER_CLOSE_S
        )
        if consumer is not None:
            consumer.cancel()
            await asyncio.wait({consumer}, timeout=FORWARDER_CLOSE_S)
    except Exception as exc:
        log_failure(logger, "close the forwarder", exc)
