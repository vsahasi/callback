"""The LiveKit worker: one AgentSession per call.

The settings are read when this module is imported, because the rtc_session decorator needs the
agent name, and the detector, Ramp and voice settings are all required. agent.py loads the
repo-root .env before it imports this. Tests do not import this module, so what a call is built
from lives in session_wiring.py, where they can reach it, and this module only puts it together.
"""

import asyncio
import logging
import os

from livekit import rtc
from livekit.agents import AgentServer, AgentSession, JobContext, JobProcess, TurnHandlingOptions
from livekit.plugins import cartesia, deepgram, openai, silero

from callback_agent.agents import CallbackAgent, FrameTap
from callback_agent.audio_check import BandwidthProbe
from callback_agent.config import Settings
from callback_agent.safelog import install_pii_filter
from callback_agent.session_wiring import (
    TOOL_HANDLING,
    RoomHangup,
    build_callback_leg,
    build_forwarding,
    build_ramp,
    build_room_options,
    build_tools,
    call_state_for,
    finish_call,
    log_demo_scenario,
    turn_logging_enabled,
    wire_hangups,
    wire_risk_updates,
    wire_turn_forwarding,
)
from callback_agent.state import CallState

logger = logging.getLogger(__name__)

settings = Settings.from_env(os.environ)
server = AgentServer()
server.on("worker_started", lambda: log_demo_scenario(settings))


def prewarm(proc: JobProcess) -> None:
    # LiveKit logs transcripts and tool arguments as lk.pii.* extras, and its formatters print them.
    # A job process has added its log handler by now, so this is where the filter can go on it.
    install_pii_filter()
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


def bandwidth_tap(participant: rtc.RemoteParticipant) -> FrameTap:
    """A frame tap that measures the caller's audio and logs one bandwidth verdict per call."""
    probe: BandwidthProbe | None = None
    logged = False

    def tap(frame: rtc.AudioFrame) -> None:
        nonlocal probe, logged
        if logged:
            return
        if probe is None:
            probe = BandwidthProbe(frame.sample_rate)
        probe.add(frame)
        result = probe.verdict()
        if result is None:
            return
        logged = True
        # The caller's number is personal data: the log keeps its last four digits and nothing else.
        phone = participant.attributes.get("sip.phoneNumber") or ""
        logger.info(
            "bandwidth probe: %s, high band ratio %.3g",
            result["verdict"],
            result["high_band_ratio"],
            extra={
                "participant_kind": rtc.ParticipantKind.Name(participant.kind),
                "phone_last4": phone[-4:] or None,
            },
        )

    return tap


@server.rtc_session(agent_name=settings.agent_name)
async def entrypoint(ctx: JobContext) -> None:
    install_pii_filter()  # again: it is safe to repeat, and it covers handlers added since prewarm
    # Wait for the caller first, so the disclosure never plays to an empty room.
    participant = await ctx.wait_for_participant()
    state = call_state_for(ctx.room.name, participant)
    forwarding = build_forwarding(settings)
    if forwarding is None:
        # Settings.from_env requires both detector settings, so this is a bug and not a setting.
        raise RuntimeError("DETECTOR_URL and DETECTOR_TOKEN are required, and are not set")
    detector, forwarder = forwarding
    ramp = build_ramp(settings)
    session = AgentSession[CallState](
        stt=deepgram.STTv2(model="flux-general-en"),
        llm=openai.LLM(model="gpt-4.1"),
        tts=cartesia.TTS(model="sonic-3", voice=settings.cartesia_voice_id),
        vad=ctx.proc.userdata["vad"],
        # Flux decides when a turn ends. Preemptive generation stays off so that no tool can fire
        # on a half-finished turn.
        turn_handling=TurnHandlingOptions(
            turn_detection="stt", preemptive_generation={"enabled": False}
        ),
        tool_handling=TOOL_HANDLING,
        userdata=state,
    )
    # Nothing goes out to the detector until identify_caller attaches a session: until then the
    # forwarder only buffers (D9). This also sets state.closing when the session closes.
    wire_turn_forwarding(session, state, forwarder, log_turns=turn_logging_enabled(os.environ))
    wire_risk_updates(ctx.room, state)
    # Every request a tool makes goes through the gate. Its lines are said through tasks
    # (LiveKitVoice), the step-ups the detector asks for are real, and a deny hangs up by deleting
    # the room. The enrollment tool runs inside the gate's lock. All of it is built in
    # build_tools, where the tests can reach it. So is the callback to the number on file, with
    # a leg that exists only when the trunk and the Telnyx number are set: without one, that
    # step-up fails closed.
    call = RoomHangup(ctx, state)
    # The session is started with close_on_disconnect off, so that a callback can hang up on the
    # caller without ending the call. That leaves ending it, when the caller hangs up, to this:
    # it marks the call as ending, closes the session and deletes the room. Wired before the
    # session starts, so that a caller who hangs up while it starts is not missed.
    wire_hangups(ctx.room, session, state, call)
    callback_leg = build_callback_leg(ctx, session, state, settings)
    tools = build_tools(
        session,
        state,
        detector=detector,
        forwarder=forwarder,
        ramp=ramp,
        call=call,
        callback_leg=callback_leg,
    )
    # In livekit-agents 1.8.10 the job's shutdown awaits session.aclose() (up to 60 s) before it
    # runs any shutdown callback, so this one starts once the session has closed, or been given up
    # on, and its wait for the close is a safety net. It then sends the last turns, stops the gate
    # and closes both HTTP clients. The parent's 10 s shutdown_process_timeout starts only when the
    # job reports that it is shutting down, which comes after that close.
    session_closed = asyncio.Event()
    session.on("close", lambda _: session_closed.set())
    consumer = asyncio.create_task(forwarder.run(), name="turn forwarder")

    async def end_forwarding() -> None:
        await finish_call(
            state,
            forwarder,
            detector,
            consumer=consumer,
            session_closed=session_closed,
            gate=tools.gate,
            ramp=ramp,
        )

    ctx.add_shutdown_callback(end_forwarding)
    await session.start(
        agent=CallbackAgent(toolbox=tools.toolbox, call=call, frame_tap=bandwidth_tap(participant)),
        room=ctx.room,
        # Linked to the caller who joined, with the chat topic closed to typed input.
        room_options=build_room_options(participant.identity),
        # The other way in for typed input: unless told not to, LiveKit hosts the session on the
        # lk.agent.session topic, where a run_input request from any participant becomes a user
        # turn that would answer a pending ask (an SMS code, the consent). With both closed,
        # every answer is spoken, where the detector can hear the voice.
        session_host=False,
        # Recording is fully off: transcripts hold spoken SMS codes and account numbers.
        record=False,
    )
