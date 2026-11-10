"""The agents. Every Agent and AgentTask in this codebase inherits TimedSttMixin."""

import functools
import logging
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from contextlib import aclosing, suppress
from time import time as wall_clock  # by name, so that a test can pin the clock the anchor reads
from typing import Any

from livekit import rtc
from livekit.agents import Agent, ModelSettings, RunContext, function_tool, stt
from livekit.agents.types import NOT_GIVEN
from livekit.agents.utils import is_given

from callback_agent import prompts
from callback_agent.audio_segments import SegmentCapture
from callback_agent.gate import CallControl
from callback_agent.safelog import log_failure
from callback_agent.toolbox import Toolbox
from callback_agent.turns import EpochAnchor, FinalSegment

logger = logging.getLogger(__name__)

FrameTap = Callable[[rtc.AudioFrame], None]

CAUSALITY_SLACK_S = 0.05  # a word may end this long after the transcript that holds it arrived
MIN_GROWTH_S = 0.001  # the offset has grown if it is more than this above the first frame's


class _SttClock:
    """The clock of one stt_node call: its anchor, and what the framework added to its times."""

    def __init__(self) -> None:
        self.anchor = EpochAnchor()
        # Seconds. Read off the stream at the first frame. None until then, and for good if the
        # stream can't be read, when the first word that carries an offset supplies it instead.
        self.framework_offset: float | None = None
        self.from_stream = False  # framework_offset was read off the stream: the primary estimate
        # Seconds the attempt feeding the stream had run when its audio began. It is in the growth
        # of the offset after a retry, but no audio and no Deepgram clock ran during it. It stays
        # in every later growth, which counts that attempt whole.
        self.waited_s = 0.0
        # The offset on the final that caused the last switch. The connection that final came on
        # follows it; the next connection is judged against framework_offset again.
        self.switched_at: float | None = None
        # Seconds of audio this call has seen: where the next frame starts on the anchor's clock.
        self.heard_s = 0.0


def _current_stream() -> Any | None:
    """The STT stream this stt_node call feeds, or None if it can't be found.

    Agent.default.stt_node publishes its stream as the pipeline's `_recognize_stream` before it
    starts pulling frames, and the pump that runs stt_node keeps the pipeline in a context variable.
    So at the first frame the variable leads to this call's own stream. Both names are private to
    livekit-agents 1.8 (uv.lock pins 1.8.10), which is why the callers guard the read.
    """
    from livekit.agents.voice.audio_recognition import _STTPipelineContextVar

    pipeline = _STTPipelineContextVar.get(None)
    return getattr(pipeline, "_recognize_stream", None)


def _stream_start_time_offset() -> float | None:
    """The stream's start_time_offset: what the plugin adds to its times, before a retry grows it.

    None if the stream can't be found.
    """
    stream = _current_stream()
    return None if stream is None else float(stream.start_time_offset)


def _stream_started_at() -> float | None:
    """When the attempt that feeds the stream began: its public `start_time`.

    The base class seeds it when the stream is made and again at the start of every retry (and the
    Deepgram plugin leaves it alone). None if the stream can't be found.
    """
    stream = _current_stream()
    return None if stream is None else float(stream.start_time)


def _framework_offset(data: stt.SpeechData) -> float | None:
    """The offset the words of `data` carry, if they say. Each TimedString names it.

    This is the offset at the time of the event, so it already includes any growth. A stream that
    could not be read takes it from the first word, and then a reconnect before that word puts
    every later word early by the age of the first connection. A final that causes a switch is
    placed by the offset it carries, which removes the growth of a connection that failed while
    connecting.
    """
    for w in data.words or ():
        offset = getattr(w, "start_time_offset", NOT_GIVEN)
        if is_given(offset):
            return float(offset)
    return None


def _spoken_until(data: stt.SpeechData, offset: float) -> float:
    """When the last word of `data` ends, on the stream's clock with `offset` taken out."""
    words = data.words or ()
    return (max(float(w.end_time) for w in words) if words else data.end_time) - offset


def _offset_to_take_out(clock: _SttClock, data: stt.SpeechData, arrived_at: float) -> float:
    """The offset to take out of the times of a final that arrived at `arrived_at`.

    It is the offset read at the first frame. Once the offset has grown, the growth is mostly time
    the audio went through, so it stays in, except for the time the stream waited for its first
    frame, which is in the growth but was never on Deepgram's clock. If even that is impossible, a
    word ends after the transcript that holds it arrived, the growth was not audio (a connection
    that failed while connecting), and the stream switches: this final is placed by the offset it
    carries, which takes all the growth out, and so is every later final of the same connection.

    The switch ends with that connection. It first caught up on the frames that waited for it, so
    by the time the next connection starts the dead time the switch took out is audio, and the
    next connection's finals are judged again against the first frame's offset.
    """
    base = clock.framework_offset or 0.0
    live = _framework_offset(data)
    if not clock.from_stream or live is None or clock.anchor.t0 is None:
        return base
    if clock.switched_at is not None and live - clock.switched_at <= MIN_GROWTH_S:
        return clock.switched_at
    grown = live - base
    if grown <= MIN_GROWTH_S:
        return base
    estimate = base + min(clock.waited_s, grown)
    if clock.anchor.t0 + _spoken_until(data, estimate) <= arrived_at + CAUSALITY_SLACK_S:
        return estimate
    clock.switched_at = live
    logger.warning(
        "a word ended after its transcript arrived with the STT stream's first-frame offset, "
        "which has since grown by %.3f s; the times on this connection now follow the offset on "
        "its transcripts",
        grown,
    )
    return live


class TimedSttMixin:
    """Taps every audio frame on its way to the STT stream, and keeps what the stream hears.

    Every Agent and AgentTask in this codebase inherits this. A task gets its own stt_node, and
    LiveKit shares one STT stream across a handoff only when both agents use the default node, so
    an agent without the mixin would run its stretch of the call untapped.

    The frame tap feeds the HD-voice probe. The same pass gives every frame, with its start on the
    anchor's clock, to the call's audio capture (CallState.audio_capture, which the session wiring
    cuts into speech segments for the detector), and places each FINAL_TRANSCRIPT in epoch time.
    Flux word times exist only inside stt_node (the session's events carry plain text), so the
    mixin turns every final into a FinalSegment on the call's CallState, where the session wiring
    picks it up when the user's message arrives. The LiveKit notes (Q3) explain the rule.

    Epoch ms is "STT stream start + Deepgram word offset" (the contract). The stream starts with
    the first frame this stt_node call sees, so every call has an anchor of its own: a handoff
    starts a new stream with a new Deepgram clock. Agent.default.stt_node gives its stream a
    start_time_offset (the time since the audio input started), which the Deepgram plugin adds to
    every time it reports. The mixin reads that offset off the stream at the first frame and takes
    it out, or every word would run late by it.

    The offset grows when the plugin reconnects: by the old attempt's lifetime and the wait before
    the retry (0.1 s the first time, 2 s after that). Not all of that is time the audio went
    through. The base class dates each attempt from its own start, so the lifetime holds the time
    the stream waited for its first frame, when no audio and no Deepgram clock ran: at the start of
    a call the stream is made inside session.start() and the audio is forwarded only once start()
    has returned. The mixin measures that wait at the first frame, from RecognizeStream.start_time
    (seeded when the stream is made and again at each retry) to the first sample of audio, and
    leaves it out of the growth.

    What is left depends on how the old attempt ended. One that dropped after taking audio leaves
    the new connection to start where it stopped, so what is left is that audio and the wait before
    the retry, and words are late by about that wait: 0.1 s after the first retry. One that failed
    while connecting took no frame (the plugin connects before it sends), so the new connection
    starts at the first frame, and what is left is dead time that would put every word late by it,
    up to the 10 s connect timeout.

    The two are told apart by causality. When, with the wait left out, the last word of a final
    ends more than 50 ms after that final arrived, what is left cannot be audio, and the stream
    switches: that final, and the rest of the connection it came on, are placed by the offset they
    carry, which takes all the growth out. The switch ends with that connection. It first caught
    up on the frames that waited for it, so when the next one starts the dead time is audio, and
    the next one's finals are judged again against the first frame's offset: a drop that carries
    audio stays in, with the words late by about the wait before the retry, and a failed connect
    is removed again. The check cannot see dead time, beyond the wait, that fits inside the STT's
    final latency plus 50 ms, so a connect failure puts words late by at most that, and by nothing
    once it is seen.

    The price: the 2 s wait of a second reconnect in a row can look like dead time on a stream
    whose first connection did carry audio, and the switch then takes that audio out too, so the
    words of that one connection are early by what the first carried. The next connection is
    judged again.

    If the stream can't be read, the offset on the first word that carries one is used instead
    (there is no switch to make), and a reconnect before that word puts every later word early by
    the age of the first connection.
    """

    def __init__(self, *, frame_tap: FrameTap | None = None, **kwargs: Any) -> None:
        self._frame_tap = frame_tap
        super().__init__(**kwargs)

    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> AsyncIterator[stt.SpeechEvent | str]:
        clock = _SttClock()

        async def tapped() -> AsyncIterator[rtc.AudioFrame]:
            first = True
            capture: SegmentCapture | None = None
            async for frame in audio:
                if first:
                    first = False
                    self._anchor(clock, frame)
                    self._read_offset(clock)
                    capture = self._capture()
                self._tap(frame)
                if capture is not None:
                    capture = self._feed(capture, clock, frame)
                yield frame

        async with aclosing(Agent.default.stt_node(self, tapped(), model_settings)) as events:
            async for event in events:
                self._record_final(clock, event)
                yield event

    def _tap(self, frame: rtc.AudioFrame) -> None:
        if self._frame_tap is None:
            return
        try:
            self._frame_tap(frame)
        except Exception:
            # The tap sits in the audio path. If it breaks, give it up rather than deafen the call.
            logger.exception("frame tap failed, dropping it so the audio keeps flowing")
            self._frame_tap = None

    def _capture(self) -> SegmentCapture | None:
        """The call's audio capture, read once per stt_node call. None if there is none."""
        try:
            return self.session.userdata.audio_capture
        except Exception as exc:
            log_failure(logger, "find the call's audio capture, so this stretch is not heard", exc)
            return None

    def _feed(
        self, capture: SegmentCapture, clock: _SttClock, frame: rtc.AudioFrame
    ) -> SegmentCapture | None:
        """Give the capture one frame, with its start on the anchor's clock.

        A frame starts where the audio heard before it ended, so the time is the anchor plus the
        audio heard so far, whatever the wall clock says: the same clock the turns' words are on.
        Returns the capture to keep feeding, or None once it has failed: it sits in the audio
        path, and a capture that breaks is given up rather than deafen the call.
        """
        try:
            duration = frame.duration
            t0 = clock.anchor.t0
            start_s = (wall_clock() - duration) if t0 is None else t0 + clock.heard_s
            clock.heard_s += duration
            capture.add_frame(frame, start_s)
            return capture
        except Exception as exc:
            log_failure(logger, "keep a frame of the caller's audio, giving the capture up", exc)
            return None

    def _anchor(self, clock: _SttClock, first_frame: rtc.AudioFrame) -> None:
        try:
            clock.anchor.observe_frame(first_frame.duration, wall_clock())
        except Exception as exc:
            # Same rule as the tap. This call's segments fall back to the time their event was made.
            log_failure(logger, "anchor the STT clock on the first frame", exc)

    def _read_offset(self, clock: _SttClock) -> None:
        """Read the stream's offset into the clock. If that fails, the first word's will do."""
        try:
            offset = _stream_start_time_offset()
        except Exception as exc:
            log_failure(logger, "read the STT stream's offset, using the first word's", exc)
            return
        if offset is None:
            logger.warning(
                "the STT stream's offset is not readable here, so the first word's is used"
            )
        else:
            clock.framework_offset, clock.from_stream = offset, True
            clock.waited_s = self._wait_for_first_frame(clock)

    def _wait_for_first_frame(self, clock: _SttClock) -> float:
        """How long the attempt feeding the stream had run when the stream's audio began.

        Zero if that can't be read, which leaves the wait in the growth, as it was before it was
        measured: the cost is a switch that may come too early, not a stream that stops.
        """
        try:
            began = _stream_started_at()
        except Exception as exc:
            log_failure(logger, "read when the STT stream's attempt began, counting no wait", exc)
            return 0.0
        if began is None or clock.anchor.t0 is None:
            return 0.0
        return max(0.0, clock.anchor.t0 - began)

    def _record_final(self, clock: _SttClock, event: stt.SpeechEvent | str) -> None:
        """Keep a FINAL_TRANSCRIPT as a FinalSegment on the call's CallState.

        It sits in the STT path, so it never raises: a segment that can't be recorded is logged and
        dropped, and the user's turn then goes out as text alone.
        """
        try:
            if not isinstance(event, stt.SpeechEvent) or not event.alternatives:
                return
            data = event.alternatives[0]
            if clock.framework_offset is None:
                clock.framework_offset = _framework_offset(data)  # any event with words will do
            if event.type != stt.SpeechEventType.FINAL_TRANSCRIPT or not data.text:
                return
            offset = _offset_to_take_out(clock, data, arrived_at=wall_clock())
            start_s, end_s = data.start_time - offset, data.end_time - offset
            words = tuple(
                (str(w), float(w.start_time) - offset, float(w.end_time) - offset)
                for w in data.words or ()
            )
            t0 = clock.anchor.t0
            if t0 is None:
                # No frame has been seen. The notes' fallback ends the segment at created_at.
                t0 = event.created_at - end_s
            segment = FinalSegment(text=data.text, start_s=start_s, end_s=end_s, words=words, t0=t0)
            self.session.userdata.pending_segments.append(segment)
        except Exception as exc:
            log_failure(logger, "record a final transcript, dropping it", exc)


def _hold_the_floor(ctx: RunContext) -> bool:
    """Stop a barge-in from cancelling this tool halfway. False if the reply was cut off already."""
    try:
        ctx.disallow_interruptions()
    except RuntimeError:
        return False
    return True


def _tool(method: Callable[..., Awaitable[str]]) -> Callable[..., Awaitable[str]]:
    """Make a method the body of a tool: it can't be interrupted, and it always answers with text.

    When the caller talks over the reply that called a tool, LiveKit stops waiting for the tool and
    lets it run on in the background, and the caller's new turn gets a reply of its own. For a
    request in flight at the gate that means a second reply starting mid-verification, and the
    answer arriving later as an afterthought. So the first thing a tool does is refuse
    interruptions. If the reply was already cut off, nothing runs: the caller has moved on. Whatever
    else goes wrong becomes a reply, logged by type and place (its message could quote a number).
    Cancellation, which LiveKit uses to sweep a tool left over when the session closes, still gets
    through.
    """

    @functools.wraps(method)
    async def run(self: "CallbackAgent", ctx: RunContext, *args: Any, **kwargs: Any) -> str:
        if not _hold_the_floor(ctx):
            return prompts.INTERRUPTED
        try:
            return await method(self, ctx, *args, **kwargs)
        except Exception as exc:
            log_failure(logger, f"run the {method.__name__} tool", exc)
            return prompts.TOOL_FAILED

    return run


class CallbackAgent(TimedSttMixin, Agent):
    """The phone agent. Its tools are thin delegates: one per Toolbox method, plus end_call.

    The docstring of a tool is the LLM's only description of it. The agent holds no Ramp client and
    no gate: everything a tool does is done by the toolbox, which asks the gate. LiveKit runs the
    tool calls of one reply side by side, so the tools that identify, write or enroll
    (identify_caller, the bank change, the two releases and start_voice_enrollment) are marked
    on_duplicate="reject": a second call of one, while the first is still running, is turned away
    instead of queuing on the gate's lock behind it and being allowed and written (or enrolled) as
    well. Holds and reads may repeat. `call` hangs up for end_call. Without one the session is
    shut down instead, which doesn't delete the room, so a phone caller would stay on the line:
    the worker always passes one.
    """

    def __init__(
        self,
        *,
        toolbox: Toolbox,
        call: CallControl | None = None,
        frame_tap: FrameTap | None = None,
    ) -> None:
        super().__init__(instructions=prompts.AGENT_INSTRUCTIONS, frame_tap=frame_tap)
        self._toolbox = toolbox
        self._call = call

    async def on_enter(self) -> None:
        # The recording disclosure goes first, and nothing can talk over it.
        await self.session.say(prompts.DISCLOSURE, allow_interruptions=False)
        self.session.generate_reply(instructions=prompts.ASK_WHO)

    @function_tool(on_duplicate="reject")
    @_tool
    async def identify_caller(self, ctx: RunContext, full_name: str) -> str:
        """Look up who is calling by their full name. Call this as soon as the caller gives their
        full name, before you discuss the account.

        Args:
            full_name: The caller's full name, first and last, as they said it.
        """
        return await self._toolbox.identify_caller(full_name=full_name)

    @function_tool()
    @_tool
    async def list_vendors(self, ctx: RunContext) -> str:
        """List the vendors in the company's Ramp account."""
        return await self._toolbox.list_vendors()

    @function_tool()
    @_tool
    async def get_vendor_bank_account(self, ctx: RunContext, vendor_name: str) -> str:
        """Read the bank account a vendor is paid to. The result gives the last four digits only.

        Args:
            vendor_name: The vendor's name, as the caller said it.
        """
        return await self._toolbox.get_vendor_bank_account(vendor_name=vendor_name)

    @function_tool()
    @_tool
    async def list_open_bills(self, ctx: RunContext, vendor_name: str) -> str:
        """List a vendor's open bills. Each bill in the result has a bill id, which
        release_bill_hold needs. Never read a bill id aloud: name the bill by its invoice number
        and amount.

        Args:
            vendor_name: The vendor's name, as the caller said it.
        """
        return await self._toolbox.list_open_bills(vendor_name=vendor_name)

    @function_tool()
    @_tool
    async def hold_vendor(self, ctx: RunContext, vendor_name: str, reason: str) -> str:
        """Put a vendor on hold right away, so that no payments go to it. Holds never need
        verification, so do this as soon as the caller asks.

        Args:
            vendor_name: The vendor's name, as the caller said it.
            reason: Why the vendor is being put on hold, in a few words, such as "caller asked".
                Leave account numbers out of it.
        """
        return await self._toolbox.hold_vendor(vendor_name=vendor_name, reason=reason)

    @function_tool()
    @_tool
    async def hold_bill(
        self, ctx: RunContext, vendor_name: str, invoice_number: str, reason: str
    ) -> str:
        """Put one bill on hold right away, named by its vendor and its invoice number. Holds never
        need verification, and there is no need to list the bills first.

        Args:
            vendor_name: The name of the vendor the bill is from, as the caller said it.
            invoice_number: The bill's invoice number, as the caller said it, such as INV-1001.
            reason: Why the bill is being put on hold, in a few words, such as "caller asked".
                Leave account numbers out of it.
        """
        return await self._toolbox.hold_bill(
            vendor_name=vendor_name, invoice_number=invoice_number, reason=reason
        )

    @function_tool(on_duplicate="reject")
    @_tool
    async def change_vendor_bank_account(
        self, ctx: RunContext, vendor_name: str, routing_number: str, account_number: str
    ) -> str:
        """Change the bank account a vendor is paid to. Call it once, and only after you have read
        back the last four digits of the routing number and of the account number and the caller
        said yes. Never read the full numbers aloud.

        Args:
            vendor_name: The vendor's name, as the caller said it.
            routing_number: The 9-digit routing number, digits only, keeping any leading zeros.
            account_number: The account number, digits only, keeping any leading zeros.
        """
        return await self._toolbox.change_vendor_bank_account(
            vendor_name=vendor_name, routing_number=routing_number, account_number=account_number
        )

    @function_tool(on_duplicate="reject")
    @_tool
    async def release_vendor_hold(self, ctx: RunContext, vendor_name: str) -> str:
        """Take a vendor off hold, so that payments can go to it again.

        Args:
            vendor_name: The vendor's name, as the caller said it.
        """
        return await self._toolbox.release_vendor_hold(vendor_name=vendor_name)

    @function_tool(on_duplicate="reject")
    @_tool
    async def release_bill_hold(self, ctx: RunContext, bill_id: str) -> str:
        """Take one bill off hold. The bill must be one from a list_open_bills result on this
        call.

        Args:
            bill_id: The bill id from the list_open_bills result. Never read it aloud.
        """
        return await self._toolbox.release_bill_hold(bill_id=bill_id)

    @function_tool(on_duplicate="reject")
    @_tool
    async def start_voice_enrollment(self, ctx: RunContext) -> str:
        """Enroll the caller's voice when they ask to set up voice verification."""
        return await self._toolbox.start_voice_enrollment()

    @function_tool()
    @_tool
    async def end_call(self, ctx: RunContext) -> str:
        """Hang up the phone call. Say goodbye first, then call this once the caller has nothing
        more to ask."""
        # The goodbye has to finish playing before the line drops. With no reply there is nothing
        # to wait for, and that raises RuntimeError.
        with suppress(RuntimeError):
            await ctx.wait_for_playout()
        if self._call is not None:
            await self._call.end_call()
        else:
            ctx.session.shutdown()
        return prompts.CALL_ENDING
