"""Turns and speech events on their way to the detection service.

The builders turn what LiveKit hands the agent into the contract's models, with every time in Unix
epoch ms. Forwarder carries the results to the detector from a background task, so that a slow or
broken detector can never hold up the conversation. Where each time comes from, and why the epoch
anchor belongs to one stt_node call, is in the LiveKit notes (docs/research_notes/Callback agent
build research/livekit_agents_framework.md, Q3 and Q4). The D-numbers point at the decision table
in docs/superpowers/plans/2026-11-05-voice-agent.md.

Logs name an item by its kind and time, never by its text: a turn can hold a spoken code.
"""

import asyncio
import logging
from asyncio import sleep  # imported by name so that a test can replace the wait before a retry
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from callback_shared.contract import Role, SpeechEvent, SpeechEventType, Turn, Word

from callback_agent.audio_segments import AudioSegment
from callback_agent.detector import DetectorClient, DetectorError

logger = logging.getLogger(__name__)

RETRY_DELAY_S = 0.2  # a post that fails is tried once more after this long, then dropped
# How much of the caller's audio waits in the buffer for a session. Older segments go first, and
# the count of what went is logged when the forwarder closes. 60 s is a minute of speech at the
# start of a call, before identify_caller: more than any caller says before giving a name.
MAX_BUFFERED_AUDIO_S = 60.0

Item = Turn | SpeechEvent | AudioSegment


def _epoch_ms(epoch_s: float) -> int:
    return round(epoch_s * 1000)


class EpochAnchor:
    """Turns seconds on one STT stream's clock into Unix epoch ms.

    The stream's clock starts at the first audio frame to enter it. The framework's own rule for
    when that was is `time.time() - frame.duration`, because a frame has already been captured when
    it arrives. Deepgram restarts its clock whenever an agent or task handoff starts a new stt_node,
    so every stt_node call needs an anchor of its own: one anchor for the whole call would stamp
    every later agent's words with the wrong time.
    """

    def __init__(self) -> None:
        self.t0: float | None = None  # Unix seconds at stream time 0; None until the first frame

    def observe_frame(self, frame_duration_s: float, now_s: float) -> None:
        """Anchor on the first frame, which arrived at `now_s`. Later frames change nothing."""
        if self.t0 is None:
            self.t0 = now_s - frame_duration_s

    def to_ms(self, rel_s: float) -> int:
        """Unix epoch ms of the moment `rel_s` seconds into the stream."""
        if self.t0 is None:
            raise ValueError("no audio frame has reached this anchor yet, so it has no t0")
        return _epoch_ms(self.t0 + rel_s)


@dataclass(frozen=True)
class FinalSegment:
    """One FINAL_TRANSCRIPT out of stt_node: a stretch of the caller's speech.

    Times are seconds on the stream's clock, `words` holds (word, start_s, end_s), and `t0` is the
    anchor of the stt_node call that produced the segment.
    """

    text: str
    start_s: float
    end_s: float
    words: tuple[tuple[str, float, float], ...]
    t0: float


def build_user_turn(segments: Sequence[FinalSegment], *, interrupted: bool) -> Turn:
    """The caller's turn, from the FINAL_TRANSCRIPTs that made it up (usually just one).

    Each segment is placed with its own t0, because a handoff in the middle of a turn starts a new
    STT clock.
    """
    if not segments:
        raise ValueError("a user turn needs at least one segment")
    first, last = segments[0], segments[-1]
    return Turn(
        role=Role.USER,
        text=" ".join(s.text for s in segments),
        start_ms=_epoch_ms(first.t0 + first.start_s),
        end_ms=_epoch_ms(last.t0 + last.end_s),
        words=[
            Word(w=word, start_ms=_epoch_ms(s.t0 + start), end_ms=_epoch_ms(s.t0 + end))
            for s in segments
            for word, start, end in s.words
        ],
        interrupted=interrupted,
    )


def build_agent_turn(
    text: str, started_at_s: float, stopped_at_s: float, *, interrupted: bool
) -> Turn:
    """The agent's turn, from its message and the playout bounds in the message's metrics.

    Those bounds are Unix seconds already. The agent's words carry no times (D7), so `words` stays
    empty.
    """
    return Turn(
        role=Role.AGENT,
        text=text,
        start_ms=_epoch_ms(started_at_s),
        end_ms=_epoch_ms(stopped_at_s),
        words=[],
        interrupted=interrupted,
    )


def speech_event(kind: SpeechEventType, created_at_s: float) -> SpeechEvent:
    """A speech event stamped with the `created_at` (Unix seconds) of the session event."""
    return SpeechEvent(type=kind, ts_ms=_epoch_ms(created_at_s))


def _describe(item: Item) -> str:
    """What an item is, for a log line: its kind and time, never its text or its sound."""
    if isinstance(item, Turn):
        return f"{item.role.value} turn at {item.start_ms} ms"
    if isinstance(item, AudioSegment):
        return str(item)
    return f"{item.type.value} event at {item.ts_ms} ms"


class Forwarder:
    """Sends finished turns, speech events and speech segments to the detector without ever
    holding up the call.

    submit() only appends to a bounded buffer, so any event handler can call it: it never blocks and
    never raises, whatever state the detector is in. One background task, run(), posts the buffered
    items one at a time, in the order they were submitted. Nothing goes out before attach(): the
    detector's session exists only once the caller is identified (D9), so earlier items wait in the
    buffer and are flushed then. A full buffer drops its oldest item, and the buffer holds at most
    MAX_BUFFERED_AUDIO_S of audio, dropping the oldest segments first.

    Audio fails quietly. The deployed detector may refuse /events (mock mode, a session it does not
    know), and that would repeat on every segment, so the first refusal of a call is one WARNING
    with the status and the rest are counted; the counts are logged once, when the forwarder
    closes. A segment is never retried: a refusal is deterministic, and the next segment is a
    better use of the time. The turns' retry stays as it was.
    """

    def __init__(self, detector: DetectorClient, *, max_buffer: int = 200) -> None:
        if max_buffer < 1:
            raise ValueError("max_buffer must be at least 1")
        self._detector = detector
        self._buffer: deque[Item] = deque(maxlen=max_buffer)
        self._session_id: str | None = None
        self._claimed_user_id: str | None = None
        self._audio_sequence = 0  # /events wants a strictly increasing sequence per session
        self._audio_posted = 0
        self._audio_failed = 0
        self._audio_dropped = 0  # unsent: over the budget, or still buffered at close
        self._audio_warned = False
        self._closed = False
        self._wake = asyncio.Event()  # rung when run() may have work to do, or has to stop
        self._idle = asyncio.Event()  # set while nothing is buffered and nothing is being posted
        self._idle.set()
        self._stopped = asyncio.Event()  # set while run() is not running
        self._stopped.set()

    def attach(self, session_id: str, claimed_user_id: str | None = None) -> None:
        """Name the detector's session, and the user the caller claims to be, which goes with
        every speech segment. The buffered items go out, then every later one."""
        self._session_id = session_id
        self._claimed_user_id = claimed_user_id
        self._wake.set()

    def submit(self, item: Item) -> None:
        """Queue an item for the detector. Never blocks, never raises."""
        if self._closed:
            return
        if len(self._buffer) == self._buffer.maxlen:
            oldest = self._buffer[0]
            logger.warning("forwarder buffer is full: dropping the %s", _describe(oldest))
            if isinstance(oldest, AudioSegment):
                self._audio_dropped += 1
        self._buffer.append(item)
        if isinstance(item, AudioSegment):
            self._keep_audio_within_budget()
        self._idle.clear()
        self._wake.set()

    def _keep_audio_within_budget(self) -> None:
        """Drop the oldest buffered segments until at most MAX_BUFFERED_AUDIO_S of audio waits."""
        budget_ms = MAX_BUFFERED_AUDIO_S * 1000
        audio = [i for i in self._buffer if isinstance(i, AudioSegment)]
        held_ms = sum(s.duration_ms for s in audio)
        for oldest in audio:
            if held_ms <= budget_ms or len(audio) == 1:
                break
            self._buffer.remove(oldest)
            held_ms -= oldest.duration_ms
            self._audio_dropped += 1

    async def run(self) -> None:
        """The single consumer: start it once, as a task, with the call. It ends at aclose()."""
        self._stopped.clear()
        try:
            while not self._closed:
                if self._session_id is None or not self._buffer:
                    self._wake.clear()
                    await self._wake.wait()
                    continue
                item = self._buffer.popleft()  # in flight now, so a full buffer can't drop it
                await self._deliver(self._session_id, item)
                if not self._buffer:
                    self._idle.set()
        finally:
            self._stopped.set()

    async def drain(self, timeout_s: float = 2.0) -> None:
        """Wait up to timeout_s for everything submitted to be posted or dropped.

        Meant for the end of a call. Running out of time is logged, not raised: what is still
        unsent stays behind.
        """
        try:
            async with asyncio.timeout(timeout_s):
                await self._idle.wait()
        except TimeoutError:
            logger.warning(
                "drain gave up after %g s: the detector is behind (%d still buffered)",
                timeout_s,
                len(self._buffer),
            )

    async def aclose(self) -> None:
        """Stop run(): drop what is still buffered and wait for the post in flight, if any.

        After this the forwarder ignores submit(). Once run() has returned, the detector is no
        longer called, so it is safe to close it.
        """
        self._closed = True
        self._audio_dropped += sum(isinstance(i, AudioSegment) for i in self._buffer)
        self._buffer.clear()
        self._idle.set()
        self._wake.set()
        await self._stopped.wait()
        if self._audio_posted or self._audio_failed or self._audio_dropped:
            logger.info(
                "speech segments this call: %d posted, %d refused or failed, %d dropped unsent",
                self._audio_posted,
                self._audio_failed,
                self._audio_dropped,
            )

    async def _deliver(self, session_id: str, item: Item) -> None:
        """Post one item. A DetectorError earns one more try after RETRY_DELAY_S, then a drop."""
        if isinstance(item, AudioSegment):
            await self._deliver_audio(session_id, item)
            return
        what = _describe(item)
        for attempt in (1, 2):
            try:
                if isinstance(item, Turn):
                    await self._detector.post_turn(session_id, item)
                else:
                    await self._detector.post_event(session_id, item)
                return
            except DetectorError as exc:
                if attempt == 1:
                    logger.warning("could not forward the %s, retrying once: %s", what, exc)
                    await sleep(RETRY_DELAY_S)
                else:
                    logger.error("dropping the %s, it failed twice: %s", what, exc)
            except Exception:
                # A bug rather than an outage, so a second try would fail the same way.
                logger.exception("dropping the %s after an unexpected error", what)
                return

    async def _deliver_audio(self, session_id: str, segment: AudioSegment) -> None:
        """Post one speech segment, once. The first failure of the call is logged, the rest
        counted. The sequence counts every attempt, so the detector never sees one twice."""
        sequence = self._audio_sequence
        self._audio_sequence += 1
        try:
            await self._detector.post_speech_segment(
                session_id, segment, sequence=sequence, claimed_user_id=self._claimed_user_id
            )
        except Exception as exc:
            self._audio_failed += 1
            if not self._audio_warned:
                self._audio_warned = True
                # A DetectorError names the call and the status, never the body. Anything else is
                # a bug: its type is enough, as its message could say what was being handled.
                why = str(exc) if isinstance(exc, DetectorError) else type(exc).__name__
                logger.warning(
                    "%s refused by the detector (%s); the call goes on, later refusals are counted",
                    _describe(segment),
                    why,
                )
            return
        self._audio_posted += 1
