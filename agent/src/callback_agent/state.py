"""What the agent knows about one call. The gate and the tools share one CallState per call."""

from collections import deque
from dataclasses import dataclass, field

from callback_shared.contract import Channel, RiskUpdate, UserSummary

from callback_agent.audio_segments import CallTiming, SegmentCapture
from callback_agent.turns import FinalSegment

AGENT_SPEECH_HISTORY = 32  # how many stretches of the agent speaking a call remembers


@dataclass
class SpeechInterval:
    """One stretch of the agent speaking, in Unix epoch ms. `end_ms` stays None while it goes on."""

    start_ms: int
    end_ms: int | None = None

    def covers(self, at_ms: int) -> bool:
        """True from the first millisecond of the speech up to, but not including, its end."""
        return self.start_ms <= at_ms and (self.end_ms is None or at_ms < self.end_ms)


@dataclass
class CallState:
    room_name: str
    caller_identity: str
    channel: Channel
    session_id: str | None = None  # the detector's session, once the caller is identified
    claimed_user: UserSummary | None = None
    detector_down: bool = False  # the detector failed before a session could be made
    # A deny happened: no tier 1 or 2 request goes through for the rest of the call.
    locked: bool = False
    closing: bool = False  # the call is ending
    # The caller's next turn answers a secret ask (a voice challenge or an enroll prompt), so its
    # runs of 4 to 6 digits are the challenge and are forwarded as spoken. Set just before the ask
    # and cleared right after it. The turn forwarding also clears it on the first user turn, so the
    # exemption covers one turn and no more.
    answer_exempt: bool = False
    # A callback is under way: up from CALLBACK_LINE, which tells the caller they'll be called
    # back, until the dial is over. The caller leaving is planned then, so it doesn't end the call.
    callback_in_progress: bool = False
    # The FINAL_TRANSCRIPTs TimedSttMixin heard that no user message has claimed yet. The session
    # wiring turns them into the user's turn when the message arrives.
    pending_segments: list[FinalSegment] = field(default_factory=list)
    # Where TimedSttMixin puts every frame of the caller's audio, with its time on the frame clock.
    # The session wiring sets it and cuts it into speech segments for the detector; None (a call
    # built without the wiring) means no audio is kept. On the state, and not on an agent, so that
    # every task's stt_node feeds the same capture.
    audio_capture: SegmentCapture | None = None
    # The conversation's timing so far (reply latencies, durations, barge-ins), which every speech
    # segment carries to the detector for its agent_behaviour signal. Numbers only.
    timing: CallTiming = field(default_factory=CallTiming)
    # The agent's recent speech, oldest first. A user turn is judged against it when its message
    # arrives: it began over the agent if its first word fell inside one of these.
    agent_speech: deque[SpeechInterval] = field(
        default_factory=lambda: deque(maxlen=AGENT_SPEECH_HISTORY)
    )
    # The detector's latest risk update, kept for the logs (the worker stores each risk packet).
    latest_risk: RiskUpdate | None = None
    # The open bills list_open_bills has told the LLM about on this call: bill id to the id of the
    # bill's vendor. release_bill_hold only takes an id that is a key here. hold_bill does not use
    # this: a hold needs no verification and a list does, so it names a bill by vendor and invoice.
    bills_seen: dict[str, str] = field(default_factory=dict)
