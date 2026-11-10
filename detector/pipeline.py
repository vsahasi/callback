"""From an AudioRef on an event to a list of Factors.

This is the join between `detector.audio` (fetch and decode, I/O) and
`detector.analysis` (measure, CPU). It lives outside `detector.state` on
purpose. The state machine runs inside the store's read-modify-write, which
the store may replay against fresher state if it loses a commit race, so
everything in there has to be synchronous, deterministic and cheap. A network
fetch is none of those. The route does this first, then hands the finished
factors to a state machine that stays exactly as deterministic as it was.

The CPU half runs in a worker thread. The DSP is numpy over a few seconds of
audio, which is milliseconds rather than seconds, but it is still a blocking
loop and the event loop is also serving every other call on the container.

Failure is not an error
-----------------------
A span that will not fetch or will not decode does not fail the request. The
call is live and a 422 in the middle of it helps nobody. The event applies,
no new evidence is recorded, and the reason says what happened. The
fail-closed property comes out of the scoring layer rather than out of an
exception: a session that reaches a decision with nothing scored blends to
1.0 and is refused.

Finding the reference
---------------------
`voice_match` needs the enrolled print of whoever the caller claims to be. The
identity comes off the event when the agent repeats it, and otherwise off the
session, where it was recorded when the call started. That second lookup is
why `identity_of` is a callable handed in at construction rather than a store
this module reaches into: resolving it costs a session read, and it only
happens for a span that is actually going to be analysed.

An identity with nothing enrolled leaves `voice_match` at `not_run`, which is
the state every call was in before this store existed. It carries the largest
weight, 0.35, so such a call is judged on the remaining four signals with the
weights renormalised over them. That renormalisation is exactly the condition
that makes a diluted FAIL dangerous, which is why the guard in
`detector.scoring` exists and why enrolling a vendor matters: it is the
difference between four signals and five.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

from detector.analysis.behaviour import CallMetadata
from detector.analysis.engine import AnalysisEngine, AnalysisReport
from detector.analysis.voiceprint import VoicePrint
from detector.audio import AudioPolicy, AudioUnavailable, LoadedAudio, load_span
from detector.config import Settings
from detector.identity import EVENTS_CHANNEL, candidate_identities
from shared.contract import EventRequest, EventType, Factor

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    # Imported lazily so the pipeline does not pull the web layer in with it.
    from detector.enrolment import VoicePrintStore

__all__ = ["AudioPipeline", "SpanResult"]

logger = logging.getLogger("callback.detector.pipeline")


@dataclass(frozen=True)
class SpanResult:
    """What one event's audio produced, including when it produced nothing.

    `note` is written for a human reading an evidence bundle, so it says what
    went wrong in words rather than carrying an exception type. It never
    contains the uri, which can be a signed url with a credential in it.
    """

    factors: list[Factor] | None = None
    note: str | None = None
    source: str | None = None
    duration_s: float = 0.0
    bytes_read: int = 0
    elapsed_ms: float = 0.0
    #: The span reduced to a voice print, for the cross-leg check in
    #: `detector.legs`. The samples it was made from are gone by the time this
    #: result exists; nothing downstream of the engine ever sees audio.
    voice: VoicePrint | None = None

    @property
    def analysed(self) -> bool:
        return bool(self.factors)


class AudioPipeline:
    """Fetch, decode and analyse the audio on an event. Never raises.

    One instance per app. The engine is stateless and the httpx client is
    safe to share, so a single pipeline serves every concurrent call.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        engine: AnalysisEngine | None = None,
        client: httpx.AsyncClient | None = None,
        voiceprints: VoicePrintStore | None = None,
        identity_of: Callable[[str], str | None] | None = None,
        channel: str | None = EVENTS_CHANNEL,
    ) -> None:
        self.policy = AudioPolicy.from_settings(settings)
        self.mock_mode = settings.mock_mode
        self._engine = engine or AnalysisEngine()
        self._client = client
        self._owns_client = client is None
        self._voiceprints = voiceprints
        self._identity_of = identity_of
        self._channel = channel

    @property
    def active(self) -> bool:
        """Whether this pipeline will ever do anything.

        Mock mode scores from scripts, so there is nothing to fetch. With
        audio disabled there is nothing to fetch either. Either way the route
        skips straight past, at the cost of one boolean.
        """

        return not self.mock_mode and self.policy.enabled

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def http(self) -> httpx.AsyncClient:
        """The shared keep-alive client, created on first use.

        Public because the enrolment router fetches under the same policy and
        should not open a second connection pool for it. Lazy because a
        deployment in mock mode, or with audio switched off, never fetches
        anything and should not hold a client open for the whole process.
        """

        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.policy.timeout_seconds, follow_redirects=False
            )
        return self._client

    async def for_event(self, request: EventRequest) -> SpanResult:
        """Analyse this event's span, if it has one worth analysing."""

        if not self.active:
            return SpanResult()
        if request.event is not EventType.SPEECH_SEGMENT:
            return SpanResult()
        if request.audio is None:
            return SpanResult(note="This segment carried no audio reference.")
        if not request.audio.uri:
            return SpanResult(note="This segment's audio reference had no uri.")

        started = asyncio.get_running_loop().time()
        try:
            loaded = await load_span(request.audio, self.policy, client=self.http())
        except AudioUnavailable as exc:
            logger.warning(
                "audio span unavailable",
                extra={"session_id": request.session_id, "reason": str(exc)},
            )
            return SpanResult(note=f"Audio could not be read: {exc}")
        except Exception as exc:
            # Anything load_span did not anticipate. A live call must not die
            # because a decoder met input it had no branch for, and the
            # decision still fails closed without this span.
            logger.exception("audio span raised an unexpected error")
            return SpanResult(note=f"Audio could not be read: {type(exc).__name__}")

        try:
            report = await self._analyse(loaded, request)
        except Exception as exc:
            logger.exception("audio analysis raised")
            return SpanResult(note=f"Audio analysis failed: {type(exc).__name__}")

        elapsed = (asyncio.get_running_loop().time() - started) * 1000.0
        return SpanResult(
            factors=report.factors,
            source=loaded.source,
            duration_s=loaded.duration_s,
            bytes_read=loaded.bytes_read,
            elapsed_ms=round(elapsed, 2),
            voice=report.voice,
        )

    async def _analyse(
        self, loaded: LoadedAudio, request: EventRequest
    ) -> AnalysisReport:
        metadata = CallMetadata.from_mapping(request.metadata)
        return await asyncio.to_thread(
            self._engine.analyse_report,
            loaded.samples,
            loaded.sample_rate,
            metadata=metadata if not metadata.is_empty else None,
            reference=self._reference_for(request),
        )

    def _reference_for(self, request: EventRequest) -> VoicePrint | None:
        """The enrolled print of whoever this caller claims to be.

        Returns None for an identity with nothing enrolled, and for a store
        that cannot answer. A store failure must not end a live call: without
        a reference the comparison does not run, which removes a signal and
        pushes the decision toward deny, so the failure direction is the safe
        one.
        """

        if self._voiceprints is None:
            return None
        identity = request.claimed_user_id
        if not identity and self._identity_of is not None:
            try:
                identity = self._identity_of(request.session_id)
            except Exception:
                logger.exception("claimed identity lookup raised")
                return None
        if not identity:
            return None

        # Prints are stored per channel, so this looks for the phone-channel
        # print first and an unscoped one second. Reading the bare key alone
        # would miss every print the /v1 enrollment flow writes, which are
        # keyed "{user_id}:{channel}" by ruling section 2.15.
        for key in candidate_identities(identity, self._channel):
            try:
                enrolment = self._voiceprints.get(key)
            except Exception:
                # A malformed stored record or an unreachable dict. Logged,
                # not raised: see the docstring.
                logger.exception("voice print lookup failed")
                return None
            if enrolment is not None:
                return enrolment.voice
        return None
