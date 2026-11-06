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

What this cannot do yet
-----------------------
`voice_match` needs an enrolment recording of the claimed identity and there
is no enrolment store, so in real mode that factor reports `not_run` rather
than a guess. It carries the largest weight, 0.35, so a real-mode call is
currently judged on the remaining four signals with the weights renormalised
over them. That renormalisation is exactly the condition that made a diluted
FAIL dangerous, which is why the guard in `detector.scoring` exists and why it
matters more in real mode than it ever did in mock.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx

from detector.analysis.behaviour import CallMetadata
from detector.analysis.engine import AnalysisEngine
from detector.audio import AudioPolicy, AudioUnavailable, LoadedAudio, load_span
from detector.config import Settings
from shared.contract import EventRequest, EventType, Factor

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
    ) -> None:
        self.policy = AudioPolicy.from_settings(settings)
        self.mock_mode = settings.mock_mode
        self._engine = engine or AnalysisEngine()
        self._client = client
        self._owns_client = client is None

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

    def _http(self) -> httpx.AsyncClient:
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
            loaded = await load_span(request.audio, self.policy, client=self._http())
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
            factors = await self._analyse(loaded, request)
        except Exception as exc:
            logger.exception("audio analysis raised")
            return SpanResult(note=f"Audio analysis failed: {type(exc).__name__}")

        elapsed = (asyncio.get_running_loop().time() - started) * 1000.0
        return SpanResult(
            factors=factors,
            source=loaded.source,
            duration_s=loaded.duration_s,
            bytes_read=loaded.bytes_read,
            elapsed_ms=round(elapsed, 2),
        )

    async def _analyse(
        self, loaded: LoadedAudio, request: EventRequest
    ) -> list[Factor]:
        metadata = CallMetadata.from_mapping(request.metadata)
        return await asyncio.to_thread(
            self._engine.analyse,
            loaded.samples,
            loaded.sample_rate,
            metadata=metadata if not metadata.is_empty else None,
            # No enrolment store yet, so voice_match reports not_run rather
            # than being handed the caller's own audio as its own reference,
            # which would score 1.00 for everyone including the attacker.
            reference_audio=None,
        )
