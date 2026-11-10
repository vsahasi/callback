"""POST /events over the life of a real phone call.

The voice agent drives the detector like this. `/v1/sessions` creates session
S. The agent posts one `speech_segment` per caller turn to `/events` under the
same id, 6 to 15 of them over 2 to 4 minutes, track "caller". It then hangs up,
dials the number on file, and 2 to 4 more segments arrive under track
"callback", a minute or more later. The whole time, `/v1 .../authorize` reads
`DetectorEngine.factors_for(S)` through `detector/bridge.py` and decides.

The engine used to make its own ALLOW terminal after two clean spans. Every
later `/events` was then `409 session_terminal`, so the callback leg was never
analysed and the factors `/v1` read were frozen at span two. Real mode is the
engine being read as evidence rather than as the verdict, so in real mode an
open session keeps hearing after its own allow. Mock mode keeps the published
dev-API behaviour, which `test_policy.py` pins.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import httpx
import numpy as np
import pytest

from detector.analysis.synthetic import live_speech, tts_speech
from detector.app import create_app
from detector.bridge import signals_from_engine
from detector.config import Settings
from detector.errors import SequenceConflict, SessionTerminal
from detector.state import DetectorEngine
from shared.contract import (
    Decision,
    EventRequest,
    EventType,
    Factor,
    FactorName,
    FactorVerdict,
    NextAction,
    SessionState,
)

TOKEN = "lifecycle-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
RATE = 8000
SID = "sess-lifecycle-000001"
OCCURRED_AT = datetime(2026, 11, 9, 20, 0, 0, tzinfo=UTC)


def real_settings(**overrides) -> Settings:
    base = {
        "api_token": TOKEN,
        "mock_mode": False,
        "session_ttl_seconds": 900,
        "otp_max_attempts": 3,
        "otp_ttl_seconds": 120,
        "mock_jitter_min_ms": 0,
        "mock_jitter_max_ms": 0,
        "speech_segments_before_decision": 2,
        "audio_enabled": True,
        "audio_allow_data_uris": True,
    }
    base.update(overrides)
    return Settings(**base)


# ------------------------------------------------------------- factor shapes
#
# What the analysis engine hands the state machine for one span, in miniature.
# Five signals, equal weights, so the blend is easy to reason about.


def _span(**scores: float) -> list[Factor]:
    defaults = {
        FactorName.VOICE_MATCH: 0.95,
        FactorName.SYNTHETIC_SPEECH: 0.95,
        FactorName.REPLAY: 0.95,
        FactorName.AGENT_BEHAVIOUR: 0.95,
        FactorName.LIVENESS: 0.95,
    }
    factors = []
    for name, default in defaults.items():
        score = scores.get(name.value, default)
        if score <= 0.3:
            verdict = FactorVerdict.FAIL
        elif score < 0.6:
            verdict = FactorVerdict.INCONCLUSIVE
        else:
            verdict = FactorVerdict.PASS
        factors.append(
            Factor(name=name, score=score, weight=0.2, verdict=verdict, detail="test")
        )
    return factors


CLEAN = _span()
SYNTHETIC = _span(synthetic_speech=0.05)  # a FAIL, but not a fatal factor
REPLAY = _span(replay=0.05)  # fatal: at or below 0.10


def segment(sequence: int, track: str = "caller") -> EventRequest:
    return EventRequest(
        session_id=SID,
        event=EventType.SPEECH_SEGMENT,
        sequence=sequence,
        occurred_at=OCCURRED_AT + timedelta(seconds=sequence * 10),
        audio={
            "track_id": track,
            "start_ms": sequence * 10_000,
            "end_ms": sequence * 10_000 + 4_000,
            "uri": "data:audio/l16;base64,AAAA",
        },
    )


def hear(engine: DetectorEngine, request: EventRequest, factors: list[Factor]):
    return engine.handle_event(
        request,
        scenario_header=None,
        idempotency_key=None,
        observed_factors=[f.model_copy(deep=True) for f in factors],
    )


# ------------------------------------------------------- the engine, directly


def test_a_real_session_keeps_hearing_after_its_own_allow() -> None:
    """12 caller spans, then 3 callback spans: all accepted, all analysed."""

    engine = DetectorEngine(real_settings())

    for sequence in range(1, 13):
        response = hear(engine, segment(sequence, "caller"), CLEAN)
        assert response.decision is (
            Decision.PENDING if sequence < 2 else Decision.ALLOW
        ), (sequence, response.reason)

    for sequence in range(13, 16):
        response = hear(engine, segment(sequence, "callback"), CLEAN)
        assert response.state is SessionState.ALLOWED
        assert response.decision is Decision.ALLOW
        assert response.next_action is NextAction.COMPLETE_ALLOW

    session = engine.store.get(SID)
    assert session.speech_segments == 15
    assert session.spans_analysed == 15
    assert "over 15 audio spans" in (engine.factors_for(SID).reason or "")


def test_a_synthetic_span_after_two_clean_ones_denies() -> None:
    """A FAIL that is not fatal. Step-up is gone once an allow was published,
    so the middle band falls closed rather than issuing a code nobody asked
    for."""

    engine = DetectorEngine(real_settings())
    hear(engine, segment(1), CLEAN)
    allowed = hear(engine, segment(2), CLEAN)
    assert allowed.decision is Decision.ALLOW

    denied = hear(engine, segment(3), SYNTHETIC)
    assert denied.state is SessionState.DENIED
    assert denied.decision is Decision.DENY
    assert denied.next_action is NextAction.COMPLETE_DENY
    assert denied.otp is None, "no code is issued after the fact"
    synthetic = next(f for f in denied.factors if f.name is FactorName.SYNTHETIC_SPEECH)
    assert synthetic.verdict is FactorVerdict.FAIL

    # A deny is still terminal: nothing can lift it, so nothing more is heard.
    with pytest.raises(SessionTerminal):
        hear(engine, segment(4), CLEAN)


def test_a_fatal_span_after_allow_flips_to_deny() -> None:
    engine = DetectorEngine(real_settings())
    hear(engine, segment(1), CLEAN)
    hear(engine, segment(2), CLEAN)

    denied = hear(engine, segment(3, "callback"), REPLAY)
    assert denied.decision is Decision.DENY
    assert "replay is unambiguous attack evidence" in (denied.reason or "")
    assert engine.factors_for(SID).decision is Decision.DENY


def test_factors_for_reflects_the_latest_span() -> None:
    """Worst-of still applies after the allow, and the bridge sees it."""

    engine = DetectorEngine(real_settings())
    hear(engine, segment(1), CLEAN)
    hear(engine, segment(2), CLEAN)
    before = signals_from_engine(engine, SID)
    assert before.voiceprint == pytest.approx(0.95)

    # The callback leg's voice is further from the enrolment.
    still_allowed = hear(engine, segment(3, "callback"), _span(voice_match=0.4))
    assert still_allowed.decision is Decision.ALLOW, "inconclusive is not a FAIL"

    after = engine.factors_for(SID)
    voice = next(f for f in after.factors if f.name is FactorName.VOICE_MATCH)
    assert voice.score == pytest.approx(0.4)
    assert after.events_seen == 3
    assert "over 3 audio spans" in (after.reason or "")
    assert signals_from_engine(engine, SID).voiceprint == pytest.approx(0.4)


def test_an_unreadable_span_after_allow_keeps_the_evidence() -> None:
    """No audio on a later turn is not new evidence and not a reset either."""

    engine = DetectorEngine(real_settings())
    hear(engine, segment(1), CLEAN)
    hear(engine, segment(2), CLEAN)

    response = engine.handle_event(
        segment(3),
        scenario_header=None,
        idempotency_key=None,
        observed_factors=None,
        audio_note="Audio could not be read: test",
    )
    assert response.decision is Decision.ALLOW
    assert len(response.factors) == 5
    assert engine.store.get(SID).spans_analysed == 2


def test_the_sequence_runs_on_across_both_legs() -> None:
    """Monotonic per session. The callback leg must not restart its counter."""

    engine = DetectorEngine(real_settings())
    for sequence in range(1, 4):
        hear(engine, segment(sequence), CLEAN)

    with pytest.raises(SequenceConflict):
        hear(engine, segment(1, "callback"), CLEAN)
    accepted = hear(engine, segment(4, "callback"), CLEAN)
    assert accepted.decision is Decision.ALLOW


def test_mock_mode_still_closes_the_session_on_allow() -> None:
    """The published dev-API behaviour. test_policy.py pins it over HTTP."""

    engine = DetectorEngine(real_settings(mock_mode=True, audio_enabled=False))
    hear(engine, segment(1), CLEAN)
    assert hear(engine, segment(2), CLEAN).decision is Decision.ALLOW
    with pytest.raises(SessionTerminal):
        hear(engine, segment(3), CLEAN)


# ------------------------------------------------------------------- the TTL


def test_the_default_ttl_covers_a_call_and_its_callback(monkeypatch) -> None:
    """Four minutes of caller, a hang-up, a dial, and the callback leg."""

    for name in ("CALLBACK_SESSION_TTL_SECONDS", "CALLBACK_REVIEW_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DETECTOR_TOKEN", "ttl-default-token")
    assert Settings.from_env().session_ttl_seconds >= 600


def test_a_callback_leg_past_the_ttl_is_refused_but_the_signals_stay() -> None:
    """The clock runs from the first /events post, not from the first turn of
    the callback leg. Past it, /events refuses and /v1 keeps reading what was
    measured; the record goes only at session_ttl + eviction_grace."""

    engine = DetectorEngine(real_settings(session_ttl_seconds=60))
    hear(engine, segment(1), CLEAN)
    hear(engine, segment(2), CLEAN)
    engine.store.get(SID).created_at -= timedelta(seconds=120)

    with pytest.raises(SessionTerminal):
        hear(engine, segment(3, "callback"), CLEAN)

    answer = engine.factors_for(SID)
    assert answer.state is SessionState.ALLOWED
    assert signals_from_engine(engine, SID).heard is True


# ------------------------------------------------------------ over the wire


def pcm_data_uri(samples: np.ndarray) -> str:
    clipped = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 0.999969)
    raw = (clipped * 32768.0).astype("<i2").tobytes()
    return f"data:audio/l16;base64,{base64.b64encode(raw).decode('ascii')}"


@pytest.fixture(scope="module")
def live_uri() -> str:
    return pcm_data_uri(live_speech(duration_s=4.0, sample_rate=RATE, seed=1))


@pytest.fixture(scope="module")
def tts_uri() -> str:
    return pcm_data_uri(tts_speech(duration_s=4.0, sample_rate=RATE, seed=1))


@pytest.fixture
async def real_client() -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(real_settings())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://lifecycle.test",
        headers=AUTH,
    ) as client:
        client.app = app  # type: ignore[attr-defined]
        yield client


async def post_segment(
    client: httpx.AsyncClient, sequence: int, track: str, uri: str
) -> httpx.Response:
    body = {
        "session_id": "sess-lifecycle-http-01",
        "sequence": sequence,
        "event": "speech_segment",
        "occurred_at": (OCCURRED_AT + timedelta(seconds=sequence * 12)).isoformat(),
        "claimed_user_id": "cfo@northwind.example",
        "audio": {
            "track_id": track,
            "start_ms": sequence * 12_000,
            "end_ms": sequence * 12_000 + 4_000,
            "sample_rate_hz": RATE,
            "encoding": "pcm_s16le",
            "uri": uri,
        },
    }
    return await client.post("/events", json=body)


async def test_the_whole_call_is_heard_over_http(
    real_client: httpx.AsyncClient, live_uri: str, tts_uri: str
) -> None:
    """12 caller turns, 3 callback turns, every one analysed; then a
    synthetic turn, and the session that had allowed denies."""

    for sequence in range(1, 13):
        response = await post_segment(real_client, sequence, "caller", live_uri)
        assert response.status_code == 200, (sequence, response.text)
    for sequence in range(13, 16):
        response = await post_segment(real_client, sequence, "callback", live_uri)
        assert response.status_code == 200, (sequence, response.text)
        body = response.json()
        assert body["decision"] == Decision.ALLOW.value, body["reason"]
        assert body["state"] == SessionState.ALLOWED.value

    assert "over 15 audio spans" in body["reason"]

    engine: DetectorEngine = real_client.app.state.engine  # type: ignore[attr-defined]
    assert engine.store.get("sess-lifecycle-http-01").spans_analysed == 15

    denied = await post_segment(real_client, 16, "callback", tts_uri)
    assert denied.status_code == 200, denied.text
    body = denied.json()
    assert body["decision"] == Decision.DENY.value, body["reason"]
    names = {f["name"]: f for f in body["factors"]}
    assert names["synthetic_speech"]["verdict"] == "fail"
