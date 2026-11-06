"""Real mode, end to end: audio in, decision out.

Mock mode answers from a script. These drive the same endpoints with
`mock_mode=False` and actual waveforms, so the path under test is the one
that matters in production: fetch the span, decode it, measure it, score it.

The audio arrives as `data:` uris. That is a real supported transport, it
needs no network and no fixture files, and it exercises every layer above the
socket. The https path and everything it refuses is covered in
`test_audio.py`.

Always in process: a deployed mock has no real mode to point at.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import httpx
import numpy as np
import pytest

from detector import scoring
from detector.analysis.synthetic import live_speech, replay_of, tts_speech
from detector.app import create_app
from detector.config import Settings
from detector.evidence import (
    NO_AUDIO_REASON,
    analysed_evidence,
    combine_factors,
    empty_evidence,
    scripted_evidence,
)
from shared.contract import (
    MOCK_OTP_CODE,
    Decision,
    Factor,
    FactorName,
    FactorVerdict,
    Scenario,
    SessionState,
)

TOKEN = "real-mode-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
RATE = 8000


# ------------------------------------------------------------------ fixtures


def pcm_data_uri(samples: np.ndarray) -> str:
    clipped = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 0.999969)
    raw = (clipped * 32768.0).astype("<i2").tobytes()
    return f"data:audio/l16;base64,{base64.b64encode(raw).decode('ascii')}"


@pytest.fixture(scope="module")
def live_uri() -> str:
    return pcm_data_uri(live_speech(duration_s=4.0, sample_rate=RATE, seed=1))


@pytest.fixture(scope="module")
def replay_uri() -> str:
    live = live_speech(duration_s=4.0, sample_rate=RATE, seed=1)
    return pcm_data_uri(replay_of(live, sample_rate=RATE))


@pytest.fixture(scope="module")
def tts_uri() -> str:
    return pcm_data_uri(tts_speech(duration_s=4.0, sample_rate=RATE, seed=1))


def real_settings(**overrides) -> Settings:
    base = {
        "api_token": TOKEN,
        "mock_mode": False,
        "session_ttl_seconds": 300,
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


@pytest.fixture
async def real_client() -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(real_settings())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://real.test",
        headers=AUTH,
    ) as client:
        yield client


class Call:
    """Drive one session, one event at a time."""

    def __init__(self, client: httpx.AsyncClient, session_id: str) -> None:
        self.client = client
        self.session_id = session_id
        self.sequence = 0
        self.start = datetime.now(UTC) - timedelta(minutes=1)

    async def send(self, event: str, **extra) -> httpx.Response:
        self.sequence += 1
        body: dict[str, object] = {
            "session_id": self.session_id,
            "sequence": self.sequence,
            "event": event,
            "occurred_at": (
                self.start + timedelta(seconds=self.sequence * 4)
            ).isoformat(),
        }
        body.update(extra)
        return await self.client.post("/events", json=body)

    async def speech(self, uri: str | None = None, **extra) -> httpx.Response:
        audio = None
        if uri is not None:
            audio = {
                "track_id": "caller-0",
                "start_ms": self.sequence * 4000,
                "end_ms": (self.sequence + 1) * 4000,
                "sample_rate_hz": RATE,
                "encoding": "pcm_s16le",
                "uri": uri,
            }
        return await self.send("speech_segment", audio=audio, **extra)

    async def run(self, uri: str | None, segments: int = 2) -> dict:
        await self.send("session_start", claimed_user_id="cfo@northwind.example")
        last: httpx.Response | None = None
        for _ in range(segments):
            last = await self.speech(uri)
        assert last is not None
        assert last.status_code == 200, last.text
        return last.json()


# ---------------------------------------------------- the evidence algebra


def f(
    name: FactorName,
    score: float,
    verdict: FactorVerdict = FactorVerdict.PASS,
    weight: float = 0.2,
) -> Factor:
    return Factor(
        name=name, score=score, weight=weight, verdict=verdict, detail="synthetic"
    )


def test_the_worst_span_wins() -> None:
    """One bad span is evidence. Averaging it away is how a burst hides."""

    first = [f(FactorName.REPLAY, 0.90), f(FactorName.LIVENESS, 0.40)]
    second = [f(FactorName.REPLAY, 0.20), f(FactorName.LIVENESS, 0.80)]

    merged = combine_factors(first, second)
    scores = {factor.name: factor.score for factor in merged}

    assert scores[FactorName.REPLAY] == 0.20
    assert scores[FactorName.LIVENESS] == 0.40


def test_a_detector_that_ran_beats_one_that_did_not() -> None:
    """NOT_RUN carries a placeholder score, so it must not win on being low."""

    earlier = [f(FactorName.VOICE_MATCH, 0.50, FactorVerdict.NOT_RUN)]
    later = [f(FactorName.VOICE_MATCH, 0.95, FactorVerdict.PASS)]

    merged = combine_factors(earlier, later)
    assert merged[0].verdict is FactorVerdict.PASS
    assert merged[0].score == 0.95

    # And the other way round: a real verdict is not overwritten by not_run.
    merged_back = combine_factors(later, earlier)
    assert merged_back[0].verdict is FactorVerdict.PASS
    assert merged_back[0].score == 0.95


def test_a_factor_only_the_earlier_span_had_is_kept() -> None:
    earlier = [f(FactorName.VOICE_MATCH, 0.30, FactorVerdict.FAIL)]
    later = [f(FactorName.REPLAY, 0.90)]

    merged = combine_factors(earlier, later)
    names = [factor.name for factor in merged]
    assert FactorName.VOICE_MATCH in names
    assert FactorName.REPLAY in names
    # Incoming order first, so the engine's stable order survives.
    assert names[0] is FactorName.REPLAY


def test_confidence_is_the_share_of_weight_that_actually_ran() -> None:
    factors = [
        f(FactorName.VOICE_MATCH, 0.5, FactorVerdict.NOT_RUN, weight=0.35),
        f(FactorName.SYNTHETIC_SPEECH, 0.9, FactorVerdict.PASS, weight=0.25),
        f(FactorName.REPLAY, 0.9, FactorVerdict.PASS, weight=0.15),
        f(FactorName.AGENT_BEHAVIOUR, 0.5, FactorVerdict.NOT_RUN, weight=0.15),
        f(FactorName.LIVENESS, 0.9, FactorVerdict.PASS, weight=0.10),
    ]

    evidence = analysed_evidence(factors)

    # 0.25 + 0.15 + 0.10 of 1.0 measured.
    assert evidence.confidence == pytest.approx(0.50)
    assert "partial answer" in evidence.reason


def test_nothing_measured_is_not_nothing_wrong() -> None:
    """The second fail-open, pinned at the evidence layer this time."""

    evidence = empty_evidence()

    assert evidence.factors == []
    assert evidence.confidence == 0.0
    assert evidence.otp_can_rescue is False
    assert scoring.blend_risk(evidence.factors) == 1.0
    assert scoring.decide(scoring.blend_risk(evidence.factors)) is Decision.DENY


@pytest.mark.parametrize(
    ("name", "rescuable"),
    [
        (FactorName.REPLAY, False),
        (FactorName.AGENT_BEHAVIOUR, False),
        (FactorName.VOICE_MATCH, True),
        (FactorName.SYNTHETIC_SPEECH, True),
        (FactorName.LIVENESS, True),
    ],
)
def test_which_failures_a_code_can_talk_past(name: FactorName, rescuable: bool) -> None:
    """A code proves possession of a phone, not that a human is speaking.

    Someone playing a recording down the line, or a bot driving the call, is
    not made legitimate by also holding the handset. A failing voice or
    synthesis signal is exactly the case step-up exists for.
    """

    factors = [f(other, 0.9) for other in FactorName if other is not name]
    factors.append(f(name, 0.25, FactorVerdict.FAIL))

    assert analysed_evidence(factors).otp_can_rescue is rescuable


def test_mock_evidence_is_unchanged() -> None:
    """The seam must not have moved mock mode."""

    for scenario in Scenario:
        evidence = scripted_evidence(scenario)
        assert evidence.factors
        assert 0.0 <= evidence.confidence <= 1.0
        assert evidence.reason


# -------------------------------------------------------------- end to end


async def test_live_speech_is_allowed(
    real_client: httpx.AsyncClient, live_uri: str
) -> None:
    body = await Call(real_client, "sess-real-live-0001").run(live_uri)

    assert body["decision"] == Decision.ALLOW.value, body["reason"]
    assert body["state"] == SessionState.ALLOWED.value
    assert body["risk_score"] < scoring.ALLOW_BELOW
    assert body["scenario"] is None, "real mode must not report a mock scenario"
    assert body["factors"], "a real decision has to show its working"


async def test_a_replayed_recording_is_denied(
    real_client: httpx.AsyncClient, replay_uri: str
) -> None:
    body = await Call(real_client, "sess-real-replay-001").run(replay_uri)

    assert body["decision"] == Decision.DENY.value, body["reason"]
    replay = next(f for f in body["factors"] if f["name"] == "replay")
    assert replay["verdict"] == "fail"

    # This is the diluted-FAIL case, happening for real rather than in a
    # crafted fixture. The replay scores 0.12, just above the 0.10 fatal
    # line, so the fatal rule does not fire. Only three of five detectors
    # run without an enrolment reference or call metadata, so the weights
    # renormalise and the blend lands at roughly 0.30, under the allow
    # threshold. Without the guard that any FAIL blocks a clean allow, this
    # replayed recording would have been allowed.
    assert replay["score"] > scoring.FATAL_THRESHOLD, "not a fatal deny"
    assert body["risk_score"] < scoring.DENY_AT_OR_ABOVE, "not a threshold deny"
    assert scoring.decide(body["risk_score"]) is Decision.ALLOW, (
        "the blend alone would have allowed this"
    )


async def test_synthetic_speech_is_denied(
    real_client: httpx.AsyncClient, tts_uri: str
) -> None:
    body = await Call(real_client, "sess-real-tts-0001").run(tts_uri)

    assert body["decision"] == Decision.DENY.value, body["reason"]
    names = {f["name"]: f for f in body["factors"]}
    assert names["synthetic_speech"]["verdict"] == "fail"


async def test_one_bad_span_among_good_ones_still_denies(
    real_client: httpx.AsyncClient, live_uri: str, replay_uri: str
) -> None:
    """The combination rule, over HTTP rather than in a unit test."""

    call = Call(real_client, "sess-real-mixed-001")
    await call.send("session_start", claimed_user_id="cfo@northwind.example")
    await call.speech(live_uri)
    response = await call.speech(replay_uri)

    body = response.json()
    assert body["decision"] == Decision.DENY.value, body["reason"]


async def test_the_bad_span_still_counts_when_it_came_first(
    real_client: httpx.AsyncClient, live_uri: str, replay_uri: str
) -> None:
    call = Call(real_client, "sess-real-mixed-002")
    await call.send("session_start", claimed_user_id="cfo@northwind.example")
    await call.speech(replay_uri)
    response = await call.speech(live_uri)

    body = response.json()
    assert body["decision"] == Decision.DENY.value, body["reason"]


# ------------------------------------------------- when there is no audio


async def test_no_audio_fails_closed(real_client: httpx.AsyncClient) -> None:
    """Real mode with nothing to analyse refuses. It does not shrug."""

    body = await Call(real_client, "sess-real-silent-01").run(None)

    assert body["decision"] == Decision.DENY.value
    assert body["risk_score"] == 1.0
    assert NO_AUDIO_REASON.split(",")[0] in body["reason"]


async def test_an_unreadable_span_still_applies_the_event(
    real_client: httpx.AsyncClient,
) -> None:
    """A bad uri must not 4xx in the middle of a live phone call.

    The event applies, no evidence is recorded, the decision fails closed,
    and the reason says what was wrong so it can be fixed.
    """

    body = await Call(real_client, "sess-real-badref-1").run(
        "data:audio/l16;base64,!!!not base64!!!"
    )

    assert body["decision"] == Decision.DENY.value
    assert "base64" in body["reason"], body["reason"]


async def test_a_span_pointing_at_cloud_metadata_is_refused(
    real_client: httpx.AsyncClient,
) -> None:
    """The SSRF case, through the real route rather than the loader.

    No host is allowlisted on this deployment, so the fetch never happens and
    the call denies for want of evidence.
    """

    body = await Call(real_client, "sess-real-ssrf-001").run(
        "https://169.254.169.254/latest/meta-data/iam/security-credentials/"
    )

    assert body["decision"] == Decision.DENY.value
    assert "not allowed" in body["reason"], body["reason"]


async def test_audio_disabled_denies_every_real_call(live_uri: str) -> None:
    """Real mode without CALLBACK_AUDIO_ENABLED is not quietly permissive."""

    app = create_app(real_settings(audio_enabled=False))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://off.test",
        headers=AUTH,
    ) as client:
        body = await Call(client, "sess-real-off-0001").run(live_uri)

    assert body["decision"] == Decision.DENY.value
    assert body["risk_score"] == 1.0


# ------------------------------------------------------------ the step-up


async def test_a_correct_code_does_not_rescue_a_replay(
    real_client: httpx.AsyncClient, replay_uri: str
) -> None:
    """The first fail-open, now on the real path.

    The replay is a hard deny, so the session is terminal and will not even
    issue a challenge. Belt and braces: the code is sent anyway and refused.
    """

    call = Call(real_client, "sess-real-otp-0001")
    body = await call.run(replay_uri)
    assert body["decision"] == Decision.DENY.value

    refused = await call.send("otp_request")
    assert refused.status_code == 409, refused.text


async def test_a_code_sent_before_any_speech_does_not_authenticate(
    real_client: httpx.AsyncClient, replay_uri: str
) -> None:
    """The exact ordering that used to allow a replay at risk 0.0."""

    call = Call(real_client, "sess-real-otp-0002")
    await call.send("session_start", claimed_user_id="cfo@northwind.example")
    issued = await call.send("otp_request")
    assert issued.status_code == 200, issued.text

    # Now the speech arrives, while the challenge is open.
    await call.speech(replay_uri)

    submitted = await call.send("otp_submit", otp_code=MOCK_OTP_CODE)
    body = submitted.json() if submitted.status_code == 200 else {}
    if submitted.status_code == 200:
        assert body["decision"] != Decision.ALLOW.value, body
    else:
        # Terminal already, because the replay denied on arrival.
        assert submitted.status_code == 409, submitted.text


async def test_real_mode_never_uses_the_mock_code(
    real_client: httpx.AsyncClient, live_uri: str
) -> None:
    """424242 is published in this repo. It must not work off mock mode."""

    call = Call(real_client, "sess-real-code-0001")
    await call.send("session_start", claimed_user_id="cfo@northwind.example")
    await call.speech(live_uri)
    issued = await call.send("otp_request")
    assert issued.status_code == 200, issued.text

    submitted = await call.send("otp_submit", otp_code=MOCK_OTP_CODE)
    assert submitted.status_code == 200, submitted.text
    assert submitted.json()["decision"] != Decision.ALLOW.value
