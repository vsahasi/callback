"""Real mode: a phrase or digits challenge resolves from the turn and the evidence.

The agent posts the caller's answer as a `user` turn to /v1 and the caller's
speech to POST /events under the same session id. The challenge binds to the
first user turn after issue that is an attempt at the answer (`answers`),
records whether the words were right (a bool, never the words), and settles
700 ms later from the signals the /events engine measured by then: words wrong
fails `phrase_mismatch`, words right below the voiceprint band fails
`speaker_mismatch`, otherwise it passes. Nothing heard keeps `no_audio`, which
is never a voice fail (ruling F1). A settled result never changes.
"""

from __future__ import annotations

import base64
import io
import wave
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest
from callback_shared.contract import (
    ChallengeKind,
    Channel,
    Decision,
    Role,
    SessionCreate,
    Turn,
)
from fastapi import FastAPI

from detector.analysis.dsp import build_context
from detector.analysis.synthetic import live_speech
from detector.analysis.voiceprint import voiceprint_from
from detector.enrolment import InMemoryVoicePrintStore, storable_enrolment
from detector.errors import DetectorError
from detector.v1.challenges import (
    MIN_ANSWER_TOKENS,
    MIN_EXPECTED_HITS,
    answers,
    words_match,
)
from detector.v1.policy import SPOOF_LOW, VOICEPRINT_FAIL
from detector.v1.service import V1Service
from shared.contract import (
    AudioRef,
    EventRequest,
    EventType,
    Factor,
    FactorName,
    FactorVerdict,
)
from tests.v1.conftest import (
    FakeClock,
    base_settings,
    build_app,
    client_for,
    create_session,
    turn_body,
)

RATE = 8000
USER = "u_eric"
PHRASE_DIGITS = ("4", "7", "2", "9", "1", "3")

READ = {
    "action": "list_vendor_accounts",
    "params": {"vendor_id": "v1", "vendor_name": "Acme Supplies"},
}


# ---------------------------------------------------------------- helpers


def factor(
    name: FactorName,
    score: float,
    verdict: FactorVerdict | None = None,
    weight: float = 0.2,
) -> Factor:
    if verdict is None:
        verdict = FactorVerdict.PASS if score >= 0.5 else FactorVerdict.FAIL
    return Factor(name=name, score=score, weight=weight, verdict=verdict, detail="t")


def clean_human() -> list[Factor]:
    """What the analysis engine says about a live, enrolled human."""

    return [
        factor(FactorName.VOICE_MATCH, 1.0),
        factor(FactorName.SYNTHETIC_SPEECH, 1.0),
        factor(FactorName.REPLAY, 1.0),
        factor(FactorName.AGENT_BEHAVIOUR, 0.5, FactorVerdict.NOT_RUN),
        factor(FactorName.LIVENESS, 1.0),
    ]


def with_factor(factors: list[Factor], replacement: Factor) -> list[Factor]:
    return [replacement if f.name is replacement.name else f for f in factors]


def real_app(
    clock: FakeClock, log_stream: io.StringIO | None = None, **overrides: Any
) -> FastAPI:
    return build_app(
        base_settings(mock_mode=False, audio_enabled=True, **overrides),
        clock,
        log_stream if log_stream is not None else io.StringIO(),
    )


def post_evidence(
    app: FastAPI, session_id: str, factors: list[Factor] | None, sequence: int = 1
) -> None:
    """One analysed caller span, the way the /events route hands it over."""

    request = EventRequest(
        session_id=session_id,
        event=EventType.SPEECH_SEGMENT,
        sequence=sequence,
        occurred_at=datetime.now(UTC),
        claimed_user_id=USER,
        audio=AudioRef(track_id="caller", start_ms=0, end_ms=3000),
    )
    app.state.engine.handle_event(
        request,
        scenario_header=None,
        idempotency_key=None,
        observed_factors=factors,
        audio_note=None if factors else "test: nothing analysable",
    )


async def authorize(
    client: httpx.AsyncClient, session_id: str, key: str = "k1"
) -> Decision:
    response = await client.post(
        f"/v1/sessions/{session_id}/authorize",
        json={**READ, "idempotency_key": key},
    )
    assert response.status_code == 200, response.text
    return Decision.model_validate(response.json())


async def issue(
    client: httpx.AsyncClient, session_id: str, kind: str = "phrase"
) -> dict[str, Any]:
    response = await client.post(
        f"/v1/sessions/{session_id}/challenges", json={"kind": kind}
    )
    assert response.status_code == 200, response.text
    return response.json()


async def say(
    client: httpx.AsyncClient, session_id: str, clock: FakeClock, text: str
) -> None:
    """The agent posting the caller's turn, as callback_agent.turns builds it."""

    response = await client.post(
        f"/v1/sessions/{session_id}/turns", json=turn_body(clock, text=text)
    )
    assert response.status_code == 204, response.text


async def status_of(
    client: httpx.AsyncClient, session_id: str, cid: str
) -> dict[str, Any]:
    response = await client.get(f"/v1/sessions/{session_id}/challenges/{cid}")
    assert response.status_code == 200, response.text
    return response.json()


def spoken(issued: dict[str, Any]) -> str:
    """The caller repeating the `say` verbatim."""

    return issued["say"]


class StubEngine:
    """Enough of DetectorEngine for the service: factors_for, counted.

    `unknown` raises DetectorError, which is the engine for a session nobody
    has posted to. `fail_reads` makes that many reads raise something else,
    which is a store that cannot answer right now.
    """

    def __init__(
        self, factors: list[Factor] | None = None, *, unknown: bool = False
    ) -> None:
        self.factors = factors or []
        self.unknown = unknown
        self.fail_reads = 0
        self.reads = 0

    def factors_for(self, session_id: str) -> object:
        self.reads += 1
        if self.fail_reads:
            self.fail_reads -= 1
            raise RuntimeError("store unavailable")
        if self.unknown:
            raise DetectorError("session_not_found")
        return SimpleNamespace(factors=self.factors)


def real_service(engine: object | None) -> tuple[V1Service, FakeClock]:
    clock = FakeClock()
    service = V1Service(
        base_settings(mock_mode=False),
        voiceprints=InMemoryVoicePrintStore(),
        clock=clock,
        engine=engine,
    )
    return service, clock


def a_session_id(service: V1Service, clock: FakeClock) -> str:
    created = service.create_session(
        SessionCreate(
            room=f"room-{clock.now}",
            caller_identity="caller-0",
            claimed_user_id=USER,
            channel=Channel.PHONE,
        ),
        None,
    )
    return created.session_id


def user_turn(clock: FakeClock, text: str) -> Turn:
    start = clock.ms()
    return Turn(role=Role.USER, text=text, start_ms=start, end_ms=start + 1200)


# ------------------------------------------------------------- word match


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        ("four seven two, nine one three, amber, candle, zebra", True),
        ("4 7 2 9 1 3 amber candle zebra", True),
        ("472913 Amber Candle Zebra.", True),
        ("sure, four seven two nine one three amber candle zebra okay", True),
        ("four seven two nine one three amber zebra candle", False),  # order
        ("four seven two nine one three amber candle", False),  # a word missing
        ("four seven two nine one four amber candle zebra", False),  # a digit off
        ("", False),
    ],
)
def test_words_match(text: str, ok: bool) -> None:
    expected = (*PHRASE_DIGITS, "amber", "candle", "zebra")
    assert words_match(expected, text) is ok


def test_words_match_needs_an_expectation() -> None:
    assert words_match((), "four seven two") is False
    assert words_match(None, "four seven two") is False


# ------------------------------------------------ what counts as an answer


EXPECTED = (*PHRASE_DIGITS, "amber", "candle", "zebra")


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        ("okay", False),
        ("um", False),
        ("yes", False),
        ("", False),
        ("one moment", False),  # one expected digit by accident
        ("hold on a second", False),  # two digit words, one hit
        ("I didn't catch that", False),  # four tokens, no hit
        ("sorry can you say that again", True),  # six tokens: an attempt
        ("four seven", True),  # two expected tokens: an attempt
        ("472 913", True),  # a digit run is six tokens
        ("amber candle", True),
        ("five five five five five five", True),  # wrong, but six tokens
        ("four seven two nine one three amber candle zebra", True),
    ],
)
def test_answers(text: str, ok: bool) -> None:
    assert answers(EXPECTED, text) is ok


def test_answers_thresholds_are_the_shortest_answer() -> None:
    """Six digits is the shortest answer; two hits is more than an accident."""

    assert MIN_ANSWER_TOKENS == 6
    assert MIN_EXPECTED_HITS == 2
    # A forgotten expectation (after /end) still binds on length alone.
    assert answers(None, "one two three four five six") is True
    assert answers(None, "one two") is False


# ------------------------------------------------- binding, over the wire


async def test_a_stray_turn_does_not_bind_a_verification_challenge(
    clock: FakeClock,
) -> None:
    """'okay' and 'one moment' leave the challenge waiting for the answer."""

    app = real_app(clock)
    async with client_for(app) as client:
        session_id = await create_session(client)
        post_evidence(app, session_id, clean_human())
        issued = await issue(client, session_id)
        cid = issued["challenge_id"]
        await say(client, session_id, clock, "okay")
        clock.advance(1.0)
        assert (await status_of(client, session_id, cid))["status"] == "pending"
        await say(client, session_id, clock, "one moment")
        clock.advance(1.0)
        assert (await status_of(client, session_id, cid))["status"] == "pending"
        record = app.state.v1.registry.get(session_id).challenges[cid]
        assert record.answered_at is None
        answered_at = clock.now
        await say(client, session_id, clock, spoken(issued))
        assert record.answered_at == answered_at
        assert record.words_right is True
        clock.advance(0.7)
        status = await status_of(client, session_id, cid)
    assert status["status"] == "passed"


async def test_a_wrong_attempt_binds_and_fails_phrase_mismatch(
    clock: FakeClock,
) -> None:
    """Six wrong digits is still an answer: it binds, and it fails."""

    app = real_app(clock)
    async with client_for(app) as client:
        session_id = await create_session(client)
        post_evidence(app, session_id, clean_human())
        issued = await issue(client, session_id, "digits")
        wrong = " ".join("five" if d != "5" else "six" for d in PHRASE_DIGITS)
        await say(client, session_id, clock, wrong)
        clock.advance(0.7)
        status = await status_of(client, session_id, issued["challenge_id"])
    assert status["status"] == "failed"
    assert status["reason"] == "phrase_mismatch"
    assert status["phrase_match"] is False


async def test_a_turn_before_issue_never_binds(clock: FakeClock) -> None:
    app = real_app(clock)
    async with client_for(app) as client:
        session_id = await create_session(client)
        post_evidence(app, session_id, clean_human())
        await say(client, session_id, clock, "four seven two nine one three")
        clock.advance(0.1)
        cid = (await issue(client, session_id, "digits"))["challenge_id"]
        clock.advance(5)
        assert (await status_of(client, session_id, cid))["status"] == "pending"


async def test_the_binding_rule_is_the_same_in_mock_mode(
    v1: httpx.AsyncClient, clock: FakeClock
) -> None:
    """The agent is tested against mock mode, so a stray turn behaves alike."""

    session_id = await create_session(v1)
    cid = (await issue(v1, session_id))["challenge_id"]
    await say(v1, session_id, clock, "okay")
    clock.advance(1.0)
    assert (await status_of(v1, session_id, cid))["status"] == "pending"
    await say(v1, session_id, clock, "four seven two nine one three")
    clock.advance(0.7)
    status = await status_of(v1, session_id, cid)
    assert status["status"] == "passed"  # the script, as before
    assert status["speaker_score"] == 0.83


async def test_an_enroll_prompt_still_binds_the_first_user_turn(
    clock: FakeClock,
) -> None:
    """Enroll records a voice from the turn's clip; its text is not an answer.

    And the room evidence posted to /events is not that clip: the prompt
    stays no_audio without one.
    """

    app = real_app(clock)
    async with client_for(app) as client:
        session_id = await create_session(client)
        post_evidence(app, session_id, clean_human())
        session = app.state.v1.registry.get(session_id)
        session.otp.verified = True
        response = await client.post(
            "/v1/enrollments",
            json={
                "session_id": session_id,
                "user_id": USER,
                "channel": "phone",
                "consent": {"version": "2026-11-04", "verbatim": "yes", "ts_ms": 1},
            },
        )
        assert response.status_code == 200, response.text
        issued = await issue(client, session_id, "enroll")
        await say(client, session_id, clock, "okay")
        clock.advance(0.7)
        status = await status_of(client, session_id, issued["challenge_id"])
    assert status["status"] == "failed"
    assert status["reason"] == "no_audio"


# ------------------------------------------------------ settling, service


async def test_the_phrase_passes_with_the_words_right_and_evidence(
    clock: FakeClock,
) -> None:
    app = real_app(clock)
    async with client_for(app) as client:
        session_id = await create_session(client)
        post_evidence(app, session_id, clean_human())
        issued = await issue(client, session_id)
        await say(client, session_id, clock, spoken(issued))
        cid = issued["challenge_id"]
        assert (await status_of(client, session_id, cid))["status"] == "pending"
        clock.advance(0.7)
        assert await status_of(client, session_id, cid) == {
            "status": "passed",
            "phrase_match": True,
            "speaker_score": 1.0,
            "reason": None,
        }
        # A fresh pass with low spoof: tier 1 allows (policy rule 4).
        decision = await authorize(client, session_id)
    assert decision.decision == "allow"
    assert decision.signals.voiceprint is not None
    assert decision.signals.voiceprint.phrase_match is True


async def test_the_phrase_fails_with_the_words_wrong(clock: FakeClock) -> None:
    app = real_app(clock)
    async with client_for(app) as client:
        session_id = await create_session(client)
        post_evidence(app, session_id, clean_human())
        issued = await issue(client, session_id)
        await say(client, session_id, clock, "one two three four five six apple")
        clock.advance(0.7)
        assert await status_of(client, session_id, issued["challenge_id"]) == {
            "status": "failed",
            "phrase_match": False,
            "speaker_score": 1.0,
            "reason": "phrase_mismatch",
        }
        decision = await authorize(client, session_id)
    assert decision.decision == "deny"
    assert decision.reasons[:2] == ["challenge_failed_replay", "phrase_match false"]


async def test_the_words_right_below_the_voiceprint_band_is_speaker_mismatch(
    clock: FakeClock,
) -> None:
    app = real_app(clock)
    async with client_for(app) as client:
        session_id = await create_session(client)
        post_evidence(
            app,
            session_id,
            with_factor(clean_human(), factor(FactorName.VOICE_MATCH, 0.29)),
        )
        issued = await issue(client, session_id)
        await say(client, session_id, clock, spoken(issued))
        clock.advance(0.7)
        assert await status_of(client, session_id, issued["challenge_id"]) == {
            "status": "failed",
            "phrase_match": True,
            "speaker_score": 0.29,
            "reason": "speaker_mismatch",
        }
        decision = await authorize(client, session_id)
    assert decision.decision == "deny"
    assert decision.reasons[0] == "speaker_mismatch"
    assert any(line.startswith("voiceprint_fail") for line in decision.reasons)


async def test_the_phrase_passes_without_a_voiceprint_on_file(
    clock: FakeClock,
) -> None:
    """No enrolment: voice_match is not_run, the words still decide."""

    app = real_app(clock)
    async with client_for(app) as client:
        session_id = await create_session(client)
        post_evidence(
            app,
            session_id,
            with_factor(
                clean_human(),
                factor(FactorName.VOICE_MATCH, 0.5, FactorVerdict.NOT_RUN),
            ),
        )
        issued = await issue(client, session_id)
        await say(client, session_id, clock, spoken(issued))
        clock.advance(0.7)
        status = await status_of(client, session_id, issued["challenge_id"])
    assert status["status"] == "passed"
    assert status["phrase_match"] is True
    assert status["speaker_score"] is None


async def test_no_evidence_keeps_the_no_audio_outcome(clock: FakeClock) -> None:
    """Ruling F1: a no_audio failure is not a voice fail, however the words went."""

    app = real_app(clock)
    async with client_for(app) as client:
        session_id = await create_session(client)
        issued = await issue(client, session_id)
        await say(client, session_id, clock, spoken(issued))
        clock.advance(0.7)
        status = await status_of(client, session_id, issued["challenge_id"])
        assert status == {
            "status": "failed",
            "phrase_match": None,
            "speaker_score": None,
            "reason": "no_audio",
        }
        # Evidence that lands after the result was read does not rewrite it.
        post_evidence(app, session_id, clean_human())
        assert (await status_of(client, session_id, issued["challenge_id"]))[
            "reason"
        ] == "no_audio"
        decision = await authorize(client, session_id)
    assert decision.reasons[0] not in ("speaker_mismatch", "challenge_failed_replay")


async def test_evidence_arriving_before_resolution_counts(clock: FakeClock) -> None:
    """The text turn lands first, the span a moment later, both inside 700 ms."""

    app = real_app(clock)
    async with client_for(app) as client:
        session_id = await create_session(client)
        issued = await issue(client, session_id)
        await say(client, session_id, clock, spoken(issued))
        clock.advance(0.3)
        assert (await status_of(client, session_id, issued["challenge_id"]))[
            "status"
        ] == "pending"
        post_evidence(app, session_id, clean_human())
        clock.advance(0.4)
        status = await status_of(client, session_id, issued["challenge_id"])
    assert status["status"] == "passed"


async def test_a_passed_challenge_does_not_flip_when_later_spans_fail(
    clock: FakeClock,
) -> None:
    app = real_app(clock)
    async with client_for(app) as client:
        session_id = await create_session(client)
        post_evidence(app, session_id, clean_human())
        issued = await issue(client, session_id)
        await say(client, session_id, clock, spoken(issued))
        clock.advance(0.7)
        assert (await status_of(client, session_id, issued["challenge_id"]))[
            "status"
        ] == "passed"
        post_evidence(
            app,
            session_id,
            with_factor(clean_human(), factor(FactorName.VOICE_MATCH, 0.1)),
            sequence=2,
        )
        status = await status_of(client, session_id, issued["challenge_id"])
    assert status["status"] == "passed"
    assert status["speaker_score"] == 1.0


async def test_an_answered_challenge_resolves_before_a_new_one_supersedes_it(
    clock: FakeClock,
) -> None:
    """Past its 700 ms the old challenge is a result, not a pending one."""

    app = real_app(clock)
    async with client_for(app) as client:
        session_id = await create_session(client)
        post_evidence(app, session_id, clean_human())
        first = await issue(client, session_id)
        await say(client, session_id, clock, spoken(first))
        clock.advance(0.7)
        # No poll in between: the agent went straight to a new challenge.
        await issue(client, session_id)
        status = await status_of(client, session_id, first["challenge_id"])
    assert status["status"] == "passed"


# ------------------------------------------- a store that cannot answer


async def test_a_transient_store_error_leaves_the_challenge_pending() -> None:
    """The result is fixed only once the evidence was read; a hiccup waits."""

    engine = StubEngine(clean_human())
    service, clock = real_service(engine)
    session_id = a_session_id(service, clock)
    issued = service.create_challenge(session_id, ChallengeKind.PHRASE)
    await service.add_turn(session_id, user_turn(clock, issued.say))
    clock.advance(0.7)

    engine.fail_reads = 1
    first = service.challenge_status(session_id, issued.challenge_id)
    assert first.status.value == "pending"
    record = service.registry.get(session_id).challenges[issued.challenge_id]
    assert record.outcome_fixed is False

    second = service.challenge_status(session_id, issued.challenge_id)
    assert second.status.value == "passed"
    assert second.speaker_score == 1.0
    assert record.outcome_fixed is True


async def test_a_store_still_broken_at_expiry_fixes_no_audio() -> None:
    """Pending cannot outlive the challenge: at 45 s it fails closed."""

    engine = StubEngine(clean_human())
    service, clock = real_service(engine)
    session_id = a_session_id(service, clock)
    issued = service.create_challenge(session_id, ChallengeKind.DIGITS)
    await service.add_turn(session_id, user_turn(clock, issued.say))
    clock.advance(45.0)
    engine.fail_reads = 10
    status = service.challenge_status(session_id, issued.challenge_id)
    assert status.status.value == "failed"
    assert status.reason == "no_audio"
    assert status.phrase_match is None
    # Fixed: a recovered store does not turn it into a pass.
    engine.fail_reads = 0
    assert (
        service.challenge_status(session_id, issued.challenge_id).reason == "no_audio"
    )


async def test_an_unknown_engine_session_fixes_no_audio_at_once() -> None:
    """DetectorError is the engine saying nothing was posted: not a hiccup."""

    service, clock = real_service(StubEngine(unknown=True))
    session_id = a_session_id(service, clock)
    issued = service.create_challenge(session_id, ChallengeKind.PHRASE)
    await service.add_turn(session_id, user_turn(clock, issued.say))
    clock.advance(0.7)
    status = service.challenge_status(session_id, issued.challenge_id)
    assert status.status.value == "failed"
    assert status.reason == "no_audio"
    record = service.registry.get(session_id).challenges[issued.challenge_id]
    assert record.outcome_fixed is True


# ------------------------------------------------ one read per request


async def test_view_reads_the_engine_once_however_many_challenges() -> None:
    engine = StubEngine(clean_human())
    service, clock = real_service(engine)
    session_id = a_session_id(service, clock)
    session = service.registry.get(session_id)

    first = service.create_challenge(session_id, ChallengeKind.PHRASE)
    await service.add_turn(session_id, user_turn(clock, first.say))
    clock.advance(0.7)
    service.challenge_status(session_id, first.challenge_id)  # settled: 1 read
    second = service.create_challenge(session_id, ChallengeKind.DIGITS)
    await service.add_turn(session_id, user_turn(clock, second.say))
    clock.advance(0.7)  # due, not yet settled

    engine.reads = 0
    view = service.view(session, clock.now)
    assert engine.reads == 1
    assert [c.status.value for c in view.challenges] == ["passed", "passed"]

    engine.reads = 0
    service.challenge_status(session_id, first.challenge_id)
    service.challenge_status(session_id, second.challenge_id)
    assert engine.reads == 0  # both fixed: nothing left to read


# ----------------------------------------------------------- mock mode


async def test_mock_mode_never_consults_the_engine_for_a_challenge(
    clock: FakeClock,
) -> None:
    app = build_app(base_settings(), clock, io.StringIO())

    class Exploding:
        def factors_for(self, session_id: str) -> object:
            raise AssertionError("mock mode read the engine")

    app.state.v1.engine = Exploding()
    async with client_for(app) as client:
        session_id = await create_session(client, scenario="real_cfo")
        issued = await issue(client, session_id)
        await say(client, session_id, clock, spoken(issued))
        clock.advance(0.7)
        status = await status_of(client, session_id, issued["challenge_id"])
        assert status["status"] == "passed"
        assert status["speaker_score"] == 0.83  # the script, untouched
        decision = await authorize(client, session_id)
    assert decision.decision == "allow"


# --------------------------------------------------- over HTTP, end to end


def wav_data_uri(samples: np.ndarray, rate: int) -> str:
    pcm = (np.clip(samples, -1.0, 0.999969) * 32768.0).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def enrol_reference(app: FastAPI, samples: np.ndarray) -> None:
    """A print on file for u_eric's phone channel, as a /v1 enrollment writes it."""

    enrolment = storable_enrolment(
        USER,
        voiceprint_from(build_context(samples, RATE)),
        source="test",
        channel="phone",  # keyed "u_eric:phone", as _store_enrollment_print does
    )
    assert enrolment is not None
    app.state.voiceprints.put(enrolment, replace=True)


async def post_span(
    client: httpx.AsyncClient, session_id: str, samples: np.ndarray, sequence: int
) -> None:
    body = {
        "session_id": session_id,
        "event": "speech_segment",
        "sequence": sequence,
        "occurred_at": datetime.now(UTC).isoformat(),
        "claimed_user_id": USER,
        "audio": {
            "track_id": "caller",
            "start_ms": 0,
            "end_ms": 4000,
            "sample_rate_hz": RATE,
            "encoding": "pcm_s16le",
            "uri": wav_data_uri(samples, RATE),
        },
    }
    posted = await client.post("/events", json=body)
    assert posted.status_code == 200, posted.text
    assert posted.json()["factors"], "the span was not analysed"


async def test_end_to_end_the_right_words_pass_and_tier_1_allows(
    clock: FakeClock,
) -> None:
    """POST /events with a WAV span, the challenge, the agent's turn, authorize."""

    log_stream = io.StringIO()
    app = real_app(clock, log_stream)
    voice = live_speech(4.0, RATE, seed=0)
    enrol_reference(app, voice)
    async with client_for(app) as client:
        session_id = await create_session(client)
        await post_span(client, session_id, voice, sequence=1)
        issued = await issue(client, session_id)
        await say(client, session_id, clock, spoken(issued))
        clock.advance(0.7)
        status = await status_of(client, session_id, issued["challenge_id"])
        assert status["status"] == "passed", status
        assert status["phrase_match"] is True
        assert status["speaker_score"] is not None
        assert status["speaker_score"] >= VOICEPRINT_FAIL
        decision = await authorize(client, session_id)
    assert decision.decision == "allow", decision.reasons
    assert decision.signals.deepfake is not None
    assert decision.signals.deepfake.spoof_prob < SPOOF_LOW
    # Neither the say nor the caller's words reached a log line.
    for word in issued["say"].replace(",", " ").split():
        if word.isalpha():
            assert word not in log_stream.getvalue()


async def test_end_to_end_the_wrong_words_deny_challenge_failed_replay(
    clock: FakeClock,
) -> None:
    app = real_app(clock)
    voice = live_speech(4.0, RATE, seed=0)
    enrol_reference(app, voice)
    async with client_for(app) as client:
        session_id = await create_session(client)
        await post_span(client, session_id, voice, sequence=1)
        issued = await issue(client, session_id)
        await say(client, session_id, clock, "one two three four five six apple")
        clock.advance(0.7)
        status = await status_of(client, session_id, issued["challenge_id"])
        assert status["status"] == "failed"
        assert status["reason"] == "phrase_mismatch"
        decision = await authorize(client, session_id)
    assert decision.decision == "deny"
    assert decision.reasons[0] == "challenge_failed_replay"


async def test_end_to_end_no_spans_keeps_no_audio(clock: FakeClock) -> None:
    app = real_app(clock)
    async with client_for(app) as client:
        session_id = await create_session(client)
        issued = await issue(client, session_id)
        await say(client, session_id, clock, spoken(issued))
        clock.advance(0.7)
        status = await status_of(client, session_id, issued["challenge_id"])
        assert status == {
            "status": "failed",
            "phrase_match": None,
            "speaker_score": None,
            "reason": "no_audio",
        }
        decision = await authorize(client, session_id)
    assert decision.decision == "deny"
    assert decision.reasons[0] == "no_audio"
