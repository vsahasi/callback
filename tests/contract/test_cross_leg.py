"""The cross-leg speaker check: the caller must be the person who answered.

A callback session has two voices on it. Spans on the `callback` track are
the person who picked up when the service rang the number on file, which is
the one voice in the whole flow that is trustworthy by construction. Spans on
the `caller` track are whoever rang in. If those are two different people,
nothing else about the call matters: someone claiming to be the vendor rang
in, and the real vendor answered the callback and said yes.

Two gates, both in seconds of speech per leg. At 2.0 s the legs are compared
and a mismatch steps the caller up. At 3.0 s, with the comparator confident,
a mismatch refuses the call outright and no code rescues it. The numbers
behind those gates are in `detector.legs`.

These drive the real-mode app end to end with two synthetic voices, and pin
the state-machine and codec pieces underneath.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import httpx
import numpy as np
import pytest

from detector import legs, scoring
from detector.analysis.dsp import build_context
from detector.analysis.synthetic import SpeechStyle, live_speech
from detector.analysis.voice import compare_prints
from detector.analysis.voiceprint import VoicePrint, voiceprint_from
from detector.app import create_app
from detector.bridge import signals_from_engine, signals_from_factors
from detector.evidence import analysed_evidence
from detector.state import DetectorEngine
from detector.store import Session, session_from_json, session_to_json
from detector.v1.policy import VOICEPRINT_FAIL
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
from tests.contract.test_real_mode import AUTH, RATE, pcm_data_uri, real_settings
from tests.contract.test_store import a_fully_populated_session

# Two talkers. The second sits nine semitones above the first, which is the
# same "obviously different voice" the enrolment tests use.
LOW = SpeechStyle(f0_hz=132.0)
HIGH = SpeechStyle(f0_hz=215.0)
# The same talker, four semitones up: the reviewer's within-speaker case.
SHIFTED_4ST = SpeechStyle(f0_hz=132.0 * 2 ** (4 / 12))

# A six-second clip carries 3.6 to 4.0 s of speech on this corpus, clear of
# the 3.0 s confirmation gate at both 8 and 16 kHz. A four-second clip carries
# 2.3 to 2.8 s at 8 kHz: enough to compare, not enough to refuse on.
FULL = 6.0
THIN = 4.0

REVIEW = "/review/api/sessions"


def voice(
    style: SpeechStyle, seed: int, duration_s: float = THIN, rate: int = RATE
) -> np.ndarray:
    return live_speech(duration_s, rate, seed=seed, style=style)


@pytest.fixture
async def real_client() -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(real_settings())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://legs.test",
        headers=AUTH,
    ) as client:
        yield client


class Call:
    """Drive one session, naming the leg each span came from."""

    def __init__(self, client: httpx.AsyncClient, session_id: str) -> None:
        self.client = client
        self.session_id = session_id
        self.sequence = 0

    async def send(self, event: str, **extra) -> httpx.Response:
        self.sequence += 1
        body: dict[str, object] = {
            "session_id": self.session_id,
            "sequence": self.sequence,
            "event": event,
            "occurred_at": f"2026-11-09T10:{self.sequence:02d}:00+00:00",
        }
        body.update(extra)
        return await self.client.post("/events", json=body)

    async def start(self) -> None:
        response = await self.send(
            "session_start", claimed_user_id="eric@vendor.example"
        )
        assert response.status_code == 200, response.text

    async def speech(
        self, track_id: str, samples: np.ndarray, rate: int = RATE
    ) -> dict:
        response = await self.send(
            "speech_segment",
            audio={
                "track_id": track_id,
                "start_ms": self.sequence * 6000,
                "end_ms": (self.sequence + 1) * 6000,
                "sample_rate_hz": rate,
                "encoding": "pcm_s16le",
                "uri": pcm_data_uri(samples),
            },
        )
        assert response.status_code == 200, response.text
        return response.json()

    async def factors(self) -> dict:
        response = await self.client.get(f"/factors/{self.session_id}")
        assert response.status_code == 200, response.text
        return response.json()

    async def review(self) -> dict:
        response = await self.client.get(f"{REVIEW}/{self.session_id}")
        assert response.status_code == 200, response.text
        return response.json()


def factor(body: dict, name: FactorName) -> dict:
    return next(f for f in body["factors"] if f["name"] == name.value)


def _print(samples: np.ndarray, rate: int = RATE) -> VoicePrint:
    return voiceprint_from(build_context(samples, rate))


# ---------------------------------------------------------------- end to end


async def test_two_different_voices_on_the_two_legs_deny(
    real_client: httpx.AsyncClient,
) -> None:
    """The attack this exists for: an impostor rings in, the real vendor answers."""

    call = Call(real_client, "sess-legs-mismatch")
    await call.start()
    await call.speech("caller", voice(HIGH, seed=1, duration_s=FULL))
    body = await call.speech("callback", voice(LOW, seed=2, duration_s=FULL))

    match = factor(body, FactorName.VOICE_MATCH)
    assert match["verdict"] == FactorVerdict.FAIL.value
    assert match["score"] <= 0.32
    assert match["detail"].startswith("Cross-leg speaker mismatch")
    assert "caller" in match["detail"] and "callback" in match["detail"]

    assert body["decision"] == Decision.DENY.value
    assert body["next_action"] == NextAction.COMPLETE_DENY.value
    assert "cross-leg" in body["reason"].lower()
    assert "callback" in body["reason"]


async def test_the_mismatch_denies_even_when_every_other_signal_passes(
    real_client: httpx.AsyncClient,
) -> None:
    """Live human speech on both legs passes synthesis, replay and liveness.

    The deny has to come from the cross-leg check alone, not from the blend
    happening to land high.
    """

    call = Call(real_client, "sess-legs-only-signal")
    await call.start()
    await call.speech("caller", voice(HIGH, seed=1, duration_s=FULL))
    body = await call.speech("callback", voice(LOW, seed=2, duration_s=FULL))

    for name in (FactorName.SYNTHETIC_SPEECH, FactorName.REPLAY, FactorName.LIVENESS):
        assert factor(body, name)["verdict"] != FactorVerdict.FAIL.value, name
    assert body["decision"] == Decision.DENY.value
    # No step-up on offer: the real vendor is holding the phone that would
    # receive the code, so a correct code proves nothing about the caller.
    assert body["otp"] is None


async def test_the_same_voice_on_both_legs_passes(
    real_client: httpx.AsyncClient,
) -> None:
    call = Call(real_client, "sess-legs-match")
    await call.start()
    await call.speech("caller", voice(LOW, seed=1))
    body = await call.speech("callback", voice(LOW, seed=2))

    match = factor(body, FactorName.VOICE_MATCH)
    assert match["verdict"] == FactorVerdict.PASS.value
    assert match["score"] >= 0.62
    assert "mismatch" not in match["detail"]
    assert "caller" in match["detail"] and "callback" in match["detail"]
    assert body["decision"] != Decision.DENY.value


async def test_one_leg_alone_runs_no_cross_leg_check(
    real_client: httpx.AsyncClient,
) -> None:
    call = Call(real_client, "sess-legs-one-track")
    await call.start()
    await call.speech("caller", voice(LOW, seed=1))
    body = await call.speech("caller", voice(HIGH, seed=2))

    match = factor(body, FactorName.VOICE_MATCH)
    # Nothing is enrolled for this identity and there is no callback leg, so
    # the factor stays exactly where it was before this check existed.
    assert match["verdict"] == FactorVerdict.NOT_RUN.value
    assert "mismatch" not in (match["detail"] or "")
    assert body["decision"] != Decision.DENY.value


async def test_too_little_speech_on_a_leg_does_not_run_the_check(
    real_client: httpx.AsyncClient,
) -> None:
    """A one-second callback leg cannot carry two seconds of speech."""

    call = Call(real_client, "sess-legs-thin")
    await call.start()
    await call.speech("caller", voice(HIGH, seed=1))
    body = await call.speech("callback", voice(LOW, seed=2, duration_s=1.0))

    match = factor(body, FactorName.VOICE_MATCH)
    assert match["verdict"] == FactorVerdict.NOT_RUN.value
    assert "mismatch" not in (match["detail"] or "")


async def test_a_mismatch_on_a_thin_leg_steps_up_rather_than_denying(
    real_client: httpx.AsyncClient,
) -> None:
    """Between the 2.0 s and 3.0 s gates a mismatch is a challenge, not a deny.

    The legs differ, and the factor says so, but with under three seconds of
    speech on the callback leg the comparator's margin is too narrow to refuse
    the call on. The caller is stepped up. Once the callback leg carries
    enough speech the same mismatch becomes the deny it always was.
    """

    call = Call(real_client, "sess-legs-thin-mismatch")
    await call.start()
    await call.speech("caller", voice(HIGH, seed=1, duration_s=FULL))
    body = await call.speech("callback", voice(LOW, seed=2, duration_s=THIN))

    match = factor(body, FactorName.VOICE_MATCH)
    assert match["verdict"] == FactorVerdict.FAIL.value
    assert match["score"] <= 0.32
    assert "not confident enough" in match["detail"]
    assert "3.0" in match["detail"]
    assert body["decision"] == Decision.CHALLENGE.value
    assert body["next_action"] == NextAction.REQUEST_OTP.value
    assert body["otp"] is not None

    body = await call.speech("callback", voice(LOW, seed=3, duration_s=FULL))
    assert factor(body, FactorName.VOICE_MATCH)["detail"].startswith(
        "Cross-leg speaker mismatch"
    )
    assert body["decision"] == Decision.DENY.value
    assert body["next_action"] == NextAction.COMPLETE_DENY.value


@pytest.mark.parametrize("duration_s", [THIN, FULL])
async def test_the_same_talker_shifted_four_semitones_is_not_denied(
    real_client: httpx.AsyncClient, duration_s: float
) -> None:
    """The reviewer's margin case: one talker, pitch moved four semitones.

    Measured on this corpus at 8 kHz, seeds 1 and 2: 0.3216 at 4 s and 0.3786
    at 6 s, both inside the inconclusive band. The margin is narrow (other
    seed pairs at 6 s score 0.2891, a confident FAIL), which is why the deny
    also wants three seconds of speech on each leg and why the step-up exists.
    """

    call = Call(real_client, f"sess-legs-shifted-{int(duration_s)}")
    await call.start()
    await call.speech("caller", voice(LOW, seed=1, duration_s=duration_s))
    body = await call.speech(
        "callback", voice(SHIFTED_4ST, seed=2, duration_s=duration_s)
    )

    match = factor(body, FactorName.VOICE_MATCH)
    assert match["verdict"] == FactorVerdict.INCONCLUSIVE.value
    assert body["decision"] != Decision.DENY.value
    assert body["next_action"] != NextAction.COMPLETE_DENY.value


@pytest.mark.parametrize("rate", [8000, 16000])
async def test_speech_on_a_leg_accumulates_through_both_gates(
    real_client: httpx.AsyncClient, rate: int
) -> None:
    """Short callback spans add up, and the two gates fall in the same places.

    The agent sends 16 kHz; the rest of this file sends 8 kHz. A one-second
    synthetic clip carries 0.4 to 0.85 s of speech at either rate, so three
    clips sit under the 2.0 s compare gate, the fourth crosses it (a
    mismatch, stepped up) and the fifth crosses the 3.0 s confirmation gate
    (the deny).
    """

    call = Call(real_client, f"sess-legs-accumulate-{rate}")
    await call.start()
    await call.speech("caller", voice(HIGH, seed=1, duration_s=FULL, rate=rate), rate)
    for seed in (2, 3, 4):
        body = await call.speech(
            "callback", voice(LOW, seed=seed, duration_s=1.0, rate=rate), rate
        )
        assert factor(body, FactorName.VOICE_MATCH)["verdict"] == (
            FactorVerdict.NOT_RUN.value
        ), seed
        assert body["decision"] != Decision.DENY.value

    compared = await call.speech(
        "callback", voice(LOW, seed=5, duration_s=1.0, rate=rate), rate
    )
    assert factor(compared, FactorName.VOICE_MATCH)["verdict"] == (
        FactorVerdict.FAIL.value
    )
    assert compared["decision"] == Decision.CHALLENGE.value

    confirmed = await call.speech(
        "callback", voice(LOW, seed=6, duration_s=1.0, rate=rate), rate
    )
    # The factor on the wire may still be the thinner check's: worst score
    # wins the merge, and the earlier FAIL can score lower than the confirmed
    # one. The deny does not depend on it. It reads the session's record.
    assert factor(confirmed, FactorName.VOICE_MATCH)["verdict"] == (
        FactorVerdict.FAIL.value
    )
    assert confirmed["decision"] == Decision.DENY.value
    assert confirmed["next_action"] == NextAction.COMPLETE_DENY.value
    assert "cross-leg speaker mismatch" in confirmed["reason"].lower()


async def test_the_factor_is_readable_afterwards_and_fails_the_v1_voiceprint(
    real_client: httpx.AsyncClient,
) -> None:
    """`/v1` reads this through `factors_for` and the bridge, so it has to be
    there, and it has to read below the policy's voiceprint fail line."""

    call = Call(real_client, "sess-legs-factors")
    await call.start()
    await call.speech("caller", voice(HIGH, seed=1, duration_s=FULL))
    await call.speech("callback", voice(LOW, seed=2, duration_s=FULL))

    body = await call.factors()
    match = factor(body, FactorName.VOICE_MATCH)
    assert match["verdict"] == FactorVerdict.FAIL.value
    assert match["detail"].startswith("Cross-leg speaker mismatch")
    assert body["decision"] == Decision.DENY.value
    assert "cross-leg" in body["reason"].lower()

    signals = signals_from_factors([Factor.model_validate(f) for f in body["factors"]])
    assert signals.voiceprint is not None
    assert signals.voiceprint < VOICEPRINT_FAIL


async def test_the_decision_does_not_depend_on_which_leg_speaks_first(
    real_client: httpx.AsyncClient,
) -> None:
    call = Call(real_client, "sess-legs-callback-first")
    await call.start()
    await call.speech("callback", voice(LOW, seed=2, duration_s=FULL))
    body = await call.speech("caller", voice(HIGH, seed=1, duration_s=FULL))

    assert factor(body, FactorName.VOICE_MATCH)["verdict"] == FactorVerdict.FAIL.value
    assert body["decision"] == Decision.DENY.value


async def test_a_correct_code_does_not_rescue_a_speaker_mismatch(
    real_client: httpx.AsyncClient,
) -> None:
    """The session is terminal once the legs disagree; a code is refused."""

    call = Call(real_client, "sess-legs-no-otp")
    await call.start()
    await call.speech("caller", voice(HIGH, seed=1, duration_s=FULL))
    await call.speech("callback", voice(LOW, seed=2, duration_s=FULL))

    response = await call.send("otp_request")
    assert response.status_code != 200
    body = await call.factors()
    assert body["decision"] == Decision.DENY.value


async def test_the_review_bundle_renders_the_cross_leg_deny_as_fatal(
    real_client: httpx.AsyncClient,
) -> None:
    """A reviewer sees the FATAL flag on voice_match and is told why."""

    call = Call(real_client, "sess-legs-review")
    await call.start()
    await call.speech("caller", voice(HIGH, seed=1, duration_s=FULL))
    await call.speech("callback", voice(LOW, seed=2, duration_s=FULL))

    bundle = await call.review()
    fatal = [f for f in bundle["factors"] if f["fatal"]]
    assert [f["name"] for f in fatal] == [FactorName.VOICE_MATCH.value]
    assert fatal[0]["fatal_candidate"] is True
    assert bundle["factors"][0]["name"] == FactorName.VOICE_MATCH.value

    joined = " ".join(bundle["why"])
    assert "cross-leg" in joined.lower()
    assert "callback" in joined
    assert "does not rescue" in joined
    assert "fatal threshold" not in joined  # the 0.10 rule is not what fired
    assert bundle["human_review"]["recommendation"] == "do_not_release"


# ------------------------------------------------------- the engine, directly


def _request(session_id: str, sequence: int, event: EventType, **extra) -> EventRequest:
    return EventRequest(
        session_id=session_id,
        sequence=sequence,
        event=event,
        occurred_at=datetime(2026, 11, 9, 10, sequence, tzinfo=UTC),
        **extra,
    )


def _audio(track_id: str) -> dict:
    return {
        "track_id": track_id,
        "start_ms": 0,
        "end_ms": 6000,
        "sample_rate_hz": RATE,
        "encoding": "pcm_s16le",
    }


def _clean_span(voice_match: Factor | None = None) -> list[Factor]:
    """What the analysis engine reports for live human speech."""

    observed = [
        Factor(
            name=FactorName.SYNTHETIC_SPEECH,
            score=0.9,
            weight=0.15,
            verdict=FactorVerdict.PASS,
        ),
        Factor(
            name=FactorName.REPLAY, score=0.9, weight=0.2, verdict=FactorVerdict.PASS
        ),
        Factor(
            name=FactorName.LIVENESS, score=0.9, weight=0.1, verdict=FactorVerdict.PASS
        ),
    ]
    if voice_match is not None:
        observed.insert(0, voice_match)
    return observed


def test_an_enrolment_fail_scoring_lower_does_not_hide_the_mismatch() -> None:
    """The reviewer's first flaw.

    `combine_factors` keeps the lowest `voice_match`. When the enrolment
    comparison fails harder than the cross-leg one, the enrolment factor is
    the one on the wire. The cross-leg result lives on the session, not in
    the factor's detail string, so the deny still fires, the reason still
    names it, and `/v1` still reads a voiceprint below its fail line.
    """

    engine = DetectorEngine(real_settings())
    sid = "sess-legs-enrolment-lower"
    engine.handle_event(
        _request(
            sid, 1, EventType.SESSION_START, claimed_user_id="eric@vendor.example"
        ),
        scenario_header=None,
        idempotency_key=None,
    )
    enrolment_fail = Factor(
        name=FactorName.VOICE_MATCH,
        score=0.05,
        weight=0.35,
        verdict=FactorVerdict.FAIL,
        detail="Voice comparison: caller's pitch register is 9.0 semitones from the reference",
    )
    engine.handle_event(
        _request(sid, 2, EventType.SPEECH_SEGMENT, audio=_audio("caller")),
        scenario_header=None,
        idempotency_key=None,
        observed_factors=_clean_span(enrolment_fail),
        observed_voice=_print(voice(HIGH, seed=1, duration_s=FULL)),
    )
    response = engine.handle_event(
        _request(sid, 3, EventType.SPEECH_SEGMENT, audio=_audio("callback")),
        scenario_header=None,
        idempotency_key=None,
        observed_factors=_clean_span(),
        observed_voice=_print(voice(LOW, seed=2, duration_s=FULL)),
    )

    match = next(f for f in response.factors if f.name is FactorName.VOICE_MATCH)
    assert match.score == pytest.approx(0.05)  # worst-of kept the enrolment fail
    assert response.decision is Decision.DENY
    assert response.state is SessionState.DENIED
    assert response.otp is None
    assert "cross-leg" in (response.reason or "").lower()
    assert "callback" in (response.reason or "")

    stored = engine.store.get(sid)
    assert stored.cross_leg is not None
    assert stored.cross_leg.confirmed
    assert stored.cross_leg.verdict is FactorVerdict.FAIL

    signals = signals_from_engine(engine, sid)
    assert signals.heard
    assert signals.voiceprint is not None
    assert signals.voiceprint < VOICEPRINT_FAIL


def test_a_mismatch_on_thin_legs_is_rescuable_by_a_code() -> None:
    """Unconfirmed, the mismatch is an ordinary voice_match FAIL: a step-up."""

    engine = DetectorEngine(real_settings())
    sid = "sess-legs-thin-rescuable"
    engine.handle_event(
        _request(sid, 1, EventType.SESSION_START),
        scenario_header=None,
        idempotency_key=None,
    )
    engine.handle_event(
        _request(sid, 2, EventType.SPEECH_SEGMENT, audio=_audio("caller")),
        scenario_header=None,
        idempotency_key=None,
        observed_factors=_clean_span(),
        observed_voice=_print(voice(HIGH, seed=1, duration_s=FULL)),
    )
    response = engine.handle_event(
        _request(sid, 3, EventType.SPEECH_SEGMENT, audio=_audio("callback")),
        scenario_header=None,
        idempotency_key=None,
        observed_factors=_clean_span(),
        observed_voice=_print(voice(LOW, seed=2, duration_s=THIN)),
    )
    assert response.decision is Decision.CHALLENGE
    assert response.otp is not None
    stored = engine.store.get(sid)
    assert stored.cross_leg is not None
    assert stored.cross_leg.verdict is FactorVerdict.FAIL
    assert not stored.cross_leg.confirmed

    code = stored.otp.code
    submitted = engine.handle_event(
        _request(sid, 4, EventType.OTP_SUBMIT, otp_code=code),
        scenario_header=None,
        idempotency_key=None,
    )
    assert submitted.decision is Decision.ALLOW


# ---------------------------------------------------------------- the pieces


@pytest.mark.parametrize(
    ("track_id", "expected"),
    [
        ("caller", legs.CALLER),
        ("callback", legs.CALLBACK),
        ("Caller", legs.CALLER),
        ("caller-0", legs.CALLER),
        ("callback:1", legs.CALLBACK),
        ("CALLBACK_7", legs.CALLBACK),
        ("track-1", None),
        ("v1-turn", None),
        ("callers", None),
        ("", None),
    ],
)
def test_leg_of_reads_the_track_id(track_id: str, expected: str | None) -> None:
    assert legs.leg_of(track_id) == expected


def _legs(caller: np.ndarray, callback: np.ndarray) -> dict[str, legs.LegPrint]:
    return {
        legs.CALLER: legs.LegPrint.from_print(_print(caller)),
        legs.CALLBACK: legs.LegPrint.from_print(_print(callback)),
    }


def test_a_confident_fail_on_full_legs_is_confirmed_and_fatal() -> None:
    check = legs.compare_legs(
        _legs(voice(HIGH, seed=1, duration_s=FULL), voice(LOW, seed=2, duration_s=FULL))
    )
    assert check is not None
    assert check.verdict is FactorVerdict.FAIL
    assert check.confident
    assert check.caller_seconds >= legs.MIN_CONFIRMED_LEG_SPEECH_SECONDS
    assert check.callback_seconds >= legs.MIN_CONFIRMED_LEG_SPEECH_SECONDS
    assert check.confirmed

    result = check.factor()
    assert result.name is FactorName.VOICE_MATCH
    assert result.verdict is FactorVerdict.FAIL
    assert result.weight == pytest.approx(0.35)
    assert result.detail is not None and result.detail.startswith(
        "Cross-leg speaker mismatch"
    )
    assert len(result.detail) <= 500

    fatal = check.fatal_factor()
    assert fatal is not None
    assert scoring.fatal_factor([result], speaker_mismatch=fatal) is fatal
    # The fatal verdict does not depend on the factor list at all: an
    # enrolment fail may have replaced the cross-leg factor there.
    assert scoring.fatal_factor([], speaker_mismatch=fatal) is fatal


def test_a_fail_on_a_thin_leg_is_not_confirmed() -> None:
    check = legs.compare_legs(
        _legs(voice(HIGH, seed=1, duration_s=FULL), voice(LOW, seed=2, duration_s=THIN))
    )
    assert check is not None
    assert check.verdict is FactorVerdict.FAIL
    assert check.confident
    assert check.callback_seconds < legs.MIN_CONFIRMED_LEG_SPEECH_SECONDS
    assert not check.confirmed
    assert check.fatal_factor() is None

    result = check.factor()
    assert result.verdict is FactorVerdict.FAIL  # a step-up, like an enrolment fail
    assert "not confident enough" in (result.detail or "")
    assert scoring.fatal_factor([result]) is None


def test_a_matching_pair_is_not_fatal() -> None:
    check = legs.compare_legs(_legs(voice(LOW, seed=1), voice(LOW, seed=2)))
    assert check is not None
    assert check.verdict is FactorVerdict.PASS
    assert not check.confirmed
    assert check.fatal_factor() is None
    assert check.factor().verdict is FactorVerdict.PASS


def test_an_ordinary_voice_match_fail_is_not_fatal() -> None:
    """Only the session's cross-leg state is fatal; an enrolment fail steps up."""

    plain = Factor(
        name=FactorName.VOICE_MATCH,
        score=0.1,
        weight=0.35,
        verdict=FactorVerdict.FAIL,
        detail="Cross-leg speaker mismatch: a detail string proves nothing",
    )
    assert scoring.fatal_factor([plain]) is None
    evidence = analysed_evidence([plain])
    assert evidence.otp_can_rescue


def test_the_evidence_names_the_mismatch_over_a_lower_scoring_factor() -> None:
    """The reviewer's evidence.py:168: the reason led with the lowest score."""

    check = legs.compare_legs(
        _legs(voice(HIGH, seed=1, duration_s=FULL), voice(LOW, seed=2, duration_s=FULL))
    )
    assert check is not None and check.confirmed
    lower = Factor(
        name=FactorName.VOICE_MATCH,
        score=0.05,
        weight=0.35,
        verdict=FactorVerdict.FAIL,
        detail="Voice comparison: caller's pitch register is 9.0 semitones from the reference",
    )
    also = Factor(
        name=FactorName.LIVENESS, score=0.2, weight=0.1, verdict=FactorVerdict.FAIL
    )

    evidence = analysed_evidence(
        [lower, also], spans=2, speaker_mismatch=check.fatal_factor()
    )
    assert not evidence.otp_can_rescue
    assert evidence.reason.lower().startswith("cross-leg speaker mismatch")
    assert "callback" in evidence.reason
    assert "liveness" in evidence.reason  # the other failure is still listed
    assert "2 audio spans" in evidence.reason


def test_the_check_needs_both_legs_with_enough_speech() -> None:
    full = legs.LegPrint.from_print(_print(voice(LOW, seed=1)))
    thin = legs.LegPrint.from_print(_print(voice(HIGH, seed=2, duration_s=1.0)))

    assert legs.compare_legs({legs.CALLER: full}) is None
    assert legs.compare_legs({legs.CALLBACK: full}) is None
    assert legs.compare_legs({legs.CALLER: full, legs.CALLBACK: thin}) is None
    assert thin.voice.speech_seconds < legs.MIN_LEG_SPEECH_SECONDS


def test_compare_prints_reports_its_own_confidence() -> None:
    """The flag the confirmation gate reads, not a re-derivation of it."""

    confident = compare_prints(_print(voice(HIGH, seed=1)), _print(voice(LOW, seed=2)))
    assert confident.measurements["confident"] == 1.0
    assert confident.verdict is FactorVerdict.FAIL

    thin = compare_prints(
        _print(voice(HIGH, seed=1)), _print(voice(LOW, seed=2, duration_s=1.0))
    )
    assert thin.measurements["confident"] == 0.0
    assert thin.verdict is FactorVerdict.INCONCLUSIVE


def test_legs_combine_as_spans_arrive() -> None:
    first = _print(voice(LOW, seed=1, duration_s=1.0))
    second = _print(voice(LOW, seed=2, duration_s=1.0))

    held = legs.LegPrint.from_print(first).absorb(second)
    assert held.spans == 2
    assert held.voice.speech_seconds == pytest.approx(
        first.speech_seconds + second.speech_seconds
    )
    assert held.voice.duration_s == pytest.approx(first.duration_s + second.duration_s)


def test_legs_and_the_check_survive_the_session_codec() -> None:
    session: Session = a_fully_populated_session()
    session.legs = {
        legs.CALLER: legs.LegPrint.from_print(_print(voice(HIGH, seed=1))),
        legs.CALLBACK: legs.LegPrint.from_print(_print(voice(LOW, seed=2))).absorb(
            _print(voice(LOW, seed=3))
        ),
    }
    session.cross_leg = legs.compare_legs(session.legs)
    assert session.cross_leg is not None

    document = session_to_json(session)
    restored = session_from_json(json.loads(json.dumps(document)))

    assert restored.legs == session.legs
    assert restored.cross_leg == session.cross_leg
    assert restored == session
    assert json.dumps(session_to_json(restored), sort_keys=True) == json.dumps(
        document, sort_keys=True
    )


def test_a_document_without_legs_still_reads() -> None:
    """Written by a replica from before this check existed."""

    document = session_to_json(a_fully_populated_session())
    del document["legs"]
    del document["cross_leg"]
    restored = session_from_json(document)
    assert restored.legs == {}
    assert restored.cross_leg is None
