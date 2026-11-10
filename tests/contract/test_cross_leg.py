"""The cross-leg speaker check: the caller must be the person who answered.

A callback session has two voices on it. Spans on the `callback` track are
the person who picked up when the service rang the number on file, which is
the one voice in the whole flow that is trustworthy by construction. Spans on
the `caller` track are whoever rang in. If those are two different people,
nothing else about the call matters: someone claiming to be the vendor rang
in, and the real vendor answered the callback and said yes.

These drive the real-mode app end to end with two synthetic voices, and pin
the state-machine and codec pieces underneath.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import numpy as np
import pytest

from detector import legs, scoring
from detector.analysis.dsp import build_context
from detector.analysis.synthetic import SpeechStyle, live_speech
from detector.analysis.voiceprint import voiceprint_from
from detector.app import create_app
from detector.store import Session, session_from_json, session_to_json
from shared.contract import Decision, Factor, FactorName, FactorVerdict, NextAction
from tests.contract.test_real_mode import AUTH, RATE, pcm_data_uri, real_settings
from tests.contract.test_store import a_fully_populated_session

# Two talkers. The second sits nine semitones above the first, which is the
# same "obviously different voice" the enrolment tests use.
LOW = SpeechStyle(f0_hz=132.0)
HIGH = SpeechStyle(f0_hz=215.0)


def voice(style: SpeechStyle, seed: int, duration_s: float = 4.0) -> np.ndarray:
    return live_speech(duration_s, RATE, seed=seed, style=style)


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

    async def speech(self, track_id: str, samples: np.ndarray) -> dict:
        response = await self.send(
            "speech_segment",
            audio={
                "track_id": track_id,
                "start_ms": self.sequence * 4000,
                "end_ms": (self.sequence + 1) * 4000,
                "sample_rate_hz": RATE,
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


def factor(body: dict, name: FactorName) -> dict:
    return next(f for f in body["factors"] if f["name"] == name.value)


# ---------------------------------------------------------------- end to end


async def test_two_different_voices_on_the_two_legs_deny(
    real_client: httpx.AsyncClient,
) -> None:
    """The attack this exists for: an impostor rings in, the real vendor answers."""

    call = Call(real_client, "sess-legs-mismatch")
    await call.start()
    await call.speech("caller", voice(HIGH, seed=1))
    body = await call.speech("callback", voice(LOW, seed=2))

    match = factor(body, FactorName.VOICE_MATCH)
    assert match["verdict"] == FactorVerdict.FAIL.value
    assert match["score"] <= 0.32
    assert match["detail"].startswith("speaker_mismatch:")
    assert "caller" in match["detail"] and "callback" in match["detail"]

    assert body["decision"] == Decision.DENY.value
    assert body["next_action"] == NextAction.COMPLETE_DENY.value
    assert "speaker_mismatch" in body["reason"]
    assert "caller" in body["reason"] and "callback" in body["reason"]


async def test_the_mismatch_denies_even_when_every_other_signal_passes(
    real_client: httpx.AsyncClient,
) -> None:
    """Live human speech on both legs passes synthesis, replay and liveness.

    The deny has to come from the cross-leg factor alone, not from the blend
    happening to land high.
    """

    call = Call(real_client, "sess-legs-only-signal")
    await call.start()
    await call.speech("caller", voice(HIGH, seed=1))
    body = await call.speech("callback", voice(LOW, seed=2))

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
    assert "speaker_mismatch" not in match["detail"]
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
    assert "speaker_mismatch" not in (match["detail"] or "")
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
    assert "speaker_mismatch" not in (match["detail"] or "")


async def test_speech_on_a_leg_accumulates_until_there_is_enough(
    real_client: httpx.AsyncClient,
) -> None:
    """Short callback spans add up to one comparable print.

    A one-second synthetic clip carries about half a second of speech, so
    three of them sit just under the two-second gate and the fourth crosses it.
    """

    call = Call(real_client, "sess-legs-accumulate")
    await call.start()
    await call.speech("caller", voice(HIGH, seed=1))
    await call.speech("callback", voice(LOW, seed=2, duration_s=1.0))
    await call.speech("callback", voice(LOW, seed=3, duration_s=1.0))
    held = await call.speech("callback", voice(LOW, seed=4, duration_s=1.0))
    assert factor(held, FactorName.VOICE_MATCH)["verdict"] == (
        FactorVerdict.NOT_RUN.value
    )
    body = await call.speech("callback", voice(LOW, seed=5, duration_s=1.0))

    match = factor(body, FactorName.VOICE_MATCH)
    assert match["verdict"] == FactorVerdict.FAIL.value
    assert match["detail"].startswith("speaker_mismatch:")
    assert body["decision"] == Decision.DENY.value


async def test_the_factor_is_readable_afterwards(
    real_client: httpx.AsyncClient,
) -> None:
    """`/v1` will read this through `factors_for`, so it has to be there."""

    call = Call(real_client, "sess-legs-factors")
    await call.start()
    await call.speech("caller", voice(HIGH, seed=1))
    await call.speech("callback", voice(LOW, seed=2))

    body = await call.factors()
    match = factor(body, FactorName.VOICE_MATCH)
    assert match["verdict"] == FactorVerdict.FAIL.value
    assert match["detail"].startswith("speaker_mismatch:")
    assert body["decision"] == Decision.DENY.value
    assert "speaker_mismatch" in body["reason"]


async def test_the_decision_does_not_depend_on_which_leg_speaks_first(
    real_client: httpx.AsyncClient,
) -> None:
    call = Call(real_client, "sess-legs-callback-first")
    await call.start()
    await call.speech("callback", voice(LOW, seed=2))
    body = await call.speech("caller", voice(HIGH, seed=1))

    assert factor(body, FactorName.VOICE_MATCH)["verdict"] == FactorVerdict.FAIL.value
    assert body["decision"] == Decision.DENY.value


async def test_a_correct_code_does_not_rescue_a_speaker_mismatch(
    real_client: httpx.AsyncClient,
) -> None:
    """The session is terminal once the legs disagree; a code is refused."""

    call = Call(real_client, "sess-legs-no-otp")
    await call.start()
    await call.speech("caller", voice(HIGH, seed=1))
    await call.speech("callback", voice(LOW, seed=2))

    response = await call.send("otp_request")
    assert response.status_code != 200
    body = await call.factors()
    assert body["decision"] == Decision.DENY.value


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


def _print(samples: np.ndarray):
    return voiceprint_from(build_context(samples, RATE))


def test_the_cross_leg_factor_is_fatal_in_the_fusion() -> None:
    caller = legs.LegPrint.from_print(_print(voice(HIGH, seed=1)))
    callback = legs.LegPrint.from_print(_print(voice(LOW, seed=2)))

    result = legs.cross_leg_factor({legs.CALLER: caller, legs.CALLBACK: callback})
    assert result is not None
    assert result.name is FactorName.VOICE_MATCH
    assert result.verdict is FactorVerdict.FAIL
    assert result.weight == pytest.approx(0.35)
    assert len(result.detail or "") <= 500

    assert scoring.fatal_factor([result]) is result
    assert scoring.speaker_mismatch([result]) is result


def test_a_matching_pair_is_not_fatal() -> None:
    caller = legs.LegPrint.from_print(_print(voice(LOW, seed=1)))
    callback = legs.LegPrint.from_print(_print(voice(LOW, seed=2)))

    result = legs.cross_leg_factor({legs.CALLER: caller, legs.CALLBACK: callback})
    assert result is not None
    assert result.verdict is FactorVerdict.PASS
    assert scoring.fatal_factor([result]) is None
    assert scoring.speaker_mismatch([result]) is None


def test_an_ordinary_voice_match_fail_is_not_a_speaker_mismatch() -> None:
    """Only the cross-leg marker is fatal; an enrolment fail still steps up."""

    plain = Factor(
        name=FactorName.VOICE_MATCH,
        score=0.1,
        weight=0.35,
        verdict=FactorVerdict.FAIL,
        detail="Voice comparison: caller's pitch register is 9.0 semitones from the reference",
    )
    assert scoring.speaker_mismatch([plain]) is None
    assert scoring.fatal_factor([plain]) is None


def test_the_check_needs_both_legs_with_enough_speech() -> None:
    full = legs.LegPrint.from_print(_print(voice(LOW, seed=1)))
    thin = legs.LegPrint.from_print(_print(voice(HIGH, seed=2, duration_s=1.0)))

    assert legs.cross_leg_factor({legs.CALLER: full}) is None
    assert legs.cross_leg_factor({legs.CALLBACK: full}) is None
    assert legs.cross_leg_factor({legs.CALLER: full, legs.CALLBACK: thin}) is None
    assert thin.voice.speech_seconds < legs.MIN_LEG_SPEECH_SECONDS


def test_legs_combine_as_spans_arrive() -> None:
    first = _print(voice(LOW, seed=1, duration_s=1.0))
    second = _print(voice(LOW, seed=2, duration_s=1.0))

    held = legs.LegPrint.from_print(first).absorb(second)
    assert held.spans == 2
    assert held.voice.speech_seconds == pytest.approx(
        first.speech_seconds + second.speech_seconds
    )
    assert held.voice.duration_s == pytest.approx(first.duration_s + second.duration_s)


def test_legs_survive_the_session_codec() -> None:
    session: Session = a_fully_populated_session()
    session.legs = {
        legs.CALLER: legs.LegPrint.from_print(_print(voice(HIGH, seed=1))),
        legs.CALLBACK: legs.LegPrint.from_print(_print(voice(LOW, seed=2))).absorb(
            _print(voice(LOW, seed=3))
        ),
    }

    document = session_to_json(session)
    restored = session_from_json(json.loads(json.dumps(document)))

    assert restored.legs == session.legs
    assert restored == session
    assert json.dumps(session_to_json(restored), sort_keys=True) == json.dumps(
        document, sort_keys=True
    )


def test_a_document_without_legs_still_reads() -> None:
    """Written by a replica from before this check existed."""

    document = session_to_json(a_fully_populated_session())
    del document["legs"]
    restored = session_from_json(document)
    assert restored.legs == {}
