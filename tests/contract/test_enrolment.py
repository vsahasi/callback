"""Enrolment: the reference voice, the store that holds it, and what changes.

Three layers, in order.

The print itself, in `detector.analysis.voiceprint`: that it keeps what the
comparison reads and nothing that could reconstruct speech, that it survives a
round trip through the store byte for byte, and that the frequency grid it
records makes an 8 kHz enrolment comparable against a 16 kHz call.

The store and the endpoint: that a put will not silently replace a reference,
that an unusable recording is refused rather than stored, that an identity
never reaches a url, and that the response never carries a feature of the
voice.

What it changes about a decision, end to end. This is the part worth reading:
enrolling a vendor moves real mode onto the policy the mock has specified all
along, and one documented outcome gets weaker rather than stronger. See
`test_a_synthesiser_is_challenged_not_denied_once_the_voice_is_enrolled`.

To accept an intended change to the schema snapshot:

    CALLBACK_UPDATE_SNAPSHOT=1 uv run pytest tests/contract/test_enrolment.py

then bump ENROLMENT_VERSION in shared/enrolment.py.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pytest

from detector import scoring
from detector.analysis.dsp import build_context
from detector.analysis.engine import AnalysisEngine
from detector.analysis.synthetic import SpeechStyle, live_speech, replay_of, tts_speech
from detector.analysis.voice import analyse_voice_match
from detector.analysis.voiceprint import (
    VoicePrint,
    voiceprint_from,
    voiceprint_from_json,
    voiceprint_to_json,
)
from detector.app import create_app
from detector.enrolment import (
    DictVoicePrintStore,
    Enrolment,
    EnrolmentConflict,
    EnrolmentStoreFull,
    InMemoryVoicePrintStore,
    VoicePrintStore,
    enrolment_from_json,
    enrolment_to_json,
    quality_of,
)
from detector.store import StoreFormatError
from shared.contract import Decision, FactorName, FactorVerdict, NextAction
from shared.enrolment import EnrolmentQuality, enrolment_schema_bundle
from tests.contract.test_real_mode import AUTH, RATE, pcm_data_uri, real_settings
from tests.contract.test_store import FakeModalDict

# A different realisation of the same talker: same SpeechStyle, different seed.
# Enrolling the exact recording the caller then sends would score a perfect
# match for reasons that have nothing to do with the voice.
ENROL_SEED = 11
CALLER_SEED = 1

IDENTITY = "cfo@northwind.example"


# ------------------------------------------------------------------- helpers


def enrol_audio(duration_s: float = 6.0, rate: int = RATE) -> np.ndarray:
    return live_speech(duration_s, rate, seed=ENROL_SEED)


def caller_audio(duration_s: float = 4.0, rate: int = RATE) -> np.ndarray:
    return live_speech(duration_s, rate, seed=CALLER_SEED)


def impostor_audio() -> np.ndarray:
    """A different talker reading the same script, nine semitones higher."""

    return live_speech(4.0, RATE, seed=CALLER_SEED, style=SpeechStyle(f0_hz=215.0))


def print_of(samples: np.ndarray, rate: int = RATE) -> VoicePrint:
    return voiceprint_from(build_context(samples, rate))


def audio_ref(samples: np.ndarray, rate: int = RATE) -> dict[str, object]:
    return {
        "track_id": "enrol-0",
        "start_ms": 0,
        "end_ms": int(len(samples) / rate * 1000),
        "sample_rate_hz": rate,
        "encoding": "pcm_s16le",
        "uri": pcm_data_uri(samples),
    }


def an_enrolment(
    identity: str = IDENTITY, samples: np.ndarray | None = None
) -> Enrolment:
    voice = print_of(enrol_audio() if samples is None else samples)
    return Enrolment(
        identity=identity,
        voice=voice,
        enrolled_at=datetime(2026, 3, 2, 9, 30, tzinfo=UTC),
        quality=quality_of(voice),
        source="data:",
        label="invoice call 2026-03-02",
    )


@pytest.fixture
def store() -> InMemoryVoicePrintStore:
    return InMemoryVoicePrintStore()


@pytest.fixture
async def enrolled_client(
    store: InMemoryVoicePrintStore,
) -> AsyncIterator[httpx.AsyncClient]:
    """Real mode with this vendor already enrolled."""

    store.put(an_enrolment(), replace=False)
    app = create_app(real_settings(), voiceprints=store)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://enrol.test",
        headers=AUTH,
    ) as client:
        yield client


@pytest.fixture
async def bare_client(
    store: InMemoryVoicePrintStore,
) -> AsyncIterator[httpx.AsyncClient]:
    """Real mode with an empty enrolment store."""

    app = create_app(real_settings(), voiceprints=store)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://enrol.test",
        headers=AUTH,
    ) as client:
        yield client


class Call:
    """Drive one session. Only `session_start` names the identity."""

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
            "occurred_at": f"2026-11-06T09:{self.sequence:02d}:00+00:00",
        }
        body.update(extra)
        return await self.client.post("/events", json=body)

    async def run(
        self, samples: np.ndarray, identity: str = IDENTITY, segments: int = 2
    ) -> dict:
        await self.send("session_start", claimed_user_id=identity)
        ref = audio_ref(samples)
        last: httpx.Response | None = None
        for _ in range(segments):
            last = await self.send("speech_segment", audio=ref)
        assert last is not None
        assert last.status_code == 200, last.text
        return last.json()


def factor(body: dict, name: FactorName) -> dict:
    return next(f for f in body["factors"] if f["name"] == name.value)


# ------------------------------------------------------- the print it keeps


def test_a_print_keeps_what_the_comparison_reads() -> None:
    voice = print_of(enrol_audio())

    assert voice.usable
    assert voice.has_pitch
    assert voice.has_spectrum
    assert voice.comparable
    assert 100.0 < voice.median_f0_hz < 180.0
    assert voice.voiced_seconds > 1.5
    assert voice.duration_s == pytest.approx(6.0, abs=0.05)


def test_a_print_cannot_reconstruct_the_recording() -> None:
    """The privacy claim, as a number rather than a promise.

    Everything that carries what was said is averaged away: the spectrum is a
    mean over every speech frame, with no phase and no time axis. What is left
    is small enough that the claim is self-evident.
    """

    samples = enrol_audio()
    doc = voiceprint_to_json(print_of(samples))

    floats = len(doc["ltas_db"]) + sum(
        1 for key, value in doc.items() if isinstance(value, (int, float))
    )
    assert floats < len(samples) / 100

    # No field holds a waveform, a frame sequence or any phase information.
    assert set(doc) == {
        "median_f0_hz",
        "voiced_seconds",
        "speech_seconds",
        "duration_s",
        "sample_rate",
        "bin_hz",
        "lo_hz",
        "ltas_db",
        "usable",
        "notes",
    }


def test_a_print_survives_the_store_unchanged() -> None:
    original = print_of(enrol_audio())
    restored = voiceprint_from_json(
        json.loads(json.dumps(voiceprint_to_json(original)))
    )

    assert restored == original

    probe = build_context(caller_audio(), RATE)
    assert analyse_voice_match(probe, restored).score == pytest.approx(
        analyse_voice_match(probe, original).score
    )


def test_an_eight_kilohertz_enrolment_compares_against_a_sixteen_kilohertz_call() -> (
    None
):
    """The grid property the spectral cue depends on, asserted rather than assumed.

    `FrameGrid.n_fft` doubles the window, which lands the bin spacing at the
    same 15.625 Hz at both rates. If that ever stops holding, this fails here
    instead of silently correlating the wrong frequencies in production.
    """

    narrow = print_of(enrol_audio(rate=RATE), rate=RATE)
    wide = print_of(enrol_audio(rate=16000), rate=16000)

    assert narrow.bin_hz == pytest.approx(wide.bin_hz, abs=0.01)
    assert narrow.lo_hz == pytest.approx(wide.lo_hz, abs=0.01)
    assert narrow.shares_grid_with(wide)

    result = analyse_voice_match(build_context(caller_audio(rate=16000), 16000), narrow)
    assert result.verdict is FactorVerdict.PASS


def test_a_mismatched_grid_declines_the_spectral_cue_instead_of_guessing() -> None:
    """Correlating two different frequency grids would be a number, not a fact."""

    reference = print_of(enrol_audio())
    off_grid = replace(reference, bin_hz=reference.bin_hz * 1.4)

    assert not reference.shares_grid_with(off_grid)

    result = analyse_voice_match(build_context(caller_audio(), RATE), off_grid)
    cues = {cue.name: cue for cue in result.cues}

    assert not cues["spectral_shape"].measurable
    assert "not comparable" in cues["spectral_shape"].note
    # The pitch cue still carries 0.7, so the comparison is weaker, not absent.
    assert cues["pitch_register"].measurable
    assert result.verdict is not FactorVerdict.FAIL


def test_silence_produces_a_print_that_knows_it_is_useless() -> None:
    voice = print_of(np.zeros(3 * RATE))

    assert not voice.comparable
    assert quality_of(voice) is EnrolmentQuality.UNUSABLE


def test_a_print_from_a_clipped_fragment_is_not_usable() -> None:
    voice = print_of(caller_audio()[:800])

    assert not voice.usable
    assert quality_of(voice) is EnrolmentQuality.UNUSABLE


def test_a_short_enrolment_is_stored_but_flagged_thin() -> None:
    """Worth keeping: it still catches a wildly different talker."""

    voice = print_of(caller_audio(duration_s=1.6))

    assert voice.comparable
    assert quality_of(voice) is EnrolmentQuality.THIN


def test_a_good_enrolment_is_called_good() -> None:
    assert quality_of(print_of(enrol_audio())) is EnrolmentQuality.GOOD


def test_a_record_written_by_another_build_is_refused_not_guessed() -> None:
    doc = enrolment_to_json(an_enrolment())
    doc["format"] = 99

    with pytest.raises(StoreFormatError, match="format 99"):
        enrolment_from_json(doc)


def test_a_truncated_record_is_refused() -> None:
    doc = enrolment_to_json(an_enrolment())
    del doc["voice"]

    with pytest.raises(StoreFormatError):
        enrolment_from_json(doc)


# --------------------------------------------------------------- the stores


def every_backend() -> list[VoicePrintStore]:
    return [
        InMemoryVoicePrintStore(),
        DictVoicePrintStore(backing=FakeModalDict()),
    ]


@pytest.mark.parametrize("backend", every_backend(), ids=["memory", "dict"])
def test_a_stored_print_comes_back_identical(backend: VoicePrintStore) -> None:
    original = an_enrolment()
    backend.put(original, replace=False)

    found = backend.get(IDENTITY)
    assert found is not None
    assert found.voice == original.voice
    assert found.quality is original.quality
    assert found.label == original.label
    assert found.enrolled_at == original.enrolled_at


@pytest.mark.parametrize("backend", every_backend(), ids=["memory", "dict"])
def test_a_missing_identity_is_none_not_an_error(backend: VoicePrintStore) -> None:
    assert backend.get("nobody@northwind.example") is None


@pytest.mark.parametrize("backend", every_backend(), ids=["memory", "dict"])
def test_a_second_enrolment_will_not_silently_replace_the_first(
    backend: VoicePrintStore,
) -> None:
    """The attack this guards: swap in your own voice and pass forever."""

    backend.put(an_enrolment(), replace=False)

    with pytest.raises(EnrolmentConflict):
        backend.put(an_enrolment(samples=impostor_audio()), replace=False)

    kept = backend.get(IDENTITY)
    assert kept is not None
    assert kept.voice.median_f0_hz < 180.0


@pytest.mark.parametrize("backend", every_backend(), ids=["memory", "dict"])
def test_replacing_is_allowed_when_it_is_asked_for(backend: VoicePrintStore) -> None:
    backend.put(an_enrolment(), replace=False)
    backend.put(an_enrolment(samples=impostor_audio()), replace=True)

    kept = backend.get(IDENTITY)
    assert kept is not None
    assert kept.voice.median_f0_hz > 180.0


@pytest.mark.parametrize("backend", every_backend(), ids=["memory", "dict"])
def test_erasing_is_idempotent(backend: VoicePrintStore) -> None:
    backend.put(an_enrolment(), replace=False)

    assert backend.delete(IDENTITY) is True
    assert backend.delete(IDENTITY) is False
    assert backend.get(IDENTITY) is None


@pytest.mark.parametrize("backend", every_backend(), ids=["memory", "dict"])
def test_identities_are_matched_byte_for_byte_after_trimming(
    backend: VoicePrintStore,
) -> None:
    """Trim, do not canonicalise. A missed lookup is safe, a collision is not."""

    backend.put(an_enrolment(identity="  cfo@northwind.example  "), replace=False)

    assert backend.get("cfo@northwind.example") is not None
    assert backend.get("CFO@Northwind.example") is None


def test_a_full_store_refuses_rather_than_evicting_someone_else() -> None:
    """Evicting would remove a signal from a call nobody is watching."""

    small = InMemoryVoicePrintStore(max_identities=2)
    small.put(an_enrolment(identity="one"), replace=False)
    small.put(an_enrolment(identity="two"), replace=False)

    with pytest.raises(EnrolmentStoreFull, match="identity limit"):
        small.put(an_enrolment(identity="three"), replace=False)

    # The two that were there are untouched, and replacing one still works.
    assert small.get("one") is not None
    small.put(an_enrolment(identity="one"), replace=True)
    assert len(small) == 2


def test_the_distributed_store_loses_a_race_rather_than_clobbering() -> None:
    """`skip_if_exists` is the one atomic primitive modal.Dict offers.

    Two operators enrolling the same identity at the same moment, both
    believing nothing is stored: the loser is told, rather than overwriting
    what the winner stored.
    """

    backing = FakeModalDict()
    first = DictVoicePrintStore(backing=backing)
    second = DictVoicePrintStore(backing=backing)

    first.put(an_enrolment(), replace=False)
    # `second` read before `first` wrote, so it still thinks it is creating.
    with pytest.raises(EnrolmentConflict):
        second.put(an_enrolment(samples=impostor_audio()), replace=False)


def test_the_distributed_store_lists_only_its_own_keys() -> None:
    backing = FakeModalDict()
    backing.put("s:sess-1", {"not": "a voice print"})
    backed = DictVoicePrintStore(backing=backing)
    backed.put(an_enrolment(identity="one"), replace=False)

    assert sorted(backed.identities()) == ["one"]
    assert len(backed) == 1


def test_an_unreadable_stored_record_raises_rather_than_reading_as_absent() -> None:
    """An absent reference silently disables the heaviest signal.

    So a corrupt record must not look like one. The pipeline catches this and
    carries on without a reference, which is logged; the store itself refuses
    to guess.
    """

    backing = FakeModalDict()
    backing.put("vp:cfo@northwind.example", {"format": 1, "identity": "x"})

    with pytest.raises(StoreFormatError):
        DictVoicePrintStore(backing=backing).get(IDENTITY)


# ------------------------------------------------------------- the endpoint


async def test_enrolling_needs_the_service_token(
    bare_client: httpx.AsyncClient, store: InMemoryVoicePrintStore
) -> None:
    response = await bare_client.post(
        "/enrol/voice",
        json={"identity": IDENTITY, "audio": audio_ref(enrol_audio())},
        headers={"Authorization": "Bearer wrong"},
    )

    assert response.status_code == 401
    assert len(store) == 0


async def test_mock_mode_refuses_an_enrolment_it_would_never_read() -> None:
    """Storing one here would look like coverage and be inert."""

    app = create_app(real_settings(mock_mode=True))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://mock.test",
        headers=AUTH,
    ) as client:
        response = await client.post(
            "/enrol/voice",
            json={"identity": IDENTITY, "audio": audio_ref(enrol_audio())},
        )

    assert response.status_code == 409
    assert "mock mode" in response.json()["error"]["message"]


async def test_an_enrolment_is_stored_and_described_without_its_features(
    bare_client: httpx.AsyncClient, store: InMemoryVoicePrintStore
) -> None:
    response = await bare_client.post(
        "/enrol/voice",
        json={
            "identity": IDENTITY,
            "audio": audio_ref(enrol_audio()),
            "label": "invoice call 2026-03-02",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["stored"] is True
    assert body["record"]["quality"] == "good"
    assert body["record"]["source"] == "data:"
    assert body["record"]["label"] == "invoice call 2026-03-02"
    assert body["record"]["voiced_seconds"] > 1.5
    assert store.get(IDENTITY) is not None

    # Nothing in the response describes the voice itself.
    rendered = response.text
    for leak in ("median_f0", "ltas", "spectrum", "bin_hz", "lo_hz"):
        assert leak not in rendered
    stored = store.get(IDENTITY)
    assert stored is not None
    assert f"{stored.voice.median_f0_hz:.0f}" not in rendered


async def test_the_uri_never_reaches_the_response(
    bare_client: httpx.AsyncClient,
) -> None:
    """`source` is a scheme and a host. A signed url carries a credential."""

    ref = audio_ref(enrol_audio())
    response = await bare_client.post(
        "/enrol/voice", json={"identity": IDENTITY, "audio": ref}
    )

    assert response.status_code == 200
    assert str(ref["uri"])[:64] not in response.text


async def test_an_unusable_recording_is_refused_rather_than_stored(
    bare_client: httpx.AsyncClient, store: InMemoryVoicePrintStore
) -> None:
    response = await bare_client.post(
        "/enrol/voice",
        json={"identity": IDENTITY, "audio": audio_ref(np.zeros(3 * RATE))},
    )

    assert response.status_code == 422
    assert "cannot support a comparison" in response.json()["error"]["message"]
    assert len(store) == 0


async def test_a_span_this_deployment_will_not_fetch_is_refused(
    bare_client: httpx.AsyncClient, store: InMemoryVoicePrintStore
) -> None:
    """The enrolment path fetches under the same policy as call audio."""

    ref = audio_ref(enrol_audio())
    ref["uri"] = "https://169.254.169.254/latest/meta-data/"
    response = await bare_client.post(
        "/enrol/voice", json={"identity": IDENTITY, "audio": ref}
    )

    assert response.status_code == 422
    assert "could not be read" in response.json()["error"]["message"]
    assert len(store) == 0


async def test_enrolling_twice_conflicts_unless_replace_is_set(
    bare_client: httpx.AsyncClient,
) -> None:
    body = {"identity": IDENTITY, "audio": audio_ref(enrol_audio())}

    assert (await bare_client.post("/enrol/voice", json=body)).status_code == 200
    again = await bare_client.post("/enrol/voice", json=body)
    assert again.status_code == 409
    assert "replace=true" in again.json()["error"]["detail"]

    replaced = await bare_client.post("/enrol/voice", json={**body, "replace": True})
    assert replaced.status_code == 200


async def test_status_answers_before_and_after(
    bare_client: httpx.AsyncClient,
) -> None:
    before = await bare_client.post("/enrol/voice/status", json={"identity": IDENTITY})
    assert before.status_code == 200
    assert before.json()["enrolled"] is False
    assert before.json()["record"] is None

    await bare_client.post(
        "/enrol/voice", json={"identity": IDENTITY, "audio": audio_ref(enrol_audio())}
    )

    after = await bare_client.post("/enrol/voice/status", json={"identity": IDENTITY})
    assert after.json()["enrolled"] is True
    assert after.json()["record"]["identity"] == IDENTITY


async def test_erasing_removes_the_reference(
    bare_client: httpx.AsyncClient, store: InMemoryVoicePrintStore
) -> None:
    await bare_client.post(
        "/enrol/voice", json={"identity": IDENTITY, "audio": audio_ref(enrol_audio())}
    )

    erased = await bare_client.post("/enrol/voice/erase", json={"identity": IDENTITY})
    assert erased.status_code == 200
    assert erased.json()["deleted"] is True
    assert store.get(IDENTITY) is None

    again = await bare_client.post("/enrol/voice/erase", json={"identity": IDENTITY})
    assert again.json()["deleted"] is False


async def test_reading_and_erasing_also_need_the_token(
    bare_client: httpx.AsyncClient,
) -> None:
    for path in ("/enrol/voice/status", "/enrol/voice/erase"):
        response = await bare_client.post(
            path,
            json={"identity": IDENTITY},
            headers={"Authorization": "Bearer wrong"},
        )
        assert response.status_code == 401, path


# ------------------------------------------- what it changes about a decision


async def test_voice_match_runs_once_the_vendor_is_enrolled(
    enrolled_client: httpx.AsyncClient,
) -> None:
    body = await Call(enrolled_client, "sess-enrolled-live").run(caller_audio())
    voice = factor(body, FactorName.VOICE_MATCH)

    assert voice["verdict"] == FactorVerdict.PASS.value
    assert voice["score"] > 0.8
    assert body["decision"] == Decision.ALLOW.value
    assert body["risk_score"] < scoring.ALLOW_BELOW
    # Four of five signals now, rather than three.
    assert body["confidence"] == pytest.approx(0.85, abs=0.01)


async def test_without_an_enrolment_nothing_changes(
    bare_client: httpx.AsyncClient,
) -> None:
    body = await Call(bare_client, "sess-bare-live").run(caller_audio())
    voice = factor(body, FactorName.VOICE_MATCH)

    assert voice["verdict"] == FactorVerdict.NOT_RUN.value
    assert body["decision"] == Decision.ALLOW.value
    assert body["confidence"] == pytest.approx(0.50, abs=0.01)


async def test_the_identity_comes_off_the_session_not_just_the_event(
    enrolled_client: httpx.AsyncClient,
) -> None:
    """`Call.run` names the identity only on session_start.

    The agent is not required to repeat `claimed_user_id` on every speech
    segment, so the reference lookup has to find it on the session. If this
    regresses, voice_match goes quietly back to not_run.
    """

    body = await Call(enrolled_client, "sess-enrolled-session-id").run(caller_audio())

    assert (
        factor(body, FactorName.VOICE_MATCH)["verdict"] != FactorVerdict.NOT_RUN.value
    )


async def test_a_different_talker_reading_the_script_is_caught(
    enrolled_client: httpx.AsyncClient,
) -> None:
    """The one attack classical features genuinely catch: human_impostor.

    Nothing else in the model notices this caller. Synthesis, replay and
    liveness all pass, because a live human reading a script is exactly what
    they are looking for.
    """

    body = await Call(enrolled_client, "sess-enrolled-impostor").run(impostor_audio())
    voice = factor(body, FactorName.VOICE_MATCH)

    assert voice["verdict"] == FactorVerdict.FAIL.value
    assert voice["score"] < 0.3
    assert "semitones from the reference" in voice["detail"]
    assert factor(body, FactorName.SYNTHETIC_SPEECH)["verdict"] == (
        FactorVerdict.PASS.value
    )
    assert body["decision"] == Decision.CHALLENGE.value
    assert body["next_action"] == NextAction.REQUEST_OTP.value


async def test_a_replayed_recording_still_denies_through_the_unrescuable_rule(
    enrolled_client: httpx.AsyncClient,
) -> None:
    """A replay is the right voice, so voice_match passes. It must still deny.

    And the mechanism matters. The replay scores 0.12, above the 0.10 fatal
    line, and the blend lands at 0.18, below the 0.30 allow threshold. So this
    deny comes from neither: the failing factor blocks the allow, and `replay`
    being unrescuable means the step-up is not offered, which drops the middle
    band to deny. Remove any one of those three and a replayed recording is
    allowed or merely challenged.
    """

    body = await Call(enrolled_client, "sess-enrolled-replay").run(
        replay_of(caller_audio(), sample_rate=RATE)
    )
    replay = factor(body, FactorName.REPLAY)

    assert factor(body, FactorName.VOICE_MATCH)["verdict"] == FactorVerdict.PASS.value
    assert replay["verdict"] == FactorVerdict.FAIL.value
    assert replay["score"] > scoring.FATAL_THRESHOLD
    assert body["risk_score"] < scoring.ALLOW_BELOW
    assert body["decision"] == Decision.DENY.value
    assert body["next_action"] == NextAction.COMPLETE_DENY.value


async def test_a_synthesiser_is_challenged_not_denied_once_the_voice_is_enrolled(
    enrolled_client: httpx.AsyncClient,
) -> None:
    """Read this one. Enrolling a vendor makes this outcome weaker.

    Without an enrolment a synthesiser denied outright, at a blended risk of
    0.59. With one it is challenged, at 0.35. Nothing about the synthesiser
    changed; what changed is that `voice_match` now runs, passes, and carries
    0.35 of the weight, so the two failing signals are a smaller share of the
    total.

    That is not a bug, it is this product's stated policy arriving in real
    mode. `detector.mock`'s `cloned_voice` scenario has always said so:
    voice_match 0.88 PASS, "Embedding matches, which a high-quality clone is
    expected to do", synthetic_speech FAIL, and a reason that ends "Step-up
    required" with otp_can_rescue True. The old deny was an artefact of
    voice_match never running, which inflated every real-mode risk score by
    removing a passing factor from the renormalisation.

    The control that catches a clone is therefore the one-time code sent to
    the number on the old invoice, not the blended score. Whether a cloned
    voice should instead be unrescuable, as replay and agent_behaviour are, is
    a product decision and is not made here.
    """

    body = await Call(enrolled_client, "sess-enrolled-tts").run(
        tts_speech(4.0, RATE, seed=CALLER_SEED)
    )

    assert factor(body, FactorName.VOICE_MATCH)["verdict"] == FactorVerdict.PASS.value
    assert factor(body, FactorName.SYNTHETIC_SPEECH)["verdict"] == (
        FactorVerdict.FAIL.value
    )
    assert scoring.ALLOW_BELOW <= body["risk_score"] < scoring.DENY_AT_OR_ABOVE
    assert body["decision"] == Decision.CHALLENGE.value
    assert body["next_action"] == NextAction.REQUEST_OTP.value


async def test_a_store_that_cannot_answer_does_not_end_the_call() -> None:
    """A lookup failure removes a signal. It must not raise into a live call."""

    class Broken(InMemoryVoicePrintStore):
        def get(self, identity: str) -> Enrolment | None:
            raise RuntimeError("the dict is unreachable")

    app = create_app(real_settings(), voiceprints=Broken())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://broken.test",
        headers=AUTH,
    ) as client:
        body = await Call(client, "sess-broken-store").run(caller_audio())

    assert (
        factor(body, FactorName.VOICE_MATCH)["verdict"] == FactorVerdict.NOT_RUN.value
    )
    assert body["decision"] == Decision.ALLOW.value


async def test_an_enrolment_for_a_different_identity_is_not_used(
    enrolled_client: httpx.AsyncClient,
) -> None:
    """Otherwise one vendor's reference would judge another vendor's call."""

    body = await Call(enrolled_client, "sess-other-identity").run(
        caller_audio(), identity="ap@southwind.example"
    )

    assert (
        factor(body, FactorName.VOICE_MATCH)["verdict"] == FactorVerdict.NOT_RUN.value
    )


async def test_the_engine_prefers_a_stored_print_over_extracting_one() -> None:
    """Both routes exist; the stored one wins, and they agree."""

    engine = AnalysisEngine()
    caller = caller_audio()
    reference = enrol_audio()

    from_print = engine.analyse(caller, RATE, reference=print_of(reference))
    from_audio = engine.analyse(caller, RATE, reference_audio=reference)

    assert [f.score for f in from_print] == [f.score for f in from_audio]


# ---------------------------------------------------------- schema snapshot

SNAPSHOT = Path(__file__).resolve().parents[2] / "schemas" / "enrolment.schema.json"


def test_enrolment_schema_snapshot_matches() -> None:
    current = json.dumps(enrolment_schema_bundle(), indent=2, sort_keys=True) + "\n"

    if os.environ.get("CALLBACK_UPDATE_SNAPSHOT") == "1":
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(current, encoding="utf-8")
        pytest.skip(f"snapshot rewritten at {SNAPSHOT}")

    assert SNAPSHOT.exists(), (
        f"no enrolment snapshot at {SNAPSHOT}. Create it with "
        "CALLBACK_UPDATE_SNAPSHOT=1 uv run pytest tests/contract/test_enrolment.py"
    )

    recorded = SNAPSHOT.read_text(encoding="utf-8")
    if recorded != current:
        recorded_obj = json.loads(recorded)
        current_obj = json.loads(current)
        added = sorted(set(current_obj["models"]) - set(recorded_obj["models"]))
        removed = sorted(set(recorded_obj["models"]) - set(current_obj["models"]))
        changed = sorted(
            name
            for name in set(current_obj["models"]) & set(recorded_obj["models"])
            if current_obj["models"][name] != recorded_obj["models"][name]
        )
        pytest.fail(
            "enrolment schema drifted from the snapshot.\n"
            f"  version recorded: {recorded_obj['enrolment_version']}\n"
            f"  version now:      {current_obj['enrolment_version']}\n"
            f"  models added:     {added}\n"
            f"  models removed:   {removed}\n"
            f"  models changed:   {changed}\n"
            "Accept with CALLBACK_UPDATE_SNAPSHOT=1 and bump ENROLMENT_VERSION."
        )


def test_the_contract_snapshot_is_untouched_by_this_surface() -> None:
    """Enrolment must not move CONTRACT_VERSION or the frozen bundle.

    The agent pins the contract. This surface is additive and lives in its own
    module and its own snapshot precisely so that adding it cannot invalidate
    what Veer codes against.
    """

    from shared.contract import CONTRACT_VERSION, schema_bundle

    contract_snapshot = (
        Path(__file__).resolve().parents[2] / "schemas" / "contract.schema.json"
    )
    recorded = json.loads(contract_snapshot.read_text(encoding="utf-8"))

    assert recorded["contract_version"] == CONTRACT_VERSION
    assert recorded == json.loads(json.dumps(schema_bundle()))
