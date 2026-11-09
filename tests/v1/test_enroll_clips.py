"""Enrolling a voice from a clip the agent uploads with a turn (contract 1.2.0).

`/v1` carries no audio in 1.1.0. The design is a hidden listener in the room,
and until that exists an `enroll` prompt can never pass, so a voiceprint can
never be recorded. `Turn.audio` is the way round that: the agent already
writes call audio to its own bucket, so it points the detector at the span it
just posted a turn for. The detector fetches it under the same policy as call
audio, reduces it to a print, and keeps no audio.

The security property this file exists to hold
----------------------------------------------
**Only an `enroll` prompt may pass because audio arrived.** A `digits`,
`phrase` or `liveness` challenge asks what was said and by whom. If uploading
audio passed one of those, anyone holding the service token could clear a
step-up with any recording at all, which would defeat the gate the whole
service is. An `enroll` prompt asks for neither: it records a voice. So
"audio arrived and it is measurable" is the whole of its pass condition and
none of theirs.

`test_a_verification_challenge_never_passes_on_audio_alone` and
`test_a_clip_on_a_verification_challenge_is_not_even_measured` are the two
that matter. If either goes red, stop.
"""

from __future__ import annotations

import io
import pathlib
from typing import Any

import httpx
import numpy as np
import pytest
from callback_shared.contract import (
    ChallengeKind,
    ChallengeState,
    Channel,
    ClipRef,
    Consent,
    EnrollmentStart,
    Role,
    SessionCreate,
    Turn,
)
from fastapi import FastAPI

from detector.analysis.synthetic import live_speech
from detector.app import create_app
from detector.audio import AudioPolicy
from detector.enrolment import InMemoryVoicePrintStore, VoicePrintStore
from detector.identity import candidate_identities, store_identity
from detector.observability import ObservabilityConfig
from detector.v1.enrollments import PASSES_TO_ENROLL
from detector.v1.errors import NoOpenEnrollment
from detector.v1.service import V1Service
from detector.v1.sessions import V1Session
from tests.v1.conftest import HEADERS, FakeClock, base_settings

RATE = 8000
USER = "u_eric"
HOST = "clips.example.com"
CONSENT = Consent(
    version="2026-11-04", verbatim="yes I agree, sentinel-consent-5f3a", ts_ms=1
)


def pcm(samples: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 0.999969)
    return (clipped * 32768.0).astype("<i2").tobytes()


# The clips the fake bucket serves, by path. Tests over the https transport
# rather than a data: uri, because https is what the agent will actually send
# and it exercises the allowlist, the redirect refusal and the byte cap on the
# way through.
BUCKET: dict[str, bytes] = {
    f"/clip-{seed}.pcm": pcm(live_speech(3.0, RATE, seed=seed)) for seed in range(1, 40)
}
BUCKET["/silence.pcm"] = pcm(np.zeros(3 * RATE))


def bucket(request: httpx.Request) -> httpx.Response:
    body = BUCKET.get(request.url.path)
    if body is None:
        return httpx.Response(404)
    return httpx.Response(200, content=body)


@pytest.fixture(autouse=True)
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the allowlisted host resolve somewhere globally routable.

    Autouse: every test here either fetches a clip or does not care, and the
    alternative is the same fixture named in a dozen signatures.
    """

    import socket as socket_module

    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket_module.AF_INET, None, None, "", ("93.184.216.34", port))]

    monkeypatch.setattr("detector.audio.socket.getaddrinfo", fake_getaddrinfo)


def a_clip(seed: int = 1, path: str | None = None) -> ClipRef:
    return ClipRef(
        uri=f"https://{HOST}{path or f'/clip-{seed}.pcm'}",
        sample_rate_hz=RATE,
        encoding="pcm_s16le",
    )


def a_turn(clock: FakeClock, *, audio: ClipRef | None, text: str = "ready") -> Turn:
    at = round(clock.now * 1000)
    return Turn(
        role=Role.USER,
        text=f"{text} {at}",
        start_ms=at,
        end_ms=at + 3000,
        audio=audio,
    )


def service_for(
    *,
    store: VoicePrintStore | None = None,
    mock_mode: bool = False,
    audio_enabled: bool = True,
    handler: Any = None,
) -> tuple[V1Service, FakeClock, VoicePrintStore]:
    clock = FakeClock()
    # Never `store or ...`: a VoicePrintStore defines __len__, so an empty one
    # is falsy and the service would get a different store from the test.
    prints = store if store is not None else InMemoryVoicePrintStore()
    settings = base_settings(
        mock_mode=mock_mode,
        audio_enabled=audio_enabled,
        audio_allowed_hosts=(HOST,),
    )
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler or bucket), follow_redirects=False
    )
    service = V1Service(
        settings,
        voiceprints=prints,
        clock=clock,
        audio_policy=AudioPolicy.from_settings(settings),
        client_of=lambda: client,
    )
    return service, clock, prints


def a_session(
    service: V1Service, clock: FakeClock, channel: Channel = Channel.PHONE
) -> V1Session:
    created = service.create_session(
        SessionCreate(
            room=f"room-{clock.now}",
            caller_identity="caller-0",
            claimed_user_id=USER,
            channel=channel,
        ),
        None,
    )
    session = service.registry.lookup(created.session_id)
    session.otp.verified = True  # Twilio is not in a unit test
    return session


async def enroll_prompt(
    service: V1Service,
    clock: FakeClock,
    session: V1Session,
    *,
    audio: ClipRef | None,
) -> ChallengeState:
    """Issue one enroll prompt, answer it, and report how it resolved."""

    issued = service.create_challenge(session.session_id, ChallengeKind.ENROLL)
    await service.add_turn(session.session_id, a_turn(clock, audio=audio))
    clock.advance(1.0)
    return service.challenge_status(session.session_id, issued.challenge_id).status


async def full_enrollment(
    service: V1Service,
    clock: FakeClock,
    session: V1Session,
    *,
    prompts: int = PASSES_TO_ENROLL,
    audio: bool = True,
) -> str:
    started = service.start_enrollment(
        EnrollmentStart(
            session_id=session.session_id,
            user_id=USER,
            channel=session.channel,
            consent=CONSENT,
        )
    )
    for index in range(prompts):
        await enroll_prompt(
            service,
            clock,
            session,
            audio=a_clip(seed=11 + index) if audio else None,
        )
    return started.enrollment_id


# ------------------------------------------------- the security property


async def test_a_verification_challenge_never_passes_on_audio_alone() -> None:
    """Read this one first.

    If uploading audio could pass a digits or phrase challenge, anyone with
    the service token would clear a step-up with any recording at all.
    """

    service, clock, _ = service_for()
    session = a_session(service, clock)

    for kind in (ChallengeKind.DIGITS, ChallengeKind.PHRASE, ChallengeKind.LIVENESS):
        issued = service.create_challenge(session.session_id, kind)
        await service.add_turn(session.session_id, a_turn(clock, audio=a_clip()))
        clock.advance(1.0)
        status = service.challenge_status(session.session_id, issued.challenge_id)

        assert status.status is ChallengeState.FAILED, kind
        assert status.reason == "no_audio", kind


async def test_a_clip_on_a_verification_challenge_is_not_even_measured() -> None:
    """Not just "does not pass": the audio is not looked at.

    Measuring it would put a print of the caller on a challenge record that
    an enrollment could later read, which is a way to enrol a voice with no
    consent at all.
    """

    service, clock, _ = service_for()
    session = a_session(service, clock)
    issued = service.create_challenge(session.session_id, ChallengeKind.DIGITS)
    await service.add_turn(session.session_id, a_turn(clock, audio=a_clip()))

    assert session.challenges[issued.challenge_id].voice is None


async def test_a_clip_cannot_enrol_without_an_open_enrollment() -> None:
    """An enroll challenge needs an enrollment, which needs consent and a
    verified code. Audio alone opens nothing."""

    service, clock, prints = service_for()
    session = a_session(service, clock)

    with pytest.raises(NoOpenEnrollment):
        service.create_challenge(session.session_id, ChallengeKind.ENROLL)

    assert len(prints) == 0


async def test_a_locked_session_records_no_voice() -> None:
    """A denied caller's voice is never enrolled, clip or no clip."""

    service, clock, prints = service_for()
    session = a_session(service, clock)
    service.start_enrollment(
        EnrollmentStart(
            session_id=session.session_id,
            user_id=USER,
            channel=Channel.PHONE,
            consent=CONSENT,
        )
    )
    issued = service.create_challenge(session.session_id, ChallengeKind.ENROLL)
    session.locked_code = "voice_fail"
    await service.add_turn(session.session_id, a_turn(clock, audio=a_clip()))

    assert session.challenges[issued.challenge_id].voice is None
    assert len(prints) == 0


# ----------------------------------------------------- the happy path


async def test_an_enroll_prompt_passes_when_the_clip_is_measurable() -> None:
    service, clock, _ = service_for()
    session = a_session(service, clock)
    service.start_enrollment(
        EnrollmentStart(
            session_id=session.session_id,
            user_id=USER,
            channel=Channel.PHONE,
            consent=CONSENT,
        )
    )

    assert (
        await enroll_prompt(service, clock, session, audio=a_clip())
        is ChallengeState.PASSED
    )


async def test_five_clips_enrol_a_voice_the_pipeline_can_find() -> None:
    """The whole point, end to end: a /v1 enrollment writes a real print."""

    service, clock, prints = service_for()
    session = a_session(service, clock)
    enrollment_id = await full_enrollment(service, clock, session)

    result = service.complete_enrollment(enrollment_id)
    assert result.status == "enrolled"
    assert result.prompts_passed == PASSES_TO_ENROLL

    record = prints.get(store_identity(USER, "phone"))
    assert record is not None
    assert record.source == "listener"
    assert record.channel == "phone"
    assert record.voice.has_pitch
    # Five three-second clips, pooled.
    assert record.voice.duration_s == pytest.approx(15.0, abs=0.5)
    assert record.voice.voiced_seconds > 4.0

    # And it is under a key the /events pipeline actually looks at.
    assert store_identity(USER, "phone") in candidate_identities(USER)


async def test_no_clip_means_no_pass_and_no_print() -> None:
    """Which is every 1.1.0 agent: the field is absent, so nothing changes."""

    service, clock, prints = service_for()
    session = a_session(service, clock)
    enrollment_id = await full_enrollment(service, clock, session, audio=False)

    assert service.complete_enrollment(enrollment_id).status == "failed"
    assert len(prints) == 0


async def test_an_unfetchable_clip_fails_the_prompt_rather_than_the_call() -> None:
    """A bad uri costs a retry. The enrollment flow has three per prompt."""

    service, clock, prints = service_for()
    session = a_session(service, clock)
    service.start_enrollment(
        EnrollmentStart(
            session_id=session.session_id,
            user_id=USER,
            channel=Channel.PHONE,
            consent=CONSENT,
        )
    )

    status = await enroll_prompt(
        service,
        clock,
        session,
        audio=ClipRef(uri="https://169.254.169.254/latest/meta-data/"),
    )

    assert status is ChallengeState.FAILED
    assert len(prints) == 0


async def test_a_silent_clip_does_not_pass_the_prompt() -> None:
    service, clock, prints = service_for()
    session = a_session(service, clock)
    service.start_enrollment(
        EnrollmentStart(
            session_id=session.session_id,
            user_id=USER,
            channel=Channel.PHONE,
            consent=CONSENT,
        )
    )

    status = await enroll_prompt(
        service,
        clock,
        session,
        audio=a_clip(path="/silence.pcm"),
    )

    assert status is ChallengeState.FAILED
    assert len(prints) == 0


async def test_audio_switched_off_ignores_clips_entirely() -> None:
    """One switch governs every fetch in the service, enrollment included."""

    service, clock, prints = service_for(audio_enabled=False)
    session = a_session(service, clock)
    enrollment_id = await full_enrollment(service, clock, session)

    assert service.complete_enrollment(enrollment_id).status == "failed"
    assert len(prints) == 0


async def test_mock_mode_ignores_clips() -> None:
    """Mock mode answers from a script and writes no print, by ruling."""

    service, clock, prints = service_for(mock_mode=True)
    session = a_session(service, clock)
    enrollment_id = await full_enrollment(service, clock, session)

    service.complete_enrollment(enrollment_id)
    assert len(prints) == 0
    assert all(challenge.voice is None for challenge in session.challenges.values())


async def test_a_second_clip_for_one_prompt_does_not_replace_the_first() -> None:
    """A retry issues a new challenge, so one record is measured once."""

    service, clock, _ = service_for()
    session = a_session(service, clock)
    service.start_enrollment(
        EnrollmentStart(
            session_id=session.session_id,
            user_id=USER,
            channel=Channel.PHONE,
            consent=CONSENT,
        )
    )
    issued = service.create_challenge(session.session_id, ChallengeKind.ENROLL)
    await service.add_turn(session.session_id, a_turn(clock, audio=a_clip(seed=1)))
    first = session.challenges[issued.challenge_id].voice
    clock.advance(0.1)
    await service.add_turn(
        session.session_id, a_turn(clock, audio=a_clip(seed=99), text="again")
    )

    assert session.challenges[issued.challenge_id].voice is first


async def test_an_agent_turn_with_audio_is_ignored() -> None:
    """Only the caller's voice is ever enrolled, not the agent's."""

    service, clock, _ = service_for()
    session = a_session(service, clock)
    service.start_enrollment(
        EnrollmentStart(
            session_id=session.session_id,
            user_id=USER,
            channel=Channel.PHONE,
            consent=CONSENT,
        )
    )
    issued = service.create_challenge(session.session_id, ChallengeKind.ENROLL)
    at = round(clock.now * 1000)
    await service.add_turn(
        session.session_id,
        Turn(
            role=Role.AGENT,
            text="please repeat after me",
            start_ms=at,
            end_ms=at + 1000,
            audio=a_clip(),
        ),
    )

    assert session.challenges[issued.challenge_id].voice is None


async def test_the_browser_channel_enrols_separately() -> None:
    service, clock, prints = service_for()
    session = a_session(service, clock, channel=Channel.BROWSER)
    enrollment_id = await full_enrollment(service, clock, session)

    assert service.complete_enrollment(enrollment_id).status == "enrolled"
    assert prints.get(store_identity(USER, "browser")) is not None
    assert prints.get(store_identity(USER, "phone")) is None


async def test_a_clip_is_read_once_and_no_audio_is_kept() -> None:
    """The record holds a print. Nothing anywhere holds the samples."""

    service, clock, prints = service_for()
    session = a_session(service, clock)
    enrollment_id = await full_enrollment(service, clock, session)
    service.complete_enrollment(enrollment_id)

    for challenge in session.challenges.values():
        assert not hasattr(challenge.voice, "samples")
    record = prints.get(store_identity(USER, "phone"))
    assert record is not None
    # A print is a few hundred floats. Three seconds at 8 kHz is 24000
    # samples, five clips is 120000, so this is the claim as a number.
    assert len(record.voice.ltas_db) < 1000


async def test_what_a_measured_clip_logs() -> None:
    class Spy:
        def __init__(self) -> None:
            self.lines: list[tuple[str, str, dict[str, Any]]] = []

        def log(self, event: str, *, level: str = "info", **fields: Any) -> None:
            self.lines.append((event, level, fields))

    service, clock, _ = service_for()
    spy = Spy()
    service.bind_logger(spy)
    session = a_session(service, clock)
    await full_enrollment(service, clock, session, prompts=1)

    measured = [line for line in spy.lines if line[0] == "v1_enroll_clip_measured"]
    assert len(measured) == 1
    _, level, fields = measured[0]
    assert level == "info"
    assert fields["prompt"] == 1
    # A scheme and a host, never the uri: a signed object-store url carries a
    # credential in its query string, and this value reaches the logs.
    assert fields["source"] == f"https://{HOST}"
    assert "clip-" not in fields["source"]
    assert "uri" not in fields


# ------------------------------------------- over HTTP, in real mode


@pytest.fixture
def clip_root(tmp_path: pathlib.Path) -> pathlib.Path:
    """Seven clips on disk, read through `file:` uris.

    `file:` rather than https so this needs no injected transport: the point
    of these two tests is the HTTP layer above the fetch, not the fetch.
    """

    for seed in range(1, 8):
        samples = np.clip(live_speech(3.0, RATE, seed=20 + seed), -1.0, 0.999969)
        (tmp_path / f"clip-{seed}.pcm").write_bytes(
            (samples * 32768.0).astype("<i2").tobytes()
        )
    return tmp_path


def real_app(
    clip_root: pathlib.Path, store: VoicePrintStore, clock: FakeClock
) -> FastAPI:
    settings = base_settings(
        mock_mode=False,
        audio_enabled=True,
        audio_allow_file_uris=True,
        audio_file_root=str(clip_root),
    )
    return create_app(
        settings,
        voiceprints=store,
        v1_clock=clock,
        log_stream=io.StringIO(),
        observability=ObservabilityConfig(),
    )


async def test_the_whole_enrollment_works_over_http_in_real_mode(
    clip_root: pathlib.Path,
) -> None:
    """The one test that goes through the router, the version header, the auth
    and the Turn model, in real mode, and ends with a stored print.

    The OTP flag is set directly: real mode sends the code through Twilio
    Verify, which is not in a test, and the gate itself is covered in
    tests/v1/test_enrollments.py.
    """

    store = InMemoryVoicePrintStore()
    clock = FakeClock()
    app = real_app(clip_root, store, clock)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://real.test",
        headers=HEADERS,
    ) as client:
        created = await client.post(
            "/v1/sessions",
            json={
                "room": "room-http-1",
                "caller_identity": "caller-0",
                "claimed_user_id": USER,
                "channel": "phone",
            },
        )
        assert created.status_code == 200, created.text
        session_id = created.json()["session_id"]
        app.state.v1.registry.lookup(session_id).otp.verified = True

        started = await client.post(
            "/v1/enrollments",
            json={
                "session_id": session_id,
                "user_id": USER,
                "channel": "phone",
                "consent": CONSENT.model_dump(),
            },
        )
        assert started.status_code == 200, started.text
        enrollment_id = started.json()["enrollment_id"]

        passed = 0
        for prompt in range(1, 8):
            issued = await client.post(
                f"/v1/sessions/{session_id}/challenges", json={"kind": "enroll"}
            )
            assert issued.status_code == 200, issued.text
            at = 1_000_000 + prompt * 4000
            posted = await client.post(
                f"/v1/sessions/{session_id}/turns",
                json={
                    "role": "user",
                    "text": f"answer {prompt}",
                    "start_ms": at,
                    "end_ms": at + 3000,
                    "audio": {
                        "uri": (clip_root / f"clip-{prompt}.pcm").as_uri(),
                        "sample_rate_hz": RATE,
                        "encoding": "pcm_s16le",
                    },
                },
            )
            assert posted.status_code == 204, posted.text
            # A challenge resolves 700 ms after the turn is received. Stepping
            # the clock beats sleeping seven times.
            clock.advance(1.0)
            status = await client.get(
                f"/v1/sessions/{session_id}/challenges/{issued.json()['challenge_id']}"
            )
            if status.json()["status"] == "passed":
                passed += 1

        assert passed == 7

        done = await client.post(f"/v1/enrollments/{enrollment_id}/complete", json={})
        assert done.status_code == 200, done.text
        assert done.json() == {"status": "enrolled", "prompts_passed": 7}

        # And the operator surface can see it, under the channel key.
        asked = await client.post(
            "/enrol/voice/status",
            json={"identity": USER, "channel": "phone"},
            headers={"Authorization": HEADERS["Authorization"]},
        )
        assert asked.status_code == 200, asked.text
        assert asked.json()["enrolled"] is True
        assert asked.json()["record"]["source"] == "listener"

    record = store.get(store_identity(USER, "phone"))
    assert record is not None
    assert record.voice.duration_s == pytest.approx(21.0, abs=0.5)


async def test_a_1_1_0_agent_sending_no_clip_is_served_unchanged(
    clip_root: pathlib.Path,
) -> None:
    """The compatibility claim behind the minor bump, as a test.

    An agent on the previous contract sends no `audio` and an older version
    header. It is served, the prompt fails `no_audio` exactly as it did
    before, and nothing is enrolled.
    """

    store = InMemoryVoicePrintStore()
    clock = FakeClock()
    app = real_app(clip_root, store, clock)
    old = {**HEADERS, "X-Contract-Version": "1.1.0"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://old.test",
        headers=old,
    ) as client:
        created = await client.post(
            "/v1/sessions",
            json={
                "room": "room-old-1",
                "caller_identity": "caller-0",
                "claimed_user_id": USER,
                "channel": "phone",
            },
        )
        assert created.status_code == 200, created.text
        session_id = created.json()["session_id"]
        app.state.v1.registry.lookup(session_id).otp.verified = True

        started = await client.post(
            "/v1/enrollments",
            json={
                "session_id": session_id,
                "user_id": USER,
                "channel": "phone",
                "consent": CONSENT.model_dump(),
            },
        )
        assert started.status_code == 200
        issued = await client.post(
            f"/v1/sessions/{session_id}/challenges", json={"kind": "enroll"}
        )
        posted = await client.post(
            f"/v1/sessions/{session_id}/turns",
            json={
                "role": "user",
                "text": "answer",
                "start_ms": 1_000_000,
                "end_ms": 1_003_000,
            },
        )
        assert posted.status_code == 204
        clock.advance(1.0)
        status = await client.get(
            f"/v1/sessions/{session_id}/challenges/{issued.json()['challenge_id']}"
        )

    assert status.json()["status"] == "failed"
    assert status.json()["reason"] == "no_audio"
    assert len(store) == 0
