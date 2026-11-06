"""The audio transport: what it decodes, and what it refuses to fetch.

Always in process, even when CALLBACK_BASE_URL is set: these assert on the
loader's behaviour against a transport this file controls, which a remote
deployment cannot hand back.

The refusals matter more than the decodes. `uri` arrives in a request body,
so every one of these tests is a case where a caller with a valid token tries
to point the service somewhere it should not go.
"""

from __future__ import annotations

import base64
import io
import wave
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import pytest

from detector.audio import (
    AudioPolicy,
    AudioUnavailable,
    decode_span,
    load_span,
)
from shared.contract import AudioEncoding, AudioRef

# --------------------------------------------------------------- fixtures


def pcm_bytes(samples: np.ndarray) -> bytes:
    """float32 in [-1, 1) to raw little-endian int16."""

    clipped = np.clip(samples, -1.0, 0.999969)
    return (clipped * 32768.0).astype("<i2").tobytes()


def wav_bytes(samples: np.ndarray, rate: int = 8000, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm_bytes(samples))
    return buffer.getvalue()


def data_uri(raw: bytes, media: str = "audio/l16") -> str:
    return f"data:{media};base64,{base64.b64encode(raw).decode('ascii')}"


def ref(
    uri: str | None,
    *,
    encoding: AudioEncoding = AudioEncoding.PCM_S16LE,
    rate: int = 8000,
) -> AudioRef:
    return AudioRef(
        track_id="track-1",
        start_ms=0,
        end_ms=4000,
        sample_rate_hz=rate,
        encoding=encoding,
        uri=uri,
    )


@pytest.fixture
def tone() -> np.ndarray:
    t = np.arange(8000 * 2, dtype=np.float64) / 8000.0
    return (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


@pytest.fixture
def open_policy() -> AudioPolicy:
    """Enabled, data uris on, nothing else."""

    return AudioPolicy(enabled=True, allow_data_uris=True)


# ------------------------------------------------------------- the defaults


def test_audio_is_off_until_it_is_switched_on() -> None:
    """A deployment that never configured audio fetches nothing."""

    assert AudioPolicy().enabled is False


async def test_disabled_policy_refuses_even_a_data_uri(tone: np.ndarray) -> None:
    with pytest.raises(AudioUnavailable, match="disabled"):
        await load_span(ref(data_uri(pcm_bytes(tone))), AudioPolicy())


async def test_an_empty_allowlist_fetches_nothing_over_the_network() -> None:
    """Enabled, but no hosts named. The network half stays shut."""

    policy = AudioPolicy(enabled=True)
    with pytest.raises(AudioUnavailable, match="not allowed"):
        await load_span(ref("https://recordings.example.com/a.wav"), policy)


# ---------------------------------------------------------------- decoding


async def test_raw_pcm_round_trips(tone: np.ndarray, open_policy: AudioPolicy) -> None:
    loaded = await load_span(ref(data_uri(pcm_bytes(tone))), open_policy)

    assert loaded.sample_rate == 8000
    assert loaded.samples.dtype == np.float32
    assert len(loaded.samples) == len(tone)
    assert loaded.duration_s == pytest.approx(2.0, abs=0.01)
    # int16 quantisation is the only loss.
    assert np.max(np.abs(loaded.samples - tone)) < 1.0 / 32768.0 * 2


async def test_a_wav_container_is_detected_by_its_magic(
    tone: np.ndarray, open_policy: AudioPolicy
) -> None:
    """The span says pcm_s16le and hands over a .wav, which agents do.

    Reading the 44-byte RIFF header as samples produces a click loud enough
    for the liveness detector to score, so the magic wins over the declared
    encoding.
    """

    loaded = await load_span(
        ref(data_uri(wav_bytes(tone, rate=16000)), rate=8000), open_policy
    )

    # The rate came from the container, not from the declared 8000.
    assert loaded.sample_rate == 16000
    assert len(loaded.samples) == len(tone)
    # Sample for sample, so the 44-byte header was parsed and not scored.
    # Reading it as audio would shift everything by 22 samples and put a
    # square edge at the front, which is what the liveness detector would
    # have measured.
    assert np.max(np.abs(loaded.samples - tone)) < 2.0 / 32768.0


async def test_stereo_wav_is_downmixed(open_policy: AudioPolicy) -> None:
    left = np.full(800, 0.5, dtype=np.float32)
    right = np.full(800, -0.1, dtype=np.float32)
    interleaved = np.empty(1600, dtype=np.float32)
    interleaved[0::2] = left
    interleaved[1::2] = right

    loaded = await load_span(
        ref(data_uri(wav_bytes(interleaved, channels=2))), open_policy
    )

    assert len(loaded.samples) == 800
    assert loaded.samples.mean() == pytest.approx(0.2, abs=0.01)


async def test_a_trailing_half_sample_is_refused(open_policy: AudioPolicy) -> None:
    """One odd byte shifts every following sample and turns speech to noise."""

    raw = pcm_bytes(np.zeros(100, dtype=np.float32)) + b"\x01"
    with pytest.raises(AudioUnavailable, match="partial sample"):
        await load_span(ref(data_uri(raw)), open_policy)


async def test_opus_is_refused_rather_than_guessed(
    tone: np.ndarray, open_policy: AudioPolicy
) -> None:
    """There is no opus decoder here, and a guess would be scored as audio."""

    with pytest.raises(AudioUnavailable, match="opus"):
        await load_span(
            ref(data_uri(pcm_bytes(tone)), encoding=AudioEncoding.OPUS),
            open_policy,
        )


async def test_flac_decodes_when_soundfile_is_present(
    tone: np.ndarray, open_policy: AudioPolicy
) -> None:
    soundfile = pytest.importorskip("soundfile")
    buffer = io.BytesIO()
    soundfile.write(buffer, tone, 8000, format="FLAC")

    loaded = await load_span(
        ref(data_uri(buffer.getvalue()), encoding=AudioEncoding.FLAC), open_policy
    )

    assert loaded.sample_rate == 8000
    assert len(loaded.samples) == len(tone)


async def test_garbage_is_refused_not_scored(open_policy: AudioPolicy) -> None:
    """A WAV magic with nothing behind it must not become samples."""

    with pytest.raises(AudioUnavailable, match="WAV"):
        await load_span(ref(data_uri(b"RIFF\x00\x00\x00\x00WAVEnope")), open_policy)


def test_non_finite_samples_never_reach_the_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A NaN propagates through every mean and ratio in the analysis.

    No decoder here can currently produce one: int16 has no NaN, the WAV path
    is 16-bit only, and FLAC is integer coded. The guard is for the next
    decoder, so the only honest way to exercise it is to make a decoder
    return one. Keeping it and saying so beats deleting a cheap check that a
    float-capable decoder would immediately need.
    """

    bad = np.array([0.1, np.nan, 0.2, np.inf], dtype=np.float32)
    monkeypatch.setattr("detector.audio._decode_flac", lambda raw: (bad, 8000))

    with pytest.raises(AudioUnavailable, match="non-finite"):
        decode_span(
            b"fLaC-not-really",
            ref(None, encoding=AudioEncoding.FLAC),
            AudioPolicy(enabled=True),
        )


async def test_a_span_longer_than_the_limit_is_refused() -> None:
    policy = AudioPolicy(enabled=True, max_seconds=1.0)
    long = np.zeros(8000 * 3, dtype=np.float32)
    with pytest.raises(AudioUnavailable, match="over the"):
        await load_span(ref(data_uri(pcm_bytes(long))), policy)


# ------------------------------------------------------------- data: limits


async def test_an_oversize_data_uri_is_refused_before_decoding() -> None:
    policy = AudioPolicy(enabled=True, max_bytes=1024)
    big = np.zeros(4000, dtype=np.float32)
    with pytest.raises(AudioUnavailable, match="larger than the limit"):
        await load_span(ref(data_uri(pcm_bytes(big))), policy)


async def test_a_data_uri_that_is_not_base64_is_refused(
    open_policy: AudioPolicy,
) -> None:
    with pytest.raises(AudioUnavailable, match="base64"):
        await load_span(ref("data:audio/l16;base64,!!!not base64!!!"), open_policy)


async def test_a_plain_data_uri_without_base64_is_refused(
    open_policy: AudioPolicy,
) -> None:
    with pytest.raises(AudioUnavailable, match="base64"):
        await load_span(ref("data:audio/l16,0000"), open_policy)


async def test_data_uris_can_be_switched_off(
    tone: np.ndarray,
) -> None:
    policy = AudioPolicy(enabled=True, allow_data_uris=False)
    with pytest.raises(AudioUnavailable, match="not enabled"):
        await load_span(ref(data_uri(pcm_bytes(tone))), policy)


# ------------------------------------------------------------------- file:


@dataclass(frozen=True)
class FileLayout:
    """A span inside a permitted root, and one outside it.

    Built in a synchronous fixture so the async tests below do no blocking
    filesystem work of their own.
    """

    root: Path
    inside: Path
    outside: Path

    def policy(self, *, rooted: bool = True) -> AudioPolicy:
        return AudioPolicy(
            enabled=True,
            allow_file_uris=True,
            file_root=self.root if rooted else None,
        )


@pytest.fixture
def layout(tmp_path: Path, tone: np.ndarray) -> FileLayout:
    root = tmp_path / "allowed"
    root.mkdir()
    inside = root / "span.wav"
    inside.write_bytes(wav_bytes(tone))
    outside = tmp_path / "secret.wav"
    outside.write_bytes(wav_bytes(tone))
    return FileLayout(root=root.resolve(), inside=inside, outside=outside)


async def test_file_uris_are_off_by_default(layout: FileLayout) -> None:
    with pytest.raises(AudioUnavailable, match="not enabled"):
        await load_span(ref(layout.inside.as_uri()), AudioPolicy(enabled=True))


async def test_a_file_uri_inside_the_root_works(
    layout: FileLayout, tone: np.ndarray
) -> None:
    loaded = await load_span(ref(layout.inside.as_uri()), layout.policy())

    assert len(loaded.samples) == len(tone)
    assert loaded.source == "file:"


async def test_a_file_uri_outside_the_root_is_refused(layout: FileLayout) -> None:
    """Without the root check this reads anything the process can read."""

    with pytest.raises(AudioUnavailable, match="outside the permitted root"):
        await load_span(ref(layout.outside.as_uri()), layout.policy())


async def test_a_traversing_file_uri_is_refused(layout: FileLayout) -> None:
    traversal = (layout.root / ".." / "secret.wav").as_uri()

    with pytest.raises(AudioUnavailable, match="outside the permitted root"):
        await load_span(ref(traversal), layout.policy())


async def test_file_uris_need_a_root(layout: FileLayout) -> None:
    with pytest.raises(AudioUnavailable, match="CALLBACK_AUDIO_FILE_ROOT"):
        await load_span(ref(layout.inside.as_uri()), layout.policy(rooted=False))


# ------------------------------------------------------- the network, abused

ALLOWED = AudioPolicy(
    enabled=True,
    allowed_hosts=frozenset({"recordings.example.com"}),
    max_bytes=64 * 1024,
)


def mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_http_is_never_fetched() -> None:
    with pytest.raises(AudioUnavailable, match="https, not http"):
        await load_span(ref("http://recordings.example.com/a.wav"), ALLOWED)


async def test_an_unlisted_host_is_refused() -> None:
    with pytest.raises(AudioUnavailable, match="not allowed"):
        await load_span(ref("https://evil.example.net/a.wav"), ALLOWED)


async def test_the_refusal_does_not_enumerate_the_allowlist() -> None:
    """A caller probing hosts must not be handed the list."""

    with pytest.raises(AudioUnavailable) as caught:
        await load_span(ref("https://evil.example.net/a.wav"), ALLOWED)
    assert "recordings.example.com" not in str(caught.value)


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",  # itself
        "169.254.169.254",  # cloud metadata
        "10.0.0.5",  # inside the vpc
        "192.168.1.1",
        "172.16.0.1",
    ],
)
async def test_an_allowed_host_resolving_somewhere_private_is_refused(
    monkeypatch: pytest.MonkeyPatch, address: str
) -> None:
    """The allowlist is the control; this is the layer behind it.

    A name that is on the allowlist but points inside the network is the
    whole SSRF game, and the metadata address is the prize.
    """

    import socket as socket_module

    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket_module.AF_INET, None, None, "", (address, port))]

    monkeypatch.setattr("detector.audio.socket.getaddrinfo", fake_getaddrinfo)

    with pytest.raises(AudioUnavailable, match="non-routable"):
        await load_span(ref("https://recordings.example.com/a.wav"), ALLOWED)


async def test_a_host_that_does_not_resolve_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args, **kwargs):
        raise OSError("no such host")

    monkeypatch.setattr("detector.audio.socket.getaddrinfo", boom)
    with pytest.raises(AudioUnavailable, match="did not resolve"):
        await load_span(ref("https://recordings.example.com/a.wav"), ALLOWED)


@pytest.fixture
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the allowlisted host resolve somewhere globally routable."""

    import socket as socket_module

    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket_module.AF_INET, None, None, "", ("93.184.216.34", port))]

    monkeypatch.setattr("detector.audio.socket.getaddrinfo", fake_getaddrinfo)


async def test_a_redirect_is_an_error_not_a_hop(public_dns: None) -> None:
    """A 302 is how you leave an allowlist."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://evil.example.net/a"})

    async with mock_client(handler) as client:
        with pytest.raises(AudioUnavailable, match="redirect"):
            await load_span(
                ref("https://recordings.example.com/a.wav"), ALLOWED, client=client
            )


async def test_a_non_200_is_refused(public_dns: None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, content=b"nope")

    async with mock_client(handler) as client:
        with pytest.raises(AudioUnavailable, match="HTTP 403"):
            await load_span(
                ref("https://recordings.example.com/a.wav"), ALLOWED, client=client
            )


async def test_a_fetch_that_times_out_is_refused(public_dns: None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    async with mock_client(handler) as client:
        with pytest.raises(AudioUnavailable, match="timed out"):
            await load_span(
                ref("https://recordings.example.com/a.wav"), ALLOWED, client=client
            )


async def test_a_transport_error_is_refused(public_dns: None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    async with mock_client(handler) as client:
        with pytest.raises(AudioUnavailable, match="fetch failed"):
            await load_span(
                ref("https://recordings.example.com/a.wav"), ALLOWED, client=client
            )


async def test_an_endless_body_is_abandoned_at_the_limit(public_dns: None) -> None:
    """A URL that streams forever costs a bounded amount of memory."""

    served = {"chunks": 0}

    async def stream():
        # Async, because httpx asserts an AsyncByteStream on an async client.
        # A sync generator here fails inside httpx rather than in the loader.
        while True:
            served["chunks"] += 1
            yield b"\x00" * 8192

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=stream())

    async with mock_client(handler) as client:
        with pytest.raises(AudioUnavailable, match="larger than the limit"):
            await load_span(
                ref("https://recordings.example.com/a.wav"), ALLOWED, client=client
            )

    # 64 KiB cap over 8 KiB chunks: it stopped reading rather than finishing.
    assert served["chunks"] <= 16, served["chunks"]


async def test_a_good_https_span_decodes(public_dns: None, tone: np.ndarray) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=wav_bytes(tone))

    async with mock_client(handler) as client:
        loaded = await load_span(
            ref("https://recordings.example.com/a.wav"), ALLOWED, client=client
        )

    assert len(loaded.samples) == len(tone)
    assert loaded.source == "https://recordings.example.com"


async def test_the_source_never_carries_the_signed_query(
    public_dns: None, tone: np.ndarray
) -> None:
    """`source` reaches the log line and the evidence bundle."""

    signed = (
        "https://recordings.example.com/a.wav"
        "?X-Amz-Signature=deadbeefdeadbeefdeadbeef&X-Amz-Expires=900"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=wav_bytes(tone))

    async with mock_client(handler) as client:
        loaded = await load_span(ref(signed), ALLOWED, client=client)

    assert loaded.source == "https://recordings.example.com"
    assert "Signature" not in loaded.source
    assert "deadbeef" not in loaded.source


async def test_an_unsupported_scheme_is_refused() -> None:
    for uri in ("s3://bucket/key", "gs://bucket/key", "ftp://host/a.wav"):
        with pytest.raises(AudioUnavailable, match="unsupported audio uri scheme"):
            await load_span(ref(uri), ALLOWED)


async def test_a_span_with_no_uri_is_refused(open_policy: AudioPolicy) -> None:
    with pytest.raises(AudioUnavailable, match="no uri"):
        await load_span(ref(None), open_policy)
