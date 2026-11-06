"""Getting call audio from an AudioRef into a numpy array.

`AudioRef` names a span of the call and, optionally, where the bytes live.
Nothing fetched them, so `detector.analysis` had no input and real mode could
not run. This module is that missing half: resolve the URI, fetch the bytes
under a budget, decode them to mono float32, and hand the samples to the
analysis engine.

It does the fetching and nothing else. No scoring, no session state, no
decision. `detector.evidence` turns the samples into factors and
`detector.state` turns factors into a decision.

The threat model, stated plainly
--------------------------------
`uri` arrives in a request body. Anyone who can reach `POST /events` with a
valid token picks the URL this service connects to, which is a server-side
request forgery primitive pointed straight at whatever the container can
reach: cloud metadata on 169.254.169.254, the service itself, anything else
inside the VPC. The response body never goes back to the caller, but a blind
SSRF is still an SSRF, and the factors and the latency are both side channels.

So fetching is off unless it is switched on, and switching it on means naming
the hosts:

    CALLBACK_AUDIO_ENABLED=1
    CALLBACK_AUDIO_ALLOWED_HOSTS=recordings.example.com,cdn.example.net

An empty allowlist fetches nothing over the network. That is the default, and
it is deliberate: a misconfigured deployment refuses audio rather than
becoming a proxy. On top of the allowlist:

* The scheme must be https, or data:, or file: when that is separately
  enabled. Plain http is never fetched.
* The host is resolved and every address it resolves to must be globally
  routable. A name in the allowlist that points at 127.0.0.1 or at
  169.254.169.254 is refused.
* Redirects are not followed. A 302 is a way to leave the allowlist, so a
  redirect is an error rather than a hop.
* The body is streamed and abandoned past `audio_max_bytes`, so a URL that
  serves an endless stream costs a bounded amount of memory.
* The whole fetch runs under `audio_fetch_timeout_ms`, which defaults well
  inside the 800 ms budget `shared.client` gives the call.

Residual, and written down rather than hidden: the address check and the
connection are two separate steps, so a name that resolves to a public address
for the check and a private one for the connection defeats it. Closing that
properly means connecting to the pinned address and carrying the hostname
through TLS separately, which breaks certificate verification unless the
transport is replaced. The allowlist is the real control here; the address
check is defence in depth behind it.

What it can decode
------------------
    pcm_s16le   raw interleaved little-endian int16, and WAV containing it
    flac        via soundfile, when soundfile is importable
    opus        not supported, and it says so rather than guessing

`AudioEncoding.PCM_S16LE` is the contract default and the format LiveKit hands
over, so that is the path that matters. FLAC works wherever soundfile
installed. Opus needs libopus, which is not in the serving image: a span that
claims opus is refused with a clear code, because a decoder that guesses would
feed noise to the analysis engine and the engine would score the noise.
"""

from __future__ import annotations

import base64
import binascii
import io
import ipaddress
import socket
import urllib.parse
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import numpy as np

from detector.config import Settings
from detector.errors import DetectorError
from shared.contract import AudioEncoding, AudioRef, ErrorCode

__all__ = [
    "MAX_REASONABLE_SECONDS",
    "AudioPolicy",
    "AudioUnavailable",
    "LoadedAudio",
    "decode_span",
    "load_span",
]


# A span longer than this is a configuration error, not a phone call. The
# analysis engine is happy with four seconds; thirty is generous.
MAX_REASONABLE_SECONDS: Final = 30.0

# int16 full scale. Dividing by this puts samples in [-1, 1).
_INT16_SCALE: Final = 32768.0

_HTTPS: Final = "https"
_DATA: Final = "data"
_FILE: Final = "file"


class AudioUnavailable(DetectorError):
    """The span could not be fetched or decoded.

    Always a 422 rather than a 5xx: the span is part of the request, and a URI
    the service will not fetch is the caller's problem to fix. The message
    says what was wrong with it and never echoes the URI, which can carry a
    signed query string.
    """

    http_status = 422
    code = ErrorCode.VALIDATION_ERROR


@dataclass(frozen=True)
class AudioPolicy:
    """Every limit on fetching, in one place.

    Built from `Settings` so the whole policy travels as one value and a test
    can hand the loader a different one without touching the environment.
    """

    enabled: bool = False
    allowed_hosts: frozenset[str] = frozenset()
    allow_file_uris: bool = False
    file_root: Path | None = None
    allow_data_uris: bool = True
    max_bytes: int = 8 * 1024 * 1024
    fetch_timeout_ms: int = 400
    max_seconds: float = MAX_REASONABLE_SECONDS

    @classmethod
    def from_settings(cls, settings: Settings) -> AudioPolicy:
        root = settings.audio_file_root
        return cls(
            enabled=settings.audio_enabled,
            allowed_hosts=frozenset(
                host.strip().lower()
                for host in settings.audio_allowed_hosts
                if host.strip()
            ),
            allow_file_uris=settings.audio_allow_file_uris,
            file_root=Path(root).resolve() if root else None,
            allow_data_uris=settings.audio_allow_data_uris,
            max_bytes=max(1024, settings.audio_max_bytes),
            fetch_timeout_ms=max(50, settings.audio_fetch_timeout_ms),
            max_seconds=min(
                MAX_REASONABLE_SECONDS, max(0.1, settings.audio_max_seconds)
            ),
        )

    @property
    def timeout_seconds(self) -> float:
        return self.fetch_timeout_ms / 1000.0


@dataclass(frozen=True)
class LoadedAudio:
    """Decoded mono samples plus how they were obtained.

    `source` is a scheme and host, never the full URI: a signed object-store
    URL carries a credential in its query string and this value reaches the
    log line and the evidence bundle.
    """

    samples: np.ndarray
    sample_rate: int
    source: str
    bytes_read: int

    @property
    def duration_s(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return float(len(self.samples)) / float(self.sample_rate)


# --------------------------------------------------------------- url checks


def _describe(parsed: urllib.parse.SplitResult) -> str:
    """A loggable name for a URL. Scheme and host only, never the query."""

    if parsed.scheme == _DATA:
        return "data:"
    if parsed.scheme == _FILE:
        return "file:"
    return f"{parsed.scheme}://{parsed.hostname or '?'}"


def _check_network_url(parsed: urllib.parse.SplitResult, policy: AudioPolicy) -> None:
    """Refuse anything that is not an allowlisted, globally routable host."""

    host = (parsed.hostname or "").lower()
    if not host:
        raise AudioUnavailable("audio uri has no host")
    if host not in policy.allowed_hosts:
        # Deliberately does not list the allowlist back: that is deployment
        # configuration and the caller has no business enumerating it.
        raise AudioUnavailable(f"audio host {host!r} is not allowed")

    port = parsed.port or 443
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise AudioUnavailable(f"audio host {host!r} did not resolve") from exc

    addresses = {info[4][0] for info in infos}
    if not addresses:
        raise AudioUnavailable(f"audio host {host!r} did not resolve")

    for raw in addresses:
        try:
            address = ipaddress.ip_address(raw.split("%", 1)[0])
        except ValueError as exc:  # pragma: no cover - getaddrinfo is well formed
            raise AudioUnavailable(
                "audio host resolved to an unreadable address"
            ) from exc
        if not address.is_global:
            # Covers loopback, private ranges, link-local (and so cloud
            # metadata), multicast and the reserved blocks, in one test.
            raise AudioUnavailable(
                f"audio host {host!r} resolves to a non-routable address"
            )


def _resolve_file_path(parsed: urllib.parse.SplitResult, policy: AudioPolicy) -> Path:
    if not policy.allow_file_uris:
        raise AudioUnavailable("file: audio uris are not enabled")
    if parsed.hostname not in (None, "", "localhost"):
        raise AudioUnavailable("file: audio uris must be local")

    path = Path(urllib.parse.unquote(parsed.path))
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise AudioUnavailable("audio file does not exist") from exc

    root = policy.file_root
    if root is None:
        raise AudioUnavailable("file: audio uris need CALLBACK_AUDIO_FILE_ROOT")
    if not resolved.is_relative_to(root):
        # Without this, a file: uri reads any file the process can read. The
        # body never comes back, but the factors are still a side channel.
        raise AudioUnavailable("audio file is outside the permitted root")
    if not resolved.is_file():
        raise AudioUnavailable("audio file is not a regular file")
    return resolved


# ----------------------------------------------------------------- fetching


def _read_data_uri(uri: str, policy: AudioPolicy) -> bytes:
    if not policy.allow_data_uris:
        raise AudioUnavailable("data: audio uris are not enabled")
    header, _, payload = uri.partition(",")
    if not _:
        raise AudioUnavailable("malformed data: audio uri")
    if ";base64" not in header:
        raise AudioUnavailable("data: audio uris must be base64")
    # Check the encoded length before decoding: base64 is 4/3 of the payload,
    # so this bounds the allocation rather than discovering it afterwards.
    if len(payload) > policy.max_bytes * 4 // 3 + 4:
        raise AudioUnavailable("audio span is larger than the limit")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AudioUnavailable("data: audio uri is not valid base64") from exc
    if len(raw) > policy.max_bytes:
        raise AudioUnavailable("audio span is larger than the limit")
    return raw


def _read_file(path: Path, policy: AudioPolicy) -> bytes:
    size = path.stat().st_size
    if size > policy.max_bytes:
        raise AudioUnavailable("audio span is larger than the limit")
    return path.read_bytes()


async def _fetch_https(
    uri: str, policy: AudioPolicy, client: httpx.AsyncClient | None
) -> bytes:
    owned = client is None
    active = client or httpx.AsyncClient(
        timeout=policy.timeout_seconds,
        # A redirect is a way out of the allowlist, so it is an error here and
        # not a hop. The status check below turns it into one.
        follow_redirects=False,
    )
    try:
        chunks: list[bytes] = []
        total = 0
        async with active.stream(
            "GET", uri, timeout=policy.timeout_seconds, follow_redirects=False
        ) as response:
            if response.is_redirect:
                raise AudioUnavailable("audio uri redirected, which is not followed")
            if response.status_code != 200:
                raise AudioUnavailable(
                    f"audio uri returned HTTP {response.status_code}"
                )
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > policy.max_bytes:
                    # Stop reading rather than finishing the download and then
                    # complaining about it.
                    raise AudioUnavailable("audio span is larger than the limit")
                chunks.append(chunk)
        return b"".join(chunks)
    except httpx.TimeoutException as exc:
        raise AudioUnavailable("audio fetch timed out") from exc
    except httpx.HTTPError as exc:
        # Everything httpx can raise, including the decoding and protocol
        # errors it raises after a response has already started.
        raise AudioUnavailable("audio fetch failed") from exc
    finally:
        if owned:
            await active.aclose()


# ----------------------------------------------------------------- decoding


def _decode_wav(raw: bytes) -> tuple[np.ndarray, int]:
    """A RIFF container holding PCM. stdlib only."""

    try:
        with wave.open(io.BytesIO(raw), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError, ValueError) as exc:
        raise AudioUnavailable("audio is not a readable WAV file") from exc

    if width != 2:
        raise AudioUnavailable(f"WAV audio must be 16-bit, got {width * 8}-bit")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / _INT16_SCALE
    return _downmix(samples, channels), rate


def _decode_raw_pcm(raw: bytes) -> np.ndarray:
    if len(raw) < 2:
        raise AudioUnavailable("audio span is empty")
    if len(raw) % 2:
        # A trailing odd byte means the span was cut mid-sample. Dropping it
        # silently would shift every following sample by one byte and turn
        # speech into noise, which the engine would then score.
        raise AudioUnavailable("pcm_s16le audio has a trailing partial sample")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / _INT16_SCALE


def _decode_flac(raw: bytes) -> tuple[np.ndarray, int]:
    try:
        import soundfile
    except ImportError as exc:
        raise AudioUnavailable(
            "flac audio needs soundfile, which is not installed here"
        ) from exc
    try:
        samples, rate = soundfile.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception as exc:
        raise AudioUnavailable("audio is not a readable FLAC file") from exc
    mono = samples.mean(axis=1) if samples.shape[1] > 1 else samples[:, 0]
    return np.ascontiguousarray(mono, dtype=np.float32), int(rate)


def _downmix(flat: np.ndarray, channels: int) -> np.ndarray:
    if channels <= 1:
        return np.ascontiguousarray(flat, dtype=np.float32)
    usable = (len(flat) // channels) * channels
    if usable == 0:
        raise AudioUnavailable("audio span is empty")
    return np.ascontiguousarray(
        flat[:usable].reshape(-1, channels).mean(axis=1), dtype=np.float32
    )


def decode_span(raw: bytes, ref: AudioRef, policy: AudioPolicy) -> LoadedAudio:
    """Bytes to mono float32, using the encoding the span declares.

    A WAV container is detected by its magic rather than by the declared
    encoding, because `pcm_s16le` is what an agent writes in the contract when
    it hands over a .wav, and reading a RIFF header as audio samples produces
    a loud click the liveness detector would score.
    """

    if not raw:
        raise AudioUnavailable("audio span is empty")

    rate = ref.sample_rate_hz
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        samples, rate = _decode_wav(raw)
    elif ref.encoding is AudioEncoding.PCM_S16LE:
        samples = _decode_raw_pcm(raw)
    elif ref.encoding is AudioEncoding.FLAC:
        samples, rate = _decode_flac(raw)
    elif ref.encoding is AudioEncoding.OPUS:
        raise AudioUnavailable(
            "opus audio cannot be decoded here. Send pcm_s16le or flac."
        )
    else:  # pragma: no cover - the enum is closed
        raise AudioUnavailable(f"unsupported audio encoding {ref.encoding}")

    if samples.size == 0:
        raise AudioUnavailable("audio span decoded to no samples")
    if rate <= 0:
        raise AudioUnavailable("audio span has no sample rate")

    duration = len(samples) / rate
    if duration > policy.max_seconds:
        raise AudioUnavailable(
            f"audio span is {duration:.1f}s, over the {policy.max_seconds:.0f}s limit"
        )

    # Non-finite samples reach the analysis engine as NaN and propagate through
    # every mean and ratio in it, so they are rejected at the boundary.
    if not np.all(np.isfinite(samples)):
        raise AudioUnavailable("audio span contains non-finite samples")

    return LoadedAudio(
        samples=samples,
        sample_rate=int(rate),
        source="decoded",
        bytes_read=len(raw),
    )


# ------------------------------------------------------------------ the api


async def load_span(
    ref: AudioRef,
    policy: AudioPolicy,
    *,
    client: httpx.AsyncClient | None = None,
) -> LoadedAudio:
    """Fetch and decode one span. Raises `AudioUnavailable` on every failure.

    Never returns partial or guessed audio. The caller treats a raised
    `AudioUnavailable` as "this span produced no evidence", which the scoring
    layer already handles by refusing to call an unscored session safe.
    """

    if not policy.enabled:
        raise AudioUnavailable("audio fetching is disabled on this deployment")
    if not ref.uri:
        raise AudioUnavailable("audio span has no uri")

    parsed = urllib.parse.urlsplit(ref.uri)
    scheme = parsed.scheme.lower()
    source = _describe(parsed)

    if scheme == _DATA:
        raw = _read_data_uri(ref.uri, policy)
    elif scheme == _FILE:
        raw = _read_file(_resolve_file_path(parsed, policy), policy)
    elif scheme == _HTTPS:
        _check_network_url(parsed, policy)
        raw = await _fetch_https(ref.uri, policy, client)
    elif scheme == "http":
        raise AudioUnavailable("audio uris must be https, not http")
    else:
        raise AudioUnavailable(f"unsupported audio uri scheme {scheme!r}")

    decoded = decode_span(raw, ref, policy)
    return LoadedAudio(
        samples=decoded.samples,
        sample_rate=decoded.sample_rate,
        source=source,
        bytes_read=decoded.bytes_read,
    )
