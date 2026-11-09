"""Build the showcase page by measuring real audio with the real detector.

    uv run python showcase/build.py

Writes `showcase/dist/showcase.html`, a single file with no backend: the
audio, the verdicts and the page are all inside it. That file is gitignored,
because it is a few hundred kilobytes of base64 and this script rebuilds it
in seconds.

Why it drives the service over HTTP instead of calling the engine
-----------------------------------------------------------------
Calling `AnalysisEngine` directly would give the five factor scores, and the
page would then have to work out the decision itself. That is a second
implementation of the scoring rules, and a second implementation is a second
thing to be wrong. So this posts events to the real app over an in-process
ASGI transport and records the response. The decision, the blended risk, the
confidence, the next action and the sentences under each signal all come back
from the service exactly as the agent would receive them. Nothing on the page
is written by hand, so a threshold that moves moves the page with it.

Each caller is run twice, once with a reference voice on file for the vendor
and once without, because the difference between those two is the point: with
nothing enrolled, a different person reading the attacker's script is allowed.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import pathlib
import wave

import httpx
import numpy as np

from detector.analysis.dsp import build_context
from detector.analysis.synthetic import SpeechStyle, live_speech, replay_of, tts_speech
from detector.analysis.voiceprint import voiceprint_from
from detector.app import create_app
from detector.config import Settings
from detector.enrolment import InMemoryVoicePrintStore, storable_enrolment

HERE = pathlib.Path(__file__).resolve().parent
TEMPLATE = HERE / "page.template.html"
DIST = HERE / "dist"

RATE = 8000
TOKEN = "showcase-build-token"
IDENTITY = "cfo@northwind.example"

# The caller and the enrolled reference are the same talker saying different
# things, which is what a real enrolment is. Enrolling the very clip the
# caller then sends would score a perfect match for reasons that have nothing
# to do with the voice. The impostor is a different larynx, nine semitones up,
# reading the same script.
CALLER_SEED = 1
ENROL_SEED = 11
IMPOSTOR_F0 = 215.0


def settings() -> Settings:
    return Settings(
        api_token=TOKEN,
        mock_mode=False,
        session_ttl_seconds=300,
        otp_max_attempts=3,
        otp_ttl_seconds=120,
        mock_jitter_min_ms=0,
        mock_jitter_max_ms=0,
        speech_segments_before_decision=2,
        audio_enabled=True,
        audio_allow_data_uris=True,
    )


def wav_b64(samples: np.ndarray, rate: int = RATE) -> str:
    """A playable WAV, for the page's audio element and its waveform."""

    clipped = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 0.999969)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((clipped * 32768.0).astype("<i2").tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def data_uri(samples: np.ndarray) -> str:
    """Raw PCM for the detector, which reads `data:` uris in real mode."""

    clipped = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 0.999969)
    raw = (clipped * 32768.0).astype("<i2").tobytes()
    return f"data:audio/l16;base64,{base64.b64encode(raw).decode('ascii')}"


def callers() -> dict[str, dict]:
    live = live_speech(4.0, RATE, seed=CALLER_SEED)
    return {
        "live": {
            "label": "The real CFO",
            "blurb": "A live human at a handset, the person the account belongs to.",
            "samples": live,
        },
        "replay": {
            "label": "A recording played down the line",
            "blurb": "The same voice, captured earlier and played into the phone "
            "from a speaker.",
            "samples": replay_of(live, sample_rate=RATE),
        },
        "clone": {
            "label": "A synthesised clone",
            "blurb": "A text-to-speech model reading the attacker's script in the "
            "CFO's register.",
            "samples": tts_speech(4.0, RATE, seed=CALLER_SEED),
        },
        "impostor": {
            "label": "A different person",
            "blurb": "A live human, just not the right one, reading the attacker's "
            "script.",
            "samples": live_speech(
                4.0, RATE, seed=CALLER_SEED, style=SpeechStyle(f0_hz=IMPOSTOR_F0)
            ),
        },
    }


async def one_call(client: httpx.AsyncClient, session_id: str, uri: str) -> dict:
    """Open a session, send two speech segments, return the final decision."""

    await client.post(
        "/events",
        json={
            "session_id": session_id,
            "sequence": 1,
            "event": "session_start",
            "occurred_at": "2026-11-09T09:00:00+00:00",
            "claimed_user_id": IDENTITY,
        },
    )
    last: httpx.Response | None = None
    for sequence in (2, 3):
        last = await client.post(
            "/events",
            json={
                "session_id": session_id,
                "sequence": sequence,
                "event": "speech_segment",
                "occurred_at": f"2026-11-09T09:0{sequence}:00+00:00",
                "audio": {
                    "track_id": "caller-0",
                    "start_ms": sequence * 4000,
                    "end_ms": (sequence + 1) * 4000,
                    "sample_rate_hz": RATE,
                    "encoding": "pcm_s16le",
                    "uri": uri,
                },
            },
        )
    if last is None or last.status_code != 200:
        raise RuntimeError(
            f"the detector refused the call: {last.text if last else ''}"
        )
    body = last.json()
    return {
        "decision": body["decision"],
        "risk": body["risk_score"],
        "confidence": body["confidence"],
        "next_action": body["next_action"],
        "reason": body["reason"],
        "factors": [
            {
                "name": f["name"],
                "score": f["score"],
                "weight": f["weight"],
                "verdict": f["verdict"],
                "detail": f["detail"],
            }
            for f in body["factors"]
        ],
    }


async def measure() -> dict:
    cases = callers()
    reference = live_speech(6.0, RATE, seed=ENROL_SEED)
    results: dict[str, dict] = {key: {} for key in cases}

    for enrolled in (False, True):
        store = InMemoryVoicePrintStore()
        if enrolled:
            record = storable_enrolment(
                IDENTITY,
                voiceprint_from(build_context(reference, RATE)),
                source="demo",
                channel="phone",
                label="a trusted earlier call",
            )
            if record is None:
                raise RuntimeError("the reference recording was not usable")
            store.put(record, replace=True)

        app = create_app(settings(), voiceprints=store)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://showcase.test",
            headers={"Authorization": f"Bearer {TOKEN}"},
        ) as client:
            for index, (key, case) in enumerate(cases.items()):
                results[key]["enrolled" if enrolled else "bare"] = await one_call(
                    client, f"sess-{int(enrolled)}-{index}", data_uri(case["samples"])
                )

    return {
        "generated_from": "detector.app over ASGI, real mode, audio enabled",
        "sample_rate_hz": RATE,
        "identity": IDENTITY,
        "attacks": [
            {
                "key": key,
                "label": case["label"],
                "blurb": case["blurb"],
                "audio": wav_b64(case["samples"]),
                **results[key],
            }
            for key, case in cases.items()
        ],
    }


def render(payload: dict) -> pathlib.Path:
    template = TEMPLATE.read_text(encoding="utf-8")
    if "__DATA__" not in template:
        raise RuntimeError(f"{TEMPLATE.name} has no __DATA__ placeholder")
    # `</` inside a <script> would end the block early, whatever the quoting.
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    DIST.mkdir(parents=True, exist_ok=True)
    out = DIST / "showcase.html"
    out.write_text(template.replace("__DATA__", blob), encoding="utf-8")
    return out


def main() -> int:
    payload = asyncio.run(measure())
    out = render(payload)
    print(f"{out.relative_to(HERE.parent)}  {out.stat().st_size / 1024:.0f} KB")
    for item in payload["attacks"]:
        bare, enrolled = item["bare"], item["enrolled"]
        changed = (
            "  <- enrolling changes this"
            if bare["decision"] != enrolled["decision"]
            else ""
        )
        print(
            f"  {item['key']:9} "
            f"no reference: {bare['decision']:9} {bare['risk']:.2f}   "
            f"enrolled: {enrolled['decision']:9} {enrolled['risk']:.2f}{changed}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
