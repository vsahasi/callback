"""Drive a running detector in real mode, with actual audio.

`drive_mock.py` exercises the five scripted scenarios. This one sends
waveforms and lets the analysis engine decide, so what it prints is measured
rather than scripted.

    CALLBACK_API_TOKEN=kit-test \\
    CALLBACK_MOCK_MODE=0 \\
    CALLBACK_AUDIO_ENABLED=1 \\
    uv run uvicorn detector.app:app --port 8021

    uv run python examples/drive_real.py --base-url http://127.0.0.1:8021 \\
        --token kit-test

Every case runs twice: once with nothing enrolled for the vendor, then again
after enrolling a reference recording. That is the point of the table. With no
reference, `voice_match` does not run and its 0.35 of the weight is spread
over the four signals that did, which inflates every risk score. Enrolling the
vendor is what makes the model complete.

Two of the rows are worth looking at closely.

The impostor is only caught once the vendor is enrolled. A live human reading
the attacker's script passes synthesis, replay and liveness, because a live
human is exactly what they are looking for. `voice_match` is the only signal
that notices, and without a reference it is not running.

The synthesiser gets a *weaker* outcome once the vendor is enrolled: deny
becomes challenge. A clone of the vendor's voice is supposed to match on
`voice_match`, so a passing 0.35 signal dilutes the two that failed. That is
this product's stated policy rather than a regression: the `cloned_voice`
scenario in `detector.mock` has always said voice matches, synthesis fails,
step up. What catches a clone is the one-time code sent to the number on the
old invoice, not the blended score.

The audio travels as `data:` uris, which is a supported transport and needs
no object store to try. In production the agent writes each span to its
bucket and sends an https url, with the bucket host named in
CALLBACK_AUDIO_ALLOWED_HOSTS.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx
import numpy as np

from detector.analysis.synthetic import SpeechStyle, live_speech, replay_of, tts_speech
from shared.client import new_session_id

RATE = 8000
DURATION_S = 4.0
IDENTITY = "cfo@northwind.example"

# A different realisation of the same talker: same voice, different words.
# Enrolling the exact recording the caller then sends would score a perfect
# match for reasons that have nothing to do with the voice.
ENROL_SEED = 11
CALLER_SEED = 1


def pcm_data_uri(samples: np.ndarray) -> str:
    clipped = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 0.999969)
    raw = (clipped * 32768.0).astype("<i2").tobytes()
    return f"data:audio/l16;base64,{base64.b64encode(raw).decode('ascii')}"


def audio_ref(samples: np.ndarray) -> dict[str, object]:
    return {
        "track_id": "caller-0",
        "start_ms": 0,
        "end_ms": int(len(samples) / RATE * 1000),
        "sample_rate_hz": RATE,
        "encoding": "pcm_s16le",
        "uri": pcm_data_uri(samples),
    }


@dataclass(frozen=True)
class Case:
    label: str
    uris: tuple[str | None, ...]
    why: str


@dataclass(frozen=True)
class Outcome:
    decision: str
    risk: float
    confidence: float
    voice_match: str
    failed: tuple[str, ...]
    measured: int
    reason: str
    error: str | None = None


def build_cases() -> list[Case]:
    live = live_speech(duration_s=DURATION_S, sample_rate=RATE, seed=CALLER_SEED)
    live_uri = pcm_data_uri(live)
    replay_uri = pcm_data_uri(replay_of(live, sample_rate=RATE))
    return [
        Case(
            "a live caller",
            (live_uri, live_uri),
            "nothing failed",
        ),
        Case(
            "a different person reading the script",
            tuple(
                [
                    pcm_data_uri(
                        live_speech(
                            duration_s=DURATION_S,
                            sample_rate=RATE,
                            seed=CALLER_SEED,
                            style=SpeechStyle(f0_hz=215.0),
                        )
                    )
                ]
                * 2
            ),
            "only voice_match notices this one",
        ),
        Case(
            "the same voice, played back down the line",
            (replay_uri, replay_uri),
            "replay cues in the room acoustics",
        ),
        Case(
            "a synthesiser reading the same script",
            tuple(
                [
                    pcm_data_uri(
                        tts_speech(
                            duration_s=DURATION_S, sample_rate=RATE, seed=CALLER_SEED
                        )
                    )
                ]
                * 2
            ),
            "no micro-variation, digital noise floor",
        ),
        Case(
            "a recording first, then a live voice",
            (replay_uri, live_uri),
            "the worst span wins",
        ),
        Case(
            "no audio at all",
            (None, None),
            "nothing to verify against",
        ),
        Case(
            "a span this deployment will not fetch",
            (
                "https://169.254.169.254/latest/meta-data/",
                "https://169.254.169.254/latest/meta-data/",
            ),
            "cloud metadata, refused before any connection",
        ),
    ]


class Call:
    def __init__(self, client: httpx.AsyncClient, session_id: str) -> None:
        self.client = client
        self.session_id = session_id
        self.sequence = 0
        self.start = datetime.now(UTC)

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

    async def speech(self, uri: str | None) -> httpx.Response:
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
        return await self.send("speech_segment", audio=audio)


async def run_case(client: httpx.AsyncClient, session_id: str, case: Case) -> Outcome:
    call = Call(client, session_id)
    started = await call.send("session_start", claimed_user_id=IDENTITY)
    if started.status_code != 200:
        return _failed(f"session_start: HTTP {started.status_code}")

    last: httpx.Response | None = None
    for uri in case.uris:
        last = await call.speech(uri)
    if last is None or last.status_code != 200:
        code = last.status_code if last else "?"
        return _failed(f"speech_segment: HTTP {code}")

    body = last.json()
    verdicts = {f["name"]: f["verdict"] for f in body["factors"]}
    return Outcome(
        decision=body["decision"],
        risk=body["risk_score"],
        confidence=body["confidence"],
        voice_match=verdicts.get("voice_match", "absent"),
        failed=tuple(name for name, v in verdicts.items() if v == "fail"),
        measured=sum(1 for v in verdicts.values() if v != "not_run"),
        reason=body["reason"] or "",
    )


def _failed(message: str) -> Outcome:
    return Outcome("ERROR", 0.0, 0.0, "absent", (), 0, "", error=message)


async def enrol(client: httpx.AsyncClient) -> str:
    """Store a reference voice for the vendor. Returns a one-line summary."""

    reference = live_speech(duration_s=6.0, sample_rate=RATE, seed=ENROL_SEED)
    response = await client.post(
        "/enrol/voice",
        json={
            "identity": IDENTITY,
            "audio": audio_ref(reference),
            "label": "a trusted earlier call with this vendor",
            "replace": True,
        },
    )
    if response.status_code != 200:
        return f"enrolment failed: HTTP {response.status_code} {response.text[:200]}"
    record = response.json()["record"]
    return (
        f"enrolled {record['identity']} from {record['duration_s']:.0f}s of audio, "
        f"{record['voiced_seconds']:.1f}s of it voiced, quality {record['quality']}"
    )


async def erase(client: httpx.AsyncClient) -> None:
    await client.post("/enrol/voice/erase", json={"identity": IDENTITY})


async def run(args: argparse.Namespace) -> int:
    headers = {"Authorization": f"Bearer {args.token}"}
    cases = build_cases()

    async with httpx.AsyncClient(
        base_url=args.base_url, headers=headers, timeout=30.0
    ) as client:
        health = await client.get("/healthz")
        if health.status_code != 200:
            print(f"detector is not healthy: HTTP {health.status_code}")
            return 2
        info = health.json()
        print(f"detector: {args.base_url}")
        print(f"contract: {info['contract_version']}   mock_mode: {info['mock_mode']}")
        if info["mock_mode"]:
            print()
            print("This detector is in mock mode, so nothing below is measured.")
            print("Restart it with CALLBACK_MOCK_MODE=0 CALLBACK_AUDIO_ENABLED=1.")
            return 2
        print()

        # Fresh ids every run. A fixed id would be terminal from the previous
        # run and every event would come back 409, which is the state machine
        # working correctly and a demo that only runs once.
        run_id = new_session_id("run")[-8:]

        # Pass one: no reference voice for this vendor.
        await erase(client)
        print("no enrolment for this vendor yet, so voice_match cannot run.")
        bare = [
            await run_case(client, f"sess-bare-{run_id}-{index:02d}", case)
            for index, case in enumerate(cases)
        ]

        print(await enrol(client))
        print()
        enrolled = [
            await run_case(client, f"sess-live-{run_id}-{index:02d}", case)
            for index, case in enumerate(cases)
        ]

    return report(cases, bare, enrolled)


def report(cases: list[Case], bare: list[Outcome], enrolled: list[Outcome]) -> int:
    width = max(len(case.label) for case in cases)
    print(
        f"{'what the caller sent':{width}}  "
        f"{'not enrolled':>18}  {'enrolled':>18}  voice_match"
    )
    print("-" * (width + 62))

    errors = 0
    for case, before, after in zip(cases, bare, enrolled, strict=True):
        if before.error or after.error:
            errors += 1
            message = before.error or after.error
            print(f"{case.label:{width}}  ERROR: {message}")
            continue
        left = f"{before.decision} {before.risk:.2f}"
        right = f"{after.decision} {after.risk:.2f}"
        print(f"{case.label:{width}}  {left:>18}  {right:>18}  {after.voice_match}")

    print()
    print("signals measured, out of five:")
    for case, before, after in zip(cases, bare, enrolled, strict=True):
        if before.error or after.error:
            continue
        failed = ", ".join(after.failed) or "none"
        print(
            f"  {case.label:{width}}  {before.measured} then {after.measured}, "
            f"confidence {before.confidence:.2f} then {after.confidence:.2f}, "
            f"failing: {failed}"
        )

    print()
    for case, _, after in zip(cases, bare, enrolled, strict=True):
        if after.error:
            continue
        print(f"{case.label}:")
        print(f"    {after.reason}")

    print()
    changed = [
        case.label
        for case, before, after in zip(cases, bare, enrolled, strict=True)
        if not (before.error or after.error) and before.decision != after.decision
    ]
    if errors:
        print(f"{errors} of {len(cases)} cases could not be driven.")
        return 1
    print(
        f"enrolling the vendor changed the decision on: {', '.join(changed) or 'none'}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("CALLBACK_BASE_URL", "http://127.0.0.1:8021"),
        help="Detector base URL. Defaults to $CALLBACK_BASE_URL.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("CALLBACK_API_TOKEN", "kit-test"),
        help="Bearer token. Defaults to $CALLBACK_API_TOKEN.",
    )
    return parser.parse_args()


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    sys.exit(main())
