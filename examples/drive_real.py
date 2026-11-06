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

The audio travels as `data:` uris, which is a supported transport and needs
no object store to try. In production the agent writes each span to its
bucket and sends an https url, with the bucket host named in
CALLBACK_AUDIO_ALLOWED_HOSTS.

The last two cases are refusals rather than decisions: a span this service
will not fetch, and a span pointing at the cloud metadata address. Both end
in deny, because a caller nothing could be measured against is not a caller
who passed.
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

from detector.analysis.synthetic import live_speech, replay_of, tts_speech

RATE = 8000
DURATION_S = 4.0


def pcm_data_uri(samples: np.ndarray) -> str:
    clipped = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 0.999969)
    raw = (clipped * 32768.0).astype("<i2").tobytes()
    return f"data:audio/l16;base64,{base64.b64encode(raw).decode('ascii')}"


@dataclass(frozen=True)
class Case:
    label: str
    uri: str | None
    expect: str
    why: str


def build_cases() -> list[Case]:
    live = live_speech(duration_s=DURATION_S, sample_rate=RATE, seed=1)
    return [
        Case(
            "a live caller",
            pcm_data_uri(live),
            "allow",
            "nothing failed",
        ),
        Case(
            "the same voice, played back down the line",
            pcm_data_uri(replay_of(live, sample_rate=RATE)),
            "deny",
            "replay cues in the room acoustics",
        ),
        Case(
            "a synthesiser reading the same script",
            pcm_data_uri(tts_speech(duration_s=DURATION_S, sample_rate=RATE, seed=1)),
            "deny",
            "no micro-variation, digital noise floor",
        ),
        Case(
            "a recording first, then a live voice",
            None,  # handled specially below
            "deny",
            "the worst span wins",
        ),
        Case(
            "no audio at all",
            None,
            "deny",
            "nothing to verify against",
        ),
        Case(
            "a span this deployment will not fetch",
            "https://169.254.169.254/latest/meta-data/",
            "deny",
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


async def run_case(
    client: httpx.AsyncClient, index: int, case: Case, mixed: tuple[str, str] | None
) -> tuple[Case, dict | None, str | None]:
    call = Call(client, f"sess-real-demo-{index:04d}")
    started = await call.send("session_start", claimed_user_id="cfo@northwind.example")
    if started.status_code != 200:
        return case, None, f"session_start: HTTP {started.status_code} {started.text}"

    uris = [case.uri, case.uri] if mixed is None else list(mixed)
    last: httpx.Response | None = None
    for uri in uris:
        last = await call.speech(uri)
    if last is None or last.status_code != 200:
        code = last.status_code if last else "?"
        text = last.text if last else ""
        return case, None, f"speech_segment: HTTP {code} {text}"
    return case, last.json(), None


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

        live = live_speech(duration_s=DURATION_S, sample_rate=RATE, seed=1)
        mixed_pair = (
            pcm_data_uri(replay_of(live, sample_rate=RATE)),
            pcm_data_uri(live),
        )

        rows: list[tuple[Case, dict | None, str | None]] = []
        for index, case in enumerate(cases):
            pair = mixed_pair if case.label.startswith("a recording first") else None
            rows.append(await run_case(client, index, case, pair))

    width = max(len(case.label) for case in cases)
    print(f"{'what the caller sent':{width}}  {'decision':9}  {'risk':>5}  signals")
    print("-" * (width + 60))

    failures = 0
    for case, body, error in rows:
        if error is not None:
            print(f"{case.label:{width}}  {'ERROR':9}  {'-':>5}  {error}")
            failures += 1
            continue
        decision = body["decision"]
        risk = body["risk_score"]
        verdicts = {f["name"]: f["verdict"] for f in body["factors"]}
        failed = [name for name, v in verdicts.items() if v == "fail"]
        ran = [name for name, v in verdicts.items() if v != "not_run"]
        summary = (
            f"fail: {', '.join(failed)}"
            if failed
            else (f"{len(ran)} measured, none failed" if ran else "none measured")
        )
        mark = "" if decision == case.expect else "   <-- UNEXPECTED"
        if decision != case.expect:
            failures += 1
        print(f"{case.label:{width}}  {decision:9}  {risk:5.2f}  {summary}{mark}")

    print()
    for case, body, error in rows:
        if error or body is None:
            continue
        print(f"{case.label}:")
        print(f"    {body['reason']}")

    print()
    if failures:
        print(f"{failures} of {len(rows)} cases did not behave as documented.")
        return 1
    print(f"all {len(rows)} cases behaved as documented.")
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
