"""An opt-in local transcript of each call, for reading back what happened in a rehearsal.

Off unless CALLBACK_TRANSCRIPT_DIR names a directory. Then each call appends JSON lines to its own
file there: one per turn (role, times, whether it was interrupted, the text, and the latency
LiveKit measured for it) and one per batch of tool calls (each tool's name, how long it ran and
whether it failed).

The session itself is never recorded (worker.py starts it with record=False), because callers say
SMS codes and bank numbers. This keeps to that: every number in a turn is redacted to [NUM], with
no exemption, so even the answer to a secret ask, which reaches the detector with its digits, is
[NUM] here. A tool call is written without its arguments or its output. A write that fails is
logged and dropped, and never reaches the call.
"""

import json
import logging
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from callback_shared.contract import Turn
from livekit.agents.voice.events import FunctionToolsExecutedEvent

from callback_agent.redact import redact_turn

logger = logging.getLogger(__name__)

TRANSCRIPT_DIR_ENV = "CALLBACK_TRANSCRIPT_DIR"

# The parts of LiveKit's per-message MetricsReport that are durations, in seconds.
LATENCY_KEYS = (
    "transcription_delay",
    "end_of_turn_delay",
    "on_user_turn_completed_delay",
    "llm_node_ttft",
    "llm_node_ttfs",
    "tts_node_ttfb",
    "playback_latency",
)

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


class Transcript:
    """Appends one call's records to `path`, one JSON object per line."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def turn(self, turn: Turn, metrics: Mapping[str, Any] | None = None) -> None:
        redacted = redact_turn(turn, exempt_short_runs=False)
        latency = {
            key: round(value * 1000)
            for key, value in (metrics or {}).items()
            if key in LATENCY_KEYS and isinstance(value, int | float)
        }
        self._write(
            {
                "kind": "turn",
                "role": redacted.role.value,
                "start_ms": redacted.start_ms,
                "end_ms": redacted.end_ms,
                "interrupted": redacted.interrupted,
                "text": redacted.text,
                "latency_ms": latency,
            }
        )

    def tools(self, event: FunctionToolsExecutedEvent) -> None:
        errors = {output.call_id: output.is_error for output in event.function_call_outputs}
        self._write(
            {
                "kind": "tools",
                "at_ms": round(event.created_at * 1000),
                "calls": [
                    {
                        "name": call.name,
                        "ms": round((event.created_at - call.created_at) * 1000),
                        "error": errors.get(call.call_id, False),
                    }
                    for call in event.function_calls
                ],
            }
        )

    def _write(self, record: dict[str, Any]) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as out:
                out.write(json.dumps(record) + "\n")
        except OSError as exc:
            logger.warning("could not write the transcript: %s", type(exc).__name__)


def open_transcript(env: Mapping[str, str], room_name: str) -> Transcript | None:
    """The transcript of the call in `room_name`, or None when CALLBACK_TRANSCRIPT_DIR is blank."""
    directory = (env.get(TRANSCRIPT_DIR_ENV) or "").strip()
    if not directory:
        return None
    folder = Path(directory).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y%m%d-%H%M%S")
    return Transcript(folder / f"{started}-{_UNSAFE.sub('', room_name)}.jsonl")
