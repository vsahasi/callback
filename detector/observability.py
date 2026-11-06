"""Structured logging, Prometheus metrics and the per-session journal.

This module is the detector's observability layer. It owns three things and
nothing else:

    Redactor          one place that decides what must never be written down
    StructuredLogger  one JSON line per request, on stdout
    MetricsRegistry   hand-rolled Prometheus text exposition
    SessionJournal    a bounded, append-only record of what happened on a call

The journal exists because the session store has no listing API and keeps no
timeline: `Session` holds the current verdict, not the history that produced
it. A human reviewing a held payment needs the history, so the observability
layer records it as requests pass through, redacted at the boundary. Nothing
secret ever enters the journal, which is why the evidence bundle in
`detector.review` can read it without a second scrub.

Nothing here imports `detector.app`. The wiring lives in `detector.review`,
which calls `mount(app, ...)`.

Cost, and turning it off
------------------------
On the hot path a request costs: one uuid4, one `time.perf_counter` pair, a
JSON parse of bodies we already hold in memory, a few dict writes and one
`json.dumps`. No network, no thread, no background task. Set
`ObservabilityConfig(log_enabled=False, metrics_enabled=False,
journal_enabled=False)` and the middleware becomes a direct pass-through with
a single boolean test. Each switch is independent, so metrics can stay on
while logging goes quiet.

Redaction, stated plainly
-------------------------
Three layers, in this order:

    1. Allowlist. The log record and the journal entry are built field by
       field from a fixed list. Request bodies are never copied wholesale, and
       `otp_code` is read for presence only; its value is never touched.
    2. Key rules. Any field whose name looks like a secret or a bank detail is
       replaced, however deeply nested and whatever it arrived in.
    3. Value rules. Known secret literals (the bearer token, the mock OTP
       code) are scrubbed out of every string, as are IBAN-shaped tokens and
       runs of nine or more digits.

Layer 3 is what catches a secret that arrives in a field nobody expected. It
is a backstop, not the plan: an eight-digit account number in free text is not
matched by the digit rule, which is why layers 1 and 2 come first.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import os
import re
import secrets
import sys
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Annotated, Any, TextIO

from fastapi import APIRouter, Depends, Header, Response

PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

REQUEST_ID_HEADER = "X-Callback-Request-Id"

JSON_PARSE_ERRORS = (json.JSONDecodeError, UnicodeDecodeError, ValueError, TypeError)


def _utcnow() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------- redaction

REDACTED = "[REDACTED]"
REDACTED_BANK = "[REDACTED:BANK_DETAIL]"

#: Below this length a "secret" is too short to scrub safely: replacing every
#: occurrence of a three-character string would shred ordinary prose.
MIN_SCRUBBABLE_SECRET = 6

#: A field whose normalised name contains any of these is replaced outright.
SENSITIVE_KEY_NEEDLES: tuple[str, ...] = (
    "otp",
    "token",
    "secret",
    "password",
    "passphrase",
    "credential",
    "apikey",
    "authorization",
    "bearer",
    "account",
    "iban",
    "routing",
    "sortcode",
    "swift",
    "bic",
    "cardnumber",
    "cvv",
    "cvc",
    "ssn",
    "taxid",
    "beneficiary",
    "bank",
)

#: Names that are sensitive on their own but far too common as substrings to
#: use as needles. "code" cannot be a needle: `error_code` and `scenario_code`
#: are exactly the fields an operator needs to see.
SENSITIVE_KEY_EXACT: frozenset[str] = frozenset(
    {
        "code",
        "pin",
        "pincode",
        "passcode",
        "verificationcode",
        "challengecode",
        "key",
        "auth",
    }
)

#: Field names this module produces that match a sensitive needle but carry no
#: secret. Checked first, so the allowlist beats the needle. Every one of these
#: is *about* the one-time code without being the code: its length, how many
#: attempts were made, how it came out. Note what is absent: `otp_code`, and a
#: bare `otp`, both of which stay sensitive, because a value under either of
#: those names in arbitrary metadata could well be the code itself.
SAFE_KEY_NAMES: frozenset[str] = frozenset(
    {
        "otpsubmitted",
        "otpoutcome",
        "otpattempts",
        "otpattemptsused",
        "otpattemptsremaining",
        "otpmaxattempts",
        "otpchallenged",
        "otpchallenges",
        "otpresolved",
        "otpttlseconds",
        "otpweight",
        "codelength",
        "codewithheld",
        "activesessions",
    }
)

#: Identifiers. Secret literals are still scrubbed out of these, but the
#: bank-detail *patterns* are not applied, because an identifier is allowed to
#: look like a long number. Without this exemption a session id that happens to
#: contain nine consecutive hex digits came back mangled, and a mangled id is a
#: record nobody can look up.
IDENTIFIER_KEY_NAMES: frozenset[str] = frozenset(
    {
        "sessionid",
        "requestid",
        "claimeduserid",
        "idempotencykey",
        "trackid",
        "endpoint",
        "path",
        "nextaction",
        "callevent",
        "event",
        "occurredat",
        "ts",
        "at",
    }
)

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

#: ISO 13616 shape: two letters, two check digits, then the national part.
_IBAN_RE = re.compile(r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{8,30}\b")

#: Nine or more consecutive digits. Nine, not eight, so that an ISO date, a
#: port number and a millisecond latency survive. Applied to prose only, never
#: to the identifier fields above. A backstop, not the plan: an eight-digit
#: account number in free text slips through it, which is why the key rules
#: come first.
_LONG_DIGITS_RE = re.compile(r"\b\d{9,}\b")


def normalise_key(key: str) -> str:
    """Lowercase, strip every non-alphanumeric character.

    So `Bank-Account_Number`, `bankAccountNumber` and `bank account number`
    all collapse to the same needle-matchable string.
    """

    return _NON_ALNUM.sub("", key.lower())


class Redactor:
    """Decides what never gets written down.

    `extra_secrets` is how a real-mode OTP code or a rotated token is added at
    runtime: the per-session code is random, so it cannot be known here at
    construction time. Key rules already stop `otp_code` reaching a log, and
    the value never leaves `OtpState`; this is the belt to that braces.
    """

    def __init__(
        self,
        secret_values: Iterable[str] = (),
        *,
        max_depth: int = 6,
        max_string: int = 2000,
        scrub_patterns: bool = True,
    ) -> None:
        self._secrets: tuple[str, ...] = ()
        self._max_depth = max_depth
        self._max_string = max_string
        self._scrub_patterns = scrub_patterns
        self._lock = threading.Lock()
        self.add_secrets(secret_values)

    # ------------------------------------------------------------- management

    def add_secrets(self, values: Iterable[str]) -> None:
        """Register literal values that must never appear in output."""

        with self._lock:
            known = set(self._secrets)
            for value in values:
                if value and len(value) >= MIN_SCRUBBABLE_SECRET:
                    known.add(value)
            # Longest first, so a token that contains another is scrubbed whole.
            self._secrets = tuple(sorted(known, key=len, reverse=True))

    @property
    def secret_count(self) -> int:
        return len(self._secrets)

    # ----------------------------------------------------------------- checks

    def is_sensitive_key(self, key: str) -> bool:
        normalised = normalise_key(key)
        if not normalised:
            return False
        if normalised in SAFE_KEY_NAMES:
            return False
        if normalised in SENSITIVE_KEY_EXACT:
            return True
        return any(needle in normalised for needle in SENSITIVE_KEY_NEEDLES)

    # ------------------------------------------------------------------ action

    def scrub_secrets(self, text: str) -> str:
        """Remove registered secret literals from one string. Patterns are not
        applied, so an identifier keeps its shape.
        """

        if not text:
            return text
        if len(text) > self._max_string:
            text = text[: self._max_string] + "...[truncated]"
        for secret_value in self._secrets:
            if secret_value in text:
                text = text.replace(secret_value, REDACTED)
        return text

    def scrub_text(self, text: str) -> str:
        """Secret literals and bank-detail shapes, for free prose.

        Key rules are not applied, so this is the right call for prose and for
        structures whose field names are themselves part of the record. Use
        `scrub_secrets` for an identifier.
        """

        text = self.scrub_secrets(text)
        if text and self._scrub_patterns:
            text = _IBAN_RE.sub(REDACTED_BANK, text)
            text = _LONG_DIGITS_RE.sub(REDACTED_BANK, text)
        return text

    def redact(self, value: Any, *, _depth: int = 0) -> Any:
        """Recursively redact a structure by key name and by value.

        Mappings, sequences, sets, strings and scalars are all handled. Depth
        is capped so a cyclic or pathological payload cannot hang a log write.
        """

        if _depth > self._max_depth:
            return "[REDACTED:DEPTH]"

        if isinstance(value, str):
            return self.scrub_text(value)

        if isinstance(value, Mapping):
            out: dict[str, Any] = {}
            for raw_key, item in value.items():
                key = str(raw_key)
                safe_key = self.scrub_text(key)
                if self.is_sensitive_key(key):
                    out[safe_key] = REDACTED
                    continue
                if normalise_key(key) in IDENTIFIER_KEY_NAMES and isinstance(item, str):
                    out[safe_key] = self.scrub_secrets(item)
                    continue
                out[safe_key] = self.redact(item, _depth=_depth + 1)
            return out

        if isinstance(value, (list, tuple, set, frozenset)):
            return [self.redact(item, _depth=_depth + 1) for item in value]

        if isinstance(value, (bool, int, float)) or value is None:
            return value

        if isinstance(value, datetime):
            return value.isoformat()

        return self.scrub_text(str(value))

    def contains_secret(self, text: str) -> bool:
        """Whether any registered secret literal appears in `text`.

        For assertions, in tests and in a guard before anything is published.
        """

        return any(secret_value in text for secret_value in self._secrets)


# ----------------------------------------------------------------------- logging


class StructuredLogger:
    """JSON lines on stdout, one object per line, nothing buffered across calls.

    Deliberately not built on `logging`: there is no formatter to misconfigure,
    no handler to inherit, and the output is exactly the bytes a log shipper
    parses. `enabled=False` makes `log()` a single boolean test and a return.
    """

    def __init__(
        self,
        *,
        service: str = "callback-detector",
        enabled: bool = True,
        redactor: Redactor | None = None,
        stream: TextIO | None = None,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self.service = service
        self.enabled = enabled
        self.redactor = redactor or Redactor()
        self._stream = stream
        self._clock = clock
        self._lock = threading.Lock()

    @property
    def stream(self) -> TextIO:
        # Resolved per write, not pinned at construction, so a test that
        # captures stdout sees the lines.
        return self._stream if self._stream is not None else sys.stdout

    def log(self, event: str, *, level: str = "info", **fields: Any) -> dict[str, Any]:
        """Write one record. Returns the record that was written, redacted.

        Returns an empty dict when logging is off, so a caller can branch on
        truthiness without checking the flag itself.
        """

        if not self.enabled:
            return {}

        record: dict[str, Any] = {
            "ts": self._clock().isoformat(),
            "level": level,
            "service": self.service,
            "event": event,
        }
        for key, value in fields.items():
            if value is None:
                continue
            record[key] = value

        safe = self.redactor.redact(record)
        line = json.dumps(safe, default=str, separators=(",", ":"))
        with self._lock:
            stream = self.stream
            stream.write(line + "\n")
            flush = getattr(stream, "flush", None)
            if callable(flush):
                flush()
        return safe


# ----------------------------------------------------------------------- metrics


def escape_label_value(value: str) -> str:
    """Prometheus label-value escaping: backslash, double quote, newline."""

    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def escape_help(value: str) -> str:
    """Prometheus HELP escaping: backslash and newline only."""

    return value.replace("\\", "\\\\").replace("\n", "\\n")


def format_number(value: float) -> str:
    """Render a sample value the way the exposition format wants it."""

    numeric = float(value)
    if math.isnan(numeric):
        return "NaN"
    if numeric == math.inf:
        return "+Inf"
    if numeric == -math.inf:
        return "-Inf"
    if numeric.is_integer() and abs(numeric) < 1e15:
        return str(int(numeric))
    return repr(numeric)


def _render_labels(names: Sequence[str], values: Sequence[str]) -> str:
    if not names:
        return ""
    pairs = ",".join(
        f'{name}="{escape_label_value(str(value))}"'
        for name, value in zip(names, values, strict=True)
    )
    return "{" + pairs + "}"


class _Metric:
    kind = "untyped"

    def __init__(self, name: str, help_text: str, labels: Sequence[str] = ()) -> None:
        self.name = name
        self.help_text = help_text
        self.label_names = tuple(labels)
        self._lock = threading.Lock()

    def _key(self, labels: Mapping[str, Any]) -> tuple[str, ...]:
        missing = set(self.label_names) - set(labels)
        if missing:
            raise ValueError(
                f"metric {self.name!r} needs labels {sorted(self.label_names)}, "
                f"missing {sorted(missing)}"
            )
        extra = set(labels) - set(self.label_names)
        if extra:
            raise ValueError(f"metric {self.name!r} has no labels {sorted(extra)}")
        return tuple(str(labels[name]) for name in self.label_names)

    def render(self) -> list[str]:  # pragma: no cover - abstract
        raise NotImplementedError

    def _header(self, name: str | None = None) -> list[str]:
        shown = name or self.name
        return [
            f"# HELP {shown} {escape_help(self.help_text)}",
            f"# TYPE {shown} {self.kind}",
        ]


class Counter(_Metric):
    """Monotonic count. The name must end in `_total`."""

    kind = "counter"

    def __init__(self, name: str, help_text: str, labels: Sequence[str] = ()) -> None:
        if not name.endswith("_total"):
            raise ValueError(f"counter {name!r} must end in '_total'")
        super().__init__(name, help_text, labels)
        self._values: dict[tuple[str, ...], float] = {}
        if not self.label_names:
            self._values[()] = 0.0

    def inc(self, amount: float = 1.0, **labels: Any) -> None:
        if amount < 0:
            raise ValueError("a counter cannot decrease")
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: Any) -> float:
        with self._lock:
            return self._values.get(self._key(labels), 0.0)

    def render(self) -> list[str]:
        with self._lock:
            items = sorted(self._values.items())
        lines = self._header()
        lines.extend(
            f"{self.name}{_render_labels(self.label_names, key)} {format_number(value)}"
            for key, value in items
        )
        return lines


class Gauge(_Metric):
    """A value that goes up and down, or is read from a callable at scrape time."""

    kind = "gauge"

    def __init__(self, name: str, help_text: str, labels: Sequence[str] = ()) -> None:
        super().__init__(name, help_text, labels)
        self._values: dict[tuple[str, ...], float] = {}
        self._function: Callable[[], float] | None = None

    def set(self, value: float, **labels: Any) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = float(value)

    def set_function(self, source: Callable[[], float]) -> None:
        """Read the value at scrape time. Only for an unlabelled gauge."""

        if self.label_names:
            raise ValueError("set_function is only available on an unlabelled gauge")
        with self._lock:
            self._function = source

    def value(self, **labels: Any) -> float:
        with self._lock:
            if self._function is not None and not self.label_names:
                return float(self._function())
            return self._values.get(self._key(labels), 0.0)

    def render(self) -> list[str]:
        with self._lock:
            source = self._function
            items = sorted(self._values.items())
        lines = self._header()
        if source is not None and not self.label_names:
            try:
                current = float(source())
            except (TypeError, ValueError, AttributeError):
                # A broken probe must not take the scrape down with it.
                return lines
            lines.append(f"{self.name} {format_number(current)}")
            return lines
        lines.extend(
            f"{self.name}{_render_labels(self.label_names, key)} {format_number(value)}"
            for key, value in items
        )
        return lines


#: Request latency, in seconds. Tight at the bottom because the agent's whole
#: budget is 800ms (`shared.contract.CLIENT_TIMEOUT_MS`), with a bucket right
#: on that boundary so "did we blow the client's deadline" is one query.
DEFAULT_LATENCY_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    0.8,
    1.0,
    2.5,
    5.0,
)

#: Risk and confidence live in [0, 1]. Ten even buckets plus the decision
#: thresholds at 0.30 and 0.55, so the shape around a threshold is visible.
DEFAULT_SCORE_BUCKETS: tuple[float, ...] = (
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.55,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
)


class Histogram(_Metric):
    """Cumulative buckets plus `_sum` and `_count`, as the format requires."""

    kind = "histogram"

    def __init__(
        self,
        name: str,
        help_text: str,
        labels: Sequence[str] = (),
        buckets: Sequence[float] = DEFAULT_LATENCY_BUCKETS,
    ) -> None:
        if "le" in labels:
            raise ValueError("'le' is reserved for histogram buckets")
        super().__init__(name, help_text, labels)
        bounds = tuple(sorted(float(b) for b in buckets))
        if not bounds:
            raise ValueError("a histogram needs at least one bucket")
        self.buckets: tuple[float, ...] = bounds
        self._counts: dict[tuple[str, ...], list[int]] = {}
        self._sums: dict[tuple[str, ...], float] = {}
        self._totals: dict[tuple[str, ...], int] = {}

    def observe(self, value: float, **labels: Any) -> None:
        key = self._key(labels)
        numeric = float(value)
        with self._lock:
            counts = self._counts.get(key)
            if counts is None:
                counts = [0] * len(self.buckets)
                self._counts[key] = counts
                self._sums[key] = 0.0
                self._totals[key] = 0
            for index, bound in enumerate(self.buckets):
                if numeric <= bound:
                    counts[index] += 1
            self._sums[key] += numeric
            self._totals[key] += 1

    def count(self, **labels: Any) -> int:
        with self._lock:
            return self._totals.get(self._key(labels), 0)

    def total(self, **labels: Any) -> float:
        with self._lock:
            return self._sums.get(self._key(labels), 0.0)

    def render(self) -> list[str]:
        with self._lock:
            keys = sorted(self._counts)
            snapshot = {
                key: (list(self._counts[key]), self._sums[key], self._totals[key])
                for key in keys
            }
        lines = self._header()
        for key in keys:
            counts, total_sum, total_count = snapshot[key]
            for index, bound in enumerate(self.buckets):
                labels = _render_labels(
                    (*self.label_names, "le"), (*key, format_number(bound))
                )
                lines.append(f"{self.name}_bucket{labels} {counts[index]}")
            inf_labels = _render_labels((*self.label_names, "le"), (*key, "+Inf"))
            lines.append(f"{self.name}_bucket{inf_labels} {total_count}")
            base = _render_labels(self.label_names, key)
            lines.append(f"{self.name}_sum{base} {format_number(total_sum)}")
            lines.append(f"{self.name}_count{base} {total_count}")
        return lines


class MetricsRegistry:
    """Everything the service exposes, rendered in registration order."""

    def __init__(self) -> None:
        self._metrics: OrderedDict[str, _Metric] = OrderedDict()
        self._lock = threading.Lock()

    def register(self, metric: _Metric) -> _Metric:
        with self._lock:
            if metric.name in self._metrics:
                raise ValueError(f"metric {metric.name!r} is already registered")
            self._metrics[metric.name] = metric
        return metric

    def counter(self, name: str, help_text: str, labels: Sequence[str] = ()) -> Counter:
        metric = Counter(name, help_text, labels)
        self.register(metric)
        return metric

    def gauge(self, name: str, help_text: str, labels: Sequence[str] = ()) -> Gauge:
        metric = Gauge(name, help_text, labels)
        self.register(metric)
        return metric

    def histogram(
        self,
        name: str,
        help_text: str,
        labels: Sequence[str] = (),
        buckets: Sequence[float] = DEFAULT_LATENCY_BUCKETS,
    ) -> Histogram:
        metric = Histogram(name, help_text, labels, buckets)
        self.register(metric)
        return metric

    def get(self, name: str) -> _Metric | None:
        with self._lock:
            return self._metrics.get(name)

    def names(self) -> list[str]:
        with self._lock:
            return list(self._metrics)

    def render(self) -> str:
        with self._lock:
            metrics = list(self._metrics.values())
        lines: list[str] = []
        for metric in metrics:
            lines.extend(metric.render())
        return "\n".join(lines) + "\n"


# ----------------------------------------------------------- detector metrics


class DetectorMetrics:
    """The metric set this service actually publishes.

    Names follow the Prometheus conventions: a `callback_` namespace, base
    units (seconds, not milliseconds), `_total` on every counter. Label values
    are drawn from closed contract enums, so cardinality is bounded by
    construction: `endpoint` is a route template and never a raw path.
    """

    def __init__(self, registry: MetricsRegistry | None = None) -> None:
        self.registry = registry or MetricsRegistry()
        reg = self.registry

        self.build_info = reg.gauge(
            "callback_detector_build_info",
            "Always 1. Labels carry the contract version and the mock flag.",
            ("contract_version", "mock_mode"),
        )
        self.requests = reg.counter(
            "callback_http_requests_total",
            "HTTP requests handled, by route template, method and status code.",
            ("endpoint", "method", "status"),
        )
        self.latency = reg.histogram(
            "callback_http_request_duration_seconds",
            "Wall-clock time to handle one request, by route template.",
            ("endpoint",),
            DEFAULT_LATENCY_BUCKETS,
        )
        self.decisions = reg.counter(
            "callback_decisions_total",
            "Decisions returned to the agent, by decision and mock scenario. "
            "Idempotent replays are not counted twice.",
            ("decision", "scenario"),
        )
        self.sessions_terminal = reg.counter(
            "callback_sessions_terminal_total",
            "Sessions that reached a terminal state, counted once each.",
            ("state", "scenario"),
        )
        self.otp_challenges = reg.counter(
            "callback_otp_challenges_total",
            "Sessions that required a one-time-code step-up, counted once each.",
            ("scenario",),
        )
        self.otp_attempts = reg.counter(
            "callback_otp_attempts_total",
            "One-time-code submissions, by outcome. The code itself is never "
            "recorded anywhere.",
            ("outcome",),
        )
        self.fail_closed = reg.counter(
            "callback_fail_closed_total",
            "Events where the detector refused rather than answered, by cause. "
            "Every non-2xx response is one of these, because the reference "
            "client turns any non-2xx into DENY.",
            ("cause",),
        )
        self.risk_score = reg.histogram(
            "callback_risk_score",
            "Blended risk score at the moment a decision was returned.",
            ("decision",),
            DEFAULT_SCORE_BUCKETS,
        )
        self.confidence = reg.histogram(
            "callback_decision_confidence",
            "Detector confidence at the moment a decision was returned.",
            ("decision",),
            DEFAULT_SCORE_BUCKETS,
        )
        self.factor_verdicts = reg.counter(
            "callback_factor_verdicts_total",
            "Per-factor verdicts observed in returned decisions.",
            ("factor", "verdict"),
        )
        self.active_sessions = reg.gauge(
            "callback_active_sessions",
            "Sessions currently held in the session store.",
        )
        self.journal_sessions = reg.gauge(
            "callback_journal_sessions",
            "Sessions currently held in the review journal.",
        )
        self.evidence_bundles = reg.counter(
            "callback_evidence_bundles_total",
            "Evidence bundles rendered for a human reviewer, by format.",
            ("format",),
        )

    def render(self) -> str:
        return self.registry.render()


# ----------------------------------------------------------------- the journal


@dataclass(frozen=True)
class OtpAttemptRecord:
    """One code submission. Never the code."""

    attempt: int
    at: datetime
    outcome: str
    detail: str | None = None


@dataclass(frozen=True)
class JournalEntry:
    """One request against one session, as the reviewer will read it."""

    at: datetime
    request_id: str
    endpoint: str
    method: str
    status: int
    outcome: str
    latency_ms: int
    call_event: str | None = None
    sequence: int | None = None
    occurred_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None
    decision: str | None = None
    state: str | None = None
    risk_score: float | None = None
    confidence: float | None = None
    next_action: str | None = None
    reason: str | None = None
    otp_submitted: bool = False
    otp_outcome: str | None = None
    idempotent_replay: bool = False
    fail_closed_cause: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    factor_verdicts: dict[str, str] = field(default_factory=dict)
    factor_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class MediaAttachment:
    """Where the call recording and transcript live, once the agent has them."""

    recording_uri: str | None = None
    recording_duration_ms: int | None = None
    transcript_uri: str | None = None
    transcript_text: str | None = None
    note: str | None = None
    attached_at: datetime | None = None

    @property
    def attached(self) -> bool:
        return any(
            (
                self.recording_uri,
                self.transcript_uri,
                self.transcript_text,
                self.note,
            )
        )


@dataclass
class SessionRecord:
    """The journal's view of one call."""

    session_id: str
    first_seen: datetime
    last_seen: datetime
    scenario: str | None = None
    claimed_user_id: str | None = None
    decision: str | None = None
    state: str | None = None
    risk_score: float | None = None
    confidence: float | None = None
    reason: str | None = None
    requests: int = 0
    rejected_requests: int = 0
    otp_attempt_count: int = 0
    entries: deque[JournalEntry] = field(default_factory=deque)
    otp_attempts: list[OtpAttemptRecord] = field(default_factory=list)
    media: MediaAttachment = field(default_factory=MediaAttachment)
    #: Last seen per-factor readings. Kept on the record as well as on each
    #: entry so that an evidence bundle for a session the store has already
    #: evicted still shows the numbers behind the verdict.
    factor_verdicts: dict[str, str] = field(default_factory=dict)
    factor_scores: dict[str, float] = field(default_factory=dict)

    def timeline(self) -> list[JournalEntry]:
        return list(self.entries)


class SessionJournal:
    """Bounded, append-only record of what happened on each call.

    Bounded twice over, because this is memory a token holder can grow: at most
    `max_sessions` calls are remembered, evicted least-recently-touched first,
    and at most `max_entries` requests per call, dropping the oldest. A
    truncated timeline says so rather than looking complete.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_sessions: int = 256,
        max_entries: int = 200,
    ) -> None:
        self.enabled = enabled
        self.max_sessions = max(1, max_sessions)
        self.max_entries = max(1, max_entries)
        self._records: OrderedDict[str, SessionRecord] = OrderedDict()
        self._lock = threading.Lock()
        self.dropped_sessions = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self.dropped_sessions = 0

    # --------------------------------------------------------------- writing

    def _touch(self, session_id: str, now: datetime) -> SessionRecord:
        """Caller holds the lock."""

        record = self._records.get(session_id)
        if record is None:
            record = SessionRecord(
                session_id=session_id,
                first_seen=now,
                last_seen=now,
                entries=deque(maxlen=self.max_entries),
            )
            self._records[session_id] = record
            while len(self._records) > self.max_sessions:
                self._records.popitem(last=False)
                self.dropped_sessions += 1
        self._records.move_to_end(session_id)
        return record

    def record(
        self,
        session_id: str,
        entry: JournalEntry,
        *,
        scenario: str | None = None,
        claimed_user_id: str | None = None,
        otp_attempt: OtpAttemptRecord | None = None,
    ) -> SessionRecord | None:
        if not self.enabled or not session_id:
            return None
        with self._lock:
            record = self._touch(session_id, entry.at)
            record.last_seen = entry.at
            record.requests += 1
            if entry.outcome != "accepted":
                record.rejected_requests += 1
            if scenario:
                record.scenario = scenario
            if claimed_user_id:
                record.claimed_user_id = claimed_user_id
            if entry.decision:
                record.decision = entry.decision
            if entry.state:
                record.state = entry.state
            if entry.risk_score is not None:
                record.risk_score = entry.risk_score
            if entry.confidence is not None:
                record.confidence = entry.confidence
            if entry.reason:
                record.reason = entry.reason
            if entry.factor_verdicts:
                record.factor_verdicts = dict(entry.factor_verdicts)
            if entry.factor_scores:
                record.factor_scores = dict(entry.factor_scores)
            if otp_attempt is not None:
                record.otp_attempts.append(otp_attempt)
                record.otp_attempt_count += 1
            record.entries.append(entry)
            return record

    def attach_media(
        self,
        session_id: str,
        *,
        recording_uri: str | None = None,
        recording_duration_ms: int | None = None,
        transcript_uri: str | None = None,
        transcript_text: str | None = None,
        note: str | None = None,
        now: datetime | None = None,
    ) -> SessionRecord:
        """Record where the recording and transcript live.

        The agent calls this after the call ends, which is why the bundle
        carries the hook whether or not anything has arrived yet.
        """

        moment = now or _utcnow()
        with self._lock:
            record = self._touch(session_id, moment)
            media = record.media
            if recording_uri is not None:
                media.recording_uri = recording_uri
            if recording_duration_ms is not None:
                media.recording_duration_ms = recording_duration_ms
            if transcript_uri is not None:
                media.transcript_uri = transcript_uri
            if transcript_text is not None:
                media.transcript_text = transcript_text
            if note is not None:
                media.note = note
            media.attached_at = moment
            return record

    # --------------------------------------------------------------- reading

    def get(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            return self._records.get(session_id)

    def recent(self, limit: int = 50) -> list[SessionRecord]:
        """Most recently touched first."""

        with self._lock:
            records = list(self._records.values())
        records.sort(key=lambda r: r.last_seen, reverse=True)
        return records[: max(0, limit)]

    def known(self, session_id: str) -> bool:
        return self.get(session_id) is not None


# -------------------------------------------------------------- body decoding

#: Request fields the layer is allowed to look at. `otp_code` is absent on
#: purpose: its presence is recorded, its value is never read.
_EVENT_FIELDS = ("session_id", "event", "sequence", "occurred_at", "claimed_user_id")

_ENVELOPE_FIELDS = (
    "session_id",
    "state",
    "decision",
    "risk_score",
    "confidence",
    "next_action",
    "reason",
    "scenario",
    "sequence",
    "latency_ms",
    "idempotent_replay",
)


def _decode_json(body: bytes | None) -> Any:
    if not body:
        return None
    try:
        return json.loads(body)
    except JSON_PARSE_ERRORS:
        return None


def parse_event_request(body: bytes | None, redactor: Redactor) -> dict[str, Any]:
    """Pull an allowlist of fields out of a POST /events body.

    Returns an empty dict for anything unparseable, which is the normal case
    for a request that failed contract validation.
    """

    doc = _decode_json(body)
    if not isinstance(doc, dict):
        return {}
    out: dict[str, Any] = {}
    for key in _EVENT_FIELDS:
        value = doc.get(key)
        if isinstance(value, (str, int, float)):
            # Every one of these is a structured identifier or an enum value,
            # not prose, so only secret literals are removed. Mangling a
            # session id would make the record unfindable.
            out[key] = (
                redactor.scrub_secrets(value) if isinstance(value, str) else value
            )
    out["otp_code_present"] = bool(doc.get("otp_code"))
    raw_metadata = doc.get("metadata")
    if isinstance(raw_metadata, dict) and raw_metadata:
        trimmed = dict(list(raw_metadata.items())[:32])
        out["metadata"] = redactor.redact(trimmed)
    return out


def parse_response_envelope(body: bytes | None) -> dict[str, Any]:
    """Pull the decision fields, or the error code, out of a response body."""

    doc = _decode_json(body)
    if not isinstance(doc, dict):
        return {}
    out: dict[str, Any] = {}
    error = doc.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message")
        if isinstance(code, str):
            out["error_code"] = code
        if isinstance(message, str):
            out["error_message"] = message
        return out
    for key in _ENVELOPE_FIELDS:
        if key in doc:
            out[key] = doc[key]
    factors = doc.get("factors")
    if isinstance(factors, list):
        verdicts: dict[str, str] = {}
        scores: dict[str, float] = {}
        for item in factors:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str):
                continue
            verdict = item.get("verdict")
            if isinstance(verdict, str):
                verdicts[name] = verdict
            score = item.get("score")
            if isinstance(score, (int, float)):
                scores[name] = float(score)
        if verdicts:
            out["factor_verdicts"] = verdicts
        if scores:
            out["factor_scores"] = scores
    otp = doc.get("otp")
    if isinstance(otp, dict):
        out["otp_attempts_used"] = otp.get("attempts_used")
        out["otp_attempts_remaining"] = otp.get("attempts_remaining")
    return out


# ------------------------------------------------------- endpoint normalising

#: Route templates, not raw paths. An unknown path collapses to `/unmatched`
#: so a port scanner cannot mint unbounded label values.
BASE_ENDPOINTS: frozenset[str] = frozenset(
    {"/healthz", "/events", "/factors/{session_id}", "/metrics"}
)

UNMATCHED_ENDPOINT = "/unmatched"

_SESSION_SEGMENT = "/api/sessions/"


def normalise_endpoint(path: str, known: frozenset[str] = BASE_ENDPOINTS) -> str:
    """Collapse one request path to a bounded route template."""

    if not path:
        return UNMATCHED_ENDPOINT
    trimmed = path.rstrip("/") or "/"
    if trimmed in known:
        return trimmed
    if trimmed.startswith("/factors/"):
        return "/factors/{session_id}"
    index = trimmed.find(_SESSION_SEGMENT)
    if index != -1:
        tail = trimmed[index + len(_SESSION_SEGMENT) :]
        if tail:
            parts = tail.split("/", 1)
            suffix = f"/{parts[1]}" if len(parts) > 1 and parts[1] else ""
            candidate = trimmed[:index] + _SESSION_SEGMENT + "{session_id}" + suffix
            if candidate in known:
                return candidate
    return UNMATCHED_ENDPOINT


# -------------------------------------------------------------- fail-closed map

#: Every contract error code maps to a fail-closed cause, because the
#: reference client turns any non-2xx into DENY.
FAIL_CLOSED_CAUSES: Mapping[str, str] = {
    "unauthorized": "unauthorized",
    "session_not_found": "session_not_found",
    "session_terminal": "session_terminal",
    "sequence_conflict": "sequence_conflict",
    "otp_exhausted": "otp_exhausted",
    "otp_expired": "otp_expired",
    "invalid_scenario": "invalid_scenario",
    "validation_error": "validation_error",
}


def fail_closed_cause(
    *,
    status: int,
    error_code: str | None,
    state: str | None,
    decision: str | None,
    call_event: str | None,
) -> str | None:
    """Why this request ended in a refusal, or None if it did not."""

    if status >= 500:
        return "internal_error"
    if status >= 400:
        if error_code and error_code in FAIL_CLOSED_CAUSES:
            return FAIL_CLOSED_CAUSES[error_code]
        return f"http_{status}"
    if state == "expired":
        return "session_ttl_expired"
    if decision == "deny" and call_event == "session_end":
        return "call_ended_before_verification"
    return None


def otp_outcome(
    *,
    status: int,
    error_code: str | None,
    decision: str | None,
    state: str | None,
    reason: str | None,
) -> str:
    """Classify one code submission without ever seeing the code."""

    if error_code == "otp_expired":
        return "expired"
    if error_code == "otp_exhausted":
        return "exhausted"
    if status >= 400:
        return "rejected"
    if state == "allowed":
        return "correct"
    if decision == "challenge":
        return "incorrect"
    if state == "denied":
        lowered = (reason or "").lower()
        if "exhaust" in lowered:
            return "exhausted"
        return "incorrect"
    return "unknown"


# ------------------------------------------------------------------ container


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class ObservabilityConfig:
    """Every switch, in one place. All three layers turn off independently.

    Read from the environment by `from_env()`, which is why none of this needs
    a field on `detector.config.Settings`. The names are listed in the module
    docstring of `detector.review`.
    """

    log_enabled: bool = True
    metrics_enabled: bool = True
    journal_enabled: bool = True
    #: Scraping /metrics needs the bearer token. See `metrics_router` for why
    #: this defaults to on.
    metrics_require_auth: bool = True
    journal_max_sessions: int = 256
    journal_max_entries: int = 200
    request_body_limit: int = 16 * 1024
    response_body_limit: int = 256 * 1024
    service_name: str = "callback-detector"
    dedupe_cache_size: int = 2048

    @classmethod
    def from_env(cls) -> ObservabilityConfig:
        defaults = cls()
        return cls(
            log_enabled=_env_bool("CALLBACK_LOG_ENABLED", defaults.log_enabled),
            metrics_enabled=_env_bool(
                "CALLBACK_METRICS_ENABLED", defaults.metrics_enabled
            ),
            journal_enabled=_env_bool(
                "CALLBACK_REVIEW_JOURNAL_ENABLED", defaults.journal_enabled
            ),
            metrics_require_auth=_env_bool(
                "CALLBACK_METRICS_REQUIRE_AUTH", defaults.metrics_require_auth
            ),
            journal_max_sessions=_env_int(
                "CALLBACK_REVIEW_JOURNAL_MAX_SESSIONS",
                defaults.journal_max_sessions,
            ),
            journal_max_entries=_env_int(
                "CALLBACK_REVIEW_JOURNAL_MAX_ENTRIES",
                defaults.journal_max_entries,
            ),
            service_name=os.environ.get(
                "CALLBACK_SERVICE_NAME", defaults.service_name
            ).strip()
            or defaults.service_name,
        )


class Observability:
    """Logger, metrics and journal, plus the glue that fills them.

    One instance per app. `observe_request` is the single entry point the
    middleware calls once per finished request.
    """

    def __init__(
        self,
        *,
        config: ObservabilityConfig | None = None,
        redactor: Redactor | None = None,
        stream: TextIO | None = None,
        clock: Callable[[], datetime] = _utcnow,
        known_endpoints: Iterable[str] = BASE_ENDPOINTS,
    ) -> None:
        self.config = config or ObservabilityConfig()
        self.redactor = redactor or Redactor()
        self.clock = clock
        self.known_endpoints = frozenset(known_endpoints)
        self.logger = StructuredLogger(
            service=self.config.service_name,
            enabled=self.config.log_enabled,
            redactor=self.redactor,
            stream=stream,
            clock=clock,
        )
        self.metrics = DetectorMetrics()
        self.journal = SessionJournal(
            enabled=self.config.journal_enabled,
            max_sessions=self.config.journal_max_sessions,
            max_entries=self.config.journal_max_entries,
        )
        self._once: OrderedDict[tuple[str, str], bool] = OrderedDict()
        self._once_lock = threading.Lock()

    # ------------------------------------------------------------ properties

    @property
    def active(self) -> bool:
        """Whether anything at all is being recorded."""

        cfg = self.config
        return cfg.log_enabled or cfg.metrics_enabled or cfg.journal_enabled

    def add_known_endpoints(self, endpoints: Iterable[str]) -> None:
        self.known_endpoints = self.known_endpoints | frozenset(endpoints)

    def bind_active_sessions(self, source: Callable[[], float]) -> None:
        self.metrics.active_sessions.set_function(source)

    def bind_journal_size(self) -> None:
        self.metrics.journal_sessions.set_function(lambda: float(len(self.journal)))

    def set_build_info(self, *, contract_version: str, mock_mode: bool) -> None:
        self.metrics.build_info.set(
            1.0,
            contract_version=contract_version,
            mock_mode="true" if mock_mode else "false",
        )

    # ---------------------------------------------------------------- dedupe

    def _first_time(self, session_id: str, flag: str) -> bool:
        """True exactly once per (session, flag). Bounded, so it cannot grow."""

        key = (session_id, flag)
        with self._once_lock:
            if key in self._once:
                return False
            self._once[key] = True
            while len(self._once) > self.config.dedupe_cache_size:
                self._once.popitem(last=False)
            return True

    # --------------------------------------------------------------- observe

    def observe_request(
        self,
        *,
        request_id: str,
        method: str,
        path: str,
        status: int,
        duration_s: float,
        request_body: bytes | None = None,
        response_body: bytes | None = None,
        error: BaseException | None = None,
    ) -> dict[str, Any]:
        """Record one finished request. Returns the log record, redacted."""

        if not self.active:
            return {}

        now = self.clock()
        endpoint = normalise_endpoint(path, self.known_endpoints)
        latency_ms = int(duration_s * 1000)

        request_fields = parse_event_request(request_body, self.redactor)
        envelope = parse_response_envelope(response_body)

        session_id = str(
            envelope.get("session_id") or request_fields.get("session_id") or ""
        )
        call_event = request_fields.get("event")
        call_event = call_event if isinstance(call_event, str) else None
        decision = envelope.get("decision")
        state = envelope.get("state")
        scenario = envelope.get("scenario")
        reason = envelope.get("reason")
        error_code = envelope.get("error_code")
        if error is not None and not error_code:
            error_code = "internal_error"
        accepted = 200 <= status < 300
        outcome = "accepted" if accepted else "rejected"
        cause = fail_closed_cause(
            status=status,
            error_code=error_code if isinstance(error_code, str) else None,
            state=state if isinstance(state, str) else None,
            decision=decision if isinstance(decision, str) else None,
            call_event=call_event,
        )

        submitted_otp = call_event == "otp_submit"
        otp_result: str | None = None
        attempt_record: OtpAttemptRecord | None = None
        if submitted_otp:
            otp_result = otp_outcome(
                status=status,
                error_code=error_code if isinstance(error_code, str) else None,
                decision=decision if isinstance(decision, str) else None,
                state=state if isinstance(state, str) else None,
                reason=reason if isinstance(reason, str) else None,
            )

        risk = _as_float(envelope.get("risk_score"))
        confidence = _as_float(envelope.get("confidence"))
        replay = bool(envelope.get("idempotent_replay"))

        # Scrubbed here, at the boundary, so nothing secret is ever stored in
        # the journal. The evidence bundle in `detector.review` reads these
        # back verbatim and relies on that.
        safe_reason = (
            self.redactor.scrub_text(reason) if isinstance(reason, str) else None
        )
        safe_error_message = (
            self.redactor.scrub_text(envelope["error_message"])
            if isinstance(envelope.get("error_message"), str)
            else None
        )

        if self.config.metrics_enabled:
            self._record_metrics(
                endpoint=endpoint,
                method=method,
                status=status,
                duration_s=duration_s,
                session_id=session_id,
                decision=decision if isinstance(decision, str) else None,
                state=state if isinstance(state, str) else None,
                scenario=scenario if isinstance(scenario, str) else None,
                risk=risk,
                confidence=confidence,
                factor_verdicts=envelope.get("factor_verdicts"),
                replay=replay,
                cause=cause,
                otp_result=otp_result,
            )

        # A request that failed authentication never reached the engine, and
        # its session id is attacker-chosen. Journalling it would hand an
        # unauthenticated caller a way to evict real calls from a bounded
        # journal, so the entry is dropped. The log line and the metric still
        # record the attempt; both are bounded by construction.
        authenticated = status not in {401, 403}

        if self.config.journal_enabled and session_id and authenticated:
            attempts_used = envelope.get("otp_attempts_used")
            if otp_result is not None:
                attempt_record = OtpAttemptRecord(
                    attempt=int(attempts_used) if isinstance(attempts_used, int) else 0,
                    at=now,
                    outcome=otp_result,
                    detail=None if accepted else str(error_code or f"http_{status}"),
                )
            entry = JournalEntry(
                at=now,
                request_id=request_id,
                endpoint=endpoint,
                method=method,
                status=status,
                outcome=outcome,
                latency_ms=latency_ms,
                call_event=call_event,
                sequence=_as_int(
                    envelope.get("sequence", request_fields.get("sequence"))
                ),
                occurred_at=_as_datetime(request_fields.get("occurred_at")),
                error_code=error_code if isinstance(error_code, str) else None,
                error_message=safe_error_message,
                decision=decision if isinstance(decision, str) else None,
                state=state if isinstance(state, str) else None,
                risk_score=risk,
                confidence=confidence,
                next_action=_as_str(envelope.get("next_action")),
                reason=safe_reason,
                otp_submitted=submitted_otp,
                otp_outcome=otp_result,
                idempotent_replay=replay,
                fail_closed_cause=cause,
                metadata=dict(request_fields.get("metadata") or {}),
                factor_verdicts=dict(envelope.get("factor_verdicts") or {}),
                factor_scores=dict(envelope.get("factor_scores") or {}),
            )
            self.journal.record(
                session_id,
                entry,
                scenario=scenario if isinstance(scenario, str) else None,
                claimed_user_id=_as_str(request_fields.get("claimed_user_id")),
                otp_attempt=attempt_record,
            )

        return self.logger.log(
            "http_request",
            level="error" if status >= 500 else "info",
            request_id=request_id,
            method=method,
            path=path,
            endpoint=endpoint,
            status=status,
            latency_ms=latency_ms,
            outcome=outcome,
            session_id=session_id or None,
            call_event=call_event,
            sequence=_as_int(envelope.get("sequence", request_fields.get("sequence"))),
            claimed_user_id=_as_str(request_fields.get("claimed_user_id")),
            decision=decision,
            state=state,
            next_action=envelope.get("next_action"),
            risk_score=risk,
            confidence=confidence,
            scenario=scenario,
            reason=safe_reason,
            factor_verdicts=envelope.get("factor_verdicts"),
            factor_scores=envelope.get("factor_scores"),
            otp_submitted=submitted_otp or None,
            otp_outcome=otp_result,
            otp_attempts_used=envelope.get("otp_attempts_used"),
            idempotent_replay=replay or None,
            error_code=error_code,
            error_message=safe_error_message,
            fail_closed_cause=cause,
            metadata_keys=sorted(request_fields.get("metadata") or {}) or None,
            exception=type(error).__name__ if error is not None else None,
        )

    def _record_metrics(
        self,
        *,
        endpoint: str,
        method: str,
        status: int,
        duration_s: float,
        session_id: str,
        decision: str | None,
        state: str | None,
        scenario: str | None,
        risk: float | None,
        confidence: float | None,
        factor_verdicts: Any,
        replay: bool,
        cause: str | None,
        otp_result: str | None,
    ) -> None:
        metrics = self.metrics
        metrics.requests.inc(endpoint=endpoint, method=method, status=str(status))
        metrics.latency.observe(duration_s, endpoint=endpoint)

        if cause:
            metrics.fail_closed.inc(cause=cause)
        if otp_result:
            metrics.otp_attempts.inc(outcome=otp_result)

        scenario_label = scenario or "unknown"

        if decision and endpoint == "/events" and not replay:
            metrics.decisions.inc(decision=decision, scenario=scenario_label)
            if risk is not None:
                metrics.risk_score.observe(risk, decision=decision)
            if confidence is not None:
                metrics.confidence.observe(confidence, decision=decision)
            if isinstance(factor_verdicts, dict):
                for name, verdict in factor_verdicts.items():
                    metrics.factor_verdicts.inc(factor=str(name), verdict=str(verdict))

        if not session_id:
            return
        if decision == "challenge" and self._first_time(session_id, "challenged"):
            metrics.otp_challenges.inc(scenario=scenario_label)
        if state in {"allowed", "denied", "expired"} and self._first_time(
            session_id, f"terminal:{state}"
        ):
            metrics.sessions_terminal.inc(state=state, scenario=scenario_label)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _as_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


# ---------------------------------------------------------------- middleware


def new_request_id() -> str:
    return uuid.uuid4().hex


class ObservabilityMiddleware:
    """Pure ASGI middleware. One log line, one journal entry, metrics, per request.

    Written against the raw ASGI interface rather than `BaseHTTPMiddleware`
    because the request body has to be observed without being consumed: the
    wrapped `receive` copies each chunk as the application reads it, so the
    application still sees every byte in order.

    Transparent by construction. It adds one response header
    (`X-Callback-Request-Id`) and changes nothing else about the exchange. When
    every switch is off it is a straight pass-through.
    """

    def __init__(
        self,
        app: Any,
        observability: Observability,
        *,
        capture_endpoints: Iterable[str] = ("/events", "/factors/{session_id}"),
    ) -> None:
        self.app = app
        self.obs = observability
        self.capture_endpoints = frozenset(capture_endpoints)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        obs = self.obs
        if scope.get("type") != "http" or not obs.active:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "GET")
        endpoint = normalise_endpoint(path, obs.known_endpoints)
        capture = endpoint in self.capture_endpoints
        request_id = new_request_id()
        scope["callback_request_id"] = request_id

        request_limit = obs.config.request_body_limit
        response_limit = obs.config.response_body_limit
        request_chunks: list[bytes] = []
        response_chunks: list[bytes] = []
        request_size = 0
        response_size = 0
        status = 500

        async def wrapped_receive() -> Any:
            nonlocal request_size
            message = await receive()
            if capture and message.get("type") == "http.request":
                chunk = message.get("body") or b""
                if chunk and request_size < request_limit:
                    room = request_limit - request_size
                    request_chunks.append(chunk[:room])
                    request_size += len(chunk)
            return message

        header_bytes = (REQUEST_ID_HEADER.lower().encode(), request_id.encode())

        async def wrapped_send(message: Any) -> None:
            nonlocal status, response_size
            kind = message.get("type")
            if kind == "http.response.start":
                status = int(message.get("status", 500))
                headers = list(message.get("headers") or [])
                headers.append(header_bytes)
                message = {**message, "headers": headers}
            elif kind == "http.response.body" and capture:
                chunk = message.get("body") or b""
                if chunk and response_size < response_limit:
                    room = response_limit - response_size
                    response_chunks.append(chunk[:room])
                    response_size += len(chunk)
            await send(message)

        started = time.perf_counter()
        failure: BaseException | None = None
        try:
            await self.app(scope, wrapped_receive, wrapped_send)
        except BaseException as exc:
            failure = exc
            status = 500
            raise
        finally:
            elapsed = time.perf_counter() - started
            obs.observe_request(
                request_id=request_id,
                method=method,
                path=path,
                status=status,
                duration_s=elapsed,
                request_body=b"".join(request_chunks) if request_chunks else None,
                response_body=b"".join(response_chunks) if response_chunks else None,
                error=failure,
            )


# ----------------------------------------------------------------------- auth


class ReviewUnauthorized(Exception):
    """401 for an observability or review route.

    Deliberately not a `DetectorError`: the app already owns a handler for
    those, and this one needs to add `WWW-Authenticate` so a browser can be
    prompted for the token instead of the token travelling in a query string.
    """

    def __init__(self, message: str, *, challenge: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.challenge = challenge


class TokenAuth:
    """Bearer or Basic, both carrying the same service token.

    Bearer is what the agent and Prometheus send. Basic exists for exactly one
    reason: a human opening the dashboard in a browser. A plain navigation
    cannot set an Authorization header, and putting the token in a query string
    writes it into every access log, proxy and Referer header. With Basic the
    browser prompts, the token travels in the header, and the page's own
    `fetch` calls inherit the credentials for free.
    """

    def __init__(self, token: str, *, realm: str = "Callback review") -> None:
        self.token = token
        self.realm = realm

    def _matches(self, presented: str) -> bool:
        return secrets.compare_digest(
            presented.encode("utf-8"), self.token.encode("utf-8")
        )

    def check(self, authorization: str | None, *, challenge: bool = False) -> None:
        """Raise `ReviewUnauthorized` unless the header carries the token."""

        if not authorization:
            raise ReviewUnauthorized("missing credentials", challenge=challenge)
        scheme, _, rest = authorization.partition(" ")
        value = rest.strip()
        lowered = scheme.lower()

        if lowered == "bearer":
            if value and self._matches(value):
                return
            raise ReviewUnauthorized("invalid bearer token", challenge=challenge)

        if lowered == "basic":
            try:
                decoded = base64.b64decode(value, validate=True).decode(
                    "utf-8", "replace"
                )
            except (binascii.Error, ValueError) as exc:
                raise ReviewUnauthorized(
                    "malformed basic credentials", challenge=challenge
                ) from exc
            user, _, password = decoded.partition(":")
            # Either field may carry the token, so `-u token:` and
            # `-u review:token` both work from a terminal.
            if (password and self._matches(password)) or (user and self._matches(user)):
                return
            raise ReviewUnauthorized("invalid credentials", challenge=challenge)

        raise ReviewUnauthorized(
            "unsupported authorization scheme", challenge=challenge
        )

    # ------------------------------------------------------- fastapi plumbing

    def browser_dependency(self) -> Callable[..., None]:
        """Guard that prompts a browser for credentials on failure."""

        def guard(authorization: Annotated[str | None, Header()] = None) -> None:
            self.check(authorization, challenge=True)

        return guard

    def machine_dependency(self) -> Callable[..., None]:
        """Guard for a machine caller. No browser prompt."""

        def guard(authorization: Annotated[str | None, Header()] = None) -> None:
            self.check(authorization, challenge=False)

        return guard


# ------------------------------------------------------------- metrics router


def metrics_router(observability: Observability, auth: TokenAuth) -> APIRouter:
    """`GET /metrics` in Prometheus text exposition format.

    Auth decision: required by default. The metric labels carry the decision
    mix and the mock scenario, so an open scrape publishes the deny rate and,
    worse, acts as an oracle: an attacker who has just placed a call can watch
    `callback_decisions_total{decision="deny"}` move and learn that they were
    caught. Prometheus supports `authorization` in a scrape config, so the
    operational cost of requiring the token is one line of YAML.
    `ObservabilityConfig(metrics_require_auth=False)` exists for a private
    network, and has to be asked for.
    """

    router = APIRouter(tags=["observability"])
    dependencies = (
        [Depends(auth.machine_dependency())]
        if observability.config.metrics_require_auth
        else []
    )

    @router.get(
        "/metrics",
        dependencies=dependencies,
        include_in_schema=False,
        response_class=Response,
    )
    async def metrics() -> Response:
        if not observability.config.metrics_enabled:
            return Response(
                content="# metrics are disabled\n",
                media_type=PROMETHEUS_CONTENT_TYPE,
            )
        return Response(
            content=observability.metrics.render(),
            media_type=PROMETHEUS_CONTENT_TYPE,
        )

    return router
