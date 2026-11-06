"""Logging, redaction and metrics.

These tests are always in process, even when CALLBACK_BASE_URL is set: they
assert on the log stream and the metric registry of the app under test, which
a remote deployment cannot hand back.

The Prometheus exposition is checked against a parser written here from the
format specification rather than against a golden string, so a formatting
change that a real scraper would reject fails these tests even if it looks
fine to a human.
"""

from __future__ import annotations

import io
import json
import math
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from detector.app import create_app
from detector.config import Settings
from detector.observability import (
    BASE_ENDPOINTS,
    PROMETHEUS_CONTENT_TYPE,
    REQUEST_ID_HEADER,
    Counter,
    DetectorMetrics,
    Gauge,
    Histogram,
    JournalEntry,
    MetricsRegistry,
    Observability,
    ObservabilityConfig,
    Redactor,
    SessionJournal,
    StructuredLogger,
    TokenAuth,
    fail_closed_cause,
    format_number,
    normalise_endpoint,
    normalise_key,
    otp_outcome,
)
from shared.contract import (
    MOCK_OTP_CODE,
    Decision,
    EventType,
    Scenario,
    SessionState,
)
from tests.contract.conftest import TEST_TOKEN
from tests.contract.journeys import EXPECTATIONS, Journey, new_session_id

WRONG_OTP = "000000"

AUTH = {"Authorization": f"Bearer {TEST_TOKEN}"}


# --------------------------------------------------------------------- wiring


@pytest.fixture
def log_stream() -> io.StringIO:
    return io.StringIO()


@pytest.fixture
def obs_config() -> ObservabilityConfig:
    """Explicit, never from the environment, so a stray env var cannot flip a test."""

    return ObservabilityConfig(
        log_enabled=True,
        metrics_enabled=True,
        journal_enabled=True,
        metrics_require_auth=True,
    )


@pytest.fixture
def mounted(
    settings: Settings, obs_config: ObservabilityConfig, log_stream: io.StringIO
) -> tuple[object, Observability]:
    app = create_app(settings, observability=obs_config, log_stream=log_stream)
    # Drop the mount banner so a test reads only request lines.
    log_stream.seek(0)
    log_stream.truncate(0)
    return app, app.state.observability


@pytest.fixture
def observability(mounted: tuple[object, Observability]) -> Observability:
    return mounted[1]


@pytest.fixture
async def client(
    mounted: tuple[object, Observability],
) -> AsyncIterator[httpx.AsyncClient]:
    app, _ = mounted
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://observability.test", headers=AUTH
    ) as local:
        yield local


def log_records(stream: io.StringIO) -> list[dict]:
    """Every line written so far, parsed. Fails loudly on a non-JSON line."""

    out: list[dict] = []
    for index, line in enumerate(stream.getvalue().splitlines()):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:  # pragma: no cover - diagnostic
            pytest.fail(f"log line {index} is not valid JSON: {line!r} ({exc})")
    return out


def request_records(stream: io.StringIO) -> list[dict]:
    return [r for r in log_records(stream) if r.get("event") == "http_request"]


# ------------------------------------------------------------ the redactor


def test_normalise_key_collapses_separators_and_case() -> None:
    assert normalise_key("Bank-Account_Number") == "bankaccountnumber"
    assert normalise_key("bankAccountNumber") == "bankaccountnumber"
    assert normalise_key("  IBAN  ") == "iban"
    assert normalise_key("___") == ""


def test_redactor_removes_known_secret_literals() -> None:
    redactor = Redactor([TEST_TOKEN, MOCK_OTP_CODE])

    assert TEST_TOKEN not in redactor.scrub_text(f"Bearer {TEST_TOKEN} was used")
    assert MOCK_OTP_CODE not in redactor.scrub_text(f"caller said {MOCK_OTP_CODE}")
    assert "[REDACTED]" in redactor.scrub_text(TEST_TOKEN)


def test_redactor_ignores_secrets_too_short_to_scrub_safely() -> None:
    """A three-character 'secret' would shred ordinary prose."""

    redactor = Redactor(["abc"])
    assert redactor.secret_count == 0
    assert redactor.scrub_text("abcdef") == "abcdef"


def test_redactor_redacts_by_key_name_however_it_is_spelled() -> None:
    redactor = Redactor()
    payload = {
        "otp_code": "424242",
        "OTP-Code": "424242",
        "otpCode": "424242",
        "authorization": "Bearer hunter2hunter2",
        "api_token": "live-token-value",
        "bank_account": "12345678",
        "Account Number": "87654321",
        "IBAN": "GB29NWBK60161331926819",
        "routing_number": "026009593",
        "sort_code": "12-34-56",
        "password": "correct horse",
        "beneficiary_name": "Acme Supplies",
        "code": "424242",
        "pin": "1234",
    }
    out = redactor.redact(payload)
    for key, value in out.items():
        assert value == "[REDACTED]", f"{key} was not redacted: {value!r}"


def test_redactor_distinguishes_the_code_from_facts_about_the_code() -> None:
    """The allowlist has to beat the needle, or the record loses its meaning.

    `otp_submitted` and `otp_attempts_used` say nothing secret. `otp_code` and
    a bare `otp` could both be the code itself.
    """

    redactor = Redactor()
    out = redactor.redact(
        {
            "otp_submitted": True,
            "otp_outcome": "incorrect",
            "otp_attempts_used": 2,
            "otp_attempts_remaining": 1,
            "code_length": 6,
            "code_withheld": True,
            "otp_code": MOCK_OTP_CODE,
            "otp": MOCK_OTP_CODE,
        }
    )
    assert out["otp_submitted"] is True
    assert out["otp_outcome"] == "incorrect"
    assert out["otp_attempts_used"] == 2
    assert out["otp_attempts_remaining"] == 1
    assert out["code_length"] == 6
    assert out["code_withheld"] is True
    assert out["otp_code"] == "[REDACTED]"
    assert out["otp"] == "[REDACTED]"


def test_redactor_keeps_the_fields_an_operator_needs() -> None:
    """A rule that eats error_code makes the log useless. These must survive."""

    redactor = Redactor()
    out = redactor.redact(
        {
            "error_code": "session_terminal",
            "scenario": "cloned_voice",
            "country_code": "GB",
            "status": 409,
            "decision": "deny",
            "next_action": "complete_deny",
            "latency_ms": 12,
            "risk_score": 0.42,
        }
    )
    assert out["error_code"] == "session_terminal"
    assert out["scenario"] == "cloned_voice"
    assert out["country_code"] == "GB"
    assert out["status"] == 409
    assert out["decision"] == "deny"
    assert out["next_action"] == "complete_deny"
    assert out["latency_ms"] == 12
    assert out["risk_score"] == 0.42


def test_redactor_catches_a_secret_in_a_field_nobody_expected() -> None:
    """The point of the value layer.

    None of these field names looks sensitive. The secret still must not come
    out the other side, at any nesting depth, in a key or in a value.
    """

    redactor = Redactor([TEST_TOKEN])
    payload = {
        "notes": f"operator pasted {TEST_TOKEN} into the ticket",
        "nested": {
            "harmless": {
                "deeper": [
                    "nothing here",
                    {"surprise": f"prefix-{TEST_TOKEN}-suffix"},
                ]
            }
        },
        "tuple_like": ["a", TEST_TOKEN],
        TEST_TOKEN: "the secret arrived as a key name",
        "stringified": {"obj": object()},
    }
    rendered = json.dumps(redactor.redact(payload), default=str)
    assert TEST_TOKEN not in rendered
    assert redactor.contains_secret(rendered) is False
    assert "[REDACTED]" in rendered


def test_redactor_masks_bank_shaped_values_in_free_text() -> None:
    redactor = Redactor()
    out = redactor.redact(
        {
            "free_text": "send it to GB29NWBK60161331926819 please",
            "other": "account 123456789012 at the new bank",
        }
    )
    assert "GB29NWBK60161331926819" not in out["free_text"]
    assert "123456789012" not in out["other"]
    assert "[REDACTED:BANK_DETAIL]" in out["free_text"]
    assert "[REDACTED:BANK_DETAIL]" in out["other"]


def test_redactor_leaves_identifiers_and_timestamps_intact() -> None:
    """Identifiers are allowed to look like long numbers.

    A session id that happens to contain nine consecutive digits must come back
    unchanged, because a mangled id is a record nobody can look up.
    """

    redactor = Redactor([TEST_TOKEN])
    numeric_id = "sess-1234567890abcdef"
    out = redactor.redact(
        {
            "session_id": numeric_id,
            "request_id": "9876543210abcdef",
            "claimed_user_id": "vendor-123456789",
            "ts": "2026-11-04T22:15:00.123456+00:00",
            "occurred_at": "2026-11-04T22:15:00+00:00",
            "latency_ms": 123456789,
        }
    )
    assert out["session_id"] == numeric_id
    assert out["request_id"] == "9876543210abcdef"
    assert out["claimed_user_id"] == "vendor-123456789"
    assert out["ts"] == "2026-11-04T22:15:00.123456+00:00"
    assert out["latency_ms"] == 123456789


def test_redactor_caps_recursion_depth() -> None:
    redactor = Redactor()
    deep: dict = {"v": "leaf"}
    for _ in range(40):
        deep = {"v": deep}
    rendered = json.dumps(redactor.redact(deep))
    assert "[REDACTED:DEPTH]" in rendered


def test_redactor_truncates_very_long_strings() -> None:
    redactor = Redactor(max_string=50)
    out = redactor.scrub_text("x" * 500)
    assert out.endswith("...[truncated]")
    assert len(out) < 100


# ------------------------------------------------------------ the log stream


def test_log_line_is_json_with_the_expected_envelope(log_stream: io.StringIO) -> None:
    logger = StructuredLogger(stream=log_stream, service="unit-test")
    logger.log("thing_happened", session_id="sess-abc", count=3, missing=None)

    records = log_records(log_stream)
    assert len(records) == 1
    record = records[0]
    assert record["event"] == "thing_happened"
    assert record["service"] == "unit-test"
    assert record["level"] == "info"
    assert record["session_id"] == "sess-abc"
    assert record["count"] == 3
    # None-valued fields are dropped rather than written as null.
    assert "missing" not in record
    datetime.fromisoformat(record["ts"])


def test_logging_can_be_turned_off(log_stream: io.StringIO) -> None:
    logger = StructuredLogger(stream=log_stream, enabled=False)
    assert logger.log("ignored", a=1) == {}
    assert log_stream.getvalue() == ""


def test_logger_redacts_before_writing(log_stream: io.StringIO) -> None:
    logger = StructuredLogger(
        stream=log_stream, redactor=Redactor([TEST_TOKEN, MOCK_OTP_CODE])
    )
    logger.log(
        "thing",
        otp_code=MOCK_OTP_CODE,
        note=f"token is {TEST_TOKEN}",
        authorization=f"Bearer {TEST_TOKEN}",
    )
    raw = log_stream.getvalue()
    assert TEST_TOKEN not in raw
    assert MOCK_OTP_CODE not in raw


async def test_one_log_line_per_request_with_the_required_fields(
    client: httpx.AsyncClient, log_stream: io.StringIO
) -> None:
    journey = Journey(client, Scenario.CLONED_VOICE)
    await journey.run_to_decision()

    records = request_records(log_stream)
    # session_start, caller_claim, two speech segments.
    assert len(records) == 4

    final = records[-1]
    for required in (
        "ts",
        "request_id",
        "method",
        "path",
        "endpoint",
        "status",
        "latency_ms",
        "outcome",
        "session_id",
        "call_event",
        "decision",
        "state",
        "risk_score",
        "confidence",
        "next_action",
        "factor_verdicts",
    ):
        assert required in final, f"{required} missing from {final}"

    assert final["session_id"] == journey.session_id
    assert final["endpoint"] == "/events"
    assert final["status"] == 200
    assert final["outcome"] == "accepted"
    assert final["decision"] == Decision.CHALLENGE.value
    assert final["state"] == SessionState.CHALLENGING.value
    assert final["call_event"] == EventType.SPEECH_SEGMENT.value
    assert 0.0 <= final["risk_score"] <= 1.0
    assert final["latency_ms"] >= 0
    assert final["factor_verdicts"]["synthetic_speech"] == "fail"
    assert final["factor_scores"]["synthetic_speech"] == pytest.approx(0.08)


async def test_log_never_carries_the_code_the_token_or_a_bank_detail(
    client: httpx.AsyncClient, log_stream: io.StringIO
) -> None:
    """End to end, with the secrets pushed through unexpected fields.

    The correct code goes in the contract field, the bearer token goes in a
    metadata value *and* in a metadata key name, and a bank detail goes in free
    text. None of it may reach the log.
    """

    journey = Journey(client, Scenario.CLONED_VOICE)
    await journey.run_to_decision()
    await journey.send(EventType.OTP_REQUEST)

    body = {
        "session_id": journey.session_id,
        "event": EventType.OTP_SUBMIT.value,
        "sequence": journey.sequence + 1,
        "occurred_at": datetime.now(UTC).isoformat(),
        "otp_code": MOCK_OTP_CODE,
        "metadata": {
            "operator_note": f"token {TEST_TOKEN} pasted here by mistake",
            TEST_TOKEN: "the token arrived as a key name",
            "new_bank_account": "12345678901234",
            "free_text": "pay into GB29NWBK60161331926819",
        },
    }
    response = await client.post(
        "/events", json=body, headers={"X-Callback-Scenario": Scenario.CLONED_VOICE}
    )
    assert response.status_code == 200, response.text

    raw = log_stream.getvalue()
    assert raw, "nothing was logged"
    assert MOCK_OTP_CODE not in raw
    assert TEST_TOKEN not in raw
    assert "GB29NWBK60161331926819" not in raw
    assert "12345678901234" not in raw

    final = request_records(log_stream)[-1]
    assert final["call_event"] == EventType.OTP_SUBMIT.value
    assert final["otp_submitted"] is True
    assert final["otp_outcome"] == "correct"
    # The key names are reported, redacted, so an operator can see what arrived.
    assert "operator_note" in final["metadata_keys"]


async def test_failed_requests_are_logged_with_a_cause(
    client: httpx.AsyncClient, log_stream: io.StringIO
) -> None:
    journey = Journey(client, Scenario.REAL_CFO)
    await journey.send(EventType.SESSION_START)
    # Re-using sequence 1 is a sequence conflict.
    conflict = await journey.send(EventType.CALLER_CLAIM, sequence=1)
    assert conflict.status_code == 409

    final = request_records(log_stream)[-1]
    assert final["status"] == 409
    assert final["outcome"] == "rejected"
    assert final["error_code"] == "sequence_conflict"
    assert final["fail_closed_cause"] == "sequence_conflict"
    assert final["level"] == "info"


async def test_request_id_is_returned_so_a_human_can_correlate(
    client: httpx.AsyncClient, log_stream: io.StringIO
) -> None:
    response = await client.get("/healthz")
    assert response.status_code == 200
    header_value = response.headers[REQUEST_ID_HEADER]
    assert header_value
    assert request_records(log_stream)[-1]["request_id"] == header_value


# -------------------------------------------------------- exposition parser


@dataclass
class Family:
    name: str
    help_text: str = ""
    kind: str = "untyped"
    samples: list[tuple[str, dict[str, str], float]] = field(default_factory=list)

    def value(self, **labels: str) -> float:
        for _, sample_labels, value in self.samples:
            if sample_labels == labels:
                return value
        raise AssertionError(
            f"{self.name} has no sample with labels {labels}; "
            f"have {[s[1] for s in self.samples]}"
        )

    def suffix(self, suffix: str) -> list[tuple[str, dict[str, str], float]]:
        return [s for s in self.samples if s[0] == f"{self.name}{suffix}"]


def _unescape(raw: str) -> str:
    out: list[str] = []
    index = 0
    while index < len(raw):
        char = raw[index]
        if char == "\\" and index + 1 < len(raw):
            nxt = raw[index + 1]
            out.append({"n": "\n", "\\": "\\", '"': '"'}.get(nxt, nxt))
            index += 2
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _split_label_set(rest: str) -> tuple[str, str]:
    """Everything up to the closing brace, honouring quotes and escapes."""

    collected: list[str] = []
    in_quote = False
    escaped = False
    for index, char in enumerate(rest):
        if escaped:
            escaped = False
            collected.append(char)
            continue
        if char == "\\" and in_quote:
            escaped = True
            collected.append(char)
            continue
        if char == '"':
            in_quote = not in_quote
            collected.append(char)
            continue
        if char == "}" and not in_quote:
            return "".join(collected), rest[index + 1 :]
        collected.append(char)
    raise AssertionError(f"unterminated label set in {rest!r}")


def _split_commas(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    in_quote = False
    escaped = False
    for char in text:
        if escaped:
            escaped = False
            current.append(char)
            continue
        if char == "\\" and in_quote:
            escaped = True
            current.append(char)
            continue
        if char == '"':
            in_quote = not in_quote
            current.append(char)
            continue
        if char == "," and not in_quote:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    parts.append("".join(current))
    return [p for p in parts if p.strip()]


def _parse_labels(text: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for part in _split_commas(text):
        name, sep, raw = part.partition("=")
        assert sep, f"label {part!r} has no '='"
        raw = raw.strip()
        assert raw.startswith('"') and raw.endswith('"'), f"unquoted label {part!r}"
        labels[name.strip()] = _unescape(raw[1:-1])
    return labels


def _parse_value(text: str) -> float:
    cleaned = text.strip()
    if cleaned == "+Inf":
        return math.inf
    if cleaned == "-Inf":
        return -math.inf
    if cleaned == "NaN":
        return math.nan
    return float(cleaned)


def parse_exposition(text: str) -> dict[str, Family]:
    """Parse Prometheus text exposition the way a scraper would.

    Deliberately strict: a sample whose family was never declared with HELP and
    TYPE is an error, because that is what a real scraper rejects.
    """

    assert text.endswith("\n"), "exposition must end with a newline"
    families: dict[str, Family] = {}

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        if not line:
            continue
        if line.startswith("# HELP "):
            name, _, help_text = line[len("# HELP ") :].partition(" ")
            families.setdefault(name, Family(name)).help_text = help_text
            continue
        if line.startswith("# TYPE "):
            name, _, kind = line[len("# TYPE ") :].partition(" ")
            families.setdefault(name, Family(name)).kind = kind.strip()
            continue
        if line.startswith("#"):
            continue

        if "{" in line:
            sample_name, _, rest = line.partition("{")
            label_text, value_text = _split_label_set(rest)
            labels = _parse_labels(label_text)
        else:
            sample_name, _, value_text = line.partition(" ")
            labels = {}
        value = _parse_value(value_text)

        owner = None
        if sample_name in families:
            owner = sample_name
        else:
            for suffix in ("_bucket", "_sum", "_count"):
                if sample_name.endswith(suffix):
                    candidate = sample_name[: -len(suffix)]
                    if candidate in families:
                        owner = candidate
                        break
        assert owner is not None, f"sample {sample_name!r} has no declared family"
        families[owner].samples.append((sample_name, labels, value))

    return families


def assert_exposition_is_well_formed(text: str) -> dict[str, Family]:
    """Every rule of the format this service has to satisfy."""

    families = parse_exposition(text)
    assert families, "nothing was exposed"

    for family in families.values():
        assert family.help_text, f"{family.name} has no HELP"
        assert family.kind in {"counter", "gauge", "histogram"}, (
            f"{family.name} has type {family.kind!r}"
        )

        if family.kind == "counter":
            assert family.name.endswith("_total"), (
                f"counter {family.name} does not end in _total"
            )
            for sample_name, _, value in family.samples:
                assert sample_name == family.name
                assert value >= 0

        if family.kind == "gauge":
            for sample_name, _, _ in family.samples:
                assert sample_name == family.name

        if family.kind == "histogram":
            buckets = family.suffix("_bucket")
            sums = family.suffix("_sum")
            counts = family.suffix("_count")
            if not buckets:
                assert not sums and not counts
                continue
            assert sums, f"{family.name} has buckets but no _sum"
            assert counts, f"{family.name} has buckets but no _count"

            grouped: dict[tuple, list[tuple[float, float]]] = {}
            for _, labels, value in buckets:
                assert "le" in labels, f"{family.name}_bucket without an le label"
                key = tuple(sorted((k, v) for k, v in labels.items() if k != "le"))
                grouped.setdefault(key, []).append((_parse_value(labels["le"]), value))

            for key, pairs in grouped.items():
                pairs.sort(key=lambda pair: pair[0])
                assert pairs[-1][0] == math.inf, (
                    f"{family.name} is missing its +Inf bucket for {key}"
                )
                previous = -1.0
                for bound, count in pairs:
                    assert count >= previous, (
                        f"{family.name} buckets are not cumulative at le={bound}"
                    )
                    previous = count

                labels = dict(key)
                total = next(
                    value
                    for _, sample_labels, value in counts
                    if sample_labels == labels
                )
                assert pairs[-1][1] == total, (
                    f"{family.name} +Inf bucket {pairs[-1][1]} != _count {total}"
                )

    return families


# -------------------------------------------------------------- metric units


def test_format_number_matches_the_exposition_format() -> None:
    assert format_number(0) == "0"
    assert format_number(1.0) == "1"
    assert format_number(12) == "12"
    assert format_number(0.005) == "0.005"
    assert format_number(math.inf) == "+Inf"
    assert format_number(-math.inf) == "-Inf"
    assert format_number(math.nan) == "NaN"


def test_counter_names_must_end_in_total() -> None:
    with pytest.raises(ValueError, match="_total"):
        Counter("callback_things", "no suffix")


def test_counter_refuses_to_decrease() -> None:
    counter = Counter("callback_things_total", "things")
    with pytest.raises(ValueError, match="cannot decrease"):
        counter.inc(-1)


def test_metric_rejects_wrong_labels() -> None:
    counter = Counter("callback_things_total", "things", ("kind",))
    with pytest.raises(ValueError, match="missing"):
        counter.inc()
    with pytest.raises(ValueError, match="no labels"):
        counter.inc(kind="a", other="b")


def test_histogram_refuses_the_reserved_le_label() -> None:
    with pytest.raises(ValueError, match="reserved"):
        Histogram("callback_thing_seconds", "nope", ("le",))


def test_registry_refuses_a_duplicate_name() -> None:
    registry = MetricsRegistry()
    registry.counter("callback_a_total", "a")
    with pytest.raises(ValueError, match="already registered"):
        registry.counter("callback_a_total", "a")


def test_exposition_round_trips_through_a_parser() -> None:
    registry = MetricsRegistry()
    counter = registry.counter("callback_demo_total", "A demo counter.", ("kind",))
    gauge = registry.gauge("callback_demo_gauge", "A demo gauge.")
    hist = registry.histogram(
        "callback_demo_seconds", "A demo histogram.", ("route",), (0.1, 1.0)
    )

    counter.inc(kind="alpha")
    counter.inc(2, kind="beta")
    gauge.set(7.5)
    for observation in (0.05, 0.5, 5.0):
        hist.observe(observation, route="/events")

    families = assert_exposition_is_well_formed(registry.render())

    assert families["callback_demo_total"].kind == "counter"
    assert families["callback_demo_total"].help_text == "A demo counter."
    assert families["callback_demo_total"].value(kind="alpha") == 1
    assert families["callback_demo_total"].value(kind="beta") == 2

    assert families["callback_demo_gauge"].value() == 7.5

    demo = families["callback_demo_seconds"]
    assert demo.kind == "histogram"
    buckets = {labels["le"]: value for _, labels, value in demo.suffix("_bucket")}
    assert buckets["0.1"] == 1
    assert buckets["1"] == 2
    assert buckets["+Inf"] == 3
    assert demo.suffix("_count")[0][2] == 3
    assert demo.suffix("_sum")[0][2] == pytest.approx(5.55)


def test_label_values_are_escaped_and_survive_a_round_trip() -> None:
    registry = MetricsRegistry()
    counter = registry.counter("callback_escape_total", "Escaping.", ("cause",))
    nasty = 'back\\slash and "quote" and\nnewline'
    counter.inc(cause=nasty)

    text = registry.render()
    assert "\\\\slash" in text
    assert '\\"quote\\"' in text
    assert "\\n" in text
    # One sample on one line, whatever the label value contained.
    sample_lines = [
        line for line in text.splitlines() if line.startswith("callback_escape_total")
    ]
    assert len(sample_lines) == 1

    families = parse_exposition(text)
    assert families["callback_escape_total"].value(cause=nasty) == 1


def test_help_text_is_escaped() -> None:
    registry = MetricsRegistry()
    registry.counter("callback_help_total", "line one\nline two with a \\ backslash")
    text = registry.render()
    help_lines = [line for line in text.splitlines() if line.startswith("# HELP")]
    assert len(help_lines) == 1
    assert "\\n" in help_lines[0]
    assert "\\\\" in help_lines[0]


def test_gauge_reads_a_callable_at_scrape_time() -> None:
    registry = MetricsRegistry()
    gauge = registry.gauge("callback_live_gauge", "Live.")
    source = {"n": 3}
    gauge.set_function(lambda: float(source["n"]))
    assert parse_exposition(registry.render())["callback_live_gauge"].value() == 3
    source["n"] = 9
    assert parse_exposition(registry.render())["callback_live_gauge"].value() == 9


def test_gauge_function_failure_does_not_break_the_scrape() -> None:
    registry = MetricsRegistry()
    gauge = registry.gauge("callback_broken_gauge", "Broken.")

    def explode() -> float:
        raise ValueError("probe is broken")

    gauge.set_function(explode)
    text = registry.render()
    assert "# TYPE callback_broken_gauge gauge" in text
    assert_exposition_is_well_formed(text)


def test_labelled_gauge_refuses_a_function() -> None:
    gauge = Gauge("callback_g", "g", ("kind",))
    with pytest.raises(ValueError, match="unlabelled"):
        gauge.set_function(lambda: 1.0)


def test_the_whole_detector_metric_set_is_well_formed_when_empty() -> None:
    """An untouched registry still has to parse. Empty families are legal."""

    assert_exposition_is_well_formed(DetectorMetrics().render())


# ------------------------------------------------------- the metrics endpoint


async def test_metrics_requires_the_bearer_token(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/metrics", headers={"Authorization": ""})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"

    wrong = await client.get("/metrics", headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 401

    # A machine endpoint is not a browser prompt.
    assert "www-authenticate" not in {k.lower() for k in wrong.headers}


async def test_metrics_endpoint_serves_parseable_exposition(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.headers["content-type"] == PROMETHEUS_CONTENT_TYPE

    families = assert_exposition_is_well_formed(response.text)
    assert "callback_decisions_total" in families
    assert "callback_http_request_duration_seconds" in families
    assert "callback_active_sessions" in families
    assert (
        families["callback_detector_build_info"].value(
            contract_version="0.1.0-provisional", mock_mode="true"
        )
        == 1
    )


async def test_metrics_can_be_opened_deliberately(
    settings: Settings, log_stream: io.StringIO
) -> None:
    """The open door exists, but has to be asked for by name."""

    app = create_app(
        settings,
        observability=ObservabilityConfig(metrics_require_auth=False),
        log_stream=log_stream,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://open.test"
    ) as anonymous:
        response = await anonymous.get("/metrics")
    assert response.status_code == 200
    assert_exposition_is_well_formed(response.text)


async def test_metrics_endpoint_reports_when_metrics_are_off(
    settings: Settings, log_stream: io.StringIO
) -> None:
    app = create_app(
        settings,
        observability=ObservabilityConfig(metrics_enabled=False),
        log_stream=log_stream,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://off.test", headers=AUTH
    ) as local:
        response = await local.get("/metrics")
    assert response.status_code == 200
    assert "disabled" in response.text


# --------------------------------------------------------- counters in anger


@pytest.mark.parametrize("expected", EXPECTATIONS, ids=lambda e: e.scenario.value)
async def test_decision_counter_moves_when_a_decision_happens(
    client: httpx.AsyncClient, observability: Observability, expected
) -> None:
    decisions = observability.metrics.decisions
    before = decisions.value(
        decision=expected.decision_after_speech.value, scenario=expected.scenario.value
    )

    journey = Journey(client, expected.scenario)
    await journey.run_to_decision()

    after = decisions.value(
        decision=expected.decision_after_speech.value, scenario=expected.scenario.value
    )
    assert after == before + 1

    families = assert_exposition_is_well_formed((await client.get("/metrics")).text)
    assert (
        families["callback_decisions_total"].value(
            decision=expected.decision_after_speech.value,
            scenario=expected.scenario.value,
        )
        == after
    )


async def test_terminal_state_is_counted_once_per_session(
    client: httpx.AsyncClient, observability: Observability
) -> None:
    terminal = observability.metrics.sessions_terminal
    journey = Journey(client, Scenario.REPLAY)
    await journey.run_to_decision()

    # Another event on a denied session is refused, and must not re-count it.
    refused = await journey.send(EventType.SPEECH_SEGMENT)
    assert refused.status_code == 409

    assert (
        terminal.value(state=SessionState.DENIED.value, scenario=Scenario.REPLAY.value)
        == 1
    )


async def test_otp_attempt_outcomes_are_counted_without_the_code(
    client: httpx.AsyncClient, observability: Observability
) -> None:
    attempts = observability.metrics.otp_attempts
    challenges = observability.metrics.otp_challenges

    good = Journey(client, Scenario.CLONED_VOICE)
    await good.run_to_decision()
    await good.send(EventType.OTP_REQUEST)
    allowed = await good.send(EventType.OTP_SUBMIT, otp_code=MOCK_OTP_CODE)
    assert allowed.json()["decision"] == Decision.ALLOW.value

    bad = Journey(client, Scenario.HUMAN_IMPOSTOR)
    await bad.run_to_decision()
    await bad.send(EventType.OTP_REQUEST)
    for _ in range(3):
        await bad.send(EventType.OTP_SUBMIT, otp_code=WRONG_OTP)

    assert attempts.value(outcome="correct") == 1
    assert attempts.value(outcome="incorrect") == 2
    assert attempts.value(outcome="exhausted") == 1
    assert challenges.value(scenario=Scenario.CLONED_VOICE.value) == 1
    assert challenges.value(scenario=Scenario.HUMAN_IMPOSTOR.value) == 1


async def test_fail_closed_counter_moves_on_every_refusal(
    client: httpx.AsyncClient, observability: Observability
) -> None:
    fail_closed = observability.metrics.fail_closed

    unauth = await client.post(
        "/events",
        json={
            "session_id": new_session_id(),
            "event": EventType.SESSION_START.value,
            "sequence": 1,
            "occurred_at": datetime.now(UTC).isoformat(),
        },
        headers={"Authorization": "Bearer wrong"},
    )
    assert unauth.status_code == 401

    missing = await client.get(f"/factors/{new_session_id()}")
    assert missing.status_code == 404

    bad_scenario = await client.post(
        "/events",
        json={
            "session_id": new_session_id(),
            "event": EventType.SESSION_START.value,
            "sequence": 1,
            "occurred_at": datetime.now(UTC).isoformat(),
        },
        headers={"X-Callback-Scenario": "not-a-scenario"},
    )
    assert bad_scenario.status_code == 400

    invalid = await client.post("/events", json={"nope": True})
    assert invalid.status_code == 422

    assert fail_closed.value(cause="unauthorized") == 1
    assert fail_closed.value(cause="session_not_found") == 1
    assert fail_closed.value(cause="invalid_scenario") == 1
    assert fail_closed.value(cause="validation_error") == 1


async def test_session_end_before_verification_is_a_fail_closed_cause(
    client: httpx.AsyncClient, observability: Observability
) -> None:
    journey = Journey(client, Scenario.REAL_CFO)
    await journey.send(EventType.SESSION_START)
    ended = await journey.send(EventType.SESSION_END)
    assert ended.json()["decision"] == Decision.DENY.value

    assert (
        observability.metrics.fail_closed.value(cause="call_ended_before_verification")
        == 1
    )


async def test_active_session_gauge_follows_the_store(
    client: httpx.AsyncClient, observability: Observability, mounted
) -> None:
    app, _ = mounted
    assert observability.metrics.active_sessions.value() == 0

    for _ in range(3):
        await Journey(client, Scenario.REAL_CFO).run_to_decision()

    assert observability.metrics.active_sessions.value() == 3
    assert len(app.state.engine.store) == 3

    families = assert_exposition_is_well_formed((await client.get("/metrics")).text)
    assert families["callback_active_sessions"].value() == 3


async def test_latency_histogram_records_every_request(
    client: httpx.AsyncClient, observability: Observability
) -> None:
    await Journey(client, Scenario.REAL_CFO).run_to_decision()
    latency = observability.metrics.latency
    assert latency.count(endpoint="/events") == 4
    assert latency.total(endpoint="/events") >= 0.0


async def test_factor_verdicts_are_counted(
    client: httpx.AsyncClient, observability: Observability
) -> None:
    await Journey(client, Scenario.AI_AGENT).run_to_decision()
    verdicts = observability.metrics.factor_verdicts
    assert verdicts.value(factor="agent_behaviour", verdict="fail") == 1
    assert verdicts.value(factor="replay", verdict="pass") == 1


# -------------------------------------------------- classification functions


def test_fail_closed_cause_classification() -> None:
    assert (
        fail_closed_cause(
            status=401,
            error_code="unauthorized",
            state=None,
            decision=None,
            call_event=None,
        )
        == "unauthorized"
    )
    assert (
        fail_closed_cause(
            status=500, error_code=None, state=None, decision=None, call_event=None
        )
        == "internal_error"
    )
    assert (
        fail_closed_cause(
            status=418, error_code=None, state=None, decision=None, call_event=None
        )
        == "http_418"
    )
    assert (
        fail_closed_cause(
            status=200,
            error_code=None,
            state="expired",
            decision="deny",
            call_event="speech_segment",
        )
        == "session_ttl_expired"
    )
    assert (
        fail_closed_cause(
            status=200,
            error_code=None,
            state="allowed",
            decision="allow",
            call_event="speech_segment",
        )
        is None
    )


def test_otp_outcome_classification() -> None:
    assert (
        otp_outcome(
            status=200, error_code=None, decision="allow", state="allowed", reason=None
        )
        == "correct"
    )
    assert (
        otp_outcome(
            status=200,
            error_code=None,
            decision="challenge",
            state="challenging",
            reason="Incorrect code. 2 attempt(s) remaining.",
        )
        == "incorrect"
    )
    assert (
        otp_outcome(
            status=200,
            error_code=None,
            decision="deny",
            state="denied",
            reason="Step-up failed. OTP attempts exhausted.",
        )
        == "exhausted"
    )
    assert (
        otp_outcome(
            status=410,
            error_code="otp_expired",
            decision=None,
            state=None,
            reason=None,
        )
        == "expired"
    )


def test_endpoint_normalisation_keeps_label_cardinality_bounded() -> None:
    known = BASE_ENDPOINTS | {
        "/review",
        "/review/api/sessions",
        "/review/api/sessions/{session_id}",
        "/review/api/sessions/{session_id}/text",
    }
    assert normalise_endpoint("/events", known) == "/events"
    assert normalise_endpoint("/factors/sess-abc", known) == "/factors/{session_id}"
    assert normalise_endpoint("/healthz/", known) == "/healthz"
    assert (
        normalise_endpoint("/review/api/sessions/sess-abc", known)
        == "/review/api/sessions/{session_id}"
    )
    assert (
        normalise_endpoint("/review/api/sessions/sess-abc/text", known)
        == "/review/api/sessions/{session_id}/text"
    )
    # Anything a scanner invents collapses to one label value.
    assert normalise_endpoint("/wp-login.php", known) == "/unmatched"
    assert normalise_endpoint("/a/b/c/d/e", known) == "/unmatched"
    assert normalise_endpoint("", known) == "/unmatched"


# ---------------------------------------------------------------- the journal


def _entry(at: datetime, request_id: str = "r") -> JournalEntry:
    return JournalEntry(
        at=at,
        request_id=request_id,
        endpoint="/events",
        method="POST",
        status=200,
        outcome="accepted",
        latency_ms=1,
    )


def test_journal_bounds_both_sessions_and_entries() -> None:
    journal = SessionJournal(max_sessions=3, max_entries=2)
    base = datetime.now(UTC)

    for index in range(5):
        journal.record(f"sess-{index}", _entry(base + timedelta(seconds=index)))

    assert len(journal) == 3
    assert journal.dropped_sessions == 2
    assert journal.get("sess-0") is None
    assert journal.get("sess-4") is not None

    for index in range(5):
        journal.record("sess-4", _entry(base + timedelta(minutes=index), str(index)))
    record = journal.get("sess-4")
    assert record is not None
    assert len(record.entries) == 2
    assert record.requests == 6
    assert [e.request_id for e in record.entries] == ["3", "4"]


def test_journal_evicts_least_recently_touched_first() -> None:
    journal = SessionJournal(max_sessions=2)
    base = datetime.now(UTC)
    journal.record("a", _entry(base))
    journal.record("b", _entry(base + timedelta(seconds=1)))
    # Touching "a" makes "b" the oldest.
    journal.record("a", _entry(base + timedelta(seconds=2)))
    journal.record("c", _entry(base + timedelta(seconds=3)))

    assert journal.get("b") is None
    assert journal.get("a") is not None
    assert journal.get("c") is not None


def test_journal_recent_is_newest_first() -> None:
    journal = SessionJournal()
    base = datetime.now(UTC)
    journal.record("old", _entry(base))
    journal.record("new", _entry(base + timedelta(seconds=30)))
    assert [r.session_id for r in journal.recent()] == ["new", "old"]
    assert [r.session_id for r in journal.recent(limit=1)] == ["new"]


def test_journal_can_be_turned_off() -> None:
    journal = SessionJournal(enabled=False)
    assert journal.record("a", _entry(datetime.now(UTC))) is None
    assert len(journal) == 0


async def test_unauthenticated_requests_cannot_fill_the_journal(
    client: httpx.AsyncClient, observability: Observability
) -> None:
    """A 401 never reached the engine, so its session id is attacker-chosen.

    Journalling it would let anyone evict real calls from a bounded journal.
    """

    for index in range(5):
        response = await client.post(
            "/events",
            json={
                "session_id": f"attacker-session-{index:04d}",
                "event": EventType.SESSION_START.value,
                "sequence": 1,
                "occurred_at": datetime.now(UTC).isoformat(),
            },
            headers={"Authorization": "Bearer wrong"},
        )
        assert response.status_code == 401

    assert len(observability.journal) == 0
    # The attempts are still counted and logged.
    assert observability.metrics.fail_closed.value(cause="unauthorized") == 5


# ------------------------------------------------------------- transparency


async def test_mounting_changes_nothing_about_the_existing_endpoints(
    settings: Settings, log_stream: io.StringIO
) -> None:
    """The middleware has to be invisible to the contract.

    Same bodies, same statuses whether the layer is recording or switched all
    the way off. The layer is always mounted now, so "plain" means every
    switch off, which is the closest thing to an app without it: the
    middleware is still in the stack and still has to change nothing.
    """

    plain = create_app(
        settings,
        observability=ObservabilityConfig(
            log_enabled=False, metrics_enabled=False, journal_enabled=False
        ),
        log_stream=log_stream,
    )
    instrumented = create_app(
        settings, observability=ObservabilityConfig(), log_stream=io.StringIO()
    )

    def stable(body: dict) -> dict:
        """Drop the fields that are wall-clock or timing dependent."""

        for volatile in ("latency_ms", "created_at", "updated_at"):
            body.pop(volatile, None)
        if isinstance(body.get("otp"), dict):
            body["otp"].pop("expires_at", None)
        return body

    async def exercise(app: object) -> list[tuple[int, object]]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://compare.test", headers=AUTH
        ) as local:
            results: list[tuple[int, object]] = []

            health = await local.get("/healthz")
            results.append((health.status_code, health.json()))

            journey = Journey(
                local, Scenario.CLONED_VOICE, session_id="sess-fixed-0001"
            )
            decided = await journey.run_to_decision()
            results.append((decided.status_code, stable(decided.json())))

            factors = await local.get(f"/factors/{journey.session_id}")
            results.append((factors.status_code, stable(factors.json())))

            unauth = await local.post(
                "/events", json={}, headers={"Authorization": "Bearer no"}
            )
            results.append((unauth.status_code, unauth.json()))

            unknown = await local.get("/factors/sess-does-not-exist")
            results.append((unknown.status_code, unknown.json()["error"]["code"]))

            return results

    assert await exercise(plain) == await exercise(instrumented)


async def test_everything_can_be_switched_off(
    settings: Settings, log_stream: io.StringIO
) -> None:
    """With every switch off the layer is a pass-through."""

    app = create_app(
        settings,
        observability=ObservabilityConfig(
            log_enabled=False, metrics_enabled=False, journal_enabled=False
        ),
        log_stream=log_stream,
    )
    observability = app.state.observability
    assert observability.active is False

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://quiet.test", headers=AUTH
    ) as local:
        decided = await Journey(local, Scenario.REAL_CFO).run_to_decision()

    assert decided.status_code == 200
    assert log_stream.getvalue() == ""
    assert len(observability.journal) == 0
    assert (
        observability.metrics.decisions.value(
            decision=Decision.ALLOW.value, scenario=Scenario.REAL_CFO.value
        )
        == 0
    )


# ------------------------------------------------------------------- the auth


def test_token_auth_accepts_bearer_and_basic() -> None:
    import base64

    auth = TokenAuth(TEST_TOKEN)
    auth.check(f"Bearer {TEST_TOKEN}")

    encoded = base64.b64encode(f"reviewer:{TEST_TOKEN}".encode()).decode()
    auth.check(f"Basic {encoded}")

    token_as_user = base64.b64encode(f"{TEST_TOKEN}:".encode()).decode()
    auth.check(f"Basic {token_as_user}")


def test_token_auth_rejects_everything_else() -> None:
    import base64

    from detector.observability import ReviewUnauthorized

    auth = TokenAuth(TEST_TOKEN)
    for header in (
        None,
        "",
        "Bearer",
        "Bearer wrong",
        "Basic not-base64!!",
        f"Basic {base64.b64encode(b'user:wrong').decode()}",
        f"Digest {TEST_TOKEN}",
        f"Bearer {TEST_TOKEN} extra",
    ):
        with pytest.raises(ReviewUnauthorized):
            auth.check(header)


def test_token_auth_survives_a_non_ascii_header() -> None:
    """compare_digest raises TypeError on non-ASCII str. A 401 is the answer."""

    from detector.observability import ReviewUnauthorized

    auth = TokenAuth(TEST_TOKEN)
    with pytest.raises(ReviewUnauthorized):
        auth.check("Bearer ü")
    with pytest.raises(ReviewUnauthorized):
        auth.check("Basic üü")
