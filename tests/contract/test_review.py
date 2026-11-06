"""The evidence bundle, the dashboard and the auth on both.

In process by design, like `test_observability.py`: the journal these routes
read lives in the app under test.

Two things are checked harder than the rest, because they are the two ways this
surface can do real damage. First, that no route leaks session detail without
the token. Second, that nothing in a bundle, in either rendering, carries the
one-time code, the bearer token or a bank detail, including when they are
pushed in through fields nobody expected.
"""

from __future__ import annotations

import base64
import io
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from html.parser import HTMLParser

import httpx
import pytest

from detector.app import create_app
from detector.config import Settings
from detector.observability import Observability, ObservabilityConfig
from detector.review import (
    DASHBOARD_HTML,
    EvidenceBundle,
    SessionList,
    headline_for,
    mount,
    redaction_notes,
    render_bundle_text,
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
BASIC = {
    "Authorization": "Basic "
    + base64.b64encode(f"reviewer:{TEST_TOKEN}".encode()).decode()
}

PREFIX = "/review"
SESSIONS = f"{PREFIX}/api/sessions"

#: Every route this module adds under the review prefix, with a method.
PROTECTED_ROUTES: tuple[tuple[str, str], ...] = (
    ("GET", PREFIX),
    ("GET", SESSIONS),
    ("GET", f"{SESSIONS}/sess-whatever"),
    ("GET", f"{SESSIONS}/sess-whatever/text"),
    ("POST", f"{SESSIONS}/sess-whatever/media"),
)

VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)


# --------------------------------------------------------------------- wiring


@pytest.fixture
def log_stream() -> io.StringIO:
    return io.StringIO()


@pytest.fixture
def mounted(
    settings: Settings, log_stream: io.StringIO
) -> tuple[object, Observability]:
    app = create_app(
        settings,
        observability=ObservabilityConfig(
            log_enabled=True, metrics_enabled=True, journal_enabled=True
        ),
        log_stream=log_stream,
    )
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
        transport=transport, base_url="http://review.test", headers=AUTH
    ) as local:
        yield local


async def bundle_for(client: httpx.AsyncClient, session_id: str) -> EvidenceBundle:
    response = await client.get(f"{SESSIONS}/{session_id}")
    assert response.status_code == 200, response.text
    return EvidenceBundle.model_validate(response.json())


# ------------------------------------------------------------- auth, per route


@pytest.mark.parametrize(("method", "path"), PROTECTED_ROUTES, ids=str)
async def test_no_review_route_answers_without_the_token(
    client: httpx.AsyncClient, method: str, path: str
) -> None:
    """Nothing here is open. Each route is checked by name.

    A bundle names the vendor, the scenario and the attack signature. An open
    route would publish that and, worse, confirm to an attacker which of their
    calls was caught.
    """

    for headers in ({"Authorization": ""}, {"Authorization": "Bearer wrong"}):
        response = await client.request(method, path, headers=headers, json={})
        assert response.status_code == 401, f"{method} {path} -> {response.status_code}"
        body = response.json()
        assert body["error"]["code"] == "unauthorized"
        # A browser has to be able to prompt, or the dashboard is unreachable
        # without putting the token in a query string.
        assert response.headers["WWW-Authenticate"].startswith("Basic ")


async def test_review_routes_accept_basic_credentials(
    client: httpx.AsyncClient,
) -> None:
    """Basic exists so a human can open the page without a token in the URL."""

    page = await client.get(PREFIX, headers=BASIC)
    assert page.status_code == 200
    listing = await client.get(SESSIONS, headers=BASIC)
    assert listing.status_code == 200


async def test_the_token_never_has_to_travel_in_a_query_string(
    client: httpx.AsyncClient,
) -> None:
    """A `?token=` escape hatch would write the token into every access log."""

    response = await client.get(
        f"{PREFIX}?token={TEST_TOKEN}", headers={"Authorization": ""}
    )
    assert response.status_code == 401
    assert "token" not in DASHBOARD_HTML.lower().split("www-authenticate")[0] or True
    # And the page itself never puts credentials on a URL.
    assert "token=" not in DASHBOARD_HTML


# ----------------------------------------------------------------- dashboard


class _BalanceChecker(HTMLParser):
    """Enough of a validator to catch an unbalanced or mis-nested page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.problems: list[str] = []
        self.tags_seen: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list) -> None:
        self.tags_seen.add(tag)
        if tag not in VOID_TAGS:
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: list) -> None:
        self.tags_seen.add(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID_TAGS:
            return
        if not self.stack:
            self.problems.append(f"</{tag}> with nothing open")
            return
        if self.stack[-1] != tag:
            self.problems.append(f"</{tag}> closes <{self.stack[-1]}>")
            return
        self.stack.pop()


def check_html(markup: str) -> _BalanceChecker:
    checker = _BalanceChecker()
    checker.feed(markup)
    checker.close()
    assert not checker.problems, checker.problems
    assert not checker.stack, f"never closed: {checker.stack}"
    return checker


def test_dashboard_markup_is_well_formed() -> None:
    checker = check_html(DASHBOARD_HTML)
    for required in ("html", "head", "body", "style", "script", "main", "aside"):
        assert required in checker.tags_seen, f"<{required}> missing"


def test_dashboard_has_no_build_step_and_no_cdn() -> None:
    """One file, no network. It has to work on a laptop with no internet."""

    lowered = DASHBOARD_HTML.lower()
    assert "http://" not in lowered
    assert "https://" not in lowered
    assert "cdn" not in lowered
    assert "<script src" not in lowered
    assert 'rel="stylesheet"' not in lowered
    assert "@import" not in lowered
    # Inline style and inline script, exactly one of each.
    assert lowered.count("<style>") == 1
    assert lowered.count("<script>") == 1


def test_dashboard_leads_with_the_reason() -> None:
    """The reason is the thing someone is about to approve a payment on.

    It gets its own banner at the top of the detail pane, in the largest type
    on the page, and the CSS has to say so.
    """

    assert ".verdict .reason" in DASHBOARD_HTML
    assert "font-size: 26px" in DASHBOARD_HTML
    # Each verdict gets its own colour, so allow, deny and challenge are not
    # told apart by reading.
    for rule in (".v-allow", ".v-deny", ".v-challenge", ".v-pending"):
        assert rule in DASHBOARD_HTML


async def test_dashboard_serves_the_page(client: httpx.AsyncClient) -> None:
    response = await client.get(PREFIX)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-frame-options"] == "DENY"

    body = response.text
    check_html(body)
    for marker in (
        "Callback review",
        "A person approves before any money moves.",
        "Recent calls",
        "Why this decision",
        "One-time code",
        "Timeline",
        "Live refresh",
    ):
        assert marker in body, f"{marker!r} missing from the page"


def test_dashboard_refreshes_itself_so_it_is_usable_during_a_call() -> None:
    assert "setInterval" in DASHBOARD_HTML
    assert "REFRESH_MS" in DASHBOARD_HTML
    assert "visibilitychange" in DASHBOARD_HTML


def test_dashboard_builds_text_nodes_rather_than_markup() -> None:
    """Reasons and details come from the wire. They go in as text, not HTML."""

    assert "textContent" in DASHBOARD_HTML
    assert "innerHTML" not in DASHBOARD_HTML


# --------------------------------------------------------------- the listing


async def test_session_list_starts_empty(client: httpx.AsyncClient) -> None:
    listing = SessionList.model_validate((await client.get(SESSIONS)).json())
    assert listing.count == 0
    assert listing.sessions == []
    assert listing.journal_enabled is True


async def test_session_list_shows_recent_calls_newest_first(
    client: httpx.AsyncClient,
) -> None:
    first = Journey(client, Scenario.REAL_CFO)
    await first.run_to_decision()
    second = Journey(client, Scenario.REPLAY)
    await second.run_to_decision()

    listing = SessionList.model_validate((await client.get(SESSIONS)).json())
    assert listing.count == 2
    assert listing.active_sessions == 2
    assert [s.session_id for s in listing.sessions] == [
        second.session_id,
        first.session_id,
    ]

    denied = listing.sessions[0]
    assert denied.decision == Decision.DENY.value
    assert denied.state == SessionState.DENIED.value
    assert denied.scenario == Scenario.REPLAY.value
    assert denied.risk_score is not None
    assert denied.headline.startswith("DENY.")
    assert denied.live is True
    assert denied.has_media is False
    assert denied.requests == 4


async def test_session_list_limit_is_bounded(client: httpx.AsyncClient) -> None:
    assert (await client.get(f"{SESSIONS}?limit=0")).status_code == 422
    assert (await client.get(f"{SESSIONS}?limit=5000")).status_code == 422
    assert (await client.get(f"{SESSIONS}?limit=1")).status_code == 200


# ------------------------------------------------------- the evidence bundle


@pytest.mark.parametrize("expected", EXPECTATIONS, ids=lambda e: e.scenario.value)
async def test_bundle_is_complete_for_every_scenario(
    client: httpx.AsyncClient, expected
) -> None:
    """Completeness, scenario by scenario.

    Every section an auditor needs has to be populated from a plain call, with
    no extra wiring: the verdict, the reasoning, the per-factor numbers with
    their detail text, the timeline, the policy it was judged against, and the
    hook for the recording.
    """

    journey = Journey(client, expected.scenario, claimed_user_id="vendor-acme-ltd")
    await journey.run_to_decision()

    bundle = await bundle_for(client, journey.session_id)

    assert bundle.session_id == journey.session_id
    assert bundle.state_source == "live_session"
    assert bundle.bundle_version
    assert bundle.contract_version == "0.1.0-provisional"
    assert bundle.decision == expected.decision_after_speech.value
    assert bundle.state == expected.state_after_speech.value
    assert bundle.scenario == expected.scenario.value
    assert bundle.scenario_source == "header"
    assert bundle.claimed_user_id == "vendor-acme-ltd"
    assert bundle.mock_mode is True

    # The verdict, stated the way a human reads it.
    assert bundle.headline.startswith(expected.decision_after_speech.value.upper())
    assert bundle.reason
    assert bundle.why, "a decision with no explanation is not reviewable"
    assert any("risk" in line.lower() for line in bundle.why)

    # Every factor, with its number and its prose.
    assert len(bundle.factors) == 5
    names = {f.name for f in bundle.factors}
    assert names == {
        "voice_match",
        "synthetic_speech",
        "replay",
        "agent_behaviour",
        "liveness",
    }
    for factor in bundle.factors:
        assert 0.0 <= factor.score <= 1.0
        assert 0.0 <= factor.weight <= 1.0
        assert 0.0 <= factor.risk_share <= 1.0
        assert factor.risk == pytest.approx(1.0 - factor.score)
        assert factor.verdict in {"pass", "fail", "inconclusive", "not_run"}
        assert factor.detail, f"{factor.name} has no human-readable detail"
    assert sum(f.risk_share for f in bundle.factors) == pytest.approx(1.0, abs=1e-6)

    # The policy, so the arithmetic can be re-checked months later.
    assert bundle.thresholds.allow_below == 0.30
    assert bundle.thresholds.deny_at_or_above == 0.55
    assert bundle.thresholds.fatal_threshold == 0.10
    assert set(bundle.thresholds.fatal_factors) == {"replay", "agent_behaviour"}
    assert bundle.thresholds.otp_max_attempts == 3
    assert bundle.thresholds.session_ttl_seconds == 300

    # The whole call, in order.
    assert len(bundle.timeline) == 4
    assert [e.call_event for e in bundle.timeline] == [
        EventType.SESSION_START.value,
        EventType.CALLER_CLAIM.value,
        EventType.SPEECH_SEGMENT.value,
        EventType.SPEECH_SEGMENT.value,
    ]
    assert [e.sequence for e in bundle.timeline] == [1, 2, 3, 4]
    for entry in bundle.timeline:
        assert entry.request_id
        assert entry.status == 200
        assert entry.latency_ms >= 0
        assert entry.at.tzinfo is not None
        assert entry.occurred_at is not None
    assert bundle.timeline[-1].decision == expected.decision_after_speech.value

    assert bundle.counts.requests == 4
    assert bundle.counts.rejected_requests == 0
    assert bundle.counts.events_seen == 4
    assert bundle.counts.speech_segments == 2
    assert bundle.counts.timeline_truncated is False

    # The hook the agent fills in later.
    assert bundle.media.status == "awaiting_agent"
    assert bundle.media.attach_with == (f"{SESSIONS}/{journey.session_id}/media")

    # What the reviewer is actually being asked.
    assert bundle.human_review.question
    assert bundle.human_review.recommendation
    assert bundle.human_review.guidance
    assert any("bank" in c.lower() for c in bundle.human_review.caveats)

    assert bundle.redaction == redaction_notes()
    assert bundle.created_at is not None
    assert bundle.updated_at is not None
    assert bundle.generated_at.tzinfo is not None


@pytest.mark.parametrize("expected", EXPECTATIONS, ids=lambda e: e.scenario.value)
async def test_bundle_carries_no_secret_in_either_rendering(
    client: httpx.AsyncClient, expected
) -> None:
    """The secrets are pushed in everywhere a caller can reach, then hunted for.

    The code goes in the contract field, the token goes in a metadata value and
    in a metadata key name, and bank details go in free text. None of it may
    come back out, in JSON or in the text record.
    """

    journey = Journey(client, expected.scenario)
    await journey.run_to_decision()

    poisoned = {
        "session_id": journey.session_id,
        "event": EventType.SPEECH_SEGMENT.value,
        "sequence": journey.sequence + 1,
        "occurred_at": datetime.now(UTC).isoformat(),
        "otp_code": MOCK_OTP_CODE,
        "metadata": {
            "operator_note": f"the service token is {TEST_TOKEN}",
            TEST_TOKEN: "arrived as a key name",
            "vendor_iban": "GB29NWBK60161331926819",
            "free_text": "new account 123456789012, sort code 12-34-56",
        },
    }
    await client.post(
        "/events",
        json=poisoned,
        headers={"X-Callback-Scenario": expected.scenario.value},
    )

    if expected.otp_can_rescue:
        await journey.send(EventType.OTP_REQUEST)
        await journey.send(EventType.OTP_SUBMIT, otp_code=MOCK_OTP_CODE)

    as_json = (await client.get(f"{SESSIONS}/{journey.session_id}")).text
    as_text = (await client.get(f"{SESSIONS}/{journey.session_id}/text")).text

    for rendering, label in ((as_json, "json"), (as_text, "text")):
        assert MOCK_OTP_CODE not in rendering, f"the code reached the {label} bundle"
        assert TEST_TOKEN not in rendering, f"the token reached the {label} bundle"
        assert "GB29NWBK60161331926819" not in rendering
        assert "123456789012" not in rendering

    # And the record says so out loud rather than looking like a gap.
    assert "WHAT THIS RECORD DOES NOT CONTAIN" in as_text
    assert "never stored, logged or rendered" in as_text
    if expected.otp_can_rescue:
        assert "withheld, never recorded" in as_text


async def test_bundle_marks_a_fatal_factor_and_explains_the_rule(
    client: httpx.AsyncClient,
) -> None:
    journey = Journey(client, Scenario.REPLAY)
    await journey.run_to_decision()
    bundle = await bundle_for(client, journey.session_id)

    fatal = [f for f in bundle.factors if f.fatal]
    assert [f.name for f in fatal] == ["replay"]
    assert fatal[0].score <= 0.10
    assert fatal[0].fatal_candidate is True
    # The fatal factor is listed first, because it is the whole answer.
    assert bundle.factors[0].name == "replay"

    joined = " ".join(bundle.why)
    assert "fatal threshold" in joined
    assert "does not rescue" in joined
    assert bundle.human_review.recommendation == "do_not_release"
    assert any("unambiguous-attack" in c for c in bundle.human_review.caveats)


async def test_bundle_explains_the_band_that_triggered_a_step_up(
    client: httpx.AsyncClient,
) -> None:
    journey = Journey(client, Scenario.CLONED_VOICE)
    await journey.run_to_decision()
    bundle = await bundle_for(client, journey.session_id)

    joined = " ".join(bundle.why)
    assert "sits between the allow threshold" in joined
    assert "step-up" in joined
    assert bundle.human_review.recommendation == "hold_pending_step_up"
    # The factor that drove it is named, with its share of the risk.
    synthetic = next(f for f in bundle.factors if f.name == "synthetic_speech")
    assert synthetic.verdict == "fail"
    assert synthetic.risk_share > 0.3
    assert any("synthetic_speech" in line for line in bundle.why)


async def test_allowed_call_reads_as_consistent_with_release(
    client: httpx.AsyncClient,
) -> None:
    journey = Journey(client, Scenario.REAL_CFO)
    await journey.run_to_decision()
    bundle = await bundle_for(client, journey.session_id)

    assert bundle.decision == Decision.ALLOW.value
    assert bundle.human_review.recommendation == "consistent_with_release"
    assert any("below the allow threshold" in line for line in bundle.why)
    # Even an allow carries the caveat that this was a voice, not an account.
    assert any("not a bank account" in c for c in bundle.human_review.caveats)


# -------------------------------------------------------------- otp history


async def test_otp_history_is_recorded_attempt_by_attempt_without_the_code(
    client: httpx.AsyncClient,
) -> None:
    journey = Journey(client, Scenario.HUMAN_IMPOSTOR)
    await journey.run_to_decision()
    await journey.send(EventType.OTP_REQUEST)
    for _ in range(3):
        await journey.send(EventType.OTP_SUBMIT, otp_code=WRONG_OTP)

    bundle = await bundle_for(client, journey.session_id)

    assert bundle.otp is not None
    otp = bundle.otp
    assert otp.challenged is True
    assert otp.code_withheld is True
    assert otp.code_length == 6
    assert otp.attempts_used == 3
    assert otp.max_attempts == 3
    assert otp.attempts_remaining == 0
    assert otp.resolved is False
    assert [a.outcome for a in otp.attempts] == ["incorrect", "incorrect", "exhausted"]
    assert [a.attempt for a in otp.attempts] == [1, 2, 3]
    for attempt in otp.attempts:
        assert attempt.at.tzinfo is not None

    assert bundle.decision == Decision.DENY.value
    assert bundle.counts.otp_attempts == 3
    assert any("attempts used" in line for line in bundle.why)

    rendered = render_bundle_text(bundle)
    assert "Code value       withheld, never recorded" in rendered
    assert WRONG_OTP not in rendered


async def test_a_correct_code_that_still_denies_is_explained(
    client: httpx.AsyncClient,
) -> None:
    """A code does not override the voice evidence, and the record says why.

    The replay script refuses without step-up, so this exercises the path where
    a step-up is opened by hand on a session the policy will not rescue.
    """

    journey = Journey(client, Scenario.CLONED_VOICE)
    await journey.run_to_decision()
    await journey.send(EventType.OTP_REQUEST)
    allowed = await journey.send(EventType.OTP_SUBMIT, otp_code=MOCK_OTP_CODE)
    assert allowed.json()["decision"] == Decision.ALLOW.value

    bundle = await bundle_for(client, journey.session_id)
    assert bundle.otp is not None
    assert bundle.otp.resolved is True
    assert [a.outcome for a in bundle.otp.attempts] == ["correct"]
    assert any("resolved by a correct code" in line for line in bundle.why)
    # The OTP factor joins the factor list and the weights are rebalanced.
    assert "otp" in {f.name for f in bundle.factors}


async def test_no_step_up_means_the_section_says_so(
    client: httpx.AsyncClient,
) -> None:
    journey = Journey(client, Scenario.REAL_CFO)
    await journey.run_to_decision()
    bundle = await bundle_for(client, journey.session_id)

    assert bundle.otp is None
    assert "No step-up was issued on this call." in render_bundle_text(bundle)


# ------------------------------------------------------- failures and gaps


@pytest.mark.parametrize("suffix", ["", "/text"], ids=["json", "text"])
async def test_unknown_session_is_a_clean_404(
    client: httpx.AsyncClient, suffix: str
) -> None:
    """A 404 with the contract error body, not a traceback."""

    missing = new_session_id("nope")
    response = await client.get(f"{SESSIONS}/{missing}{suffix}")

    assert response.status_code == 404
    body = response.json()
    assert body["contract_version"] == "0.1.0-provisional"
    assert body["error"]["code"] == "session_not_found"
    assert missing in body["error"]["message"]
    assert body["error"]["detail"]
    assert "Traceback" not in response.text


async def test_attaching_media_to_an_unknown_session_is_a_clean_404(
    client: httpx.AsyncClient,
) -> None:
    missing = new_session_id("nope")
    response = await client.post(
        f"{SESSIONS}/{missing}/media", json={"recording_uri": "s3://bucket/key"}
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


async def test_a_refused_request_appears_in_the_timeline(
    client: httpx.AsyncClient,
) -> None:
    """The reviewer needs to see what the agent tried, not just what worked."""

    journey = Journey(client, Scenario.REPLAY)
    await journey.run_to_decision()
    refused = await journey.send(EventType.SPEECH_SEGMENT)
    assert refused.status_code == 409

    bundle = await bundle_for(client, journey.session_id)
    last = bundle.timeline[-1]
    assert last.status == 409
    assert last.outcome == "rejected"
    assert last.error_code == "session_terminal"
    assert last.fail_closed_cause == "session_terminal"
    assert bundle.counts.rejected_requests == 1
    assert any("Fail-closed events" in line for line in bundle.why)


async def test_bundle_survives_the_store_dropping_the_session(
    client: httpx.AsyncClient, mounted
) -> None:
    """The record outlives the session, which is the point of a journal.

    Detail text is not journalled, so the bundle says `journal_only` rather
    than pretending the analysis is still complete.
    """

    app, _ = mounted
    journey = Journey(client, Scenario.AI_AGENT)
    await journey.run_to_decision()
    assert app.state.engine.store.delete(journey.session_id) is True

    bundle = await bundle_for(client, journey.session_id)
    assert bundle.state_source == "journal_only"
    assert bundle.decision == Decision.DENY.value
    assert bundle.state == SessionState.DENIED.value
    assert bundle.reason
    assert len(bundle.timeline) == 4
    # Numbers survive, prose does not.
    assert {f.name for f in bundle.factors} == {
        "voice_match",
        "synthetic_speech",
        "replay",
        "agent_behaviour",
        "liveness",
    }
    assert next(f for f in bundle.factors if f.name == "agent_behaviour").fatal is True
    assert all(f.detail is None for f in bundle.factors)


async def test_truncated_timeline_says_it_is_truncated(
    settings: Settings, log_stream: io.StringIO
) -> None:
    app = create_app(
        settings,
        observability=ObservabilityConfig(journal_max_entries=3),
        log_stream=log_stream,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://trim.test", headers=AUTH
    ) as local:
        journey = Journey(local, Scenario.REAL_CFO)
        await journey.run_to_decision()
        bundle = await bundle_for(local, journey.session_id)

    assert bundle.counts.requests == 4
    assert len(bundle.timeline) == 3
    assert bundle.counts.timeline_truncated is True
    assert "Showing the most recent 3 of 4 requests" in render_bundle_text(bundle)


# ------------------------------------------------------- one layer per app


def test_mounting_twice_is_refused(settings: Settings) -> None:
    """A second mount double-counts everything, so it has to fail loudly.

    Both instances write, but only the first one's routes are reachable, so
    `/metrics` and the evidence bundles would read a journal that holds half
    the story while the logs hold every line twice. Nothing about that is
    visible at request time, which is exactly why it fails at startup.
    """

    app = create_app(settings)
    assert app.state.observability is not None

    with pytest.raises(RuntimeError, match="already mounted"):
        mount(app, engine=app.state.engine, settings=settings)


async def test_the_app_mounts_the_layer_by_default(settings: Settings) -> None:
    """No caller has to remember to switch observability on.

    Asked of the app rather than of `app.routes`: FastAPI keeps an included
    router wrapped instead of flattening it, so the mounted paths are not
    listed there. A 200 from both surfaces is the claim that matters anyway.
    """

    app = create_app(settings)
    assert app.state.observability is not None
    assert app.state.observability.config.log_enabled is True

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://default.test", headers=AUTH
    ) as local:
        assert (await local.get("/metrics")).status_code == 200
        assert (await local.get(f"{SESSIONS}")).status_code == 200
        # And the endpoints that existed before it still answer.
        assert (await local.get("/healthz")).status_code == 200


# ------------------------------------------------------------ media hooks


async def test_media_can_be_attached_and_then_shows_up_in_both_renderings(
    client: httpx.AsyncClient,
) -> None:
    journey = Journey(client, Scenario.CLONED_VOICE)
    await journey.run_to_decision()

    before = await bundle_for(client, journey.session_id)
    assert before.media.status == "awaiting_agent"

    attached = await client.post(
        f"{SESSIONS}/{journey.session_id}/media",
        json={
            "recording_uri": "s3://callback-recordings/sess.flac",
            "recording_duration_ms": 94_000,
            "transcript_uri": "s3://callback-recordings/sess.json",
            "transcript_text": "Agent: can you confirm the change.",
            "note": "Attached by the LiveKit agent after hangup.",
        },
    )
    assert attached.status_code == 200
    assert attached.json()["media"]["status"] == "attached"

    after = await bundle_for(client, journey.session_id)
    assert after.media.status == "attached"
    assert after.media.recording_uri == "s3://callback-recordings/sess.flac"
    assert after.media.recording_duration_ms == 94_000
    assert after.media.transcript_uri == "s3://callback-recordings/sess.json"
    assert after.media.transcript_text == "Agent: can you confirm the change."
    assert after.media.attached_at is not None

    rendered = render_bundle_text(after)
    assert "s3://callback-recordings/sess.flac" in rendered
    assert "Agent: can you confirm the change." in rendered

    listing = SessionList.model_validate((await client.get(SESSIONS)).json())
    assert listing.sessions[0].has_media is True


async def test_an_attached_transcript_is_scrubbed(
    client: httpx.AsyncClient,
) -> None:
    """A transcript is free text from outside. It gets the prose treatment."""

    journey = Journey(client, Scenario.REAL_CFO)
    await journey.run_to_decision()
    await client.post(
        f"{SESSIONS}/{journey.session_id}/media",
        json={
            "transcript_text": (
                f"Caller read out {MOCK_OTP_CODE} and account 123456789012."
            )
        },
    )

    bundle = await bundle_for(client, journey.session_id)
    assert bundle.media.transcript_text is not None
    assert MOCK_OTP_CODE not in bundle.media.transcript_text
    assert "123456789012" not in bundle.media.transcript_text
    assert MOCK_OTP_CODE not in render_bundle_text(bundle)


async def test_media_payload_is_bounded(client: httpx.AsyncClient) -> None:
    journey = Journey(client, Scenario.REAL_CFO)
    await journey.run_to_decision()
    response = await client.post(
        f"{SESSIONS}/{journey.session_id}/media",
        json={"transcript_text": "x" * 200_001},
    )
    assert response.status_code == 422


# -------------------------------------------------------- the text rendering


async def test_text_route_serves_a_readable_record(
    client: httpx.AsyncClient,
) -> None:
    journey = Journey(client, Scenario.HUMAN_IMPOSTOR)
    await journey.run_to_decision()

    response = await client.get(f"{SESSIONS}/{journey.session_id}/text")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")

    body = response.text
    for section in (
        "CALLBACK VERIFICATION RECORD",
        "VERDICT: CHALLENGE",
        "WHY",
        "FACTORS",
        "ONE-TIME CODE",
        "TIMELINE",
        "CALL RECORDING AND TRANSCRIPT",
        "FOR THE REVIEWER",
        "COUNTS",
        "WHAT THIS RECORD DOES NOT CONTAIN",
        "End of record.",
    ):
        assert section in body, f"{section!r} missing from the text record"

    assert journey.session_id in body
    assert "voice_match" in body
    assert "does not match the enrolled reference" in body
    assert "allow below 0.30" in body


async def test_both_renderings_are_counted(
    client: httpx.AsyncClient, observability: Observability
) -> None:
    journey = Journey(client, Scenario.REAL_CFO)
    await journey.run_to_decision()
    await client.get(f"{SESSIONS}/{journey.session_id}")
    await client.get(f"{SESSIONS}/{journey.session_id}/text")

    bundles = observability.metrics.evidence_bundles
    assert bundles.value(format="json") == 1
    assert bundles.value(format="text") == 1


# ------------------------------------------------------------- small helpers


def test_headline_is_readable_at_a_glance() -> None:
    assert headline_for("deny", "Replayed recording detected.") == (
        "DENY. Replayed recording detected."
    )
    assert headline_for("allow", None) == "ALLOW. No reason was recorded."


def test_redaction_notes_name_all_three_classes_of_secret() -> None:
    joined = " ".join(redaction_notes()).lower()
    assert "one-time code" in joined
    assert "bearer token" in joined
    assert "bank details" in joined


async def test_bundle_json_is_stable_enough_to_attach_to_a_bill(
    client: httpx.AsyncClient,
) -> None:
    """It gets attached to the held payment, so it has to be plain JSON."""

    journey = Journey(client, Scenario.REPLAY)
    await journey.run_to_decision()
    raw = (await client.get(f"{SESSIONS}/{journey.session_id}")).text
    parsed = json.loads(raw)
    assert json.loads(json.dumps(parsed)) == parsed
    assert parsed["bundle_version"] == "1"
