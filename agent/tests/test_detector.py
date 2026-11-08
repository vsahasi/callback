import asyncio
import inspect
import json
import os
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
import respx
from callback_shared.contract import (
    AUTHORIZE_TIMEOUT_S,
    CONTRACT_VERSION,
    Action,
    AuthorizeRequest,
    CallbackResult,
    CallerUpdate,
    ChallengeIssued,
    ChallengeKind,
    ChallengeState,
    ChallengeStatus,
    Channel,
    Consent,
    Decision,
    DecisionKind,
    EnrollmentResult,
    EnrollmentStart,
    EnrollmentStarted,
    Factor,
    FactorReport,
    OtpCheckResult,
    OtpSent,
    OtpState,
    RampContext,
    Role,
    Scenario,
    SessionCreate,
    SessionCreated,
    Signals,
    SpeechEvent,
    SpeechEventType,
    Turn,
    UserOnFile,
    UserSummary,
    Word,
)

from callback_agent.detector import DetectorClient, DetectorError, fail_closed_decision

BASE = "https://detector.test"
AUTHORIZE_PATH = "/v1/sessions/s1/authorize"
BANK_REQUEST = AuthorizeRequest(
    action=Action.UPDATE_VENDOR_BANK_ACCOUNT, params={"vendor_id": "v1"}, idempotency_key="k1"
)
ALLOW = {
    "session_id": "s1",
    "action": "update_vendor_bank_account",
    "decision": "allow",
    "ttl_s": 120,
}

SESSION = SessionCreate(
    room="call-+12025550123-x1y2z3",
    caller_identity="sip_+12025550123",
    claimed_user_id="u-eric",
    channel=Channel.PHONE,
)
CALLER = CallerUpdate(caller_identity="sip_+12025550199", channel=Channel.PHONE)
TURN = Turn(
    role=Role.USER,
    text="four two",
    start_ms=1000,
    end_ms=1500,
    words=[Word(w="four", start_ms=1000, end_ms=1200)],
    interrupted=True,
)
EVENT = SpeechEvent(type=SpeechEventType.USER_SPEECH_START, ts_ms=1_700_000_000_000)
CONTEXT_REQUEST = AuthorizeRequest(  # D2: the context rides along with the action
    action=Action.UPDATE_VENDOR_BANK_ACCOUNT,
    params={"vendor_id": "v1", "new_account_last4": "6789"},
    idempotency_key="k2",
    context=RampContext(
        pending_action="update_vendor_bank_account",
        vendor_name="Acme Supplies",
        vendor_bank_last4="4321",
    ),
)
ENROLLMENT = EnrollmentStart(
    session_id="s1",
    user_id="u-eric",
    channel=Channel.BROWSER,
    consent=Consent(
        version="2026-11-05", verbatim="I agree to voice enrollment.", ts_ms=1_700_000_000_000
    ),
)
STEP_UP = {  # the spec's Decision example, trimmed
    "session_id": "s1",
    "action": "update_vendor_bank_account",
    "decision": "step_up",
    "risk_score": 0.62,
    "required_factors": ["voice_challenge", "sms_otp", "callback_on_file"],
    "signals": {"agent_likelihood": 0.3, "otp": "pending"},
    "reasons": ["Payment-detail change requested"],
    "agent_say": "For security, I'll text a code to the number we have on file.",
    "ttl_s": 120,
}
USERS = [
    dict(user_id="u-eric", display_name="Eric Catalano", role="CFO", phone_last_four="0123"),
    dict(user_id="u-erin", display_name="Erin Cole", role="Controller", phone_last_four="0145"),
]


@dataclass(frozen=True)
class Case:
    """One client method: the request it must send, the answer it is given and what it returns."""

    name: str
    invoke: Callable[[DetectorClient], Awaitable[Any]]
    method: str
    path: str
    sends: Any  # the JSON body of the request, None when it has none
    reply: Any  # the JSON body of the answer, None for a 204
    returns: Any


CASES = [
    Case(
        "create_session",
        lambda c: c.create_session(SESSION),
        "POST",
        "/v1/sessions",
        SESSION.model_dump(mode="json"),
        {"session_id": "s1"},
        SessionCreated(session_id="s1"),
    ),
    Case(
        "update_caller",
        lambda c: c.update_caller("s1", CALLER),
        "POST",
        "/v1/sessions/s1/caller",
        CALLER.model_dump(mode="json"),
        None,
        None,
    ),
    Case(
        "post_turn",
        lambda c: c.post_turn("s1", TURN),
        "POST",
        "/v1/sessions/s1/turns",
        TURN.model_dump(mode="json"),
        None,
        None,
    ),
    Case(
        "post_event",
        lambda c: c.post_event("s1", EVENT),
        "POST",
        "/v1/sessions/s1/events",
        EVENT.model_dump(mode="json"),
        None,
        None,
    ),
    Case(
        "create_challenge",
        lambda c: c.create_challenge("s1", ChallengeKind.PHRASE),
        "POST",
        "/v1/sessions/s1/challenges",
        {"kind": "phrase"},
        {"challenge_id": "c1", "say": "blue harbor seven", "expires_at": 1_700_000_030_000},
        ChallengeIssued(challenge_id="c1", say="blue harbor seven", expires_at=1_700_000_030_000),
    ),
    Case(
        "get_challenge",
        lambda c: c.get_challenge("s1", "c1"),
        "GET",
        "/v1/sessions/s1/challenges/c1",
        None,
        {"status": "passed", "phrase_match": True, "speaker_score": 0.81},
        ChallengeStatus(status=ChallengeState.PASSED, phrase_match=True, speaker_score=0.81),
    ),
    Case(
        "send_otp",
        lambda c: c.send_otp("s1"),
        "POST",
        "/v1/sessions/s1/otp/send",
        {},
        {"sent_to_last_four": "0123"},
        OtpSent(sent_to_last_four="0123"),
    ),
    Case(
        "check_otp",
        lambda c: c.check_otp("s1", "123456"),
        "POST",
        "/v1/sessions/s1/otp/check",
        {"code": "123456"},
        {"verified": True},
        OtpCheckResult(verified=True),
    ),
    Case(
        "authorize",
        lambda c: c.authorize("s1", CONTEXT_REQUEST),
        "POST",
        AUTHORIZE_PATH,
        CONTEXT_REQUEST.model_dump(mode="json"),
        STEP_UP,
        Decision(
            session_id="s1",
            action=Action.UPDATE_VENDOR_BANK_ACCOUNT,
            decision=DecisionKind.STEP_UP,
            risk_score=0.62,
            required_factors=[Factor.VOICE_CHALLENGE, Factor.SMS_OTP, Factor.CALLBACK_ON_FILE],
            signals=Signals(agent_likelihood=0.3, otp=OtpState.PENDING),
            reasons=["Payment-detail change requested"],
            agent_say="For security, I'll text a code to the number we have on file.",
            ttl_s=120,
        ),
    ),
    Case(
        "report_factor",
        lambda c: c.report_factor("s1", FactorReport(result=CallbackResult.CONFIRMED)),
        "POST",
        "/v1/sessions/s1/factors",
        {"kind": "callback", "result": "confirmed"},
        None,
        None,
    ),
    Case(
        "end_session",
        lambda c: c.end_session("s1"),
        "POST",
        "/v1/sessions/s1/end",
        None,
        None,
        None,
    ),
    Case(
        "find_users",
        lambda c: c.find_users("Eric"),
        "GET",
        "/v1/users",
        None,
        USERS,
        [UserSummary(**u) for u in USERS],
    ),
    Case(
        "get_user",
        lambda c: c.get_user("u-eric"),
        "GET",
        "/v1/users/u-eric",
        None,
        {**USERS[0], "phone_e164": "+12025550123"},
        UserOnFile(**USERS[0], phone_e164="+12025550123"),
    ),
    Case(
        "start_enrollment",
        lambda c: c.start_enrollment(ENROLLMENT),
        "POST",
        "/v1/enrollments",
        ENROLLMENT.model_dump(mode="json"),
        {"enrollment_id": "e1", "prompts_required": 7},
        EnrollmentStarted(enrollment_id="e1", prompts_required=7),
    ),
    Case(
        "complete_enrollment",
        lambda c: c.complete_enrollment("e1"),
        "POST",
        "/v1/enrollments/e1/complete",
        None,
        {"status": "enrolled", "prompts_passed": 7},
        EnrollmentResult(status="enrolled", prompts_passed=7),
    ),
]
EVERYTHING_BUT_AUTHORIZE = [c for c in CASES if c.name != "authorize"]


@asynccontextmanager
async def detector(base_url: str = BASE, token: str = "tok", **kwargs: Any):
    client = DetectorClient(base_url, token, **kwargs)
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
async def client():
    async with detector(scenario=Scenario.CLONED_VOICE, authorize_deadline_s=0.2) as c:
        yield c


@pytest.fixture
def api():
    # respx records a call only once it has an answer, so a request cancelled at the deadline never
    # counts as called. Tests that need to know a request arrived say so themselves.
    with respx.mock(base_url=BASE, assert_all_called=False) as router:
        yield router


def break_route(route: respx.Route, failure: Exception | httpx.Response) -> None:
    if isinstance(failure, Exception):
        route.mock(side_effect=failure)
    else:
        route.mock(return_value=failure)


async def perform(client: DetectorClient, api: respx.MockRouter, case: Case):
    """Mock the case's endpoint, make the call; return the request that arrived and the result."""
    route = api.route(method=case.method, path=case.path)
    if case.reply is None:
        route.respond(204)
    else:
        route.respond(200, json=case.reply)
    result = await case.invoke(client)
    assert route.call_count == 1
    return route.calls.last.request, result


# --- the failing decision ---------------------------------------------------------------------


def test_fail_closed_decision_is_a_deny():
    d = fail_closed_decision("s1", Action.HOLD_VENDOR, "no answer")
    assert d.decision is DecisionKind.DENY
    assert d.reasons == ["detector_unavailable: no answer"]
    assert (d.session_id, d.action) == ("s1", Action.HOLD_VENDOR)
    assert d.required_factors == []  # nothing the caller could do to pass


# --- what every request looks like ------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
async def test_every_call_sends_bearer_and_version(client, api, case):
    request, _ = await perform(client, api, case)
    assert request.headers["authorization"] == "Bearer tok"
    assert request.headers["x-contract-version"] == CONTRACT_VERSION


async def test_scenario_header_only_on_create_session(client, api):
    create_route = api.post("/v1/sessions").respond(200, json={"session_id": "s1"})
    turn_route = api.post("/v1/sessions/s1/turns").respond(204)
    await client.create_session(SESSION)
    await client.post_turn("s1", TURN)
    assert create_route.calls.last.request.headers["x-callback-scenario"] == "cloned_voice"
    assert "x-callback-scenario" not in turn_route.calls.last.request.headers


async def test_no_scenario_header_when_no_scenario_is_set(api):
    route = api.post("/v1/sessions").respond(200, json={"session_id": "s1"})
    async with detector() as c:
        await c.create_session(SESSION)
    assert route.call_count == 1
    assert "x-callback-scenario" not in route.calls.last.request.headers


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
async def test_each_call_hits_its_endpoint_and_parses_the_answer(client, api, case):
    request, result = await perform(client, api, case)
    if case.sends is None:
        assert request.content == b""
    else:
        assert request.headers["content-type"] == "application/json"
        assert json.loads(request.content) == case.sends
    assert result == case.returns


async def test_turn_body_matches_contract(client, api):
    route = api.post("/v1/sessions/s1/turns").respond(204)
    turn = Turn(
        role="user",
        text="four two",
        start_ms=1000,
        end_ms=1500,
        words=[Word(w="four", start_ms=1000, end_ms=1200)],
        interrupted=True,
    )
    await client.post_turn("s1", turn)
    assert json.loads(route.calls.last.request.content) == turn.model_dump(mode="json")


async def test_find_users_sends_the_name_as_a_query_parameter(client, api):
    route = api.get("/v1/users").respond(200, json=[])
    assert await client.find_users("Tom & Jerry's") == []
    assert route.calls.last.request.url.params["name"] == "Tom & Jerry's"


async def test_timeouts_are_one_second_for_polling_and_five_for_the_rest(client, api):
    poll = api.get("/v1/sessions/s1/challenges/c1").respond(200, json={"status": "pending"})
    otp = api.post("/v1/sessions/s1/otp/send").respond(200, json={"sent_to_last_four": "0123"})
    authorize = api.post(AUTHORIZE_PATH).respond(200, json=ALLOW)
    await client.get_challenge("s1", "c1")
    await client.send_otp("s1")
    await client.authorize("s1", BANK_REQUEST)

    def seconds(route: respx.Route) -> set[float]:
        return set(route.calls.last.request.extensions["timeout"].values())

    assert seconds(poll) == {1.0}
    assert seconds(otp) == {5.0}
    assert seconds(authorize) == {0.2}  # the fixture's authorize_deadline_s


async def test_a_trailing_slash_on_the_base_url_is_harmless(api):
    route = api.post("/v1/sessions/s1/end").respond(204)
    async with detector(BASE + "/") as c:
        await c.end_session("s1")
    assert route.call_count == 1


async def test_an_injected_http_client_is_used_and_closed_with_the_detector_client():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(204)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # no base_url, no headers
    async with detector(http=http) as c:
        await c.post_event("s1", EVENT)
    assert [(r.method, str(r.url), r.headers["authorization"]) for r in seen] == [
        ("POST", f"{BASE}/v1/sessions/s1/events", "Bearer tok")
    ]
    assert http.is_closed


def test_send_otp_takes_no_phone_number():
    assert list(inspect.signature(DetectorClient.send_otp).parameters) == ["self", "session_id"]


def test_authorize_deadline_defaults_to_the_contracts_timeout():
    params = inspect.signature(DetectorClient).parameters
    assert params["authorize_deadline_s"].default == AUTHORIZE_TIMEOUT_S


# --- every call but authorize raises ----------------------------------------------------------

FAILURES = {
    "http_401": lambda: httpx.Response(401),
    "http_503": lambda: httpx.Response(503),
    "connect_error": lambda: httpx.ConnectError("down"),
    "read_timeout": lambda: httpx.ReadTimeout("slow"),
}


@pytest.mark.parametrize("failure", FAILURES.values(), ids=list(FAILURES))
@pytest.mark.parametrize("case", EVERYTHING_BUT_AUTHORIZE, ids=lambda c: c.name)
async def test_other_calls_raise_on_error(client, api, case, failure):
    break_route(api.route(method=case.method, path=case.path), failure())
    with pytest.raises(DetectorError):
        await case.invoke(client)


@pytest.mark.parametrize(
    "case", [c for c in EVERYTHING_BUT_AUTHORIZE if c.reply is not None], ids=lambda c: c.name
)
async def test_an_answer_that_is_not_json_raises_detector_error(client, api, case):
    api.route(method=case.method, path=case.path).respond(200, text="not json")
    with pytest.raises(DetectorError):
        await case.invoke(client)


async def test_errors_name_the_call_but_never_echo_the_answer(client, api):
    # FastAPI answers a 422 with the offending input, which here is the code the caller read out.
    api.post("/v1/sessions/s1/otp/check").respond(
        422, json={"detail": [{"input": {"code": "123456"}}]}
    )
    with pytest.raises(DetectorError) as error:
        await client.check_otp("s1", "123456")
    assert "/v1/sessions/s1/otp/check" in str(error.value)
    assert "422" in str(error.value)
    assert "123456" not in str(error.value)


async def test_a_malformed_answer_names_the_missing_fields_but_not_the_values(client, api):
    api.get("/v1/users/u-eric").respond(
        200, json={"user_id": "u-eric", "phone_e164": "+12025550123"}
    )
    with pytest.raises(DetectorError) as error:
        await client.get_user("u-eric")
    assert "display_name" in str(error.value)
    assert "+12025550123" not in str(error.value)


# --- authorize fails closed -------------------------------------------------------------------

AUTHORIZE_FAILURES = [
    pytest.param(httpx.ConnectError("down"), id="connect_error"),
    pytest.param(httpx.ReadTimeout("slow"), id="read_timeout"),
    pytest.param(httpx.Response(500), id="http_500"),
    pytest.param(httpx.Response(409), id="http_409"),
    pytest.param(httpx.Response(409, json=ALLOW), id="http_409_whatever_the_body_says"),
    pytest.param(httpx.Response(200, text="not json"), id="not_json"),
    pytest.param(httpx.Response(200, json={"decision": "allow"}), id="allow_without_the_rest"),
    pytest.param(httpx.Response(200, json={**ALLOW, "decision": "maybe"}), id="unknown_decision"),
    pytest.param(httpx.Response(204), id="empty_body"),
]


@pytest.mark.parametrize("failure", AUTHORIZE_FAILURES)
async def test_authorize_failure_is_deny(client, api, failure):
    route = api.post(AUTHORIZE_PATH)
    break_route(route, failure)
    d = await client.authorize("s1", BANK_REQUEST)
    assert route.call_count == 1  # denied for the failure, not for a request that never left
    assert d.decision is DecisionKind.DENY
    assert d.reasons[0].startswith("detector_unavailable")
    assert (d.session_id, d.action) == ("s1", Action.UPDATE_VENDOR_BANK_ACCOUNT)


async def test_authorize_does_not_follow_a_redirect_to_an_allow(client, api):
    api.post(AUTHORIZE_PATH).respond(307, headers={"location": f"{BASE}/elsewhere"})
    elsewhere = api.post("/elsewhere").respond(200, json=ALLOW)
    d = await client.authorize("s1", BANK_REQUEST)
    assert d.decision is DecisionKind.DENY
    assert not elsewhere.called


async def test_authorize_past_deadline_is_deny(client, api):
    arrived = answered = False

    async def slow_allow(request: httpx.Request) -> httpx.Response:
        nonlocal arrived, answered
        arrived = True
        await asyncio.sleep(1)
        answered = True
        return httpx.Response(200, json=ALLOW)

    api.post(AUTHORIZE_PATH).mock(side_effect=slow_allow)
    d = await client.authorize("s1", BANK_REQUEST)
    assert d.decision is DecisionKind.DENY
    assert d.reasons[0].startswith("detector_unavailable")
    assert arrived
    assert not answered  # the request was cancelled at the deadline, not awaited to the end


async def test_authorize_allow_parses(client, api):
    api.post(AUTHORIZE_PATH).respond(200, json={**ALLOW, "field_added_later": 1})
    d = await client.authorize("s1", BANK_REQUEST)
    assert d.decision is DecisionKind.ALLOW
    assert (d.session_id, d.action, d.ttl_s) == ("s1", Action.UPDATE_VENDOR_BANK_ACCOUNT, 120)


@pytest.mark.parametrize(
    "answer, differs",
    [
        pytest.param({**ALLOW, "action": "hold_vendor"}, "action", id="allow_for_another_action"),
        pytest.param({**ALLOW, "session_id": "s2"}, "session_id", id="allow_for_another_session"),
        # Not only allows: a step_up for the wrong action would send a real SMS for it.
        pytest.param(
            {**STEP_UP, "action": "release_vendor_hold"}, "action", id="step_up_for_another_action"
        ),
    ],
)
async def test_authorize_answer_for_another_session_or_action_is_deny(client, api, answer, differs):
    route = api.post(AUTHORIZE_PATH).respond(200, json=answer)
    d = await client.authorize("s1", BANK_REQUEST)
    assert route.call_count == 1
    assert d.decision is DecisionKind.DENY
    assert d.reasons == [f"detector_unavailable: decision mismatch ({differs})"]
    assert (d.session_id, d.action) == ("s1", Action.UPDATE_VENDOR_BANK_ACCOUNT)  # ours, not theirs


async def test_authorize_denies_on_any_error_not_only_http_ones():
    class Exploding(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            raise RuntimeError("boom")

    async with detector(http=httpx.AsyncClient(transport=Exploding())) as c:
        d = await c.authorize("s1", BANK_REQUEST)
    assert d.decision is DecisionKind.DENY
    assert d.reasons[0].startswith("detector_unavailable")
    assert "RuntimeError" in d.reasons[0]


async def test_authorize_does_not_swallow_cancellation(api):
    # The gate relies on this: when the call ends under a pending authorize, the cancel has to
    # propagate. Turned into a deny, it would send the gate down its deny path.
    arrived = asyncio.Event()

    async def hang(request: httpx.Request) -> httpx.Response:
        arrived.set()
        await asyncio.sleep(30)
        return httpx.Response(200, json=ALLOW)

    api.post(AUTHORIZE_PATH).mock(side_effect=hang)
    async with detector(authorize_deadline_s=30) as c:
        task = asyncio.create_task(c.authorize("s1", BANK_REQUEST))
        await arrived.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# --- Eric's mock (Step 5) ---------------------------------------------------------------------


@pytest.mark.mock_detector
@pytest.mark.skipif(
    not os.environ.get("DETECTOR_URL", "").strip(),
    reason="DETECTOR_URL is not set: this needs a running detection service or its mock",
)
async def test_mock_real_cfo_bank_change_steps_up():
    # The mock's real_cfo script: a bank change needs the voice challenge, a code and the callback.
    url = os.environ["DETECTOR_URL"].strip()
    token = os.environ.get("DETECTOR_TOKEN", "").strip()
    async with detector(url, token, scenario=Scenario.REAL_CFO) as c:
        # The claim must be a user on file: a name the detector can look up, not a made-up id.
        [eric] = await c.find_users("Eric Catalano")
        created = await c.create_session(
            SessionCreate(
                room="smoke-room",
                caller_identity="smoke-caller",
                claimed_user_id=eric.user_id,
                channel=Channel.PHONE,
            )
        )
        try:
            decision = await c.authorize(created.session_id, BANK_REQUEST)
        finally:
            await c.end_session(created.session_id)
    assert decision.decision is DecisionKind.STEP_UP, decision.reasons
    assert decision.required_factors == [
        Factor.VOICE_CHALLENGE,
        Factor.SMS_OTP,
        Factor.CALLBACK_ON_FILE,
    ]
