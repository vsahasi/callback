"""The conformance cases, run against a stub that replays section 5 of the final ruling.

This proves the logic of conformance_support.py with no detector: the cases drive the real gate,
step-ups, enrollment and client, and the stub gives the answers the ruling lists. A passing run
here says the cases and the agent agree with section 5. Only test_conformance.py says the detector
does.
"""

import conformance_stub as stub
import httpx
import pytest
from callback_shared.contract import Channel, Scenario
from conformance_support import (
    ACCOUNTS,
    BANK,
    assert_held_and_hung_up,
    case_ai_agent,
    case_cloned_voice,
    case_detector_stopped,
    case_enrollment,
    case_human_impostor,
    case_missing_version_header,
    case_real_cfo,
    case_replay,
    case_wrong_token,
    rig,
)

URL = "http://detector.stub"
FAST = {"poll_interval_s": 0.0}


@pytest.fixture
def detector():
    return stub.StubDetector()


async def run_case(case, detector, **kwargs):
    await case(URL, stub.TOKEN, http_factory=detector.client, **FAST, **kwargs)
    assert detector.problems == []


async def test_call_1_real_cfo(detector):
    await run_case(case_real_cfo, detector)
    assert detector.sequence() == [
        "GET /v1/users",
        "POST /v1/sessions",
        "POST /v1/sessions/{id}/authorize",
        "POST /v1/sessions/{id}/challenges",
        "GET /v1/sessions/{id}/challenges/{cid}",
        "POST /v1/sessions/{id}/otp/send",
        "POST /v1/sessions/{id}/otp/check",
        "GET /v1/users/{id}",
        "POST /v1/sessions/{id}/caller",
        "POST /v1/sessions/{id}/factors",
        "POST /v1/sessions/{id}/authorize",
        "POST /v1/sessions/{id}/challenges",
        "GET /v1/sessions/{id}/challenges/{cid}",
        "POST /v1/sessions/{id}/authorize",  # round 3: allow
        "POST /v1/sessions/{id}/authorize",  # the follow-up read
        "POST /v1/sessions/{id}/authorize",  # the hold
        "POST /v1/sessions/{id}/end",
    ]
    # Four answers, forwarded in the background through the wiring. A challenge answer keeps its
    # spoken digits, and every other answer has its digits redacted.
    assert sorted(detector.turn_texts) == sorted(
        ["four seven two nine one three", "[NUM]", "yes", "four seven two nine one three"]
    )


async def test_a_challenge_answer_that_never_arrives_is_a_refusal_the_detector_denies(detector):
    # Every post of a turn fails, so the forwarder retries once and drops it. The detector then
    # sees challenges with no user turn after them. Each poll window ends pending, the gate asks
    # again, the new challenge supersedes the old, and two such losses deny (2.7 and 2.10).
    detector.fail_turns = True
    detector.model_refusals = True
    async with rig(
        URL,
        stub.TOKEN,
        Scenario.REAL_CFO,
        http_factory=detector.client,
        poll_interval_s=0.0,
        poll_timeout_s=0.02,
    ) as r:
        result = await r.run(BANK)
        assert_held_and_hung_up(r, result)
        assert [rnd[::2] for rnd in r.client.rounds] == [
            ("step_up", None),
            ("step_up", None),
            ("step_up", None),
            ("deny", "challenges_refused"),
        ]
    assert detector.turn_texts == []  # the detector never saw a user turn
    posts = [e for e in detector.sequence(turns=True) if e.endswith("/turns")]
    assert len(posts) >= 6  # each answer was tried twice
    assert detector.sequence().count("POST /v1/sessions/{id}/challenges") == 3
    assert "POST /v1/sessions/{id}/otp/send" not in detector.sequence()  # no later factor ran


async def test_call_2_cloned_voice(detector):
    await run_case(case_cloned_voice, detector, age_wait_s=0)


async def test_call_3_replay(detector):
    await run_case(case_replay, detector)


async def test_call_4_ai_agent(detector):
    await run_case(case_ai_agent, detector)


async def test_call_5_human_impostor(detector):
    await run_case(case_human_impostor, detector)


async def test_enrollment(detector):
    await run_case(case_enrollment, detector)
    prompts_asked = [e for e in detector.sequence() if e.endswith("/challenges")]
    assert len(prompts_asked) == 7
    assert "POST /v1/enrollments" in detector.sequence()
    assert "POST /v1/enrollments/{id}/complete" in detector.sequence()


async def test_every_request_carries_the_token_version_and_one_scenario_header(detector):
    seen: list[httpx.Request] = []

    def spy(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return detector.handle(request)

    def factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(spy))

    async with rig(URL, stub.TOKEN, Scenario.AI_AGENT, http_factory=factory, **FAST) as r:
        await r.run(BANK)
    assert seen
    for request in seen:
        assert request.headers["authorization"] == f"Bearer {stub.TOKEN}"
        assert request.headers["x-contract-version"] == "1.1.0"
        creating = request.method == "POST" and request.url.path == "/v1/sessions"
        assert ("x-callback-scenario" in request.headers) is creating
    assert detector.problems == []


async def test_a_wrong_token_is_a_deny_that_holds_the_vendor(detector):
    await case_wrong_token(URL, stub.TOKEN, http_factory=detector.client, **FAST)


async def test_a_missing_version_header_is_a_400():
    detector = stub.StubDetector()
    await case_missing_version_header(URL, stub.TOKEN, http=detector.client())


async def test_a_detector_that_is_down_is_a_deny_that_holds_the_vendor():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    await case_detector_stopped(
        stub.TOKEN, http=httpx.AsyncClient(transport=httpx.MockTransport(refuse))
    )


async def test_the_stub_checks_the_token_before_the_version():
    detector = stub.StubDetector()
    async with detector.client() as http:
        response = await http.get(URL + "/v1/users", params={"name": "x"})
    assert response.status_code == 401


async def test_a_browser_read_of_a_real_cfo_is_allowed_by_the_stub_script(detector):
    # Only the harness: a browser session reads through the gate and reaches Ramp on an allow.
    async with rig(
        URL,
        stub.TOKEN,
        Scenario.REAL_CFO,
        channel=Channel.BROWSER,
        http_factory=detector.client,
        **FAST,
    ) as r:
        result = await r.run(ACCOUNTS)
        assert result.status == "done"
