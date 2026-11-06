"""The integration kit: fail-closed on every failure mode, and safe retry.

Two kinds of test here.

Failure injection uses `httpx.MockTransport`, because the only way to be sure
a timeout denies is to cause a timeout. Every path out of the client is
covered: timeout, transport error, 5xx, 4xx, and a 200 whose body does not
parse.

Happy paths run the real app in process over `httpx.ASGITransport`, so the
sequence numbering, the idempotency keys and the terminal guard are checked
against the real state machine rather than a stub.

If you only read one test in this file, read
`test_a_retry_replays_rather_than_re_driving_the_transition`. It is the one
that proves a retry cannot apply an event twice.
"""

from __future__ import annotations

import json

import httpx
import pytest

from detector.app import create_app
from detector.config import Settings
from shared.client import (
    CallSession,
    DetectorClient,
    Outcome,
    RetryPolicy,
    SessionClosed,
    _error_code_of,
    _safe_text,
    new_session_id,
)
from shared.contract import (
    CONTRACT_VERSION,
    IDEMPOTENCY_HEADER,
    MOCK_OTP_CODE,
    Decision,
    ErrorCode,
    EventType,
    NextAction,
    Scenario,
    SessionState,
)

KIT_TOKEN = "client-kit-token"
WRONG_OTP = "000000"

# Retries must not sleep for real in the suite, and they must not be skipped
# for lack of budget either. A generous budget with no backoff keeps the tests
# about the policy rather than about the clock.
FAST_RETRY = RetryPolicy(
    max_attempts=2,
    total_budget_ms=5000,
    backoff_initial_ms=0,
    backoff_max_ms=0,
    jitter_ms=0,
    min_attempt_ms=1,
)


# --------------------------------------------------------------------- wiring


def kit_settings(
    *, speech_segments: int = 2, otp_ttl: int = 120, session_ttl: int = 300
) -> Settings:
    """Jitter off, so these tests are about behaviour and not about latency."""

    return Settings(
        api_token=KIT_TOKEN,
        mock_mode=True,
        session_ttl_seconds=session_ttl,
        otp_max_attempts=3,
        otp_ttl_seconds=otp_ttl,
        mock_jitter_min_ms=0,
        mock_jitter_max_ms=0,
        speech_segments_before_decision=speech_segments,
    )


def asgi_client(
    settings: Settings | None = None, *, retry: RetryPolicy | None = None
) -> DetectorClient:
    """A client wired to the real app in this process."""

    app = create_app(settings or kit_settings())
    http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://client-kit.test",
        headers={"Authorization": f"Bearer {KIT_TOKEN}"},
    )
    return DetectorClient(
        base_url="http://client-kit.test", token=KIT_TOKEN, client=http, retry=retry
    )


def mock_client(
    handler, *, retry: RetryPolicy | None = None, timeout_ms: int = 800
) -> DetectorClient:
    """A client wired to an injected failure."""

    http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://client-kit.test",
        headers={"Authorization": f"Bearer {KIT_TOKEN}"},
    )
    return DetectorClient(
        base_url="http://client-kit.test",
        token=KIT_TOKEN,
        client=http,
        timeout_ms=timeout_ms,
        retry=retry,
    )


class Recorder:
    """Records every request, then answers with whatever it was told to."""

    def __init__(self, *responses) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self._responses) - 1)
        answer = self._responses[index]
        if isinstance(answer, BaseException):
            raise answer
        if callable(answer):
            return answer(request)
        return answer

    @property
    def count(self) -> int:
        return len(self.requests)

    def bodies(self) -> list[bytes]:
        return [r.content for r in self.requests]

    def keys(self) -> list[str | None]:
        return [r.headers.get(IDEMPOTENCY_HEADER) for r in self.requests]

    def sequences(self) -> list[int]:
        return [json.loads(r.content)["sequence"] for r in self.requests]


def assert_failed_closed(outcome: Outcome) -> None:
    """Every property an agent might branch on must point at a refusal."""

    assert outcome.decision is Decision.DENY
    assert outcome.failed_closed is True
    assert outcome.response is None
    assert outcome.is_deny is True
    assert outcome.is_allow is False
    assert outcome.next_action is NextAction.COMPLETE_DENY
    assert outcome.state is SessionState.DENIED
    assert outcome.should_request_otp is False
    assert outcome.should_continue_listening is False
    assert outcome.is_final is True
    assert outcome.otp is None
    assert outcome.risk_score is None


# ------------------------------------------------------- fail closed: timeout


async def test_timeout_fails_closed() -> None:
    """The headline rule. A detector that does not answer in time denies."""

    recorder = Recorder(httpx.ReadTimeout("detector did not answer in time"))

    async with mock_client(recorder, retry=FAST_RETRY) as detector:
        call = CallSession(detector, retry=FAST_RETRY)
        outcome = await call.start()

    assert_failed_closed(outcome)
    assert "ReadTimeout" in (outcome.failure_reason or "")
    assert call.closed is True, "a fail-closed session must not be driven further"


async def test_timeout_mid_call_fails_closed_even_after_a_good_decision() -> None:
    """A session that was going well still denies the moment it goes dark.

    This is the regression that matters most: no amount of earlier ALLOW-ish
    progress may survive a detector that stops answering.
    """

    settings = kit_settings(speech_segments=2)
    app = create_app(settings)
    inner = httpx.ASGITransport(app=app)

    class DiesAfter(httpx.AsyncBaseTransport):
        def __init__(self, after: int) -> None:
            self.after = after
            self.seen = 0

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            self.seen += 1
            if self.seen > self.after:
                raise httpx.ReadTimeout("detector went dark")
            response = await inner.handle_async_request(request)
            await response.aread()
            return httpx.Response(
                response.status_code, headers=response.headers, content=response.content
            )

    transport = DiesAfter(after=2)
    http = httpx.AsyncClient(
        transport=transport,
        base_url="http://client-kit.test",
        headers={"Authorization": f"Bearer {KIT_TOKEN}"},
    )
    async with DetectorClient(
        base_url="http://client-kit.test",
        token=KIT_TOKEN,
        client=http,
        retry=FAST_RETRY,
    ) as detector:
        call = CallSession(detector, scenario=Scenario.REAL_CFO, retry=FAST_RETRY)
        first = await call.start()
        assert first.failed_closed is False
        second = await call.claim_identity("vendor-1")
        assert second.failed_closed is False

        dark = await call.submit_speech()

    assert_failed_closed(dark)


# ---------------------------------------------- fail closed: transport errors


async def test_transport_error_fails_closed() -> None:
    recorder = Recorder(httpx.ConnectError("connection refused"))

    async with mock_client(recorder, retry=FAST_RETRY) as detector:
        call = CallSession(detector, retry=FAST_RETRY)
        outcome = await call.start()

    assert_failed_closed(outcome)
    assert "ConnectError" in (outcome.failure_reason or "")
    # Cheap failure, so the bounded retry did fire and still denied.
    assert recorder.count == 2
    assert outcome.attempts == 2


async def test_read_error_mid_response_fails_closed() -> None:
    recorder = Recorder(httpx.ReadError("socket closed mid-body"))

    async with mock_client(recorder, retry=FAST_RETRY) as detector:
        outcome = await detector.push_event(
            session_id=new_session_id(), event=EventType.SPEECH_SEGMENT, sequence=1
        )

    assert_failed_closed(outcome)


# ------------------------------------------------------------ fail closed: 5xx


@pytest.mark.parametrize("status", [500, 502, 503, 504])
async def test_server_error_fails_closed(status: int) -> None:
    recorder = Recorder(httpx.Response(status, json={"detail": "boom"}))

    async with mock_client(recorder, retry=FAST_RETRY) as detector:
        call = CallSession(detector, retry=FAST_RETRY)
        outcome = await call.start()

    assert_failed_closed(outcome)
    assert str(status) in (outcome.failure_reason or "")
    assert outcome.status_code == status
    # A 5xx carries no decision, so retrying it is safe and it is retried.
    assert recorder.count == 2


# ------------------------------------------------------------ fail closed: 4xx


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (401, ErrorCode.UNAUTHORIZED),
        (400, ErrorCode.INVALID_SCENARIO),
        (404, ErrorCode.SESSION_NOT_FOUND),
        (409, ErrorCode.SESSION_TERMINAL),
        (409, ErrorCode.SEQUENCE_CONFLICT),
        (410, ErrorCode.OTP_EXPIRED),
        (422, ErrorCode.VALIDATION_ERROR),
        (429, ErrorCode.OTP_EXHAUSTED),
    ],
)
async def test_client_error_fails_closed_and_is_not_retried(
    status: int, code: ErrorCode
) -> None:
    """A 4xx is a real answer. Deny, surface the code, and do not retry it.

    Retrying would only hide a genuine 401 or 409 behind a second identical
    rejection, and none of these can be fixed by asking again.
    """

    recorder = Recorder(
        httpx.Response(
            status,
            json={
                "contract_version": CONTRACT_VERSION,
                "error": {"code": code.value, "message": "nope", "detail": None},
            },
        )
    )

    async with mock_client(recorder, retry=FAST_RETRY) as detector:
        call = CallSession(detector, retry=FAST_RETRY)
        outcome = await call.start()

    assert_failed_closed(outcome)
    assert outcome.status_code == status
    assert outcome.error_code is code
    assert recorder.count == 1, "a 4xx must not be retried"
    assert outcome.attempts == 1


# ------------------------------------------------- fail closed: malformed 200


@pytest.mark.parametrize(
    ("label", "response"),
    [
        ("not json", httpx.Response(200, text="<html>proxy error</html>")),
        ("empty body", httpx.Response(200, content=b"")),
        ("json but not the contract", httpx.Response(200, json={"hello": "world"})),
        (
            "missing required fields",
            httpx.Response(200, json={"session_id": "sess-123", "decision": "allow"}),
        ),
        (
            "decision is not in the enum",
            httpx.Response(
                200,
                json={
                    "contract_version": CONTRACT_VERSION,
                    "session_id": "sess-12345678",
                    "state": "allowed",
                    "decision": "definitely_fine",
                    "risk_score": 0.0,
                    "confidence": 1.0,
                    "factors": [],
                    "next_action": "complete_allow",
                    "sequence": 1,
                    "latency_ms": 1,
                },
            ),
        ),
    ],
)
async def test_malformed_200_fails_closed(label: str, response: httpx.Response) -> None:
    """A 200 we cannot read is worse than silence, not better.

    Named explicitly because the obvious implementation lets a parse error
    escape as an exception, and an exception in the agent's call handler is
    one `try/except` away from an allowed call.
    """

    recorder = Recorder(response)

    async with mock_client(recorder, retry=FAST_RETRY) as detector:
        call = CallSession(detector, retry=FAST_RETRY)
        outcome = await call.start()

    assert_failed_closed(outcome), label
    assert "cannot read" in (outcome.failure_reason or "")
    assert outcome.status_code == 200
    # The cached reply to a retry would be just as unreadable.
    assert recorder.count == 1, "a malformed body must not be retried"


async def test_an_allow_shaped_but_invalid_body_still_denies() -> None:
    """The adversarial case: a broken body that claims to be an ALLOW."""

    recorder = Recorder(
        httpx.Response(
            200,
            json={
                "decision": "allow",
                "next_action": "complete_allow",
                "state": "allowed",
            },
        )
    )

    async with mock_client(recorder) as detector:
        outcome = await detector.push_event(
            session_id=new_session_id(), event=EventType.SPEECH_SEGMENT, sequence=1
        )

    assert_failed_closed(outcome)


# ------------------------- fail closed: the rest of the httpx hierarchy
#
# httpx raises more than transport errors. The hierarchy is
#
#   HTTPError
#     RequestError
#       TransportError      (TimeoutException, NetworkError, ProtocolError, ...)
#       DecodingError       <- sibling, not a TransportError
#       TooManyRedirects    <- sibling, not a TransportError
#     HTTPStatusError
#
# so catching only (TimeoutException, TransportError) lets DecodingError and
# TooManyRedirects escape as exceptions. An exception is not a DENY: the
# documented `if outcome.is_allow: ... else: hang_up()` never runs, and an
# agent with a broad except around its turn handler would carry on with the
# call. These four cases used to raise. They must deny.


class _RawStream(httpx.AsyncByteStream):
    """Hands bytes over without letting `Response.__init__` decode them."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    async def __aiter__(self):
        yield self._data


def _undecodable(status: int) -> httpx.Response:
    """A response whose declared encoding does not match its bytes.

    A real cause: a proxy that sets `Content-Encoding: gzip` on a body it did
    not actually gzip. It has to be built as a stream, because passing
    `content=` would decode eagerly in the constructor and blow up in the
    test rather than in the client. Built this way, httpx decodes it inside
    `post()`, so `DecodingError` is raised from the request call itself and
    not from our parse step. Verified against httpx 0.28.1.
    """

    return httpx.Response(
        status,
        headers={"content-encoding": "gzip"},
        stream=_RawStream(b"this was never gzipped"),
    )


@pytest.mark.parametrize("status", [200, 409, 503])
async def test_an_undecodable_body_fails_closed(status: int) -> None:
    recorder = Recorder(lambda _request, code=status: _undecodable(code))

    async with mock_client(recorder, retry=FAST_RETRY) as detector:
        call = CallSession(detector, retry=FAST_RETRY)
        outcome = await call.start()

    assert_failed_closed(outcome)
    assert "DecodingError" in (outcome.failure_reason or "")
    # Bytes arrived but are unusable. Deterministic, so not retried.
    assert recorder.count == 1
    assert call.closed is True


async def test_too_many_redirects_fails_closed() -> None:
    recorder = Recorder(httpx.TooManyRedirects("exceeded maximum allowed redirects"))

    async with mock_client(recorder, retry=FAST_RETRY) as detector:
        call = CallSession(detector, retry=FAST_RETRY)
        outcome = await call.start()

    assert_failed_closed(outcome)
    assert "TooManyRedirects" in (outcome.failure_reason or "")
    assert recorder.count == 1


async def test_an_exception_no_one_anticipated_still_fails_closed() -> None:
    """The belt-and-braces catch, so a future httpx class cannot regress this.

    `push_event` promises never to raise for a detector problem. That promise
    must not depend on the exception hierarchy of a library we do not own.
    """

    class SomethingNewInHttpx(Exception):
        pass

    recorder = Recorder(SomethingNewInHttpx("a class that did not exist yet"))

    async with mock_client(recorder, retry=FAST_RETRY) as detector:
        call = CallSession(detector, retry=FAST_RETRY)
        outcome = await call.start()

    assert_failed_closed(outcome)
    assert "SomethingNewInHttpx" in (outcome.failure_reason or "")


@pytest.mark.parametrize("status", [200, 404, 503])
async def test_factors_fails_closed_on_an_undecodable_body(status: int) -> None:
    """The same hole, on the diagnostic read."""

    recorder = Recorder(lambda _request, code=status: _undecodable(code))

    async with mock_client(recorder) as detector:
        outcome = await detector.get_factors("sess-whatever-000001")

    assert outcome.decision is Decision.DENY
    assert outcome.failed_closed is True
    assert outcome.factors is None


async def test_health_returns_none_rather_than_raising_on_a_bad_body() -> None:
    recorder = Recorder(lambda _request: _undecodable(200))

    async with mock_client(recorder) as detector:
        assert await detector.health() is None


def test_the_error_body_helpers_are_total() -> None:
    """They run inside the fail-closed branches, so they must never raise.

    An unread streaming response makes both `.json()` and `.text` raise
    `httpx.ResponseNotRead`, which is a RuntimeError and not one of the
    exception types the obvious implementation catches.
    """

    broken = _undecodable(409)
    assert _error_code_of(broken) is None
    assert _safe_text(broken) == "<body could not be decoded>"

    # And they still work on a well-formed body.
    good = httpx.Response(
        409,
        json={
            "contract_version": CONTRACT_VERSION,
            "error": {
                "code": ErrorCode.SEQUENCE_CONFLICT.value,
                "message": "nope",
                "detail": None,
            },
        },
    )
    assert _error_code_of(good) is ErrorCode.SEQUENCE_CONFLICT
    assert "sequence_conflict" in _safe_text(good)


# ------------------------------------------------------- credentials preflight


async def test_verify_credentials_accepts_a_good_token() -> None:
    """A good token reads /factors and gets 404 for a session that cannot exist."""

    async with asgi_client() as detector:
        assert await detector.verify_credentials() is True


async def test_verify_credentials_rejects_a_bad_token() -> None:
    """The check health() cannot do, because /healthz has no auth.

    /healthz answering `ok` says nothing about the token, so an agent that
    only probed health() would discover a bad token in front of a vendor.
    """

    app = create_app(kit_settings())
    http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://client-kit.test",
        headers={"Authorization": "Bearer definitely-not-the-token"},
    )
    async with DetectorClient(
        base_url="http://client-kit.test", token="definitely-not-the-token", client=http
    ) as detector:
        health = await detector.health()
        credentials_ok = await detector.verify_credentials()

    # The documented trap, pinned: health is fine, the token is not.
    assert health is not None
    assert health.status == "ok"
    assert credentials_ok is False


async def test_verify_credentials_is_none_when_the_detector_is_unreachable() -> None:
    recorder = Recorder(httpx.ConnectError("gone"))

    async with mock_client(recorder) as detector:
        assert await detector.verify_credentials() is None


async def test_verify_credentials_is_none_when_the_detector_is_broken() -> None:
    """A 5xx tells us nothing about the token, so do not claim it does."""

    recorder = Recorder(httpx.Response(503, json={"detail": "boom"}))

    async with mock_client(recorder) as detector:
        assert await detector.verify_credentials() is None


# -------------------------------------------------------- sequence numbering


async def test_call_session_numbers_events_monotonically() -> None:
    """Strictly increasing from 1, one number per logical event."""

    ok = httpx.Response(
        200,
        json={
            "contract_version": CONTRACT_VERSION,
            "session_id": "sess-abcdef0123456789",
            "state": "listening",
            "decision": "pending",
            "risk_score": 0.0,
            "confidence": 0.0,
            "factors": [],
            "next_action": "continue_listening",
            "reason": None,
            "scenario": "real_cfo",
            "sequence": 1,
            "otp": None,
            "latency_ms": 1,
            "idempotent_replay": False,
        },
    )
    recorder = Recorder(ok)

    async with mock_client(recorder) as detector:
        call = CallSession(detector, session_id="sess-abcdef0123456789")
        assert call.sequence == 0
        await call.start()
        await call.claim_identity("vendor-9")
        await call.submit_speech()
        await call.submit_speech()
        await call.request_otp()
        assert call.sequence == 5

    assert recorder.sequences() == [1, 2, 3, 4, 5]
    assert [json.loads(r.content)["event"] for r in recorder.requests] == [
        "session_start",
        "caller_claim",
        "speech_segment",
        "speech_segment",
        "otp_request",
    ]
    # Every logical event gets its own key.
    assert recorder.keys() == [
        "sess-abcdef0123456789:000001",
        "sess-abcdef0123456789:000002",
        "sess-abcdef0123456789:000003",
        "sess-abcdef0123456789:000004",
        "sess-abcdef0123456789:000005",
    ]


async def test_sequence_numbering_survives_a_real_session() -> None:
    """Against the real state machine, which rejects a non-increasing sequence."""

    async with asgi_client() as detector:
        call = CallSession(detector, scenario=Scenario.REAL_CFO)
        await call.start()
        await call.claim_identity("vendor-9")
        await call.submit_speech()
        outcome = await call.submit_speech()

        assert outcome.response is not None
        assert outcome.response.sequence == 4
        readback = await call.factors()

    assert readback.factors is not None
    assert readback.factors.events_seen == 4


# ------------------------------------------------------------- retry: the key


async def test_a_retry_reuses_the_idempotency_key_and_the_exact_body() -> None:
    """The whole safety argument for retrying, in one assertion block.

    Same key plus an identical payload is the one combination the detector
    treats as a replay. A changed payload under a reused key is a 409. So the
    retry has to resend the same bytes, `occurred_at` included, and it does.
    """

    recorder = Recorder(
        httpx.ConnectError("first attempt never landed"),
        httpx.Response(
            200,
            json={
                "contract_version": CONTRACT_VERSION,
                "session_id": "sess-retry0123456789",
                "state": "listening",
                "decision": "pending",
                "risk_score": 0.0,
                "confidence": 0.0,
                "factors": [],
                "next_action": "continue_listening",
                "reason": None,
                "scenario": "real_cfo",
                "sequence": 1,
                "otp": None,
                "latency_ms": 1,
                "idempotent_replay": False,
            },
        ),
    )

    async with mock_client(recorder, retry=FAST_RETRY) as detector:
        call = CallSession(
            detector, session_id="sess-retry0123456789", retry=FAST_RETRY
        )
        outcome = await call.start()

    assert recorder.count == 2
    assert outcome.failed_closed is False
    assert outcome.decision is Decision.PENDING
    assert outcome.attempts == 2

    keys = recorder.keys()
    assert keys[0] == keys[1] == "sess-retry0123456789:000001"

    bodies = recorder.bodies()
    assert bodies[0] == bodies[1], "a retry must resend byte-identical content"
    # Spelled out, because a regenerated timestamp is the easy way to break
    # idempotency without noticing.
    assert json.loads(bodies[0])["occurred_at"] == json.loads(bodies[1])["occurred_at"]

    # The retry did not consume a sequence number either.
    assert recorder.sequences() == [1, 1]
    assert call.sequence == 1


async def test_a_retry_replays_rather_than_re_driving_the_transition() -> None:
    """The state machine must see the event once, however many attempts it took.

    The transport below lets the real app handle the first attempt, then
    throws the 200 away and answers 503, which is exactly the shape of a
    response lost in transit. The client retries with the same key and the
    same bytes, so the detector serves its cached reply instead of applying
    the event a second time.
    """

    app = create_app(kit_settings())
    inner = httpx.ASGITransport(app=app)

    class LosesTheFirstReply(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.posts = 0

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            response = await inner.handle_async_request(request)
            await response.aread()
            if request.url.path == "/events":
                self.posts += 1
                if self.posts == 1:
                    # The server applied it. We just never got to read it.
                    return httpx.Response(503, json={"detail": "lost in transit"})
            return httpx.Response(
                response.status_code, headers=response.headers, content=response.content
            )

    transport = LosesTheFirstReply()
    http = httpx.AsyncClient(
        transport=transport,
        base_url="http://client-kit.test",
        headers={"Authorization": f"Bearer {KIT_TOKEN}"},
    )

    async with DetectorClient(
        base_url="http://client-kit.test",
        token=KIT_TOKEN,
        client=http,
        retry=FAST_RETRY,
    ) as detector:
        call = CallSession(detector, scenario=Scenario.REAL_CFO, retry=FAST_RETRY)
        outcome = await call.start()

        assert transport.posts == 2
        assert outcome.failed_closed is False
        assert outcome.attempts == 2
        assert outcome.response is not None
        assert outcome.response.idempotent_replay is True, (
            "the second attempt must be served from the idempotency cache, "
            "not applied again"
        )

        readback = await call.factors()

    assert readback.factors is not None
    assert readback.factors.events_seen == 1, (
        "two HTTP attempts must advance the state machine exactly once"
    )


async def test_the_retry_cannot_outlive_the_latency_budget() -> None:
    """Retries live inside the 800 ms budget, they do not extend it.

    A live call cannot be made to wait while the client hopes. With almost no
    budget left there is no second attempt, and the client denies now.
    """

    recorder = Recorder(httpx.ConnectError("refused"))
    broke = RetryPolicy(
        max_attempts=4,
        total_budget_ms=40,
        backoff_initial_ms=30,
        backoff_max_ms=30,
        jitter_ms=0,
        min_attempt_ms=20,
    )

    async with mock_client(recorder, retry=broke) as detector:
        outcome = await detector.push_event(
            session_id=new_session_id(), event=EventType.SESSION_START, sequence=1
        )

    assert_failed_closed(outcome)
    assert recorder.count == 1, "no budget for a second attempt, so no second attempt"
    assert outcome.attempts == 1


async def test_the_default_client_still_makes_exactly_one_attempt() -> None:
    """Opting into retry is explicit. `DetectorClient` on its own does not."""

    recorder = Recorder(httpx.Response(503, json={"detail": "boom"}))

    async with mock_client(recorder) as detector:
        assert detector.retry.max_attempts == 1
        outcome = await detector.push_event(
            session_id=new_session_id(), event=EventType.SESSION_START, sequence=1
        )

    assert_failed_closed(outcome)
    assert recorder.count == 1


def test_backoff_is_bounded_and_grows() -> None:
    policy = RetryPolicy(
        backoff_initial_ms=25, backoff_multiplier=2.0, backoff_max_ms=100, jitter_ms=0
    )
    delays = [policy.backoff_seconds(attempt) for attempt in range(1, 6)]
    assert delays == pytest.approx([0.025, 0.050, 0.100, 0.100, 0.100])


def test_a_retry_policy_must_allow_at_least_one_attempt() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=0)


# ------------------------------------------------------- terminal-state guard


async def test_the_guard_blocks_driving_a_decided_session() -> None:
    async with asgi_client() as detector:
        call = CallSession(detector, scenario=Scenario.REAL_CFO)
        await call.start()
        await call.submit_speech()
        outcome = await call.submit_speech()

        assert outcome.decision is Decision.ALLOW
        assert outcome.state is SessionState.ALLOWED
        assert call.closed is True

        for event in (
            call.submit_speech(),
            call.request_otp(),
            call.submit_otp("1234"),
        ):
            with pytest.raises(SessionClosed) as raised:
                await event
            assert raised.value.outcome is outcome

        # The sequence counter did not move, so a later legitimate event is
        # not pushed out of order by a blocked one.
        assert call.sequence == 3

        # session_end is the one event a terminal session still accepts.
        ended = await call.end_call()
        assert ended.failed_closed is False
        assert ended.state is SessionState.ALLOWED
        assert call.sequence == 4

        # And calling it twice costs no further request.
        again = await call.end_call()
        assert again is ended
        assert call.sequence == 4


async def test_the_guard_also_closes_a_session_that_failed_closed() -> None:
    """A fail-closed DENY is final for the agent, not a retryable hiccup."""

    recorder = Recorder(httpx.ConnectError("gone"))

    async with mock_client(recorder, retry=RetryPolicy.none()) as detector:
        call = CallSession(detector, retry=RetryPolicy.none())
        outcome = await call.start()
        assert_failed_closed(outcome)
        assert call.closed is True

        with pytest.raises(SessionClosed):
            await call.submit_speech()


async def test_running_out_of_speech_segments_fails_closed() -> None:
    """A caller who never produces a decision is denied, not waited on."""

    async with asgi_client(kit_settings(speech_segments=99)) as detector:
        call = CallSession(detector, scenario=Scenario.REAL_CFO)
        await call.start()
        outcome = await call.run_speech_until_decision(max_segments=3)

    assert_failed_closed(outcome)
    assert "failing closed" in (outcome.failure_reason or "")
    assert call.closed is True


# -------------------------------------------- otp is null on a hard deny


@pytest.mark.parametrize(
    "scenario", [Scenario.REPLAY, Scenario.AI_AGENT], ids=lambda s: s.value
)
async def test_hard_deny_scenarios_return_a_null_otp(scenario: Scenario) -> None:
    """`replay` and `ai_agent` are refused with no challenge at all.

    Clients must not assume step-up is available. Branch on `next_action`,
    which says `complete_deny`, and never on `otp` being present.
    """

    async with asgi_client() as detector:
        call = CallSession(detector, scenario=scenario)
        await call.start()
        await call.submit_speech()
        outcome = await call.submit_speech()

        assert outcome.decision is Decision.DENY
        assert outcome.state is SessionState.DENIED
        assert outcome.next_action is NextAction.COMPLETE_DENY
        assert outcome.otp is None, "a hard deny must not issue an OTP challenge"
        assert outcome.should_request_otp is False
        assert outcome.response is not None
        assert outcome.response.otp is None

        readback = await call.factors()

    assert readback.factors is not None
    assert readback.factors.otp is None


@pytest.mark.parametrize(
    "scenario",
    [Scenario.CLONED_VOICE, Scenario.HUMAN_IMPOSTOR],
    ids=lambda s: s.value,
)
async def test_challenge_scenarios_do_carry_an_otp(scenario: Scenario) -> None:
    """The other half of the same rule, so `otp is None` means something."""

    async with asgi_client() as detector:
        call = CallSession(detector, scenario=scenario)
        await call.start()
        await call.submit_speech()
        outcome = await call.submit_speech()

        assert outcome.decision is Decision.CHALLENGE
        assert outcome.should_request_otp is True
        assert outcome.is_final is False
        assert outcome.otp is not None
        assert outcome.otp.attempts_remaining == 3
        assert outcome.otp.code_length == len(MOCK_OTP_CODE)

        resolved = await call.submit_otp(MOCK_OTP_CODE)

    assert resolved.is_allow is True
    assert resolved.next_action is NextAction.COMPLETE_ALLOW


async def test_wrong_codes_deny_and_then_the_guard_takes_over() -> None:
    async with asgi_client() as detector:
        call = CallSession(detector, scenario=Scenario.CLONED_VOICE)
        await call.start()
        await call.submit_speech()
        await call.submit_speech()

        remaining = []
        for _ in range(3):
            outcome = await call.submit_otp(WRONG_OTP)
            assert outcome.otp is not None
            remaining.append(outcome.otp.attempts_remaining)

        assert remaining == [2, 1, 0]
        assert outcome.is_deny is True
        assert outcome.state is SessionState.DENIED
        assert call.closed is True

        # A fourth code is refused locally, so the documented 409
        # session_terminal is never even reached. Either way it is a deny.
        with pytest.raises(SessionClosed):
            await call.submit_otp(WRONG_OTP)


# ---------------------------------------------------------- /factors and health


async def test_factors_read_agrees_with_the_event_stream() -> None:
    async with asgi_client() as detector:
        call = CallSession(detector, scenario=Scenario.HUMAN_IMPOSTOR)
        await call.start()
        await call.submit_speech()
        outcome = await call.submit_speech()
        readback = await call.factors()

    assert readback.failed_closed is False
    assert readback.factors is not None
    assert outcome.response is not None
    assert readback.factors.decision is outcome.response.decision
    assert readback.factors.state is outcome.response.state
    assert readback.factors.risk_score == pytest.approx(outcome.response.risk_score)
    assert readback.decision is outcome.decision


async def test_an_unreadable_factors_read_does_not_look_safe() -> None:
    """A diagnostic read still denies rather than returning an empty success."""

    async with asgi_client() as detector:
        missing = await detector.get_factors("sess-not-a-real-session")

    assert missing.decision is Decision.DENY
    assert missing.failed_closed is True
    assert missing.factors is None
    assert missing.status_code == 404
    assert missing.error_code is ErrorCode.SESSION_NOT_FOUND


async def test_factors_fails_closed_on_a_transport_error() -> None:
    recorder = Recorder(httpx.ConnectError("gone"))

    async with mock_client(recorder) as detector:
        outcome = await detector.get_factors("sess-whatever-000001")

    assert outcome.decision is Decision.DENY
    assert outcome.failed_closed is True
    assert outcome.factors is None


async def test_factors_fails_closed_on_a_malformed_body() -> None:
    recorder = Recorder(httpx.Response(200, text="not json at all"))

    async with mock_client(recorder) as detector:
        outcome = await detector.get_factors("sess-whatever-000001")

    assert outcome.decision is Decision.DENY
    assert outcome.failed_closed is True
    assert outcome.factors is None


async def test_health_answers_on_a_live_detector_and_is_none_otherwise() -> None:
    async with asgi_client() as detector:
        health = await detector.health()

    assert health is not None
    assert health.status == "ok"
    assert health.mock_mode is True
    assert health.contract_version == CONTRACT_VERSION

    recorder = Recorder(httpx.ConnectError("gone"))
    async with mock_client(recorder) as dead:
        assert await dead.health() is None


async def test_a_bad_token_fails_closed_rather_than_raising() -> None:
    """Catch a misconfiguration as a deny, not as an exception mid-call."""

    app = create_app(kit_settings())
    http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://client-kit.test",
        headers={"Authorization": "Bearer not-the-right-token"},
    )
    async with DetectorClient(
        base_url="http://client-kit.test", token="not-the-right-token", client=http
    ) as detector:
        call = CallSession(detector, scenario=Scenario.REAL_CFO)
        outcome = await call.start()

    assert_failed_closed(outcome)
    assert outcome.status_code == 401
    assert outcome.error_code is ErrorCode.UNAUTHORIZED


# -------------------------------------------------------------- housekeeping


async def test_metadata_is_merged_not_replaced() -> None:
    recorder = Recorder(
        httpx.Response(
            200,
            json={
                "contract_version": CONTRACT_VERSION,
                "session_id": "sess-meta0123456789a",
                "state": "listening",
                "decision": "pending",
                "risk_score": 0.0,
                "confidence": 0.0,
                "factors": [],
                "next_action": "continue_listening",
                "reason": None,
                "scenario": "real_cfo",
                "sequence": 1,
                "otp": None,
                "latency_ms": 1,
                "idempotent_replay": False,
            },
        )
    )

    async with mock_client(recorder) as detector:
        call = CallSession(
            detector,
            session_id="sess-meta0123456789a",
            metadata={"channel": "pstn"},
        )
        await call.start(metadata={"leg": "outbound"})

    sent = json.loads(recorder.bodies()[0])["metadata"]
    assert sent == {"channel": "pstn", "leg": "outbound"}


async def test_a_claim_sticks_for_the_rest_of_the_session() -> None:
    """So `claimed_user_id` can select a scenario without a custom header."""

    async with asgi_client() as detector:
        call = CallSession(detector, claimed_user_id=Scenario.AI_AGENT.value)
        await call.start()
        await call.submit_speech()
        outcome = await call.submit_speech()

    assert outcome.response is not None
    assert outcome.response.scenario is Scenario.AI_AGENT
    assert outcome.decision is Decision.DENY


async def test_audio_references_are_sent_not_inlined() -> None:
    from shared.contract import AudioRef

    async with asgi_client() as detector:
        call = CallSession(detector, scenario=Scenario.REAL_CFO)
        await call.start()
        outcome = await call.submit_speech(
            AudioRef(track_id="TR_kit", start_ms=0, end_ms=3000)
        )

    assert outcome.failed_closed is False
    assert outcome.decision is Decision.PENDING
