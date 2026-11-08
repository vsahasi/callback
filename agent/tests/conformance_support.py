"""The five demo calls of the final contract ruling (section 5), driven through the real gate.

Nothing here is collected by pytest. test_conformance.py runs these cases against a running
detector, and test_conformance_stub.py runs the same cases against a stub that replays section 5's
answers, so the logic of a case is proved even when no detector is up.

A case builds a Rig from real parts: a DetectorClient, the Gate, StepUps and Enrollment, a FakeRamp
and a ScriptedCapture. The capture answers the way the demo caller does and posts each answer to
the detector as the caller's turn, redacted the way session_wiring redacts it. Each case asserts
what section 5 says: every decision and its factors, the first reason of each deny, the score
lines, the vendor hold and the lock.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from uuid import uuid4

import httpx
from callback_shared.contract import (
    AuthorizeRequest,
    Channel,
    Decision,
    DecisionKind,
    ErrorCode,
    ErrorResponse,
    Role,
    Scenario,
    SessionCreate,
    Turn,
    UserSummary,
)
from fake_ramp import FakeRamp

from callback_agent import prompts
from callback_agent.actions import ActionRequest
from callback_agent.detector import DetectorClient
from callback_agent.enrollment import Enrollment
from callback_agent.gate import Gate, GateResult
from callback_agent.redact import redact_turn
from callback_agent.state import CallState
from callback_agent.stepups import StepUps

ERIC_NAME = "Eric Catalano"
ERIC_ID = "u_eric"
ROUTING = "021000021"
ACCOUNT = "000123456789"
CALLEE = "callee-eric"

# What the demo caller says. The mock needs a user turn after a challenge is issued, not the words.
OTP_ANSWER = "four two four two four two"  # the mock code, 424242
CHALLENGE_ANSWER = "four seven two nine one three"
CONFIRM_ANSWER = "yes"
CONSENT_ANSWER = "yes I agree"

# The mock's rise from a passive spoof of 0.30 to 0.92 for cloned_voice, plus margin.
CLONE_RISES_AFTER_S = 6.5

ALL_FACTORS_VALUES = ["voice_challenge", "sms_otp", "callback_on_file"]

BANK = ActionRequest.update_vendor_bank_account("v1", "Acme Supplies", ROUTING, ACCOUNT)
ACCOUNTS = ActionRequest.list_vendor_accounts("v1", "Acme Supplies")
HOLD = ActionRequest.hold_vendor("v1", "Acme Supplies", "conformance")


# Makes the httpx client of one DetectorClient. A case that opens two calls makes two clients.
HttpFactory = Callable[[], httpx.AsyncClient]


class SpyDetector(DetectorClient):
    """A DetectorClient that keeps every Decision authorize returns, in order."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decisions: list[Decision] = []

    async def authorize(self, session_id: str, req: AuthorizeRequest) -> Decision:
        decision = await super().authorize(session_id, req)
        self.decisions.append(decision)
        return decision

    @property
    def rounds(self) -> list[tuple[str, list[str], str | None]]:
        """(decision, required factors, reasons[0]) of each authorize so far."""
        return [
            (
                d.decision.value,
                [f.value for f in d.required_factors],
                d.reasons[0] if d.reasons else None,
            )
            for d in self.decisions
        ]


class ScriptedCapture:
    """The caller's side, for the step-ups, the enrollment and the gate's deny line."""

    def __init__(self, detector: DetectorClient, state: CallState) -> None:
        self.lines: list[str] = []
        self.asks: list[str] = []
        self._detector = detector
        self._state = state

    async def say(self, text: str) -> None:
        self.lines.append(text)

    async def ask(self, prompt: str, *, timeout_s: float, secret: bool = False) -> str | None:
        self.lines.append(prompt)
        self.asks.append(prompt)
        answer = self._answer_to(prompt, secret)
        now_ms = round(time.time() * 1000)
        turn = Turn(role=Role.USER, text=answer, start_ms=now_ms, end_ms=now_ms + 1200, words=[])
        # What session_wiring does to every turn before it forwards it.
        turn = redact_turn(turn, exempt_short_runs=self._state.answer_exempt)
        assert self._state.session_id is not None
        await self._detector.post_turn(self._state.session_id, turn)
        return answer

    @staticmethod
    def _answer_to(prompt: str, secret: bool) -> str:
        if prompt == prompts.CONSENT_TEXT:
            return CONSENT_ANSWER
        if prompt.startswith(prompts.CALLBACK_CONFIRM.split("{")[0]):
            return CONFIRM_ANSWER
        return CHALLENGE_ANSWER if secret else OTP_ANSWER


class FakeCallbackLeg:
    """Answers every callback with Eric, and records the numbers it was asked to dial."""

    def __init__(self) -> None:
        self.dialed: list[str] = []

    async def dial_and_link(self, phone_e164: str) -> str | None:
        self.dialed.append(phone_e164)
        return CALLEE


class FakeCall:
    def __init__(self) -> None:
        self.ended = False

    async def end_call(self) -> None:
        self.ended = True


@dataclass
class Rig:
    client: SpyDetector
    state: CallState
    ramp: FakeRamp
    capture: ScriptedCapture
    call: FakeCall
    leg: FakeCallbackLeg
    gate: Gate
    stepups: StepUps
    enrollment: Enrollment
    user: UserSummary
    others: list[str] = field(default_factory=list)

    async def authorize(self, req: ActionRequest, key: str) -> Decision:
        """One authorize straight to the detector, outside the gate."""
        assert self.state.session_id is not None
        return await self.client.authorize(
            self.state.session_id,
            AuthorizeRequest(action=req.action, params=req.detector_params(), idempotency_key=key),
        )

    async def run(self, req: ActionRequest) -> GateResult:
        return await self.gate.run(req)


@asynccontextmanager
async def rig(
    url: str,
    token: str,
    scenario: Scenario,
    *,
    channel: Channel = Channel.PHONE,
    http_factory: HttpFactory | None = None,
    poll_interval_s: float = 0.25,
) -> AsyncIterator[Rig]:
    """A call at `url`: the caller is identified as Eric and a session is open.

    The session is ended on the way out, so the detector's cap of 5 live sessions is never hit.
    """
    client = SpyDetector(
        url, token, scenario=scenario, http=http_factory() if http_factory else None
    )
    state = CallState(room_name=f"conf-{uuid4().hex}", caller_identity="sip_conf", channel=channel)
    ramp = FakeRamp()
    capture = ScriptedCapture(client, state)
    call = FakeCall()
    leg = FakeCallbackLeg()
    stepups = StepUps(client, capture, state, callback_leg=leg, poll_interval_s=poll_interval_s)
    gate = Gate(
        detector=client, ramp=ramp, speaker=capture, stepups=stepups, call=call, state=state
    )
    enrollment = Enrollment(client, capture, stepups, state)
    try:
        users = await client.find_users(ERIC_NAME)
        assert [u.user_id for u in users] == [ERIC_ID], users
        assert users[0].phone_last_four == "0100"
        created = await client.create_session(
            SessionCreate(
                room=state.room_name,
                caller_identity=state.caller_identity,
                claimed_user_id=users[0].user_id,
                channel=channel,
            )
        )
        state.session_id = created.session_id
        state.claimed_user = users[0]
        yield Rig(client, state, ramp, capture, call, leg, gate, stepups, enrollment, users[0])
    finally:
        try:
            if state.session_id is not None:
                await client.end_session(state.session_id)
        finally:
            await client.aclose()


def deny_codes(r: Rig) -> list[tuple[str, str | None]]:
    """(decision, reasons[0]) of each authorize so far."""
    return [(kind, code) for kind, _, code in r.client.rounds]


def assert_held_and_hung_up(r: Rig, result: GateResult) -> None:
    """The deny path ran: v1 is held, the line says so, the call is over and locked."""
    assert result == GateResult("denied", prompts.DENY_HELD), result
    assert ("hold_vendor", "v1") in r.ramp.calls
    assert r.capture.lines[-1] == prompts.DENY_HELD
    assert r.call.ended and r.state.locked


def assert_reasons(decision: Decision, *expected: str) -> None:
    missing = [line for line in expected if line not in decision.reasons]
    assert not missing, f"{missing} not in {decision.reasons}"


async def assert_hold_allowed(r: Rig, key: str) -> None:
    """Holding is always allowed: authorize hold_vendor is an allow with ttl 0."""
    decision = await r.authorize(HOLD, key)
    assert decision.decision is DecisionKind.ALLOW, decision
    assert decision.ttl_s == 0
    assert decision.reasons == ["tier0_hold"]


async def assert_session_locked(r: Rig, code: str, req: ActionRequest, key: str) -> None:
    decision = await r.authorize(req, key)
    assert decision.decision is DecisionKind.DENY, decision
    assert decision.reasons[:2] == ["session_locked", code], decision.reasons


# --- the five calls ----------------------------------------------------------------------------


async def case_real_cfo(url: str, token: str, **kw) -> None:
    async with rig(url, token, Scenario.REAL_CFO, **kw) as r:
        result = await r.run(BANK)
        assert result.status == "done", result
        # Round 1 asks for everything. Round 2 asks for a fresh voice pass, because the one from
        # round 1 is older than the callback's /caller. Round 3 allows.
        assert [rnd[:2] for rnd in r.client.rounds] == [
            ("step_up", ALL_FACTORS_VALUES),
            ("step_up", ["voice_challenge"]),
            ("allow", []),
        ], r.client.rounds
        first, second, third = r.client.decisions
        assert first.ttl_s == 120 and second.ttl_s == 120
        assert third.ttl_s == 60
        assert_reasons(third, "voiceprint_pass 0.83≥0.62", "spoof_low 0.04")
        assert len(r.leg.dialed) == 1 and r.leg.dialed[0].endswith("0100"), r.leg.dialed
        assert any(c[:2] == ("add_default_bank_account", "v1") for c in r.ramp.calls), r.ramp.calls
        assert not r.state.locked and not r.call.ended

        # Follow-up in the same session: a plain allow, no step-up.
        follow = await r.run(ACCOUNTS)
        assert follow.status == "done", follow
        last = r.client.decisions[-1]
        assert last.decision is DecisionKind.ALLOW and last.ttl_s == 300
        assert last.required_factors == []
        await assert_hold_allowed(r, "hold-1")


async def case_cloned_voice(url: str, token: str, *, age_wait_s: float = CLONE_RISES_AFTER_S, **kw):
    async with rig(url, token, Scenario.CLONED_VOICE, **kw) as r:
        await asyncio.sleep(age_wait_s)
        result = await r.run(BANK)
        assert_held_and_hung_up(r, result)
        assert deny_codes(r) == [("deny", "synthetic_speech")]
        [decision] = r.client.decisions
        assert_reasons(decision, "spoof_high 0.92", "voiceprint_pass 0.78≥0.62")
        assert decision.signals.voiceprint is not None and decision.signals.voiceprint.passed
        await assert_hold_allowed(r, "hold-1")
        await assert_session_locked(r, "synthetic_speech", ACCOUNTS, "lock-1")
        await assert_session_locked(r, "synthetic_speech", BANK, "lock-2")

    async with rig(url, token, Scenario.CLONED_VOICE, channel=Channel.BROWSER, **kw) as r:
        await asyncio.sleep(age_wait_s)
        result = await r.run(ACCOUNTS)
        assert_held_and_hung_up(r, result)
        assert deny_codes(r) == [("deny", "synthetic_speech")]


async def case_replay(url: str, token: str, **kw) -> None:
    async with rig(url, token, Scenario.REPLAY, **kw) as r:
        result = await r.run(BANK)
        assert_held_and_hung_up(r, result)
        assert r.client.rounds[0][:2] == ("step_up", ALL_FACTORS_VALUES)
        # The failed challenge stopped the round: no code was asked for and no call was made.
        assert deny_codes(r)[1:] == [("deny", "challenge_failed_replay")]
        assert len(r.client.rounds) == 2
        assert r.leg.dialed == []
        assert_reasons(r.client.decisions[-1], "phrase_match false", "spoof_mid 0.45")

    async with rig(url, token, Scenario.REPLAY, channel=Channel.BROWSER, **kw) as r:
        result = await r.run(ACCOUNTS)
        assert_held_and_hung_up(r, result)
        assert r.client.rounds[0][:2] == ("step_up", ["voice_challenge"])
        assert len(r.client.rounds) == 2
        assert r.client.rounds[1][::2] == ("deny", "challenge_failed_replay")


async def case_ai_agent(url: str, token: str, **kw) -> None:
    async with rig(url, token, Scenario.AI_AGENT, **kw) as r:
        result = await r.run(BANK)
        assert_held_and_hung_up(r, result)
        assert deny_codes(r) == [("deny", "automated_caller")]
        assert_reasons(r.client.decisions[0], "spoof_high 0.85", "agent_likelihood 0.90")


async def case_human_impostor(url: str, token: str, **kw) -> None:
    async with rig(url, token, Scenario.HUMAN_IMPOSTOR, **kw) as r:
        result = await r.run(BANK)
        assert_held_and_hung_up(r, result)
        assert r.client.rounds[0][:2] == ("step_up", ALL_FACTORS_VALUES)
        assert len(r.client.rounds) == 2
        assert r.client.rounds[1][::2] == ("deny", "speaker_mismatch")
        assert_reasons(
            r.client.decisions[-1],
            "voiceprint_fail 0.29<0.45",
            "text_flags: urgency, verification_refusal",
        )

    async with rig(url, token, Scenario.HUMAN_IMPOSTOR, channel=Channel.BROWSER, **kw) as r:
        result = await r.run(ACCOUNTS)
        assert_held_and_hung_up(r, result)
        assert r.client.rounds[0][:2] == ("step_up", ["voice_challenge"])
        assert r.client.rounds[-1][::2] == ("deny", "speaker_mismatch")


async def case_enrollment(url: str, token: str, **kw) -> None:
    async with rig(url, token, Scenario.REAL_CFO, **kw) as r:
        assert await r.enrollment.run() == "enrolled"
        assert prompts.CONSENT_TEXT in r.capture.asks
        assert r.capture.lines[-1] == prompts.ENROLL_DONE


# --- the failure checks -------------------------------------------------------------------------


async def case_wrong_token(url: str, token: str, **kw) -> None:
    """A wrong token gets 401, which the agent treats as a deny, and the vendor is held."""
    async with rig(url, token, Scenario.REAL_CFO, **kw) as r:
        factory = kw.get("http_factory")
        bad = SpyDetector(url, token + "-wrong", http=factory() if factory else None)
        try:
            gate = Gate(
                detector=bad,
                ramp=r.ramp,
                speaker=r.capture,
                stepups=r.stepups,
                call=r.call,
                state=r.state,
            )
            result = await gate.run(ACCOUNTS)
        finally:
            await bad.aclose()
        assert_held_and_hung_up(r, result)
        [decision] = bad.decisions
        assert decision.reasons[0].startswith("detector_unavailable"), decision.reasons
        assert "HTTP 401" in decision.reasons[0]


async def case_missing_version_header(url: str, token: str, *, http=None) -> None:
    """A request with no X-Contract-Version gets 400 contract_version_mismatch."""
    client = http or httpx.AsyncClient()
    async with client:
        response = await client.get(
            f"{url.rstrip('/')}/v1/users",
            params={"name": ERIC_NAME},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert response.status_code == 400, response.text
    error = ErrorResponse.model_validate_json(response.content)
    assert error.error.code == ErrorCode.CONTRACT_VERSION_MISMATCH


async def case_detector_stopped(token: str, *, http=None) -> None:
    """With nothing listening, a tier 1 or 2 request is denied with detector_unavailable, and the
    vendor is held. The session id is one the detector never had."""
    client = SpyDetector("http://127.0.0.1:1", token, http=http)
    state = CallState(room_name="conf-down", caller_identity="sip_conf", channel=Channel.PHONE)
    state.session_id = "ses_" + "0" * 32
    state.claimed_user = UserSummary(
        user_id=ERIC_ID, display_name=ERIC_NAME, role="CFO", phone_last_four="0100"
    )
    ramp, call = FakeRamp(), FakeCall()
    capture = ScriptedCapture(client, state)
    stepups = StepUps(client, capture, state)
    gate = Gate(
        detector=client, ramp=ramp, speaker=capture, stepups=stepups, call=call, state=state
    )
    try:
        result = await gate.run(BANK)
    finally:
        await client.aclose()
    assert result == GateResult("denied", prompts.DENY_HELD)
    assert ("hold_vendor", "v1") in ramp.calls and call.ended and state.locked
    assert client.decisions[0].reasons[0].startswith("detector_unavailable")
