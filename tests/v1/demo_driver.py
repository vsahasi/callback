"""Drives one demo call over raw httpx, the way the agent's gate would.

Used by test_demo_calls.py (ruling section 5) and test_log_hygiene.py. Every
answer is parsed with the callback_shared model the agent would parse it
with, so a body that breaks the contract fails here first.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import httpx
from callback_shared.contract import (
    ChallengeIssued,
    ChallengeState,
    ChallengeStatus,
    Decision,
    OtpCheckResult,
    OtpSent,
    SessionCreated,
    UserOnFile,
    UserSummary,
)

from tests.v1.conftest import FakeClock

VENDOR = {"vendor_id": "v1", "vendor_name": "Acme Supplies"}

# Routing 021000021, account 000123456789: only the last four of each and a
# keyed fingerprint cross the wire, never the full numbers.
BANK_CHANGE_PARAMS = {
    **VENDOR,
    "new_account_last4": "6789",
    "new_routing_last4": "0021",
    "account_fingerprint": "9f86d081884c7d65",
}

TIER2_ACTION = "update_vendor_bank_account"
TIER1_ACTION = "list_vendor_accounts"
HOLD_ACTION = "hold_vendor"
HOLD_PARAMS = {**VENDOR, "reason": "caller failed verification"}

_keys = itertools.count(1)


@dataclass
class DemoCall:
    client: httpx.AsyncClient
    clock: FakeClock
    scenario: str
    channel: str = "phone"
    room: str = "demo-room"
    caller_identity: str = "sip_demo_caller"
    answer_text: str = "four seven two nine one three"
    session_id: str = ""
    decisions: list[Decision] = field(default_factory=list)

    # ----------------------------------------------------------- plumbing

    def _path(self, tail: str) -> str:
        return f"/v1/sessions/{self.session_id}/{tail}"

    async def _ok(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        response = await self.client.request(method, path, **kwargs)
        assert response.status_code == 200, (path, response.status_code, response.text)
        return response

    async def _no_content(self, path: str, body: Any = None) -> None:
        response = await self.client.post(path, json=body)
        assert response.status_code == 204, (path, response.status_code, response.text)

    # -------------------------------------------------------------- setup

    async def identify(self, name: str = "Eric Catalano") -> UserSummary:
        response = await self._ok("GET", "/v1/users", params={"name": name})
        users = [UserSummary.model_validate(u) for u in response.json()]
        assert [u.user_id for u in users] == ["u_eric"]
        assert users[0].phone_last_four == "0100"
        return users[0]

    async def start(self) -> str:
        """Section 5's shared setup: identify, create, background traffic."""

        user = await self.identify()
        response = await self._ok(
            "POST",
            "/v1/sessions",
            json={
                "room": self.room,
                "caller_identity": self.caller_identity,
                "claimed_user_id": user.user_id,
                "channel": self.channel,
            },
            headers={"X-Callback-Scenario": self.scenario},
        )
        self.session_id = SessionCreated.model_validate(response.json()).session_id
        await self.background()
        return self.session_id

    async def background(self) -> None:
        """The turns and speech events the agent posts as the call goes on."""

        now = self.clock.ms()
        await self._no_content(
            self._path("events"), {"type": "agent_speech_start", "ts_ms": now}
        )
        await self._no_content(
            self._path("turns"),
            {
                "role": "agent",
                "text": "Thanks for calling. How can I help?",
                "start_ms": now,
                "end_ms": now + 900,
                "words": [],
            },
        )
        await self._no_content(
            self._path("events"), {"type": "agent_speech_end", "ts_ms": now + 900}
        )
        await self._no_content(
            self._path("events"), {"type": "user_speech_start", "ts_ms": now + 1100}
        )

    # ---------------------------------------------------------- authorize

    async def authorize(self, action: str, params: dict[str, Any]) -> Decision:
        response = await self._ok(
            "POST",
            self._path("authorize"),
            json={
                "action": action,
                "params": params,
                "idempotency_key": f"idem-{next(_keys)}",
            },
        )
        decision = Decision.model_validate(response.json())
        assert decision.session_id == self.session_id
        assert decision.action == action
        assert decision.agent_say == ""
        self.decisions.append(decision)
        return decision

    async def bank_change(self) -> Decision:
        return await self.authorize(TIER2_ACTION, BANK_CHANGE_PARAMS)

    async def read_accounts(self) -> Decision:
        return await self.authorize(TIER1_ACTION, VENDOR)

    async def hold(self) -> Decision:
        return await self.authorize(HOLD_ACTION, HOLD_PARAMS)

    # ---------------------------------------------------------- step-ups

    async def challenge(self, kind: str) -> tuple[ChallengeIssued, ChallengeStatus]:
        """Issue, speak, answer, poll: pending first, then the result."""

        response = await self._ok("POST", self._path("challenges"), json={"kind": kind})
        issued = ChallengeIssued.model_validate(response.json())
        assert issued.expires_at == self.clock.ms() + 45_000
        poll = f"challenges/{issued.challenge_id}"
        first = ChallengeStatus.model_validate(
            (await self._ok("GET", self._path(poll))).json()
        )
        assert first.status is ChallengeState.PENDING
        now = self.clock.ms()
        await self._no_content(
            self._path("turns"),
            {
                "role": "user",
                "text": self.answer_text,
                "start_ms": now,
                "end_ms": now + 1200,
                "words": [],
            },
        )
        still = ChallengeStatus.model_validate(
            (await self._ok("GET", self._path(poll))).json()
        )
        assert still.status is ChallengeState.PENDING
        self.clock.advance(0.7)
        final = ChallengeStatus.model_validate(
            (await self._ok("GET", self._path(poll))).json()
        )
        return issued, final

    async def sms_otp(self, code: str = "424242") -> OtpCheckResult:
        sent = OtpSent.model_validate(
            (await self._ok("POST", self._path("otp/send"), json={})).json()
        )
        assert sent.sent_to_last_four == "0100"
        checked = await self._ok("POST", self._path("otp/check"), json={"code": code})
        return OtpCheckResult.model_validate(checked.json())

    async def callback(
        self, result: str = "confirmed", callee: str = "callee-eric"
    ) -> UserOnFile:
        user = UserOnFile.model_validate(
            (await self._ok("GET", "/v1/users/u_eric")).json()
        )
        assert user.phone_e164 == "+15555550100"
        await self._no_content(
            self._path("caller"), {"caller_identity": callee, "channel": "phone"}
        )
        await self._no_content(
            self._path("factors"), {"kind": "callback", "result": result}
        )
        return user

    async def end(self) -> None:
        await self._no_content(self._path("end"))
