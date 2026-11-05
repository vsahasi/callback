"""Scenario expectations and a reusable call journey.

Data, not assertions, so the same table drives the in-process suite and a run
against a deployed mock URL.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from shared.contract import (
    SCENARIO_HEADER,
    Decision,
    EventType,
    Scenario,
    SessionState,
)


@dataclass(frozen=True)
class Expectation:
    scenario: Scenario
    # Decision after enough speech has been gathered, before any OTP.
    decision_after_speech: Decision
    state_after_speech: SessionState
    # Whether a correct OTP can rescue the caller.
    otp_can_rescue: bool
    # Terminal decision when the correct OTP is supplied (None if not applicable).
    decision_with_correct_otp: Decision | None
    # Terminal decision when every OTP attempt is wrong (None if not applicable).
    decision_with_wrong_otp: Decision | None


EXPECTATIONS: tuple[Expectation, ...] = (
    Expectation(
        scenario=Scenario.REAL_CFO,
        decision_after_speech=Decision.ALLOW,
        state_after_speech=SessionState.ALLOWED,
        otp_can_rescue=True,
        decision_with_correct_otp=None,
        decision_with_wrong_otp=None,
    ),
    Expectation(
        scenario=Scenario.CLONED_VOICE,
        decision_after_speech=Decision.CHALLENGE,
        state_after_speech=SessionState.CHALLENGING,
        otp_can_rescue=True,
        decision_with_correct_otp=Decision.ALLOW,
        decision_with_wrong_otp=Decision.DENY,
    ),
    Expectation(
        scenario=Scenario.REPLAY,
        decision_after_speech=Decision.DENY,
        state_after_speech=SessionState.DENIED,
        otp_can_rescue=False,
        decision_with_correct_otp=None,
        decision_with_wrong_otp=None,
    ),
    Expectation(
        scenario=Scenario.AI_AGENT,
        decision_after_speech=Decision.DENY,
        state_after_speech=SessionState.DENIED,
        otp_can_rescue=False,
        decision_with_correct_otp=None,
        decision_with_wrong_otp=None,
    ),
    Expectation(
        scenario=Scenario.HUMAN_IMPOSTOR,
        decision_after_speech=Decision.CHALLENGE,
        state_after_speech=SessionState.CHALLENGING,
        otp_can_rescue=True,
        decision_with_correct_otp=Decision.ALLOW,
        decision_with_wrong_otp=Decision.DENY,
    ),
)

EXPECTATION_BY_SCENARIO = {e.scenario: e for e in EXPECTATIONS}


def new_session_id(prefix: str = "sess") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Journey:
    """Drives one session through /events, tracking the sequence number."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        scenario: Scenario,
        *,
        session_id: str | None = None,
        claimed_user_id: str | None = None,
        pin_scenario_header: bool = True,
    ) -> None:
        self.client = client
        self.scenario = scenario
        self.session_id = session_id or new_session_id()
        self.claimed_user_id = claimed_user_id
        self.pin_scenario_header = pin_scenario_header
        self.sequence = 0

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = dict(extra or {})
        if self.pin_scenario_header:
            headers[SCENARIO_HEADER] = self.scenario.value
        return headers

    async def send(
        self,
        event: EventType,
        *,
        otp_code: str | None = None,
        sequence: int | None = None,
        idempotency_key: str | None = None,
        claimed_user_id: str | None = None,
    ) -> httpx.Response:
        if sequence is None:
            self.sequence += 1
            sequence = self.sequence

        body: dict[str, object] = {
            "session_id": self.session_id,
            "event": event.value,
            "sequence": sequence,
            "occurred_at": _now_iso(),
            "metadata": {},
        }
        claim = claimed_user_id if claimed_user_id is not None else self.claimed_user_id
        if claim is not None:
            body["claimed_user_id"] = claim
        if otp_code is not None:
            body["otp_code"] = otp_code

        extra = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        return await self.client.post(
            "/events", json=body, headers=self._headers(extra)
        )

    async def run_to_decision(self, speech_segments: int = 2) -> httpx.Response:
        """session_start, caller_claim, then enough speech to force a decision."""

        response = await self.send(EventType.SESSION_START)
        response.raise_for_status()
        response = await self.send(EventType.CALLER_CLAIM)
        response.raise_for_status()
        for _ in range(speech_segments):
            response = await self.send(EventType.SPEECH_SEGMENT)
            response.raise_for_status()
        return response
