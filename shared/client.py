"""Reference client for the LiveKit agent side.

The point of this module is the fail-closed rule: if the detector is slow,
unreachable or broken, the agent must treat the call as DENY. Veer can import
this directly or copy the three lines that matter.

    from shared.client import DetectorClient

    client = DetectorClient(base_url=..., token=...)
    outcome = await client.push_event(
        session_id=session_id,
        event=EventType.SPEECH_SEGMENT,
        sequence=next(counter),
    )
    if outcome.decision is Decision.DENY:
        await hang_up()
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from shared.contract import (
    CLIENT_TIMEOUT_MS,
    IDEMPOTENCY_HEADER,
    SCENARIO_HEADER,
    Decision,
    EventResponse,
    EventType,
    Scenario,
)


@dataclass(frozen=True)
class Outcome:
    """What the agent should act on.

    decision is always populated. response is None when the detector failed and
    the decision was synthesised by the fail-closed rule.
    """

    decision: Decision
    response: EventResponse | None
    failed_closed: bool = False
    failure_reason: str | None = None


class DetectorClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_ms: int = CLIENT_TIMEOUT_MS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._timeout_s = timeout_ms / 1000.0
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url,
            timeout=self._timeout_s,
            headers={"Authorization": f"Bearer {token}"},
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "DetectorClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def push_event(
        self,
        *,
        session_id: str,
        event: EventType,
        sequence: int,
        claimed_user_id: str | None = None,
        otp_code: str | None = None,
        scenario: Scenario | None = None,
        idempotency_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Outcome:
        body: dict[str, Any] = {
            "session_id": session_id,
            "event": event.value,
            "sequence": sequence,
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        if claimed_user_id is not None:
            body["claimed_user_id"] = claimed_user_id
        if otp_code is not None:
            body["otp_code"] = otp_code

        headers: dict[str, str] = {}
        if scenario is not None:
            headers[SCENARIO_HEADER] = scenario.value
        if idempotency_key is not None:
            headers[IDEMPOTENCY_HEADER] = idempotency_key

        try:
            http_response = await self._client.post(
                "/events", json=body, headers=headers, timeout=self._timeout_s
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            return Outcome(
                decision=Decision.DENY,
                response=None,
                failed_closed=True,
                failure_reason=f"detector unreachable: {type(exc).__name__}",
            )

        if http_response.status_code >= 500:
            return Outcome(
                decision=Decision.DENY,
                response=None,
                failed_closed=True,
                failure_reason=f"detector returned {http_response.status_code}",
            )

        if http_response.status_code >= 400:
            return Outcome(
                decision=Decision.DENY,
                response=None,
                failed_closed=True,
                failure_reason=(
                    f"detector rejected the request with "
                    f"{http_response.status_code}: {http_response.text[:200]}"
                ),
            )

        parsed = EventResponse.model_validate(http_response.json())
        return Outcome(decision=parsed.decision, response=parsed)
