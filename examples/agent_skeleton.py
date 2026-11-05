#!/usr/bin/env python3
"""Where a LiveKit-style voice agent talks to the detector. Read this as docs.

This file is a sketch, not a dependency. LiveKit is not imported and does not
need to be installed: the LiveKit side is stubbed by `FakeCall` at the bottom,
so the file runs end to end and you can see the control flow before you wire
it into the real agent.

    uv run python examples/agent_skeleton.py                 # verified caller
    uv run python examples/agent_skeleton.py cloned_voice    # step-up path
    uv run python examples/agent_skeleton.py replay          # hard deny
    uv run python examples/agent_skeleton.py --detector-down # fail closed

Replace every `FakeCall` method with the LiveKit equivalent and the rest of
`verify_caller` stays as it is. The parts that matter are flagged with a
`FAIL CLOSED` or `WIRE THIS UP` comment.

The shape of the integration:

    LiveKit room  ->  your agent  ->  detector /events  ->  decision
                           |
                           +-> payment hold stays in place until ALLOW,
                               and a human still approves before money moves.

The one rule: **a decision you did not receive is a DENY.** The detector is a
verification service. If it is slow, unreachable or broken, the call does not
get through. `shared/client.py` already does this on every path, so the only
way to fail open is to write the branch yourself. Do not write that branch.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow `python examples/agent_skeleton.py` from a checkout, not just `uv run`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.client import (
    CallSession,
    DetectorClient,
    Outcome,
    SessionClosed,
)
from shared.contract import (
    MOCK_OTP_CODE,
    AudioRef,
    NextAction,
    Scenario,
)

log = logging.getLogger("callback.agent")

# How many speech segments the agent will offer before it gives up. The
# detector needs two before it commits; this ceiling exists so a caller who
# never says anything useful ends in a DENY rather than an open call.
MAX_SPEECH_SEGMENTS = 4


@dataclass(frozen=True)
class Verdict:
    """What the agent hands back to the payment-hold workflow.

    `allowed` is the only field that may release the hold, and it is False by
    default: every way of constructing a Verdict that is not an explicit ALLOW
    leaves it False.
    """

    allowed: bool
    reason: str
    session_id: str
    # True when we denied because verification did not answer, rather than
    # because it answered no. Worth surfacing to the operator: it is an
    # outage, not a fraud signal.
    unverified: bool = False

    @classmethod
    def deny(cls, session_id: str, reason: str, *, unverified: bool = False) -> Verdict:
        return cls(
            allowed=False,
            reason=reason,
            session_id=session_id,
            unverified=unverified,
        )


# ---------------------------------------------------------------- the real flow


async def verify_caller(
    detector: DetectorClient,
    call: FakeCall,
    *,
    claimed_vendor_id: str,
    scenario: Scenario | None = None,
) -> Verdict:
    """Drive one outbound verification call. The whole integration is here.

    `CallSession` owns the sequence numbers, the idempotency keys and the
    terminal guard, so this function only has to answer one question per step:
    what does `next_action` tell me to do?
    """

    session = CallSession(
        detector,
        claimed_user_id=claimed_vendor_id,
        # Mock-mode only. In production you leave this None and the detector
        # runs real models.
        scenario=scenario,
        metadata={"channel": "pstn", "direction": "outbound"},
    )

    # -------------------------------------------------- 1. the call connected
    outcome = await session.start()
    # FAIL CLOSED. Before the caller has said a word we may already be denied,
    # because `start` could not reach the detector. Hang up, keep the hold.
    if outcome.failed_closed:
        return _fail_closed_verdict(session, outcome)

    # ------------------------------------------- 2. the caller claimed who
    # WIRE THIS UP: in the real agent this is the LLM turn where the vendor
    # answers "this is Dana at Northwind". Send the claim, not the transcript.
    await call.say("Hi, this is Callback confirming a change to your bank details.")
    spoken_name = await call.listen_for_identity_claim()
    outcome = await session.claim_identity(claimed_vendor_id)
    if outcome.failed_closed:
        return _fail_closed_verdict(session, outcome)
    log.info(
        "caller claims to be %r, we are verifying %r",
        spoken_name,
        claimed_vendor_id,
    )

    # ------------------------------------- 3. get them talking, push segments
    # WIRE THIS UP: `audio` is a reference to a span of the LiveKit track. The
    # detector never receives inlined audio over this API.
    await call.say("Please read the new account number back to me, digit by digit.")

    for index in range(MAX_SPEECH_SEGMENTS):
        span = await call.next_speech_span()
        if span is None:
            # Silence. Nothing left to verify with, so this is a DENY.
            await session.end_call()
            return Verdict.deny(
                session.session_id,
                "caller stopped speaking before verification completed",
            )

        outcome = await session.submit_speech(span)

        # FAIL CLOSED. Mid-call outage. Do not keep listening in the hope it
        # recovers: the budget is 800 ms per event and it is already spent.
        if outcome.failed_closed:
            return _fail_closed_verdict(session, outcome)

        # Act on next_action. Never on risk_score. The score is for the
        # operator screen; the action is the contract.
        if outcome.next_action is not NextAction.CONTINUE_LISTENING:
            break

        log.info(
            "segment %d: still pending (risk %.3f)",
            index + 1,
            outcome.risk_score or 0.0,
        )
    else:
        # Ran out of segments without a decision. Failing closed is the only
        # safe reading of "we could not tell".
        await session.end_call()
        return Verdict.deny(
            session.session_id,
            f"no decision after {MAX_SPEECH_SEGMENTS} speech segments",
            unverified=True,
        )

    # ------------------------------------------------------- 4. step-up, maybe
    #
    # Read this branch carefully, it is the one people get wrong.
    #
    # `replay` and `ai_agent` are refused outright. They come back as DENY
    # with `otp` set to **null** and `next_action = complete_deny`. So:
    #
    #   - never assume a challenge is available
    #   - never branch on `outcome.otp is None` to mean "no challenge yet"
    #   - branch on `next_action`, which is unambiguous in both cases
    #
    if outcome.should_request_otp:
        outcome = await _run_step_up(session, call, outcome)
        if outcome.failed_closed:
            return _fail_closed_verdict(session, outcome)

    # --------------------------------------------------- 5. the final answer
    verdict = _verdict_from(session, outcome)

    # session_end is the one event a terminal session still accepts, and the
    # detector denies any session that just stops. Always send it.
    await session.end_call()

    # The guard in action: a final session refuses to be driven on. This block
    # is here to document the behaviour, not because an agent would write it.
    # Real code checks `session.closed` instead of catching the exception.
    if session.closed:
        try:
            await session.submit_speech()
        except SessionClosed as exc:
            log.info("guard held, as it should: %s", exc)

    return verdict


async def _run_step_up(
    session: CallSession, call: FakeCall, outcome: Outcome
) -> Outcome:
    """Ask for the texted code until it passes or the attempts run out."""

    # WIRE THIS UP: whoever owns Twilio sends the SMS. See the open question
    # in docs/CONTRACT.md about who adjudicates the code.
    await call.send_sms_code()

    while outcome.should_request_otp:
        remaining = outcome.otp.attempts_remaining if outcome.otp else 0
        code = await call.listen_for_code(attempts_remaining=remaining)
        if code is None:
            # They will not read a code back. Hang up; the hold stays.
            return await session.end_call()

        outcome = await session.submit_otp(code)

        # FAIL CLOSED. An unanswered otp_submit is a DENY, not a retry
        # prompt. The client already retried inside the latency budget.
        if outcome.failed_closed:
            return outcome

    return outcome


def _verdict_from(session: CallSession, outcome: Outcome) -> Verdict:
    """Map the final `next_action` onto the payment workflow's answer."""

    match outcome.next_action:
        case NextAction.COMPLETE_ALLOW:
            # The ONLY path that sets allowed=True, and even here a human
            # still approves before money moves.
            return Verdict(
                allowed=True,
                reason=outcome.reason or "caller verified",
                session_id=session.session_id,
            )
        case NextAction.COMPLETE_DENY:
            return Verdict.deny(
                session.session_id, outcome.reason or "caller not verified"
            )
        case _:
            # continue_listening or request_otp here means the loops above
            # exited early. Treat an unfinished verification as a failure.
            return Verdict.deny(
                session.session_id,
                f"verification ended on {outcome.next_action.value} without a "
                f"final decision",
                unverified=True,
            )


def _fail_closed_verdict(session: CallSession, outcome: Outcome) -> Verdict:
    """The single place a detector failure becomes an answer. It is a DENY."""

    log.error(
        "detector did not answer for session %s after %d attempt(s): %s",
        session.session_id,
        outcome.attempts,
        outcome.failure_reason,
    )
    return Verdict.deny(
        session.session_id,
        f"verification unavailable: {outcome.failure_reason}",
        unverified=True,
    )


# ------------------------------------------------- the LiveKit side, stubbed
#
# Everything below is scaffolding so this file runs. Delete it and plug in the
# real room, the real STT and the real TTS.


class FakeCall:
    """Stands in for a LiveKit room, STT, TTS and the SMS leg."""

    def __init__(self, *, code_to_read: str | None = MOCK_OTP_CODE) -> None:
        self._code = code_to_read
        self._segments = 0

    async def say(self, text: str) -> None:
        # Real: await self.agent.say(text)
        log.info("agent: %s", text)

    async def listen_for_identity_claim(self) -> str:
        # Real: an LLM turn that extracts the claimed identity from the STT
        # transcript. Send the identity to the detector, not the transcript.
        return "Dana at Northwind"

    async def next_speech_span(self) -> AudioRef | None:
        """A reference to the next span of caller speech, or None on silence."""

        # Real: track the LiveKit track id and the offsets of each utterance
        # your VAD closes. Upload the span to object storage if you want the
        # detector to be able to fetch it, and put the location in `uri`.
        self._segments += 1
        if self._segments > MAX_SPEECH_SEGMENTS:
            return None
        start = self._segments * 3000
        return AudioRef(
            track_id="TR_fake_caller",
            start_ms=start,
            end_ms=start + 3000,
            sample_rate_hz=16000,
        )

    async def send_sms_code(self) -> None:
        # Real: Twilio, to the number from the oldest pre-change invoice. The
        # same line we called, never a number from the change request email.
        log.info("sms: one-time code sent to the number on file")

    async def listen_for_code(self, *, attempts_remaining: int) -> str | None:
        # Real: STT plus digit extraction. Returning None means the caller
        # would not or could not read a code back.
        log.info("agent: read me the code (%d attempts left)", attempts_remaining)
        return self._code


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the skeleton against a detector, or against nothing.",
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        type=Scenario,
        choices=list(Scenario),
        default=Scenario.REAL_CFO,
        metavar="SCENARIO",
        help="Mock scenario to drive.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8021")
    parser.add_argument("--token", default="kit-test")
    parser.add_argument(
        "--detector-down",
        action="store_true",
        help="Point at nothing, to watch the fail-closed path.",
    )
    parser.add_argument(
        "--wrong-code",
        action="store_true",
        help="Have the caller read back the wrong code.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-5s %(message)s")
    # httpx logs every request at INFO, which drowns out the flow we are
    # trying to show.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # 192.0.2.0/24 is reserved for documentation, so nothing is listening.
    base_url = "http://192.0.2.1:9" if args.detector_down else args.base_url

    async with DetectorClient(base_url=base_url, token=args.token) as detector:
        # WIRE THIS UP: do both of these at agent start-up, before you are
        # willing to take calls. health() catches a wrong URL or a detector
        # that is not up. It does NOT catch a bad token, because /healthz
        # carries no auth and answers `ok` regardless, so verify_credentials()
        # is the one that checks the thing you actually configured.
        if not args.detector_down:
            if await detector.health() is None:
                print(
                    f"no detector at {base_url}. Start one with:\n"
                    "  CALLBACK_API_TOKEN=kit-test uv run uvicorn "
                    "detector.app:app --port 8021\n"
                    "or run with --detector-down to see the fail-closed path.",
                    file=sys.stderr,
                )
                return 2
            if await detector.verify_credentials() is False:
                print(
                    f"detector at {base_url} is up but rejected the token. "
                    "Pass the one it was started with via --token.",
                    file=sys.stderr,
                )
                return 2

        call = FakeCall(code_to_read="000000" if args.wrong_code else MOCK_OTP_CODE)
        verdict = await verify_caller(
            detector,
            call,
            claimed_vendor_id="vendor-northwind-7781",
            scenario=None if args.detector_down else args.scenario,
        )

    print()
    print(f"session:    {verdict.session_id}")
    print(f"allowed:    {verdict.allowed}")
    print(f"unverified: {verdict.unverified}")
    print(f"reason:     {verdict.reason}")
    print()
    if verdict.allowed:
        print("-> release the payment hold for human approval")
    else:
        print("-> keep the payment hold, flag for review, do not move money")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
