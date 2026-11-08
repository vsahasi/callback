"""The gate: the one way a caller's request reaches Ramp.

The LLM never decides authorization. This code acts only on what the detection service answers:
- A hold (tier 0) runs at once, even on a locked or closing call. The detector is told in the
  background and its answer ignored.
- A read (tier 1) or a money move (tier 2) runs only on an allow from /authorize, after any
  step-ups the detector asks for. Factors run in order, and the first one that fails ends the
  round: the detector is asked again at once. Each round sends a fresh idempotency key. Rounds
  are capped, and a step-up on the last round runs no factor, since a deny would follow it.
- Everything else takes the deny path: a deny, an exception from the detector or a step-up, a
  step_up with no factors, and running out of rounds. The deny path holds the vendor of a tier 1
  or 2 request that names one, says whether the hold took, locks the call and hangs up. A locked
  call refuses every tier 1 or 2 request after that, but holds still run.
- A tier 1 or 2 request that finds the call closing or locked stops with a denied result,
  without speaking or sending anything. That is checked at the start, before each round, before
  each factor and before the Ramp call. The one exception is a deny that arrives after the call
  is closing (the caller is gone): it still holds the vendor and hangs up, but does not speak.
- run() raises nothing but cancellation. A line that cannot be said or a hang-up that fails is
  logged and the request goes on. Any exception from Ramp after an allow, a RampError or not, is
  a failed result, and the call is never retried. Cancellation propagates and never falls through
  to a Ramp call.
- Gated requests take turns on one lock. exclusive() hands out that same lock, so that voice
  enrollment, which also talks to the caller from start to finish, never overlaps a gated request.
  A hold does not wait for it.
"""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from itertools import count
from time import monotonic
from typing import Literal, Protocol

from callback_shared.contract import AuthorizeRequest, Decision, DecisionKind, Factor

from callback_agent import prompts
from callback_agent.actions import ActionRequest, execute, hold_vendor_fully
from callback_agent.detector import DetectorClient, fail_closed_decision
from callback_agent.ramp import RampClient, RampError
from callback_agent.state import CallState

logger = logging.getLogger(__name__)

DENY_HOLD_REASON = "Callback could not verify a request made on a call"


class Speaker(Protocol):
    async def say(self, text: str) -> None:
        """Say `text` to the caller and return once it has played. Never interruptible."""
        ...


class StepUpRunner(Protocol):
    async def run(self, factor: Factor, req: ActionRequest, decision: Decision) -> bool:
        """Run one factor the detector asked for. True if the caller passed it."""
        ...


class CallControl(Protocol):
    async def end_call(self) -> None:
        """Hang up on the caller."""
        ...


@dataclass
class GateResult:
    status: Literal["done", "pending_approval", "denied", "failed"]
    message: str  # what the tool returns to the LLM


class Gate:
    """Runs every action request for one call, by the rules in the module docstring."""

    def __init__(
        self,
        *,
        detector: DetectorClient,
        ramp: RampClient,
        speaker: Speaker,
        stepups: StepUpRunner,
        call: CallControl,
        state: CallState,
        max_rounds: int = 6,
    ) -> None:
        self._detector = detector
        self._ramp = ramp
        self._speaker = speaker
        self._stepups = stepups
        self._call = call
        self._state = state
        self._max_rounds = max_rounds
        # Round numbers never repeat within a call, so neither do idempotency keys. The detector
        # caches its decision per key, and a key sent twice would bring back the old decision.
        self._round_numbers = count()
        # LiveKit runs the tool calls of one reply concurrently. Gated requests take turns, so two
        # verifications never talk over each other.
        self._one_at_a_time = asyncio.Lock()
        self._log_tasks: set[asyncio.Task[Decision]] = set()

    async def run(self, req: ActionRequest) -> GateResult:
        if req.tier == 0:
            # Holding is always allowed, even on a locked or closing call, and never waits on the
            # detector.
            self._tell_detector_in_background(req)
            return await self._execute(req)
        async with self._one_at_a_time:
            return await self._run_gated(req)

    @contextlib.asynccontextmanager
    async def exclusive(self) -> AsyncIterator[None]:
        """Hold the lock that gated requests take, for as long as the `async with` body runs.

        A gated request that starts meanwhile waits for it, and this waits for any gated request
        that is running. A hold (tier 0) never takes the lock, so it does not wait. Whoever holds
        this must not call run() with a gated request inside the body: the lock is not reentrant.
        A deny can lock the call, or the caller can hang up, while this waits for its turn, so a
        body that must not run on such a call checks the state again once it is inside.
        """
        async with self._one_at_a_time:
            yield

    async def aclose(self) -> None:
        """Cancel any hold still being logged with the detector, and wait until each has stopped."""
        tasks = list(self._log_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_gated(self, req: ActionRequest) -> GateResult:
        if self._must_stop():
            return self._stopped(req)
        session_id = self._state.session_id
        if session_id is None:
            if self._state.detector_down:
                return await self._deny(req)  # fail closed
            return GateResult("failed", prompts.IDENTIFY_FIRST)
        if req.tier == 2:
            await self._say(prompts.VERIFYING)
        for attempt in range(1, self._max_rounds + 1):
            if self._must_stop():
                return self._stopped(req)
            decision = await self._authorize(session_id, req, attempt)
            if decision.decision is DecisionKind.ALLOW:
                if self._must_stop():
                    return self._stopped(req)
                return await self._execute(req)
            if decision.decision is not DecisionKind.STEP_UP or not decision.required_factors:
                return await self._deny(req)
            if attempt == self._max_rounds:
                # No round is left to use what a factor would prove, so a factor would only text a
                # real code or phone the number on file, and a deny would follow at once.
                logger.warning(
                    "too many step-ups: %s needed another on its last round (%d), so it is denied "
                    "without running it",
                    req.action,
                    attempt,
                )
                break
            for factor in decision.required_factors:
                if self._must_stop():
                    return self._stopped(req)
                try:
                    passed = await self._stepups.run(factor, req, decision)
                except Exception:
                    logger.exception(
                        "step-up %s for %s raised, so it is denied", factor, req.action
                    )
                    return await self._deny(req)
                if passed is not True:
                    # No later factor runs: the detector decides at once what the failure means.
                    logger.info("step-up %s for %s failed; asking again", factor, req.action)
                    break
        return await self._deny(req)  # out of rounds

    async def _authorize(self, session_id: str, req: ActionRequest, attempt: int) -> Decision:
        """Ask the detector about `req` and log one line. Any exception comes back as a deny.

        `attempt` counts the authorizes of this request from 1. The round, which feeds the
        idempotency key, counts those of the whole call. The log line has both.
        """
        round_no = next(self._round_numbers)
        started = monotonic()
        try:
            request = AuthorizeRequest(
                action=req.action,
                params=req.detector_params(),
                idempotency_key=req.idempotency_key(session_id, round_no),
                context=req.context(),
            )
            decision = await self._detector.authorize(session_id, request)
            if not isinstance(decision, Decision):  # not even a look-alike may say allow
                raise TypeError(f"the detector answered with a {type(decision).__name__}")
        except Exception as exc:
            # DetectorClient.authorize never raises, but a stand-in for it might.
            logger.exception("authorizing %s raised, so it counts as a deny", req.action)
            decision = fail_closed_decision(session_id, req.action, type(exc).__name__)
        logger.info(
            "authorize session_id=%s action=%s round=%d attempt=%d decision=%s factors=%s "
            "reasons=%r latency_ms=%d",
            session_id,
            req.action,
            round_no,
            attempt,
            decision.decision,
            ",".join(decision.required_factors) or "-",
            decision.reasons,
            round((monotonic() - started) * 1000),
        )
        return decision

    async def _execute(self, req: ActionRequest) -> GateResult:
        """Run `req` in Ramp, once, and answer a failure of any kind with a failed result. A failure
        doesn't hold or hang up, and the call is never retried."""
        try:
            outcome = await execute(self._ramp, req)
        except RampError as exc:
            # Never exc.message: Ramp may echo a submitted number back, reformatted past redaction.
            logger.warning(
                "ramp %s failed: status=%s error_code=%r trace_id=%s",
                req.action,
                exc.status,
                exc.error_code,
                exc.trace_id,
            )
            return GateResult("failed", _ramp_failure(req, exc.status))
        except Exception as exc:
            # Something other than a RampError, such as a closed HTTP client. Only its type is
            # logged: its message could echo what was sent. A write may have gone through.
            logger.error("ramp %s raised %s, not a RampError", req.action, type(exc).__name__)
            return GateResult("failed", _ramp_failure(req, None))
        return GateResult(outcome.status, outcome.summary)

    async def _deny(self, req: ActionRequest) -> GateResult:
        """Hold the vendor of a tier 1 or 2 request that names one, and say whether that took.

        Then lock the call and hang up.

        If the call is closing by the time the hold is done, the caller is gone: it says nothing.
        """
        self._state.locked = True  # first, so nothing more gets through even if a step below fails
        held = False
        target = req.hold_target()
        if target is not None and target.kind == "vendor":
            try:
                held = await hold_vendor_fully(self._ramp, target.id, DENY_HOLD_REASON)
            except Exception:
                logger.exception("the hold for a denied %s raised, so nothing is held", req.action)
        # Only a hold that was confirmed is claimed. A read gets its own wording: nothing was
        # changed, so "that change" would be wrong. Without a hold, DENY_NOT_HELD fits both.
        held_line = prompts.DENY_READ_HELD if req.tier == 1 else prompts.DENY_HELD
        line = held_line if held else prompts.DENY_NOT_HELD
        if self._state.closing:  # checked here, as the caller may have left while the hold ran
            logger.info("%s denied after the caller left, so the deny line is not said", req.action)
        else:
            await self._say(line)
        try:
            await self._call.end_call()
        except Exception:
            logger.exception("could not hang up after denying %s; it is denied anyway", req.action)
        return GateResult("denied", line)

    async def _say(self, line: str) -> None:
        """Say a fixed line. A speaker that fails is logged, and never stops the request."""
        try:
            await self._speaker.say(line)
        except Exception:
            logger.exception("could not say %r; going on without it", line)

    def _must_stop(self) -> bool:
        """The call is ending, or a deny has locked it: no gated request may go further."""
        return self._state.closing or self._state.locked

    def _stopped(self, req: ActionRequest) -> GateResult:
        why = "ending" if self._state.closing else "locked"
        logger.info("%s stopped without Ramp: the call is %s", req.action, why)
        return GateResult("denied", prompts.DENY_NOT_HELD)

    def _tell_detector_in_background(self, req: ActionRequest) -> None:
        """Log a hold with the detector without waiting. Its answer is ignored."""
        session_id = self._state.session_id
        if session_id is None:
            return  # nothing to log it against yet
        task = asyncio.create_task(self._authorize(session_id, req, 1))
        self._log_tasks.add(task)
        task.add_done_callback(self._log_tasks.discard)


def _ramp_failure(req: ActionRequest, status: int | None) -> str:
    """What the LLM should tell the caller about a Ramp call that failed.

    `status` is Ramp's HTTP status, or None when the call died some other way. A 4xx means Ramp
    refused the request, so nothing changed. Any other failure of a change (no answer, a 5xx, a 2xx
    that couldn't be read, an exception that isn't a RampError) leaves the outcome unknown: Ramp
    says a timeout doesn't prove a write failed. A read changes nothing, so its failure is always a
    plain failure.
    """
    if req.tier == 1 or (status is not None and 400 <= status < 500):
        return prompts.RAMP_FAILED
    return prompts.RAMP_UNKNOWN
