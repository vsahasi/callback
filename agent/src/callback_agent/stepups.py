"""The step-ups: what the agent does when the detector wants more proof before an action.

The gate hands each factor of a step_up decision to StepUps.run() and acts on the bool it gets
back: True if the caller passed. The gate stops a round at the first factor that returns False and
asks the detector again, so the detector, not this code, decides what a failure means. Nothing is
caught here, so an error from the detector goes up to the gate, which treats it as a deny. The one
exception is a failed poll of a challenge, which counts as still pending (see poll_challenge).

No model is involved. A factor speaks a fixed line, then takes the caller's next finished turn as
the answer (TurnCapture.ask). The agent never judges an answer itself, except the callback's yes or
no, which a fixed parser reads:
- A voice challenge makes the caller repeat what the detector issued. The detector scores the
  voice and the words, and the agent only polls how that went.
- An SMS code is texted by the detector to the number it has on file. send_otp takes a session and
  nothing else, so no number the caller gives can reach it. The caller reads the code out, and
  the detector checks it.
- A probe asks one off-script question. The detector reads when and how the caller answers.
- A callback hangs up, calls the claimed user back on the number the detector has on file, and
  asks whoever answers whether they made the request. Only a plain yes passes, and the detector
  is told how it went. The CallbackLeg passed in as `callback_leg` hangs up and dials. Without
  one, or on a browser session, it fails closed. If nobody answers once the caller has been taken
  off the line, or whoever answers hangs up on the confirm question, it raises CallbackNoAnswer
  (or CallbackCalleeHungUp, which is one), so that the gate denies on the spot.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Protocol, assert_never

from callback_shared.contract import (
    CallbackResult,
    CallerUpdate,
    ChallengeKind,
    ChallengeState,
    ChallengeStatus,
    Channel,
    Decision,
    Factor,
    FactorReport,
)

from callback_agent import prompts
from callback_agent.actions import ActionRequest
from callback_agent.detector import DetectorClient, DetectorError
from callback_agent.spoken import parse_spoken_digits, parse_yes_no
from callback_agent.state import CallState

logger = logging.getLogger(__name__)

CHALLENGE_ANSWER_TIMEOUT_S = 20.0  # how long the caller has to repeat a challenge
OTP_ANSWER_TIMEOUT_S = 30.0  # per try: the text can take a moment to arrive
OTP_TRIES = 3  # the spec's fusion rule denies after 3 failed codes
OTP_DIGITS = 6
PROBE_ANSWER_TIMEOUT_S = 15.0
CALLBACK_CONFIRM_TIMEOUT_S = 20.0  # how long whoever answers the callback has to confirm

# What a voice challenge asks for, by the tier of the request: a read repeats digits, and a money
# move adds words. The gate steps up only tiers 1 and 2, as a hold needs no proof.
_CHALLENGE_BY_TIER: dict[int, tuple[ChallengeKind, str]] = {
    1: (ChallengeKind.DIGITS, prompts.CHALLENGE_LEAD_IN_READ),
    2: (ChallengeKind.PHRASE, prompts.CHALLENGE_LEAD_IN_CHANGE),
}

class TurnCapture(Protocol):
    async def say(self, text: str) -> None:
        """Say `text` to the caller and return once it has played. Never interruptible."""
        ...

    async def ask(self, prompt: str, *, timeout_s: float, secret: bool = False) -> str | None:
        """Speak `prompt` with interruptions off, then return the text of the caller's next
        finished turn, or None if `timeout_s` seconds pass first or the caller leaves.

        The LLM never answers that turn, but it is still forwarded to the detector, which scores
        it. A `secret` prompt, as every challenge phrase is, is kept out of the chat context, so
        the LLM can never repeat it.
        """
        ...


class CallbackNoAnswer(Exception):
    """The callback got no answer, and the caller is no longer on the line.

    A step-up that fails returns False, and the gate asks the detector again. Here there is no one
    left to ask: the caller was taken off the line to be called back, so the factor the detector
    named next (a code to the number on file, another call) would reach the real user for
    nothing. Raising makes the gate take its deny path at once, which holds the vendor and hangs
    up, and says nothing to an empty line because the call is closing. The gate logs the
    traceback of a step-up that raises, so the message is fixed text.
    """


class CallbackCalleeHungUp(CallbackNoAnswer):
    """Whoever answered the callback hung up during the confirm question (R79).

    The same case as CallbackNoAnswer, one step on: the caller is off the line, and now the callee
    has gone too. It is raised after the detector has been told DENIED, and the gate takes its
    deny path on the spot: the vendor is held, nothing is said to a line nobody is on, and the
    call ends. Without it the gate would ask the detector again, and stop at the start of that
    round without holding anything. The message is fixed text, since the gate logs the traceback.
    """


class CallbackLeg(Protocol):
    async def dial_and_link(self, phone_e164: str) -> str | None:
        """Hang up on the caller, dial `phone_e164` and link the session to whoever answers.

        Returns the identity of the participant who answered, who is now the one the session
        listens to, or None if nobody answered. StepUps only ever passes the number on file.

        A leg that has taken the caller off the line marks the call as ending (state.closing)
        before it returns None, as nobody is left on the call. One that took no one off the line,
        for lack of a trunk or a number, leaves state.closing alone.
        """
        ...


class StepUps:
    """Runs the factors a step_up decision asks for, on one call.

    `capture` speaks to the caller and takes their answers. `callback_leg` hangs up on the caller,
    dials the number on file and links the session to whoever answers. `poll_interval_s` and
    `poll_timeout_s` say how often, and for how long, a challenge is polled once the caller has
    answered. `clock` and `sleep` stand in for time.monotonic and asyncio.sleep, so a test needs no
    real wait.
    """

    def __init__(
        self,
        detector: DetectorClient,
        capture: TurnCapture,
        state: CallState,
        *,
        callback_leg: CallbackLeg | None = None,
        poll_interval_s: float = 0.25,
        poll_timeout_s: float = 8.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._detector = detector
        self._capture = capture
        self._state = state
        self._callback_leg = callback_leg
        self._poll_interval_s = poll_interval_s
        self._poll_timeout_s = poll_timeout_s
        self._clock = clock
        self._sleep = sleep

    async def run(self, factor: Factor, req: ActionRequest, decision: Decision) -> bool:
        """Run one factor the detector asked for, for `req`. True if the caller passed it."""
        match factor:
            case Factor.VOICE_CHALLENGE:
                return await self.voice_challenge(req)
            case Factor.SMS_OTP:
                return await self.sms_otp()
            case Factor.PROBE:
                return await self.probe()
            case Factor.CALLBACK_ON_FILE:
                return await self.callback(req)
        assert_never(factor)

    async def voice_challenge(self, req: ActionRequest) -> bool:
        """Make the caller repeat what the detector issues, and say whether the detector passed it.

        The challenge is digits for a tier 1 request and a phrase for tier 2. The prompt is a
        lead-in and the detector's words exactly as issued, asked in secret. No answer is False,
        and nothing is polled for it.
        """
        session_id = self._session_id()
        try:
            kind, lead_in = _CHALLENGE_BY_TIER[req.tier]
        except KeyError:
            raise ValueError(f"a tier {req.tier} action takes no voice challenge") from None
        issued = await self._detector.create_challenge(session_id, kind)
        answer = await self._capture.ask(
            f"{lead_in} {issued.say}", timeout_s=CHALLENGE_ANSWER_TIMEOUT_S, secret=True
        )
        if answer is None:
            logger.info("voice challenge %s: no answer", issued.challenge_id)
            return False
        status = await self.poll_challenge(issued.challenge_id)
        logger.info(
            "voice challenge %s: %s (phrase_match=%s)",
            issued.challenge_id,
            status.status,
            status.phrase_match,
        )
        return status.status is ChallengeState.PASSED

    async def sms_otp(self) -> bool:
        """Have the detector text a code to the number on file, and take it back by voice.

        The caller gets OTP_TRIES tries. An answer that isn't six digits once spoken digits are
        read out of it is asked again without being checked, and uses a try. So does no answer
        within the wait. But if the caller has gone (state.closing), no answer fails it at once.
        Nothing is said after the last try.
        """
        session_id = self._session_id()
        sent = await self._detector.send_otp(session_id)
        prompt = prompts.OTP_PROMPT.format(last2=sent.sent_to_last_four[-2:])
        for attempt in range(1, OTP_TRIES + 1):
            answer = await self._capture.ask(prompt, timeout_s=OTP_ANSWER_TIMEOUT_S)
            if answer is None:
                if self._state.closing:
                    logger.info("sms otp try %d: the caller has gone", attempt)
                    return False
                # Nothing heard in the whole wait: it uses a try, like an answer that isn't a code.
                logger.info("sms otp try %d: no answer", attempt)
                prompt = prompts.OTP_SIX_DIGITS
                continue
            code = parse_spoken_digits(answer)
            if len(code) != OTP_DIGITS:
                logger.info(
                    "sms otp try %d: heard %d digits, not %d", attempt, len(code), OTP_DIGITS
                )
                prompt = prompts.OTP_SIX_DIGITS
                continue
            if (await self._detector.check_otp(session_id, code)).verified:
                logger.info("sms otp verified on try %d", attempt)
                return True
            logger.info("sms otp try %d: wrong code", attempt)
            prompt = prompts.OTP_RETRY
        return False

    async def probe(self) -> bool:
        """Ask one off-script question. Always True: the detector reads when and how the caller
        answers, so the agent never judges the answer, or whether there was one."""
        await self._capture.ask(prompts.PROBE_QUESTION, timeout_s=PROBE_ANSWER_TIMEOUT_S)
        return True

    async def callback(self, req: ActionRequest) -> bool:
        """Call the claimed user back on the number on file, and ask whoever answers about `req`.

        True only for a plain yes to the confirm question, as parse_yes_no reads it. The number is
        the one the detector's directory has for the claimed user (get_user), never one heard on
        the call. CALLBACK_LINE tells the caller what is about to happen, then the leg hangs up on
        them and dials. If someone answers, the detector is pointed at them first (update_caller),
        then the disclosure is said again, as this is a new call, and the confirm question names
        the pending action.

        The detector is told how it went, once: NO_ANSWER if nobody answered, CONFIRMED for a yes
        and DENIED for anything else, silence included. Without a leg, on a browser session (V3),
        or on a call that is ending (state.closing), it is False at once, with nothing said or
        sent. The call is checked again once the number is known, so a caller who hung up
        meanwhile isn't called back either. An exception propagates unreported, as in the other
        step-ups, and the gate denies.

        If nobody answers and the call is then ending, the leg having taken the caller off the
        line, it raises CallbackNoAnswer after the report instead of returning False: the gate
        would only ask the detector again, with nobody left on the line. So it does when whoever
        answered hangs up on the confirm question (the answer is None and the call is ending): it
        reports DENIED, as it does for any silence, and then raises CallbackCalleeHungUp. A callee
        who stays on the line and says nothing, or who answers and then hangs up, is still a plain
        False.

        state.callback_in_progress goes up just before CALLBACK_LINE, so a caller who hangs up on
        hearing it doesn't end the call before the dial. It is down again on every way out of
        here, whatever it was before. LiveKitCallbackLeg lowers it too, once its dial is over.
        """
        try:
            if self._state.closing:
                # The caller has gone: there is nobody to tell, and no call to come back to.
                logger.info("a callback was asked for on a call that is ending, so it fails")
                return False
            if self._callback_leg is None:
                logger.warning("a callback was asked for and none is set up, so it fails")
                return False
            if self._state.channel is Channel.BROWSER:
                # V3: a browser session has no phone leg to hang up. The toolbox already refuses a
                # browser tier 2 request, so this is the backstop.
                logger.warning("a callback was asked for on a browser session, so it fails")
                return False
            session_id = self._session_id()
            claimed = self._state.claimed_user
            if claimed is None:
                # As with the session: the gate runs no step-up before the caller is identified.
                raise RuntimeError("a callback needs the claimed user, and there is none yet")
            user = await self._detector.get_user(claimed.user_id)
            if user.user_id != claimed.user_id:
                # Never someone else's number, whatever the directory answers.
                raise DetectorError(
                    f"GET /v1/users/{claimed.user_id}: the answer is for another user"
                )
            if self._state.closing:
                # The caller hung up while the number was looked up, before hearing anything of the
                # callback: an ordinary hangup, so the call is over.
                logger.info("the caller hung up before the callback line, so it fails")
                return False
            # Up before the caller hears that they'll be hung up on, so hanging up then is planned.
            self._state.callback_in_progress = True
            await self._capture.say(prompts.CALLBACK_LINE)
            callee = await self._callback_leg.dial_and_link(user.phone_e164)
            if callee is None:
                await self._report_callback(session_id, CallbackResult.NO_ANSWER)
                if self._state.closing:
                    raise CallbackNoAnswer("the callback got no answer, and the caller has gone")
                return False
            await self._detector.update_caller(
                session_id, CallerUpdate(caller_identity=callee, channel=Channel.PHONE)
            )
            await self._capture.say(prompts.DISCLOSURE)
            answer = await self._capture.ask(
                prompts.CALLBACK_CONFIRM.format(pending_action=req.context().pending_action),
                timeout_s=CALLBACK_CONFIRM_TIMEOUT_S,
            )
            if answer is not None and parse_yes_no(answer) is True:
                await self._report_callback(session_id, CallbackResult.CONFIRMED)
                return True
            await self._report_callback(session_id, CallbackResult.DENIED)
            if answer is None and self._state.closing:
                # Silence because the callee hung up, not because they said nothing: the worker has
                # marked the call as ending. Nobody is left to ask again.
                raise CallbackCalleeHungUp("the callee hung up on the confirm question")
            return False
        finally:
            self._state.callback_in_progress = False

    async def poll_challenge(self, challenge_id: str) -> ChallengeStatus:
        """Ask the detector how a challenge went until it has an answer, for up to poll_timeout_s.

        Polls at once, then every poll_interval_s, and once more as the timeout ends. Returns the
        status it stopped on, which is still pending if the detector never settled.

        A poll that fails with a DetectorError counts as still pending: one blip costs one poll,
        and a detector that stays down costs the whole wait and ends pending, which is not passed.
        Any other exception propagates.
        """
        session_id = self._session_id()
        deadline = self._clock() + self._poll_timeout_s
        while True:
            try:
                status = await self._detector.get_challenge(session_id, challenge_id)
            except DetectorError as exc:
                # Its type only, never the message, which names the request.
                logger.warning(
                    "polling challenge %s failed with %s, so it counts as still pending",
                    challenge_id,
                    type(exc).__name__,
                )
                status = ChallengeStatus(status=ChallengeState.PENDING)
            remaining = deadline - self._clock()
            if status.status is not ChallengeState.PENDING or remaining <= 0:
                return status
            await self._sleep(min(self._poll_interval_s, remaining))

    def _session_id(self) -> str:
        session_id = self._state.session_id
        if session_id is None:
            # The gate runs no step-up before the caller is identified, so this is a bug. Raising
            # fails closed: the gate turns an exception into a deny.
            raise RuntimeError("a step-up needs the detector's session, and there is none yet")
        return session_id

    async def _report_callback(self, session_id: str, result: CallbackResult) -> None:
        logger.info("callback to the number on file: %s", result)
        await self._detector.report_factor(session_id, FactorReport(result=result))
