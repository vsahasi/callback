"""Voiceprint enrollment: an SMS code, spoken consent, then the enrollment prompts.

Enrollment records the caller's voice as the person they claim to be, so each step guards the next
and the detector is told nothing about the enrollment until the caller has agreed to it:
- The caller must be identified. If not, nothing is said and run() returns "not_identified", which
  the Toolbox answers with IDENTIFY_FIRST. That is a fourth result beside the three the Toolbox
  puts into words for the LLM.
- The SMS code must pass first (V4), or anyone could enroll their own voice as the CFO.
- Then the consent text is asked, and only a plain yes (spoken.parse_yes_no) goes on. A no, a
  question, an unclear answer and silence all decline. After a yes the detector gets the caller's
  answer word for word, with the consent version and the time.
- Then come the prompts the detector asks for, each tried up to 1 + retries_per_prompt times. A
  prompt that fails every try counts as failed, and one more failed prompt than max_failed_prompts
  aborts the enrollment, which is then never completed.
- The detector says at the end whether the caller is enrolled.
- If the call is closing (the caller hung up), the prompts stop: before each prompt and each try,
  and again before the "try that again" line and the abort line, run() returns "failed" at once,
  without making a challenge at the detector and without saying a word.

No model is involved, and the agent never judges what the caller says. It speaks the `say` of each
enroll challenge as the whole prompt, in secret like every challenge phrase, and
StepUps.poll_challenge says how the detector scored it. Only PASSED is a pass. Anything else is a
failed try, including the PENDING that poll_challenge ends on when the detector never settles, as
after a total outage. The one thing read of the answer is whether there is one (R59): if ask() gives
None, no turn ended within the wait, so the detector has nothing to score and a poll would only wait
out its window. That try is failed without a poll, and the prompt goes on to its retry as usual.

Nothing is caught here, as in StepUps: an error from the detector goes up to the caller, and
cannot read as enrolled. Logs hold ids, counts and states only, never the consent answer, the words
of a challenge or anything else the caller says.
"""

import logging
from time import time as wall_clock  # by name, so that a test can pin the consent's timestamp
from typing import Literal

from callback_shared.contract import ChallengeKind, ChallengeState, Consent, EnrollmentStart

from callback_agent import prompts
from callback_agent.detector import DetectorClient
from callback_agent.spoken import parse_yes_no
from callback_agent.state import CallState
from callback_agent.stepups import StepUps, TurnCapture

logger = logging.getLogger(__name__)

CONSENT_ANSWER_TIMEOUT_S = 30.0  # the consent text is long, and the caller may need a moment
PROMPT_ANSWER_TIMEOUT_S = 20.0  # how long the caller has to read an enrollment prompt

# What run() gives back. The Toolbox puts the first three into words for the LLM. The last means
# the caller is not identified yet, and nothing was said.
type EnrollmentOutcome = Literal["enrolled", "declined", "failed", "not_identified"]


class Enrollment:
    """Enrolls the caller's voice for the channel this call is on, by the rules above.

    `retries_per_prompt` is how many more tries a prompt gets after its first one fails, and
    `max_failed_prompts` is how many prompts may fail every try before the enrollment is aborted.
    The channel comes from `state`, so the caller enrolls phone and browser in a call of each. A
    run() talks to the caller from start to finish, so only one may go at a time.
    """

    def __init__(
        self,
        detector: DetectorClient,
        capture: TurnCapture,
        stepups: StepUps,
        state: CallState,
        *,
        retries_per_prompt: int = 2,
        max_failed_prompts: int = 2,
    ) -> None:
        self._detector = detector
        self._capture = capture
        self._stepups = stepups
        self._state = state
        self._retries_per_prompt = retries_per_prompt
        self._max_failed_prompts = max_failed_prompts

    async def run(self) -> EnrollmentOutcome:
        """Run the whole enrollment, speaking to the caller, and say how it ended."""
        session_id, user = self._state.session_id, self._state.claimed_user
        if session_id is None or user is None:
            logger.info("enrollment: the caller is not identified, so nothing is asked")
            return "not_identified"
        if not await self._stepups.sms_otp():
            logger.info("enrollment: the code did not verify, so nothing is enrolled")
            await self._capture.say(prompts.ENROLL_OTP_FAILED)
            return "failed"
        answer = await self._capture.ask(prompts.CONSENT_TEXT, timeout_s=CONSENT_ANSWER_TIMEOUT_S)
        consented_at_ms = round(wall_clock() * 1000)
        # None is silence, or a caller who has left, and parse_yes_no cannot read it. Only `is True`
        # is consent, as for any yes that lets something happen.
        if answer is None or parse_yes_no(answer) is not True:
            logger.info("enrollment: consent was not given, so nothing is enrolled")
            await self._capture.say(prompts.ENROLL_DECLINED)
            return "declined"
        started = await self._detector.start_enrollment(
            EnrollmentStart(
                session_id=session_id,
                user_id=user.user_id,
                channel=self._state.channel,
                consent=Consent(
                    version=prompts.CONSENT_VERSION, verbatim=answer, ts_ms=consented_at_ms
                ),
            )
        )
        enrollment_id = started.enrollment_id
        logger.info(
            "enrollment %s: started on %s with %d prompts",
            enrollment_id,
            self._state.channel,
            started.prompts_required,
        )
        failed_prompts = 0
        for number in range(1, started.prompts_required + 1):
            if self._state.closing:
                return self._stopped(enrollment_id)
            if await self._prompt_passes(session_id, enrollment_id, number):
                continue
            if self._state.closing:  # it failed because the caller left: no abort line for them
                return self._stopped(enrollment_id)
            failed_prompts += 1
            if failed_prompts > self._max_failed_prompts:
                logger.info(
                    "enrollment %s: %d prompts failed, so it is aborted",
                    enrollment_id,
                    failed_prompts,
                )
                await self._capture.say(prompts.ENROLL_ABORTED)
                return "failed"
        result = await self._detector.complete_enrollment(enrollment_id)
        logger.info(
            "enrollment %s: the detector says %s, with %d prompts passed",
            enrollment_id,
            result.status,
            result.prompts_passed,
        )
        if result.status == "enrolled":
            await self._capture.say(prompts.ENROLL_DONE)
            return "enrolled"
        await self._capture.say(prompts.ENROLL_ABORTED)
        return "failed"

    async def _prompt_passes(self, session_id: str, enrollment_id: str, number: int) -> bool:
        """Give one prompt its tries: True as soon as one passes, False if every try fails.

        Each try is a new challenge, and "try that again" is said before each retry. A try nobody
        answers is failed at once, with no poll. Once the call is closing no further try is made,
        and the prompt counts as failed: run() sees the call is closing and stops.
        """
        tries = 1 + self._retries_per_prompt
        for attempt in range(1, tries + 1):
            if self._state.closing:
                return False
            if attempt > 1:
                await self._capture.say(prompts.ENROLL_RETRY)
                if self._state.closing:  # they may have hung up while the line played
                    return False
            issued = await self._detector.create_challenge(session_id, ChallengeKind.ENROLL)
            # For an enroll challenge `say` is the whole prompt: no lead-in.
            self._state.answer_exempt = (
                True  # the answer is the digits, and the detector needs them
            )
            try:
                answer = await self._capture.ask(
                    issued.say, timeout_s=PROMPT_ANSWER_TIMEOUT_S, secret=True
                )
            finally:
                self._state.answer_exempt = False
            if answer is None:
                # Only None: an empty turn is still a turn, and the detector can score it.
                logger.info(
                    "enrollment %s: prompt %d try %d: no answer", enrollment_id, number, attempt
                )
                continue
            status = await self._stepups.poll_challenge(issued.challenge_id)
            logger.info(
                "enrollment %s: prompt %d try %d: %s (phrase_match=%s)",
                enrollment_id,
                number,
                attempt,
                status.status,
                status.phrase_match,
            )
            if status.status is ChallengeState.PASSED:
                return True
        return False

    def _stopped(self, enrollment_id: str) -> EnrollmentOutcome:
        """The caller has gone: end the enrollment as failed. It is never completed."""
        logger.info("enrollment %s: the call is ending, so it stops here", enrollment_id)
        return "failed"
