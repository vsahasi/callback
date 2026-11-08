"""The callback leg: hang up on the caller and call the number on file, in the same session.

LiveKitCallbackLeg is the CallbackLeg (stepups.py) on a LiveKit call. One AgentSession serves both
legs. dial_and_link does this, in order, as LiveKit's examples/telephony/amd.py does (research
notes, livekit_tasks_testing_callback.md Q3):
1. Link the session to a fresh identity for the callee, before the dial. The session stops
   listening to the caller at once, and RoomIO links the callee as they join.
2. Remove the caller from the room, which ends their phone leg. If LiveKit says not_found, they
   had already left, and the dial goes ahead. So it does after any other failure of the removal
   once they have left the room. While they are still in it, a failed removal stops the callback
   with CallbackLegError: they could hear or coach the real user during the confirm question.
3. Dial the number through the outbound trunk and wait for the answer. Busy, declined, no answer
   and a trunk failure (api.SipCallError), a dial that outlasts its timeout, and a callee who is
   not in the room when the dial returns are all no answer. By then the caller is off the line,
   removed or already gone, and nobody has taken their place, so the call is over: the leg marks
   it as ending (state.closing) before it returns None, and StepUps.callback turns that into a
   deny that says nothing to an empty line. A dial that LiveKit refuses, or whose request fails
   on the way to its API (an aiohttp.ClientError), ends the call the same way. A caller who could
   not be removed is still on the line, and the call goes on.
4. On an answer, clear the user turn and the caller's leftover transcript segments
   (state.pending_segments), so nothing heard before the switch can count as the callee's, then
   point the call state at the callee.

state.callback_in_progress is set through all of it, so the worker can tell the caller's planned
exit from a hangup. StepUps.callback raises it sooner, before CALLBACK_LINE, so it is already up
as the caller hears they'll be called back. Krisp noise cancellation stays off, so the detector
hears the raw line. A log line holds at most the last four digits of a number, and never the
caller's identity: LiveKit names an inbound SIP participant after their phone number.
"""

import asyncio
import logging
import secrets

import aiohttp
from callback_shared.contract import Channel
from livekit import api
from livekit.agents import AgentSession, JobContext

from callback_agent.config import Settings
from callback_agent.state import CallState

logger = logging.getLogger(__name__)

# How long the dial waits for an answer. livekit-api pins the ring window to 30 s when the request
# leaves it unset, and the request has to outlast the ring window.
DIAL_TIMEOUT_S = 45.0
CALLEE_PREFIX = "callback-"


class CallbackLegError(Exception):
    """The callback can't go ahead: LiveKit refused the dial for a reason other than nobody
    answering, such as a bad trunk id, or the request for it failed on the way, or the caller
    couldn't be removed and is still in the room.

    The message names the failure's type, and for a LiveKit API error its code and HTTP status,
    never its message, which can quote the number dialed or the caller's identity.
    """


class LiveKitCallbackLeg:
    """Dials from the job's room and relinks the call's session, by the steps in the module
    docstring. One instance serves a whole call."""

    def __init__(
        self, ctx: JobContext, session: AgentSession, state: CallState, settings: Settings
    ) -> None:
        self._ctx = ctx
        self._session = session
        self._state = state
        self._settings = settings

    async def dial_and_link(self, phone_e164: str) -> str | None:
        """Hang up on the caller, dial `phone_e164` and link the session to whoever answers.

        Returns the callee's identity, `callback-` and 8 hex digits, once they answer, or None if
        nobody did. The caller is off the line by then, so it marks the call as ending
        (state.closing) first. It is also None, with nothing touched and the caller still on the
        line, when the outbound trunk id or the Telnyx number is not set: nobody was removed, and
        state.closing is left alone.

        Raises CallbackLegError when LiveKit refuses the dial or the request for it fails on the
        way, which marks the call as ending too, or when the caller can't be removed and is still
        in the room, which does not. Any other error, and cancellation, propagates as it is.
        """
        trunk_id = self._settings.sip_outbound_trunk_id
        from_number = self._settings.telnyx_number
        if not trunk_id or not from_number:
            missing = [
                name
                for name, value in (
                    ("LIVEKIT_SIP_OUTBOUND_TRUNK_ID", trunk_id),
                    ("TELNYX_NUMBER", from_number),
                )
                if not value
            ]
            logger.warning("callback: no number is dialed without %s", " and ".join(missing))
            return None
        self._state.callback_in_progress = True
        try:
            # Fresh each time: LiveKit Cloud refuses a removed identity's token for 10 minutes.
            callee = CALLEE_PREFIX + secrets.token_hex(4)
            room = self._ctx.room
            self._session.room_io.set_participant(callee)
            await self._remove_caller(room.name)
            logger.info(
                "callback: dialing the number on file ending %s as %s", phone_e164[-4:], callee
            )
            try:
                await self._ctx.api.sip.create_sip_participant(
                    api.CreateSIPParticipantRequest(
                        room_name=room.name,
                        sip_trunk_id=trunk_id,
                        sip_call_to=phone_e164,
                        sip_number=from_number,
                        participant_identity=callee,
                        wait_until_answered=True,
                    ),
                    timeout=DIAL_TIMEOUT_S,
                )
            except api.SipCallError as exc:
                # The SIP code only: the message and the rest of the metadata can quote the number.
                logger.info("callback: nobody answered (SIP %s)", exc.sip_status_code)
                self._call_is_over()
                return None
            except asyncio.TimeoutError:
                logger.info("callback: nobody answered within %gs", DIAL_TIMEOUT_S)
                self._call_is_over()
                return None
            except api.ServerError as exc:
                self._call_is_over()
                # from None: the gate logs the traceback, and LiveKit's message stays out of it.
                raise CallbackLegError(f"LiveKit refused the dial: {_failure(exc)}") from None
            except aiohttp.ClientError as exc:
                # The request died on its way to LiveKit's API (a reset connection, say), so nobody
                # can say whether the phone rang. The caller is off the line all the same: it is
                # handled like a dial that was refused, and from None for the same reason.
                self._call_is_over()
                failure = _failure(exc)
                raise CallbackLegError(f"the dial request to LiveKit failed: {failure}") from None
            if callee not in room.remote_participants:
                # The call can end just before wait_until_answered returns.
                logger.info("callback: %s was not in the room when the dial returned", callee)
                self._call_is_over()
                return None
            self._session.clear_user_turn()
            # The caller's leftover FINALs go too, or the callee's first message could claim them.
            self._state.pending_segments.clear()
            self._state.caller_identity = callee
            self._state.channel = Channel.PHONE
            logger.info("callback: %s answered", callee)
            return callee
        finally:
            self._state.callback_in_progress = False

    def _call_is_over(self) -> None:
        """Mark the call as ending: the caller is off the line, and nobody took their place.

        Called only after _remove_caller has returned, which it does only once the caller is gone
        (removed, not found, or no longer in the room). With nobody to speak to or to ask again,
        the gate says nothing more, and StepUps.callback raises instead of returning False: a
        False would have the gate ask the detector again, and the next factor it named would reach
        the real user for nothing.
        """
        self._state.closing = True

    async def _remove_caller(self, room_name: str) -> None:
        """Remove the caller from the room, which ends their phone leg.

        not_found means the caller had already left, and the dial goes ahead. So it does after any
        other failure, once they are no longer in the room. A failure with the caller still in the
        room raises CallbackLegError, and nobody is dialed: they could hear the real user, or coach
        them, during the confirm question. Only the LiveKit API's own errors are handled here.
        """
        caller = self._state.caller_identity
        try:
            await self._ctx.api.room.remove_participant(
                api.RoomParticipantIdentity(room=room_name, identity=caller)
            )
        except (api.ServerError, aiohttp.ClientError, TimeoutError) as exc:
            if isinstance(exc, api.ServerError) and exc.code == api.ServerErrorCode.NOT_FOUND:
                logger.info("callback: the caller had already left the room")
                return
            if caller in self._ctx.room.remote_participants:
                raise CallbackLegError(
                    f"could not remove the caller ({_failure(exc)}), and they are still in the "
                    "room, so nobody is dialed"
                ) from None
            logger.warning(
                "callback: removing the caller failed with %s, but they have left, so the dial "
                "goes ahead",
                _failure(exc),
            )


def _failure(exc: BaseException) -> str:
    """An error by its type, and for a LiveKit API error by its code and HTTP status too. Never by
    its message, which can quote a number or the caller's identity."""
    if isinstance(exc, api.ServerError):
        return f"{type(exc).__name__} {exc.code}, HTTP {exc.status}"
    return type(exc).__name__
