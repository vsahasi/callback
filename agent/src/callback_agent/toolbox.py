"""What each LLM tool does, with no LiveKit in it. agents.py wraps every method in a @function_tool.

A method returns the text the LLM voices to the caller, and never raises: whatever goes wrong inside
becomes a reply (cancellation is the one exception, as it must be). Every action goes through
Gate.run, which asks the detector, runs any step-up and only then calls Ramp. This module reads from
Ramp in two places and writes nowhere: find_vendor turns a spoken name into a vendor, and list_bills
learns the ids of the bills the caller was just told about. tests/test_structure.py pins that no
module outside the gate and actions calls a Ramp write. Voice enrollment is the one thing that
talks to the caller outside the gate, and it runs inside Gate.exclusive(), so it never overlaps a
gated request.

Logs say what happened and never what the caller said: a name, a bank number and a reason can all be
personal data. A failure is logged by its type and place, not its message.
"""

import asyncio
import functools
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any, assert_never

from callback_shared.contract import Channel, SessionCreate

from callback_agent import prompts
from callback_agent.actions import ActionRequest
from callback_agent.detector import DetectorClient, DetectorError
from callback_agent.enrollment import Enrollment, EnrollmentOutcome
from callback_agent.gate import Gate
from callback_agent.ramp import Bill, RampClient, Vendor
from callback_agent.safelog import log_failure
from callback_agent.state import CallState
from callback_agent.turns import Forwarder

logger = logging.getLogger(__name__)

DEFAULT_HOLD_REASON = "Placed on hold by the caller on a Callback phone call"
MAX_REASON_CHARS = 200  # the reason is the caller's words, and goes to Ramp and the detector
NUMBER_MASK = "[number]"
_NUMBER_SEPARATORS = re.compile(r"[\s-]")  # a spoken number can come back as "021 000-021"
# Five or more digits in a row, even with up to three characters that are neither letters nor
# digits between each two (a space, a dot, a comma, a dash, ", " or " - "): a number read out in
# groups or one digit at a time. Four stay, since the last four of an account are not a secret. A
# punctuated amount such as $1,204.50 has five digits too, and is masked with them.
_LONG_NUMBER = re.compile(r"\d(?:[\W_]{0,3}\d){4,}")


def _always_a_reply(method: Callable[..., Awaitable[str]]) -> Callable[..., Awaitable[str]]:
    """Make a tool method return a reply whatever goes wrong inside it.

    Only Exception is caught, so cancellation still gets through. The failure is logged by type and
    place: its message could quote a bank number.
    """

    @functools.wraps(method)
    async def guarded(self: "Toolbox", *args: Any, **kwargs: Any) -> str:
        try:
            return await method(self, *args, **kwargs)
        except Exception as exc:
            log_failure(logger, f"run {method.__name__}", exc)
            return prompts.TOOL_FAILED

    return guarded


class Toolbox:
    """The tools of one call. Build one per call, over that call's state, gate and clients.

    `forwarder` is None when no detector is configured, and then identification has nothing to
    flush. In the worker it is always there, because the detector settings are required.
    `enrollment` is None on a call that can't enroll a voice, and then start_voice_enrollment says
    it isn't available. The worker always passes one.
    """

    def __init__(
        self,
        *,
        state: CallState,
        gate: Gate,
        ramp: RampClient,
        detector: DetectorClient,
        forwarder: Forwarder | None,
        enrollment: Enrollment | None = None,
    ) -> None:
        self._state = state
        self._gate = gate
        self._ramp = ramp
        self._detector = detector
        self._forwarder = forwarder
        self._enrollment = enrollment
        # LiveKit runs the tool calls of one reply side by side, so two identify_caller calls can be
        # in flight at once. The lock makes the second see the first one's identity.
        self._identify_lock = asyncio.Lock()
        # One enrollment per call (R81): set when an enrollment is started, and never cleared.
        self._enrollment_ran = False

    # --- who is calling ---

    @_always_a_reply
    async def identify_caller(self, full_name: str) -> str:
        """Look the caller up by name and open the detector's session for them.

        Exactly one match identifies the caller. No match and several matches answer alike and
        create nothing. Once someone is identified the identity is locked for the call, and the
        next name is not even looked up.
        """
        async with self._identify_lock:
            state = self._state
            if state.locked or state.closing:
                return prompts.DENY_NOT_HELD  # the call is over; nobody new is identified on it
            if state.claimed_user is not None:
                logger.info("identify_caller: refused, the caller is already identified")
                return prompts.IDENTITY_LOCKED
            name = full_name.strip()
            if not name:
                return prompts.NOT_IDENTIFIED
            try:
                users = await self._detector.find_users(name)
            except DetectorError as exc:
                return self._detector_failed(exc)
            if len(users) != 1:
                logger.info("identify_caller: %d users match, so no one is identified", len(users))
                return prompts.NOT_IDENTIFIED
            user = users[0]
            try:
                created = await self._detector.create_session(
                    SessionCreate(
                        room=state.room_name,
                        caller_identity=state.caller_identity,
                        claimed_user_id=user.user_id,
                        channel=state.channel,
                    )
                )
            except DetectorError as exc:
                return self._detector_failed(exc)
            # One step with no await in it, so the state and the forwarder always agree: the turns
            # buffered since the call began flush now, and the gate sees a session.
            state.claimed_user = user
            state.session_id = created.session_id
            state.detector_down = False
            if self._forwarder is not None:
                self._forwarder.attach(created.session_id)
            logger.info("identify_caller: identified, session %s", created.session_id)
            return prompts.IDENTIFIED.format(name=user.display_name)

    # --- reads ---

    @_always_a_reply
    async def list_vendors(self) -> str:
        return await self._ask(ActionRequest.list_vendors())

    @_always_a_reply
    async def get_vendor_bank_account(self, vendor_name: str) -> str:
        vendor = await self._active_vendor(vendor_name)
        if isinstance(vendor, str):
            return vendor
        return await self._ask(ActionRequest.list_vendor_accounts(vendor.id, _name_of(vendor)))

    @_always_a_reply
    async def list_open_bills(self, vendor_name: str) -> str:
        vendor = await self._active_vendor(vendor_name)
        if isinstance(vendor, str):
            return vendor
        result = await self._gate.run(ActionRequest.list_bills(vendor.id, _name_of(vendor)))
        if result.status == "done":
            await self._remember_bills(vendor)
        return result.message

    # --- holds ---

    @_always_a_reply
    async def hold_vendor(self, vendor_name: str, reason: str) -> str:
        vendor = await self._active_vendor(vendor_name)
        if isinstance(vendor, str):
            return vendor
        request = ActionRequest.hold_vendor(vendor.id, _name_of(vendor), _reason(reason))
        return await self._ask(request)

    @_always_a_reply
    async def hold_bill(self, vendor_name: str, invoice_number: str, reason: str) -> str:
        """Hold the vendor's open bill(s) with this invoice number.

        A hold needs no verification, but listing bills does, so the caller names the bill the way
        a caller knows it, by vendor and invoice number, and the toolbox finds it with reads of its
        own before the gate (as it does the vendor). That is why this does not use bills_seen.
        """
        vendor = await self._active_vendor(vendor_name)
        if isinstance(vendor, str):
            return vendor
        bills = await self._open_bills_numbered(vendor, invoice_number)
        if isinstance(bills, str):
            return bills
        reason = _reason(reason)
        # A resubmitted copy of an invoice is the same invoice to the caller: none is left to pay.
        replies = [
            await self._ask(ActionRequest.hold_bill(bill.id, bill.vendor.id, reason))
            for bill in bills
        ]
        return " ".join(replies)

    # --- money moves: tier 2, phone only (V3) ---

    @_always_a_reply
    async def change_vendor_bank_account(
        self, vendor_name: str, routing_number: str, account_number: str
    ) -> str:
        if not self._on_phone():
            return prompts.TIER2_PHONE_ONLY
        vendor = await self._active_vendor(vendor_name)
        if isinstance(vendor, str):
            return vendor
        try:
            request = ActionRequest.update_vendor_bank_account(
                vendor.id, _name_of(vendor), _digits(routing_number), _digits(account_number)
            )
        except ValueError:
            # Its message never repeats a number, and neither does this reply or the log.
            logger.info("change_vendor_bank_account: the numbers did not check out")
            return prompts.BANK_NUMBERS_UNCLEAR
        return await self._ask(request)

    @_always_a_reply
    async def release_vendor_hold(self, vendor_name: str) -> str:
        if not self._on_phone():
            return prompts.TIER2_PHONE_ONLY
        vendor = await self._active_vendor(vendor_name)
        if isinstance(vendor, str):
            return vendor
        return await self._ask(ActionRequest.release_vendor_hold(vendor.id, _name_of(vendor)))

    @_always_a_reply
    async def release_bill_hold(self, bill_id: str) -> str:
        if not self._on_phone():
            return prompts.TIER2_PHONE_ONLY
        bill_id = bill_id.strip()
        vendor_id = self._state.bills_seen.get(bill_id)
        if vendor_id is None:
            return prompts.BILL_NOT_LISTED
        return await self._ask(ActionRequest.release_bill_hold(bill_id, vendor_id))

    # --- voice enrollment ---

    @_always_a_reply
    async def start_voice_enrollment(self) -> str:
        """Enroll the caller's voice: an SMS code, spoken consent, then the enrollment prompts.

        The Enrollment speaks to the caller itself, from start to finish, so what this returns
        only tells the LLM how it ended. It runs inside the gate's exclusive lock, so it never
        overlaps a gated request. Nothing the LLM passes can steer it: the tool takes no
        arguments, a code only goes to the number on file, and consent is the caller's own
        spoken answer. Typed input cannot stand in for it: the chat topic and the session host
        are both closed (build_room_options, and the session start in worker.py).

        A call that is locked (a deny happened) or closing is not enrolled: the Enrollment would
        text the real person a code for a call the gate has given up on. That is checked before
        the lock is taken, and again once it is won, because a gated request that was running
        can deny, or the caller can hang up, while this waits for its turn.

        A call is enrolled once (R81), whatever the first run came to: an error or a cancellation
        counts, and so does an enrollment that failed or was declined. Each run texts the number
        on file a code, and Twilio Verify caps the sends. A second start is turned away with
        ENROLL_TOOL_ALREADY_RAN before it takes the lock or asks the detector anything. The one
        thing that does not use it up is a start before the caller is identified: that sends and
        asks nothing, and the LLM is sent to identify the caller and start again. The check is
        made again under the lock, as it is for a call that is over, because the start that was
        running when this one was called may be what used the enrollment up.
        """
        enrollment = self._enrollment
        if enrollment is None:
            logger.warning("start_voice_enrollment: refused, this call was built without one")
            return prompts.ENROLL_NOT_NOW
        if self._call_is_over():
            logger.info("start_voice_enrollment: refused, the call is locked or ending")
            return prompts.ENROLL_NOT_NOW
        if self._enrollment_ran:
            logger.info("start_voice_enrollment: refused, an enrollment already ran on this call")
            return prompts.ENROLL_TOOL_ALREADY_RAN
        async with self._gate.exclusive():
            if self._call_is_over():
                logger.info("start_voice_enrollment: refused, the call ended while it waited")
                return prompts.ENROLL_NOT_NOW
            if self._enrollment_ran:
                logger.info("start_voice_enrollment: refused, one ran while it waited")
                return prompts.ENROLL_TOOL_ALREADY_RAN
            self._enrollment_ran = True  # before the run, so that however it ends this is the one
            outcome = await enrollment.run()
            if outcome == "not_identified":
                # It sent and asked nothing, so the caller still has their enrollment.
                self._enrollment_ran = False
        logger.info("start_voice_enrollment: %s", outcome)
        return self._enrollment_reply(outcome)

    # --- plumbing ---

    def _call_is_over(self) -> bool:
        """A deny has locked the call, or it is ending: nothing new may start on it."""
        return self._state.locked or self._state.closing

    def _enrollment_reply(self, outcome: EnrollmentOutcome) -> str:
        """What the LLM is told about an enrollment that has already spoken to the caller."""
        match outcome:
            case "enrolled":
                return prompts.ENROLL_TOOL_DONE
            case "declined":
                return prompts.ENROLL_TOOL_DECLINED
            case "failed":
                return prompts.ENROLL_TOOL_FAILED
            case "not_identified":
                if self._state.detector_down:
                    return prompts.IDENTIFY_UNAVAILABLE
                return prompts.IDENTIFY_FIRST
        assert_never(outcome)

    def _detector_failed(self, exc: DetectorError) -> str:
        """The detector failed on the way to a session. The gate fails closed until there is one."""
        self._state.detector_down = True
        logger.warning("identify_caller: the detector failed: %s", exc)
        return prompts.IDENTIFY_UNAVAILABLE

    def _on_phone(self) -> bool:
        """Tier 2 needs the callback, which only a phone call has (V3). Anything else is not one."""
        return self._state.channel is Channel.PHONE

    async def _ask(self, request: ActionRequest) -> str:
        """Hand `request` to the gate, the one way to Ramp, and return what the LLM should voice."""
        return (await self._gate.run(request)).message

    async def _active_vendor(self, vendor_name: str) -> Vendor | str:
        """The active vendor the caller means, or the reply to give when there is none.

        A vendor that isn't there and one that isn't active get the same reply, and so does every
        caller: this runs before the gate, so the answer cannot depend on who is verified.
        """
        try:
            vendor = await self._ramp.find_vendor(vendor_name)
        except Exception as exc:
            log_failure(logger, "look up a vendor in Ramp", exc)
            return prompts.VENDOR_LOOKUP_FAILED
        if vendor is None or not vendor.is_active:
            logger.info("no active vendor matched the name")
            return prompts.NO_ACTIVE_VENDOR
        return vendor

    async def _open_bills_numbered(self, vendor: Vendor, invoice_number: str) -> list[Bill] | str:
        """The vendor's open bills with this invoice number, or the reply to give if there are none.

        Like _active_vendor this reads before the gate, so the reply cannot depend on who is
        verified: every number that matches nothing gets the same one. The match is exact once the
        spaces around the number are gone and case is ignored. A bill must also be the named
        vendor's own, so that a filter Ramp ignored could not hold another vendor's invoice.
        """
        wanted = invoice_number.strip().casefold()
        if not wanted:
            return prompts.NO_SUCH_BILL
        try:
            bills = await self._ramp.list_bills(vendor_id=vendor.id, payment_status="OPEN")
        except Exception as exc:
            log_failure(logger, "look up a vendor's open bills in Ramp", exc)
            return prompts.BILL_LOOKUP_FAILED
        found = [
            bill
            for bill in bills
            if bill.vendor.id == vendor.id
            and (bill.invoice_number or "").strip().casefold() == wanted
        ]
        if not found:
            logger.info("no open bill matched the invoice number")
            return prompts.NO_SUCH_BILL
        return found

    async def _remember_bills(self, vendor: Vendor) -> None:
        """Note the ids of the vendor's open bills, so that release_bill_hold accepts them.

        A GateResult carries text only, so this reads the list again, after the gate has allowed
        it. If that read fails, the caller was still told the bills, and the LLM will be sent back
        to list them before it can hold one.
        """
        try:
            bills = await self._ramp.list_bills(vendor_id=vendor.id)
        except Exception as exc:
            log_failure(logger, "read the open bills to remember their ids", exc)
            return
        for bill in bills:
            self._state.bills_seen[bill.id] = bill.vendor.id


def _name_of(vendor: Vendor) -> str:
    """The name the caller hears: the display name, or the name when there is none."""
    return (vendor.display_name or "").strip() or vendor.name


def _reason(text: str) -> str:
    """The caller's reason for a hold: on one line, any long number masked, and not too long.

    It is free text that goes to Ramp and, in the params of the hold, to the detector. A caller who
    reports an attempt to change an account may read the numbers out, and a full bank number must
    reach neither. The mask comes before the cut: cut first, a number that starts just inside the
    limit would keep its first digits. A blank reason becomes a stock one.
    """
    masked = _LONG_NUMBER.sub(NUMBER_MASK, " ".join(text.split()))
    return masked[:MAX_REASON_CHARS] or DEFAULT_HOLD_REASON


def _digits(spoken: str) -> str:
    """A spoken number without the spaces and dashes it may come with. The rest is checked later."""
    return _NUMBER_SEPARATORS.sub("", spoken)
