"""What a caller can ask Callback to do in Ramp, and the code that does it.

A tool builds an ActionRequest and hands it to the gate (gate.py), which authorizes it and then
calls execute() with that same request. Nothing here decides whether a request may run.

The detector never sees a full bank number: detector_params() turns a bank change's routing and
account numbers into their last four digits and a keyed fingerprint. A summary, what the caller
hears, names an account by its last four digits only.
"""

import hashlib
import hmac
import json
import logging
import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, assert_never

from callback_shared.contract import ACTION_TIER, Action, RampContext

from callback_agent import prompts
from callback_agent.ramp import Bill, HoldOutcome, RampClient, RampError, Vendor, VendorBankAccount

logger = logging.getLogger(__name__)

_ON_HOLD = "ON_HOLD"  # Bill.status_summary of a bill on hold
_ROUTING_NUMBER = re.compile(r"[0-9]{9}")
_ACCOUNT_NUMBER = re.compile(r"[0-9]{4,17}")
# Keyed: a bare hash, sent beside the last four digits, could be brute-forced back to the number.
_FINGERPRINT_KEY = secrets.token_bytes(32)  # one per process, and LiveKit gives each call its own
# The params each action takes, and no others: detector_params() sends every action's params but a
# bank change's as they are, so a param that isn't listed here could reach the detector unredacted.
_PARAM_KEYS: dict[Action, tuple[str, ...]] = {
    Action.HOLD_VENDOR: ("vendor_id", "vendor_name", "reason"),
    Action.HOLD_BILL: ("bill_id", "vendor_id", "reason"),
    Action.LIST_VENDORS: (),
    Action.LIST_VENDOR_ACCOUNTS: ("vendor_id", "vendor_name"),
    Action.LIST_BILLS: ("vendor_id", "vendor_name"),
    Action.UPDATE_VENDOR_BANK_ACCOUNT: (
        "vendor_id",
        "vendor_name",
        "routing_number",
        "account_number",
    ),
    Action.RELEASE_VENDOR_HOLD: ("vendor_id", "vendor_name"),
    Action.RELEASE_BILL_HOLD: ("bill_id", "vendor_id"),
}


@dataclass(frozen=True)
class HoldTarget:
    kind: Literal["vendor", "bill"]
    id: str


@dataclass(frozen=True)
class ActionOutcome:
    # "failed" means Ramp answered but the bill's status shows the change didn't take. Bill holds
    # and releases have no approval step, so that is a failure, never "pending_approval".
    status: Literal["done", "pending_approval", "failed"]
    summary: str  # what the agent may tell the caller


# Ramp answered a bill hold or release, but the bill's status says it didn't take.
_DID_NOT_TAKE = ActionOutcome("failed", prompts.RAMP_FAILED)


@dataclass(frozen=True)
class ActionRequest:
    """One Ramp action and its parameters.

    Build one with the constructor named after its action, which fixes the params keys. A request
    built directly is checked the same way: ValueError unless params are exactly the ones its
    action takes, all text, and, for a bank change, a routing number of 9 digits that pass the ABA
    checksum and an account number of 4 to 17 digits. An error never repeats a value. params is
    read-only, so the request the detector authorized is exactly the one that runs.
    """

    action: Action
    params: Mapping[str, str]

    def __post_init__(self) -> None:
        action = Action(self.action)
        params = dict(self.params)
        _check_params(action, params)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "params", MappingProxyType(params))

    def __repr__(self) -> str:
        # The redacted params, so a log line or a traceback never carries a full bank number.
        return f"ActionRequest({self.action.value}, {self.detector_params()})"

    # --- one constructor per action ---

    @classmethod
    def hold_vendor(cls, vendor_id: str, vendor_name: str, reason: str) -> "ActionRequest":
        params = {"vendor_id": vendor_id, "vendor_name": vendor_name, "reason": reason}
        return cls(Action.HOLD_VENDOR, params)

    @classmethod
    def hold_bill(cls, bill_id: str, vendor_id: str, reason: str) -> "ActionRequest":
        return cls(Action.HOLD_BILL, {"bill_id": bill_id, "vendor_id": vendor_id, "reason": reason})

    @classmethod
    def list_vendors(cls) -> "ActionRequest":
        return cls(Action.LIST_VENDORS, {})

    @classmethod
    def list_vendor_accounts(cls, vendor_id: str, vendor_name: str) -> "ActionRequest":
        params = {"vendor_id": vendor_id, "vendor_name": vendor_name}
        return cls(Action.LIST_VENDOR_ACCOUNTS, params)

    @classmethod
    def list_bills(cls, vendor_id: str, vendor_name: str) -> "ActionRequest":
        return cls(Action.LIST_BILLS, {"vendor_id": vendor_id, "vendor_name": vendor_name})

    @classmethod
    def update_vendor_bank_account(
        cls, vendor_id: str, vendor_name: str, routing_number: str, account_number: str
    ) -> "ActionRequest":
        """Raises ValueError unless the routing number is 9 digits that pass the ABA checksum and
        the account number is 4 to 17 digits. The error never repeats either number."""
        params = {
            "vendor_id": vendor_id,
            "vendor_name": vendor_name,
            "routing_number": routing_number,
            "account_number": account_number,
        }
        return cls(Action.UPDATE_VENDOR_BANK_ACCOUNT, params)

    @classmethod
    def release_vendor_hold(cls, vendor_id: str, vendor_name: str) -> "ActionRequest":
        params = {"vendor_id": vendor_id, "vendor_name": vendor_name}
        return cls(Action.RELEASE_VENDOR_HOLD, params)

    @classmethod
    def release_bill_hold(cls, bill_id: str, vendor_id: str) -> "ActionRequest":
        return cls(Action.RELEASE_BILL_HOLD, {"bill_id": bill_id, "vendor_id": vendor_id})

    # --- what the gate reads ---

    @property
    def tier(self) -> int:
        return ACTION_TIER[self.action]

    def detector_params(self) -> dict[str, str]:
        """The params as the detector sees them: unchanged, except that a bank change sends the
        last four digits of each number and a keyed fingerprint instead of the numbers (D2). The
        fingerprint stays the same for the same numbers for the life of the process."""
        if self.action is not Action.UPDATE_VENDOR_BANK_ACCOUNT:
            return dict(self.params)
        routing, account = self.params["routing_number"], self.params["account_number"]
        mac = hmac.new(_FINGERPRINT_KEY, f"{routing}|{account}".encode(), hashlib.sha256)
        return {
            "vendor_id": self.params["vendor_id"],
            "vendor_name": self.params["vendor_name"],
            "new_account_last4": account[-4:],
            "new_routing_last4": routing[-4:],
            "account_fingerprint": mac.hexdigest()[:16],
        }

    def context(self) -> RampContext:
        """What the detector's rubric is told about the request (D2). The callback reads
        pending_action back to the caller."""
        return RampContext(
            pending_action=self._pending_action(), vendor_name=self.params.get("vendor_name")
        )

    def idempotency_key(self, session_id: str, round_no: int) -> str:
        """The detector caches its decision per key, so every round needs its own (D6)."""
        params = json.dumps(self.detector_params(), sort_keys=True)
        text = f"{session_id}|{self.action}|{params}|{round_no}"
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    def hold_target(self) -> HoldTarget | None:
        """What a deny puts on hold: the vendor a tier 1 or 2 request names (for a bill release,
        the bill's vendor). Nothing for a hold, which is tier 0, and nothing for a request with no
        vendor, such as list_vendors. A denied read holds its vendor too (final ruling, C31)."""
        if self.tier >= 1 and "vendor_id" in self.params:
            return HoldTarget("vendor", self.params["vendor_id"])
        return None

    def _pending_action(self) -> str:
        """The request in the caller's terms. It fits "Did you just ask me to ...?"."""
        p = self.params
        match self.action:
            case Action.HOLD_VENDOR:
                return f"put {p['vendor_name']} on hold"
            case Action.HOLD_BILL:
                return "put a bill on hold"
            case Action.LIST_VENDORS:
                return "list the vendors"
            case Action.LIST_VENDOR_ACCOUNTS:
                return f"read {_possessive(p['vendor_name'])} bank accounts on file"
            case Action.LIST_BILLS:
                return f"list {_possessive(p['vendor_name'])} open bills"
            case Action.UPDATE_VENDOR_BANK_ACCOUNT:
                last4 = p["account_number"][-4:]
                return f"change {_possessive(p['vendor_name'])} bank account to one ending {last4}"
            case Action.RELEASE_VENDOR_HOLD:
                return f"release the hold on {p['vendor_name']}"
            case Action.RELEASE_BILL_HOLD:
                return "release the hold on a bill"
        assert_never(self.action)


async def execute(ramp: RampClient, req: ActionRequest) -> ActionOutcome:
    """Make the Ramp call `req` stands for, once, and say what came of it.

    Only the gate calls this: at once for a hold, otherwise after an allow. A RampError
    propagates, so the gate can tell the caller whether the change failed or may have gone through.
    """
    p = req.params
    match req.action:
        case Action.HOLD_VENDOR:
            return await _hold_vendor(ramp, p["vendor_id"], p["vendor_name"], p["reason"])
        case Action.HOLD_BILL:
            bill = await ramp.hold_bill(p["bill_id"], p["reason"])
            if bill.status_summary != _ON_HOLD:
                return _DID_NOT_TAKE
            return ActionOutcome("done", _bill_hold_summary(bill))
        case Action.LIST_VENDORS:
            return ActionOutcome("done", _vendors_summary(await ramp.list_vendors()))
        case Action.LIST_VENDOR_ACCOUNTS:
            accounts = await ramp.list_vendor_accounts(p["vendor_id"])
            return ActionOutcome("done", _accounts_summary(p["vendor_name"], accounts))
        case Action.LIST_BILLS:
            bills = await ramp.list_bills(vendor_id=p["vendor_id"])
            return ActionOutcome("done", _bills_summary(p["vendor_name"], bills))
        case Action.UPDATE_VENDOR_BANK_ACCOUNT:
            update = await ramp.add_default_bank_account(
                p["vendor_id"],
                routing_number=p["routing_number"],
                account_number=p["account_number"],
            )
            name, last4 = p["vendor_name"], p["account_number"][-4:]
            if update.status == "approved":
                return ActionOutcome("done", f"{name} now pays to the account ending {last4}.")
            return ActionOutcome(
                "pending_approval",
                f"The change to the account ending {last4} for {name} is waiting for approval "
                "in Ramp.",
            )
        case Action.RELEASE_VENDOR_HOLD:
            name = p["vendor_name"]
            if await ramp.release_vendor(p["vendor_id"]) is HoldOutcome.APPLIED:
                return ActionOutcome(
                    "done",
                    f"The hold on {name} is released. Payments taken off the schedule during the "
                    "hold need to be scheduled again.",
                )
            return ActionOutcome(
                "pending_approval", f"Releasing the hold on {name} is waiting for approval in Ramp."
            )
        case Action.RELEASE_BILL_HOLD:
            bill = await ramp.release_bill(p["bill_id"])
            if bill.status_summary == _ON_HOLD:
                return _DID_NOT_TAKE
            return ActionOutcome("done", f"The hold on {_bill_name(bill)} is released.")
    assert_never(req.action)


async def hold_vendor_fully(ramp: RampClient, vendor_id: str, reason: str) -> bool:
    """Put a vendor on hold, and its open bills too if the vendor hold waits for approval.

    True only if something now holds: the vendor hold applied, or at least one bill came back
    ON_HOLD (V6). It never raises a RampError, so the gate's deny path always goes on to say its
    line and hang up.
    """
    try:
        vendor, bills_held = await _hold_vendor_and_bills(ramp, vendor_id, reason)
    except RampError as exc:
        _log_ramp_error("vendor hold", exc)
        return False
    return vendor is HoldOutcome.APPLIED or bills_held > 0


async def _hold_vendor(ramp: RampClient, vendor_id: str, name: str, reason: str) -> ActionOutcome:
    """execute() for a vendor hold. It shares hold_vendor_fully's steps, but needs to know which
    of them took, and lets a failed vendor hold raise."""
    vendor, bills_held = await _hold_vendor_and_bills(ramp, vendor_id, reason)
    if vendor is HoldOutcome.APPLIED:
        return ActionOutcome("done", f"{name} is on hold.")
    if bills_held:
        return ActionOutcome(
            "pending_approval",
            f"{_possessive(name)} open bills are on hold; the vendor hold itself is waiting for "
            "approval in Ramp.",
        )
    return ActionOutcome(
        "pending_approval",
        f"The hold on {name} is waiting for approval in Ramp, so nothing is on hold yet.",
    )


async def _hold_vendor_and_bills(
    ramp: RampClient, vendor_id: str, reason: str
) -> tuple[HoldOutcome, int]:
    """Hold the vendor. If Ramp answers that the hold waits for approval, also hold each of the
    vendor's open bills, which takes effect at once.

    Returns Ramp's answer to the vendor hold and how many bills came back ON_HOLD (a Bill coming
    back proves nothing by itself). Only a failed vendor hold raises: a bill that can't be listed
    or held is logged and counted as not held, and the rest are still tried. A bill hold counts
    as failed whatever it raises, not only a RampError (the closed client of a call that is
    ending raises a RuntimeError, say), except for cancellation, which gets through.
    """
    vendor = await ramp.hold_vendor(vendor_id, reason)
    if vendor is not HoldOutcome.PENDING_APPROVAL:
        return vendor, 0
    try:
        bills = await ramp.list_bills(vendor_id=vendor_id)
    except RampError as exc:
        _log_ramp_error("listing bills to hold", exc)
        return vendor, 0
    held = 0
    for bill in bills:
        try:
            if (await ramp.hold_bill(bill.id, reason)).status_summary == _ON_HOLD:
                held += 1
        except RampError as exc:
            _log_ramp_error("bill hold", exc)
        except Exception as exc:
            # Only its type: the message could echo what was sent. The bill is not held, and the
            # next one is still tried.
            logger.error(
                "ramp bill hold raised %s, not a RampError, so the bill counts as not held",
                type(exc).__name__,
            )
    return vendor, held


def _log_ramp_error(what: str, exc: RampError) -> None:
    # Never exc.message: Ramp may echo a submitted value back in it, reformatted past redaction.
    logger.warning(
        "ramp %s failed: status=%s error_code=%r trace_id=%s",
        what,
        exc.status,
        exc.error_code,
        exc.trace_id,
    )


def _check_params(action: Action, params: dict[str, str]) -> None:
    """Raise ValueError unless `params` are exactly the ones `action` takes, and well formed.

    Every message is fixed text, or names a param the action takes. None repeats a value or a key
    the caller passed, since a value can be a bank number.
    """
    keys = _PARAM_KEYS[action]
    missing = [key for key in keys if key not in params]
    if missing:
        raise ValueError(f"{action.value} is missing {', '.join(missing)}")
    if params.keys() != set(keys):
        raise ValueError(f"{action.value} has a param it does not take")
    if not all(isinstance(value, str) for value in params.values()):
        raise ValueError(f"every param of {action.value} must be text")
    if action is Action.UPDATE_VENDOR_BANK_ACCOUNT:
        if not _valid_routing_number(params["routing_number"]):
            raise ValueError("the routing number must be 9 digits that pass the ABA checksum")
        if not _ACCOUNT_NUMBER.fullmatch(params["account_number"]):
            raise ValueError("the account number must be 4 to 17 digits")


def _valid_routing_number(routing: str) -> bool:
    """9 digits with 3(d1+d4+d7) + 7(d2+d5+d8) + (d3+d6+d9) a multiple of 10 (the ABA checksum)."""
    if not _ROUTING_NUMBER.fullmatch(routing):
        return False
    d = [int(c) for c in routing]
    return (3 * (d[0] + d[3] + d[6]) + 7 * (d[1] + d[4] + d[7]) + d[2] + d[5] + d[8]) % 10 == 0


def _vendors_summary(vendors: list[Vendor]) -> str:
    if not vendors:
        return "Ramp has no vendors."
    names = ", ".join(
        (v.display_name or v.name) + ("" if v.is_active else " (inactive)") for v in vendors
    )
    return f"Ramp has {_count(len(vendors), 'vendor')}: {names}."


def _accounts_summary(vendor_name: str, accounts: list[VendorBankAccount]) -> str:
    if not accounts:
        return f"Ramp has no payment accounts on file for {vendor_name}."
    described = "; ".join(_account(a) for a in accounts)
    return f"{_possessive(vendor_name)} payment accounts on file: {described}."


def _account(account: VendorBankAccount) -> str:
    text = account.payment_method
    if account.account_number_last_four:
        text += f" ending {account.account_number_last_four[-4:]}"
    if account.is_default:
        text += ", the default"
    if account.status != "active":
        text += f", {_words(account.status)}"
    return text


def _bills_summary(vendor_name: str, bills: list[Bill]) -> str:
    if not bills:
        return f"{vendor_name} has no open bills."
    described = "; ".join(_bill_line(b) for b in bills)
    return f"{vendor_name} has {_count(len(bills), 'open bill')}: {described}."


def _bill_line(bill: Bill) -> str:
    """One bill in a list. The id is there so the LLM can name the bill to release_bill_hold.

    hold_bill does not take it: since R56 a hold names a bill by its vendor and invoice number.
    """
    text = f"invoice {bill.invoice_number}" if bill.invoice_number else "a bill"
    text += f" for {bill.amount.spoken()}"
    if bill.due_at is not None:
        text += f", due {bill.due_at:%B} {bill.due_at.day}"
    return f"{text}, {_words(bill.status_summary)} (bill id {bill.id})"


def _bill_hold_summary(bill: Bill) -> str:
    """What a bill hold says: the vendor and the invoice number, never the amount.

    A hold needs no verification, and a bill is named by its vendor and invoice number, so a
    stranger who guessed both would otherwise be told what the invoice is for (R63). The request
    carries no display name for the vendor, so this uses the name Ramp puts on the bill.
    """
    if bill.invoice_number:
        return f"{_possessive(bill.vendor.name)} invoice {bill.invoice_number} is on hold."
    return f"A bill from {bill.vendor.name} is on hold."


def _bill_name(bill: Bill) -> str:
    """A bill as a verified caller knows it: by invoice number when it has one, never by its id.

    It says the amount, so only a release uses it (tier 2). A hold does not: see _bill_hold_summary.
    """
    amount = bill.amount.spoken()
    if bill.invoice_number:
        return f"invoice {bill.invoice_number} from {bill.vendor.name} for {amount}"
    return f"the {amount} bill from {bill.vendor.name}"


def _possessive(name: str) -> str:
    return f"{name}'" if name.endswith(("s", "S")) else f"{name}'s"


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _words(code: str) -> str:
    """A Ramp code such as PAYMENT_SCHEDULED as plain words: "payment scheduled"."""
    return code.replace("_", " ").lower()
