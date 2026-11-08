"""Put the demo's fixtures in the Ramp sandbox: the vendor Acme Supplies with a default ACH account
ending 4417, and two open bills.

Run it from agent/:

    uv run python scripts/seed_ramp.py

It reads RAMP_CLIENT_ID, RAMP_CLIENT_SECRET and, if set, RAMP_BASE_URL from the repo-root .env or
the shell (the shell wins). It writes made-up bank details, so it refuses to run against
production. Run it as often as you like: it creates only what is missing.

The bills need Bill Pay, which means an entity with a Bill Pay bank account, and reading the
entities needs the entities:read scope on the sandbox app (the agent's own scopes leave it out).
Without either the bills can't be made. The seed says so and carries on, and the demo runs on
vendor holds alone.
"""

import asyncio
import logging
import os
import sys
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, ValidationError

from callback_agent.config import Settings
from callback_agent.ramp import RampClient, RampError, VendorBankAccount

logger = logging.getLogger("seed_ramp")

PRODUCTION_HOST = "api.ramp.com"

VENDOR_NAME = "Acme Supplies"
# example.com is reserved for examples, so nothing sent to it is delivered. There is no phone
# number either.
CONTACT_EMAIL = "acme-billing@example.com"
# The routing number is the project's sample value, and the account number is made up.
ROUTING_NUMBER = "021000021"
ACCOUNT_NUMBER = "000000004417"
BANK_LAST4 = ACCOUNT_NUMBER[-4:]


class EntitiesClient(RampClient):
    """A client whose token carries only entities:read.

    GET /entities needs that scope. The agent's token leaves it out, because nothing the agent does
    reads entities, and putting it in would make every token request fail on an app that doesn't
    allow it. So the seed asks for it in a token of its own and uses it for that one call.
    """

    SCOPES = "entities:read"


@dataclass(frozen=True)
class Invoice:
    number: str
    amount: str  # in dollars, the way Ramp's line items take it
    due_in_days: int
    memo: str


INVOICES = (
    Invoice("INV-1001", "1204.50", 14, "Fall restock"),
    Invoice("INV-1002", "3980.00", 30, "Fall restock"),
)
INVOICE_NUMBERS = {invoice.number for invoice in INVOICES}


@dataclass(frozen=True)
class SeedResult:
    vendor_id: str
    # The last four digits of the seeded account, if Ramp has it on file. Empty while it waits for
    # approval, because then Acme has no such account yet.
    bank_last4: str
    # The ids of the seed's bills, INV-1001 first. Empty or short if Ramp would not make them all.
    bill_ids: list[str]


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _Created(_Model):
    id: str


class _AccountRef(_Model):
    usage_type: str | None = None
    bank_account_id: str | None = None  # how Ramp's OpenAPI spec names it, in accounts[]
    id: str | None = None  # how the Bill Payments guide names it, in bank_accounts[]


class _Entity(_Model):
    id: str
    is_primary: bool | None = None
    default_bill_pay_payment_account_id: str | None = None
    accounts: list[_AccountRef] | None = None
    bank_accounts: list[_AccountRef] | None = None

    def bill_pay_account_id(self) -> str | None:
        """The id of the bank account that funds Bill Pay, or None if the entity has none.

        Ramp describes it twice: the spec lists it in accounts[] and the guide in bank_accounts[],
        both with usage_type BILL_PAY_BANK_ACCOUNT. The entity's own default is the fallback.
        """
        for ref in [*(self.accounts or []), *(self.bank_accounts or [])]:
            found = ref.bank_account_id or ref.id
            if ref.usage_type == "BILL_PAY_BANK_ACCOUNT" and found:
                return found
        return self.default_bill_pay_payment_account_id


class _Entities(_Model):
    data: list[_Entity]


@dataclass(frozen=True)
class _Funding:
    entity_id: str
    bank_account_id: str


def _read[M: BaseModel](model: type[M], response: httpx.Response) -> M:
    """Parse a 2xx answer. One that doesn't fit is a RampError, without the body in it."""
    try:
        return model.model_validate_json(response.content)
    except ValidationError:
        raise RampError(
            response.status_code,
            message="the answer breaks the expected shape",
            trace_id=response.headers.get("x-trace-id"),
        ) from None


def _why(error: RampError) -> str:
    """What went wrong, from the status, the code and the trace only. Ramp's own message can echo
    what was sent, so it stays out of the output."""
    if not error.status:
        return "Ramp gave no answer"
    return f"HTTP {error.status} {error.error_code or '-'}, trace_id {error.trace_id or '-'}"


def _vendor_body() -> dict:
    return {
        "name": VENDOR_NAME,
        "business_vendor_contacts": {"email": CONTACT_EMAIL},
        "address": {
            "address_line_1": "100 Example Way",
            "city": "Oakland",
            "state": "CA",
            "postal_code": "94607",
            "country": "US",
        },
        "country": "US",
        "state": "CA",
    }


def _arrival(due: date) -> date:
    """The day to ask for a payment due on `due` to arrive: `due` itself, or the Monday after it
    when `due` is a Saturday or a Sunday.

    Ramp's spec says "An error will be thrown if selected date is invalid" about the arrival date
    and doesn't say what is invalid, but a payment can't arrive on a weekend. Ramp moves
    payment_date to the next available day by itself. It does not move this date.
    """
    to_monday = {5: 2, 6: 1}  # Saturday is 5 and Sunday is 6, with Monday 0. Other days stay.
    return due + timedelta(days=to_monday.get(due.weekday(), 0))


def _bill_body(
    vendor_id: str, funding: _Funding, vendor_account_id: str, invoice: Invoice, today: date
) -> dict:
    due = today + timedelta(days=invoice.due_in_days)
    return {
        "vendor_id": vendor_id,
        "entity_id": funding.entity_id,
        "invoice_number": invoice.number,
        "invoice_currency": "USD",
        "issued_at": today.isoformat(),
        "due_at": due.isoformat(),  # the calendar date, weekend or not
        "memo": invoice.memo,
        "line_items": [{"amount": invoice.amount, "memo": invoice.memo}],
        "use_default_vendor_contact": True,
        "payment_method": "ACH",
        "payment_details": {
            "payment_arrival_date": _arrival(due).isoformat(),
            "source_bank_account_id": funding.bank_account_id,
            "vendor_account_id": vendor_account_id,
        },
    }


def _seeded_account(accounts: list[VendorBankAccount]) -> VendorBankAccount | None:
    """The vendor's active ACH account ending in BANK_LAST4, if it has one."""
    return next(
        (
            account
            for account in accounts
            if account.account_number_last_four == BANK_LAST4
            and account.payment_method == "ACH"
            and account.status == "active"
        ),
        None,
    )


async def _vendor(ramp: RampClient) -> str:
    """The id of Acme Supplies, created if the sandbox doesn't have it yet.

    It looks the vendor up with find_vendor, as the agent will, so the vendor it finds is the one
    the agent will find.
    """
    found = await ramp.find_vendor(VENDOR_NAME)
    if found is not None:
        logger.info("vendor: found %s", VENDOR_NAME)
        return found.id
    response = await ramp.request("POST", "/vendors", json=_vendor_body())
    logger.info("vendor: created %s", VENDOR_NAME)
    return _read(_Created, response).id


async def _bank_account(ramp: RampClient, vendor_id: str) -> VendorBankAccount | None:
    """The vendor's account ending in BANK_LAST4, added as the default if it isn't on file yet.

    None means Ramp doesn't have it as an active account: the change is waiting for approval, or
    the account was added but isn't listed yet.
    """
    account = _seeded_account(await ramp.list_vendor_accounts(vendor_id))
    if account is not None:
        logger.info("bank account: the one ending %s is already on file", BANK_LAST4)
        if not account.is_default:
            logger.warning(
                "bank account: the one ending %s is not Acme's default. Make it the default in "
                "Ramp before the demo.",
                BANK_LAST4,
            )
        return account
    update = await ramp.add_default_bank_account(
        vendor_id, routing_number=ROUTING_NUMBER, account_number=ACCOUNT_NUMBER
    )
    if update.status == "pending_approval":
        logger.warning(
            "bank account: the one ending %s is waiting for approval in Ramp. Approve it there, "
            "then run the seed again to make the bills.",
            BANK_LAST4,
        )
        return None
    logger.info("bank account: added the one ending %s as the default", BANK_LAST4)
    account = _seeded_account(await ramp.list_vendor_accounts(vendor_id))
    if account is None:
        logger.warning(
            "bank account: Ramp doesn't list the new one yet. Run the seed again to make the bills."
        )
    return account


async def _funding(client: RampClient) -> _Funding | None:
    """The entity to bill from and its Bill Pay bank account: the first entity that has one, the
    primary entity first. None, after saying why, if the entities can't be read for want of the
    scope or no entity has the account. Any other failure of Ramp raises RampError."""
    try:
        response = await client.request("GET", "/entities")
    except RampError as error:
        if error.status not in (400, 401, 403):  # a token refused for its scope, or a call refused
            raise
        logger.warning(
            "bills: not made, Ramp would not list the entities (%s). The usual cause is the "
            "entities:read scope, which the sandbox app has to allow.",
            _why(error),
        )
        return None
    for entity in sorted(_read(_Entities, response).data, key=lambda e: not e.is_primary):
        if account_id := entity.bill_pay_account_id():
            return _Funding(entity.id, account_id)
    logger.warning("bills: not made, no entity has a Bill Pay bank account (is Bill Pay on?)")
    return None


async def _bills(
    ramp: RampClient,
    entities: RampClient,
    vendor_id: str,
    account: VendorBankAccount | None,
    today: date,
) -> AsyncIterator[tuple[str, str]]:
    """Yield (invoice number, bill id) for each of INVOICES that is open or can be made.

    A bill that is open already is yielded as it is, so a second run makes nothing twice. The
    others are made, which takes the vendor's account to pay into and an entity with a Bill Pay
    bank account. If either is missing, no more is yielded.

    Each bill stands alone: one that Ramp refuses is logged and skipped, and the next is still
    tried. A 401 or a 403 ends the bills instead, since the token or its scope is at fault and
    every later bill would be refused the same way. Any other failure of Ramp raises RampError,
    after everything yielded before it.
    """
    open_ids: dict[str, str] = {}
    for bill in await ramp.list_bills(vendor_id=vendor_id):
        if bill.invoice_number in INVOICE_NUMBERS:
            open_ids.setdefault(bill.invoice_number, bill.id)
    missing = [invoice for invoice in INVOICES if invoice.number not in open_ids]
    for number, bill_id in open_ids.items():
        logger.info("bills: %s is open already", number)
        yield number, bill_id
    if not missing:
        return
    if account is None:
        logger.warning("bills: not made, Acme has no account ending %s to pay into yet", BANK_LAST4)
        return
    funding = await _funding(entities)
    if funding is None:
        return
    for invoice in missing:
        body = _bill_body(vendor_id, funding, account.id, invoice, today)
        try:
            bill_id = _read(_Created, await ramp.request("POST", "/bills", json=body)).id
        except RampError as error:
            if error.status in (401, 403):
                raise
            logger.warning("bills: %s not made (%s)", invoice.number, _why(error))
            continue
        logger.info("bills: made %s", invoice.number)
        yield invoice.number, bill_id


async def seed(
    ramp: RampClient, *, entities: RampClient | None = None, today: date | None = None
) -> SeedResult:
    """Make sure the vendor, its account and its two bills are in the sandbox, and say which.

    Whatever is there is left alone, so running it again changes nothing. A failure to set up the
    vendor or its account raises RampError. A failure with the bills does not: the seed logs why
    and returns the bills it has, because the demo still runs on vendor holds. The bills fall due
    14 and 30 days after `today`, which is the current date unless given, and are asked to arrive
    that day, or the Monday after it when it is a weekend.

    `entities` reads the entities, which takes the entities:read scope that `ramp`'s token usually
    lacks: pass an EntitiesClient. It defaults to `ramp`.
    """
    today = today or date.today()
    vendor_id = await _vendor(ramp)
    account = await _bank_account(ramp, vendor_id)
    bill_ids: dict[str, str] = {}
    try:
        reader = ramp if entities is None else entities
        async for number, bill_id in _bills(ramp, reader, vendor_id, account, today):
            bill_ids[number] = bill_id
    except RampError as error:
        logger.warning("bills: stopped (%s). The demo can still run on vendor holds.", _why(error))
    return SeedResult(
        vendor_id=vendor_id,
        bank_last4=BANK_LAST4 if account is not None else "",
        bill_ids=[bill_ids[invoice.number] for invoice in INVOICES if invoice.number in bill_ids],
    )


def _summary(result: SeedResult) -> str:
    account = f"ending {result.bank_last4}" if result.bank_last4 else "none yet (see above)"
    bills = (
        ", ".join(result.bill_ids)
        if result.bill_ids
        else "no bills (see above). The demo runs on vendor holds alone"
    )
    return f"{VENDOR_NAME}: vendor {result.vendor_id}\nDefault account: {account}\nBills: {bills}"


async def run(env: Mapping[str, str]) -> int:
    """The command line: read the settings, seed the sandbox and print what is there.

    Returns the exit status: 0 for a seed that finished, 1 for one that stopped, 2 for settings
    that can't be used.
    """
    try:
        settings = Settings.from_env(env, required=("RAMP_CLIENT_ID", "RAMP_CLIENT_SECRET"))
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    if httpx.URL(settings.ramp_base_url).host == PRODUCTION_HOST:
        print(
            f"Not seeding {PRODUCTION_HOST}: that is production, and the seed writes made-up bank "
            "details. Leave RAMP_BASE_URL empty or point it at the sandbox.",
            file=sys.stderr,
        )
        return 2
    # from_env has checked that both credentials are set. Two clients, so two tokens: the agent's
    # scopes, and entities:read for the one call that needs it.
    credentials = (settings.ramp_client_id or "", settings.ramp_client_secret or "")
    ramp = RampClient(settings.ramp_base_url, *credentials)
    entities = EntitiesClient(settings.ramp_base_url, *credentials)
    try:
        result = await seed(ramp, entities=entities)
    except RampError as error:
        print(f"The seed stopped: {_why(error)}. Run it again to carry on.", file=sys.stderr)
        return 1
    finally:
        await entities.aclose()
        await ramp.aclose()
    print(_summary(result))
    return 0


def main() -> int:
    # Like agent.py: variables already set in the shell win over the file.
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    # Our own lines and the client's (what it sent and what came back, never a body). The root
    # stays at WARNING, which keeps httpx's per-request lines out.
    logging.basicConfig(format="%(message)s")
    for name in (logger.name, "callback_agent.ramp"):
        logging.getLogger(name).setLevel(logging.INFO)
    return asyncio.run(run(os.environ))


if __name__ == "__main__":
    sys.exit(main())
