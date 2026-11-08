"""Offline tests for scripts/seed_ramp.py, which puts the demo's vendor, account and bills in Ramp.

The script lives outside the package, so the tests load it by path, as test_attacker_agent.py does.
Ramp is replaced by FakeRampApi, a small in-memory copy of the endpoints the seed uses. It answers
over respx and refuses a body that leaves out a field the Ramp OpenAPI spec marks required, so a
passing test also means the seed sent every field the spec asks for.
"""

import importlib.util
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest
import respx

import callback_agent.ramp as ramp_module
from callback_agent.ramp import RampClient, RampError

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "seed_ramp.py"
FIXTURES = Path(__file__).parent / "fixtures" / "ramp"
BASE = "https://ramp.test"
PREFIX = "/developer/v1"
TODAY = date(2026, 11, 8)

TOKEN = json.loads((FIXTURES / "token.json").read_text())
BASE_SCOPES = ("vendors:read", "vendors:write", "bills:read", "bills:write")  # the agent's
ENTITY_ID = "ent-1"
FUNDING_ID = "bank-funding"  # the entity's Bill Pay bank account
ECHO = "ramp-echoed-this-4417-text"  # what a Ramp error message could repeat from a request
ROUTING = "021000021"
ACCOUNT = "000000004417"


def vendor_json(id_: str = "v-acme", name: str = "Acme Supplies") -> dict:
    return {"id": id_, "name": name, "display_name": None, "is_active": True}


def account_json(id_: str = "acct-4417", last_four: str = "4417", *, default: bool = True) -> dict:
    return {
        "id": id_,
        "account_number_last_four": last_four,
        "currency": "USD",
        "is_default": default,
        "payment_method": "ACH",
        "proposal_id": None,
        "status": "active",
    }


def funding(account_id: str = FUNDING_ID) -> dict:
    return {"bank_account_id": account_id, "usage_type": "BILL_PAY_BANK_ACCOUNT"}


def entity(id_: str = ENTITY_ID, *, primary: bool = True, **fields) -> dict:
    """An entity as the OpenAPI spec shows it: the Bill Pay bank account sits in accounts[]."""
    return {
        "id": id_,
        "entity_name": "Example Co",
        "is_primary": primary,
        "default_bill_pay_payment_account_id": None,
        "payment_accounts": [],
        "accounts": [
            {"bank_account_id": "bank-other", "usage_type": "CASH_ACCOUNT"},
            funding(),
        ],
        **fields,
    }


def page(*items: dict) -> httpx.Response:
    return httpx.Response(200, json={"data": list(items), "page": {"next": None}})


def refusal(status: int) -> httpx.Response:
    body = {"error_v2": {"error_code": "DEVELOPER_7001", "message": f"Ramp says {ECHO}"}}
    return httpx.Response(status, json=body, headers={"x-trace-id": "t-seed"})


def invalid(problems: list[str]) -> httpx.Response:
    body = {"error_v2": {"error_code": "DEVELOPER_7001", "message": f"invalid {problems}"}}
    return httpx.Response(422, json=body, headers={"x-trace-id": "t-fake"})


def lacking(body: dict, *keys: str) -> list[str]:
    return [key for key in keys if not body.get(key)]


class FakeRampApi:
    """The Ramp endpoints the seed touches, kept in memory. `requests` has every call it received.

    `answers[(method, path)]` lists what to do with the next calls to an endpoint (the path is
    relative to /developer/v1): None lets a call through, a number answers with that error.

    Scopes are ignored unless `strict_scopes` is set. Then, as in Ramp, a token request for a scope
    outside `app_scopes` is refused with a 400, and GET /entities answers 403 to a token without
    entities:read. `token_scopes` has the scope of each token request, oldest first.
    """

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.vendors: list[dict] = []
        self.accounts: defaultdict[str, list[dict]] = defaultdict(list)  # by vendor id
        self.bills: list[dict] = []
        self.entities: list[dict] = [entity()]
        self.bank_update = "approved"  # or "pending_approval"
        self.listing_lags = False  # an approved account is not in the listing until later
        self.strict_scopes = False
        self.app_scopes = {*BASE_SCOPES, "entities:read"}
        self.token_scopes: list[str] = []
        self.answers: dict[tuple[str, str], list[int | None]] = {}

    def posts(self, path: str) -> list[dict]:
        """The JSON bodies sent to POST `path`, oldest first."""
        return [
            json.loads(r.content)
            for r in self.requests
            if r.method == "POST" and r.url.path == PREFIX + path
        ]

    def writes(self) -> list[tuple[str, str]]:
        """Every call that could change Ramp, as (method, path): all but reads and the token."""
        return [
            (r.method, r.url.path.removeprefix(PREFIX))
            for r in self.requests
            if r.method != "GET" and not r.url.path.endswith("/token")
        ]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        method, path = request.method, request.url.path.removeprefix(PREFIX)
        if path == "/token":
            return self._token(request)
        queued = self.answers.get((method, path))
        if queued and (status := queued.pop(0)) is not None:
            return refusal(status)
        body = json.loads(request.content) if request.content else {}
        vendor_path = re.fullmatch(r"/vendors/([^/]+)/(accounts|update-bank-accounts)", path)
        if (method, path) == ("GET", "/vendors"):
            return page(*self.vendors)
        if (method, path) == ("POST", "/vendors"):
            return self._create_vendor(body)
        if vendor_path and (method, vendor_path[2]) == ("GET", "accounts"):
            return page(*self.accounts[vendor_path[1]])
        if vendor_path and (method, vendor_path[2]) == ("POST", "update-bank-accounts"):
            return self._add_account(vendor_path[1], body)
        if (method, path) == ("GET", "/entities"):
            granted = request.headers["authorization"].removeprefix("Bearer tok:").split(",")
            if self.strict_scopes and "entities:read" not in granted:
                return refusal(403)
            return page(*self.entities)
        if (method, path) == ("GET", "/bills"):
            params = request.url.params
            assert params["payment_status"] == "OPEN"
            return page(*(b for b in self.bills if b["vendor"]["id"] == params["vendor_id"]))
        if (method, path) == ("POST", "/bills"):
            return self._create_bill(body)
        raise AssertionError(f"the seed made a call the fake does not know: {method} {path}")

    def _token(self, request: httpx.Request) -> httpx.Response:
        scope = parse_qs(request.content.decode())["scope"][0]
        self.token_scopes.append(scope)
        if self.strict_scopes and not set(scope.split()) <= self.app_scopes:
            body = {"error_v2": {"error_code": "DEVELOPER_7012", "message": f"scope {ECHO}"}}
            return httpx.Response(400, json=body, headers={"x-trace-id": "t-scope"})
        # The token says what it may do, so the fake can tell the two clients' tokens apart.
        token = "tok:" + scope.replace(" ", ",")
        return httpx.Response(200, json={**TOKEN, "access_token": token})

    def _create_vendor(self, body: dict) -> httpx.Response:
        contact, address = body.get("business_vendor_contacts") or {}, body.get("address") or {}
        problems = [
            *lacking(body, "name", "business_vendor_contacts"),
            *lacking(contact, "email"),
            *lacking(address, "address_line_1", "city", "country", "postal_code", "state"),
        ]
        if problems:
            return invalid(problems)
        vendor = vendor_json("v-acme", body["name"])
        self.vendors.append(vendor)
        return httpx.Response(200, json=vendor)

    def _add_account(self, vendor_id: str, body: dict) -> httpx.Response:
        if lacking(body.get("ach_details") or {}, "routing_number", "account_number"):
            return invalid(["ach_details"])
        if self.bank_update == "approved" and not self.listing_lags:
            number = body["ach_details"]["account_number"]
            self.accounts[vendor_id].append(account_json(last_four=number[-4:]))
        pending = self.bank_update != "approved"
        return httpx.Response(
            200,
            json={
                "vendor_id": vendor_id,
                "status": self.bank_update,
                "payment_method_id": None if pending else "pm-1",
                "payment_method_type": "ach",
                "proposal_id": "prop-1" if pending else None,
            },
        )

    def _create_bill(self, body: dict) -> httpx.Response:
        ach = body.get("payment_details") or {}
        problems = [
            *lacking(body, "vendor_id", "entity_id", "invoice_number", "invoice_currency"),
            *lacking(body, "issued_at", "due_at", "line_items", "payment_method"),
            *lacking(ach, "payment_arrival_date", "source_bank_account_id", "vendor_account_id"),
        ]
        if not (body.get("vendor_contact_id") or body.get("use_default_vendor_contact")):
            problems.append("vendor_contact_id")
        # Ramp: "An error will be thrown if selected date is invalid." Only payment_date is moved on
        # by Ramp itself, so an arrival date on a Saturday or a Sunday is refused here as well.
        arrival = ach.get("payment_arrival_date")
        if arrival and date.fromisoformat(arrival).weekday() >= 5:
            problems.append("payment_arrival_date falls on a weekend")
        if problems:
            return invalid(problems)
        vendor = next(v for v in self.vendors if v["id"] == body["vendor_id"])
        cents = int(Decimal(body["line_items"][0]["amount"]) * 100)
        bill = {
            "id": f"bill-{len(self.bills) + 1}",
            "invoice_number": body["invoice_number"],
            "status": "OPEN",
            "status_summary": "PAYMENT_SCHEDULED",
            "amount": {"amount": cents, "currency_code": "USD", "minor_unit_conversion_rate": 100},
            "due_at": f"{body['due_at']}T00:00:00+00:00",
            "vendor": {"id": vendor["id"], "name": vendor["name"]},
        }
        self.bills.append(bill)
        return httpx.Response(201, json=bill)

    def have_seeded_vendor(self, vendor_id: str = "v-acme", name: str = "Acme Supplies") -> None:
        """Start from a sandbox the seed has already visited: the vendor and its account."""
        self.vendors.append(vendor_json(vendor_id, name))
        self.accounts[vendor_id].append(account_json())


@pytest.fixture
def seed_ramp(monkeypatch):
    spec = importlib.util.spec_from_file_location("seed_ramp", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "seed_ramp", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def pauses(monkeypatch):
    """A read that Ramp refuses is tried again after a pause. The pause is recorded, not slept."""
    waited: list[float] = []

    async def record(seconds: float) -> None:
        waited.append(seconds)

    monkeypatch.setattr(ramp_module, "sleep", record)
    return waited


@pytest.fixture
def fake():
    return FakeRampApi()


@pytest.fixture
def api(fake):
    # A request that no test expected is an error, so a client that wanders off fails loudly.
    with respx.mock(base_url=BASE, assert_all_called=False) as router:
        router.route().mock(side_effect=fake)
        yield router


@pytest.fixture
async def ramp(api):
    client = RampClient(BASE, "cid", "secret")
    try:
        yield client
    finally:
        await client.aclose()


# --- the three the plan names -----------------------------------------------------------------


async def test_seed_skips_an_existing_vendor(seed_ramp, ramp, fake):
    # find_vendor matches a longer name that contains the wanted one, as the agent's lookup will.
    fake.have_seeded_vendor("v-old", "Acme Supplies Inc")
    result = await seed_ramp.seed(ramp, today=TODAY)
    assert fake.posts("/vendors") == []
    assert result.vendor_id == "v-old"


async def test_seed_creates_vendor_account_and_bills(seed_ramp, ramp, fake):
    result = await seed_ramp.seed(ramp, today=TODAY)

    [vendor_body] = fake.posts("/vendors")
    assert vendor_body["name"] == "Acme Supplies"
    assert "email" in vendor_body["business_vendor_contacts"]
    [bank_body] = fake.posts("/vendors/v-acme/update-bank-accounts")
    assert bank_body["ach_details"]["account_number"].endswith("4417")
    assert bank_body["is_default"] is True
    assert len(fake.posts("/bills")) == 2
    assert result == seed_ramp.SeedResult(
        vendor_id="v-acme", bank_last4="4417", bill_ids=["bill-1", "bill-2"]
    )
    # The vendor first: the account needs its id, and the bills need both.
    assert fake.writes() == [
        ("POST", "/vendors"),
        ("POST", "/vendors/v-acme/update-bank-accounts"),
        ("POST", "/bills"),
        ("POST", "/bills"),
    ]


async def test_seed_continues_without_bills_when_bill_pay_is_missing(seed_ramp, ramp, fake):
    fake.answers[("POST", "/bills")] = [403]
    result = await seed_ramp.seed(ramp, today=TODAY)
    assert result.bill_ids == []
    assert (result.vendor_id, result.bank_last4) == ("v-acme", "4417")
    assert len(fake.posts("/bills")) == 1  # a refusal is not repeated for the second bill


# --- what the seed puts in Ramp ---------------------------------------------------------------


async def test_the_fixture_values_are_made_up(seed_ramp, ramp, fake):
    await seed_ramp.seed(ramp, today=TODAY)
    [vendor_body] = fake.posts("/vendors")
    [bank_body] = fake.posts("/vendors/v-acme/update-bank-accounts")
    contact = vendor_body["business_vendor_contacts"]
    assert contact["email"].endswith("@example.com")
    assert not {"phone", "first_name", "last_name"} & contact.keys()  # no phone, no person's name
    assert bank_body["ach_details"]["routing_number"] == ROUTING
    assert bank_body["ach_details"]["account_number"] == ACCOUNT
    address = vendor_body["address"]
    assert address["address_line_1"] == "100 Example Way"  # the address the test fixtures use


async def test_the_bills_are_ach_payments_from_the_bill_pay_account_to_the_seeded_account(
    seed_ramp, ramp, fake
):
    await seed_ramp.seed(ramp, today=TODAY)
    first, second = fake.posts("/bills")
    assert [(b["invoice_number"], b["line_items"][0]["amount"]) for b in (first, second)] == [
        ("INV-1001", "1204.50"),
        ("INV-1002", "3980.00"),
    ]
    for body in (first, second):
        assert body["vendor_id"] == "v-acme"
        assert body["entity_id"] == ENTITY_ID
        assert body["invoice_currency"] == "USD"
        assert body["payment_method"] == "ACH"
        assert body["payment_details"]["source_bank_account_id"] == FUNDING_ID
        assert body["payment_details"]["vendor_account_id"] == "acct-4417"
        assert body["use_default_vendor_contact"] is True
        assert body["issued_at"] == "2026-11-08"
    # INV-1001 falls due on Sunday 2026-11-22 and a payment can't arrive on a weekend, so it is
    # asked to arrive on the Monday. INV-1002 falls due on a Tuesday and arrives that day.
    dates = [(b["due_at"], b["payment_details"]["payment_arrival_date"]) for b in (first, second)]
    assert dates == [("2026-11-22", "2026-11-23"), ("2026-12-08", "2026-12-08")]


async def test_the_due_dates_follow_the_day_the_seed_runs(seed_ramp, ramp, fake):
    await seed_ramp.seed(ramp)  # no `today`: the real date
    for body in fake.posts("/bills"):
        assert date.fromisoformat(body["due_at"]) > date.fromisoformat(body["issued_at"])
        assert date.fromisoformat(body["issued_at"]) == date.today()


# --- the arrival date -------------------------------------------------------------------------
# Ramp refuses a payment_arrival_date that isn't valid, and moves on only the payment_date. The
# due dates are 14 and 30 days after the run, so on a Thursday, Friday, Saturday or Sunday one of
# them is a weekend. The due date stays the calendar date; the arrival date moves to the Monday.


@pytest.mark.parametrize(
    "due, day, arrival",
    [
        pytest.param(date(2026, 12, 12), "Saturday", date(2026, 12, 14), id="saturday"),
        pytest.param(date(2026, 12, 13), "Sunday", date(2026, 12, 14), id="sunday"),
    ],
)
async def test_a_weekend_arrival_date_moves_to_the_monday_and_the_due_date_stays(
    seed_ramp, ramp, fake, due, day, arrival
):
    assert (due.strftime("%A"), arrival.strftime("%A")) == (day, "Monday")  # the dates are right
    today = due - timedelta(days=seed_ramp.INVOICES[0].due_in_days)  # INV-1001 falls due on `due`
    result = await seed_ramp.seed(ramp, today=today)
    first = fake.posts("/bills")[0]
    assert first["invoice_number"] == "INV-1001"
    assert first["due_at"] == due.isoformat()
    assert first["payment_details"]["payment_arrival_date"] == arrival.isoformat()
    assert len(result.bill_ids) == 2  # and Ramp took it, which it would not on the weekend date


RUN_WEEK = [date(2026, 11, 2) + timedelta(days=n) for n in range(7)]  # Monday to Sunday


@pytest.mark.parametrize("today", RUN_WEEK, ids=lambda day: day.strftime("%A"))
async def test_no_bill_is_asked_to_arrive_on_a_weekend_whatever_day_the_seed_runs(
    seed_ramp, ramp, fake, today
):
    result = await seed_ramp.seed(ramp, today=today)
    assert len(result.bill_ids) == 2  # the fake refuses a weekend arrival date, as Ramp does
    for body, invoice in zip(fake.posts("/bills"), seed_ramp.INVOICES, strict=True):
        due = today + timedelta(days=invoice.due_in_days)
        arrival = date.fromisoformat(body["payment_details"]["payment_arrival_date"])
        assert body["due_at"] == due.isoformat()  # the due date is the calendar date
        assert arrival.weekday() < 5  # Monday to Friday
        assert timedelta(0) <= arrival - due <= timedelta(days=2)  # no later than it must be
        assert (arrival == due) == (due.weekday() < 5)  # moved only when it had to be


# --- running it again -------------------------------------------------------------------------


async def test_seeding_twice_changes_nothing_the_second_time(seed_ramp, ramp, fake):
    first = await seed_ramp.seed(ramp, today=TODAY)
    writes = fake.writes()
    second = await seed_ramp.seed(ramp, today=TODAY)
    assert second == first
    assert fake.writes() == writes
    assert len(fake.vendors) == 1 and len(fake.bills) == 2


async def test_an_existing_vendor_without_the_account_gets_it(seed_ramp, ramp, fake):
    fake.vendors.append(vendor_json("v-old", "Acme Supplies"))
    result = await seed_ramp.seed(ramp, today=TODAY)
    assert fake.posts("/vendors") == []
    assert len(fake.posts("/vendors/v-old/update-bank-accounts")) == 1
    assert result.bank_last4 == "4417" and len(result.bill_ids) == 2


async def test_another_account_is_not_taken_for_the_seeded_one(seed_ramp, ramp, fake):
    fake.vendors.append(vendor_json("v-old", "Acme Supplies"))
    fake.accounts["v-old"].append(account_json("acct-6789", "6789"))
    await seed_ramp.seed(ramp, today=TODAY)
    assert len(fake.posts("/vendors/v-old/update-bank-accounts")) == 1


async def test_only_the_missing_bill_is_created(seed_ramp, ramp, fake):
    fake.have_seeded_vendor()
    fake.bills.append(
        {
            "id": "bill-kept",
            "invoice_number": "INV-1001",
            "status": "OPEN",
            "status_summary": "ON_HOLD",
            "amount": {"amount": 120450, "currency_code": "USD", "minor_unit_conversion_rate": 100},
            "vendor": {"id": "v-acme", "name": "Acme Supplies"},
        }
    )
    result = await seed_ramp.seed(ramp, today=TODAY)
    assert [b["invoice_number"] for b in fake.posts("/bills")] == ["INV-1002"]
    assert result.bill_ids == ["bill-kept", "bill-2"]  # in invoice order, old and new


async def test_a_bill_that_exists_needs_no_entity_lookup(seed_ramp, ramp, fake):
    await seed_ramp.seed(ramp, today=TODAY)
    reads = len(fake.requests)
    await seed_ramp.seed(ramp, today=TODAY)
    asked = [r.url.path for r in fake.requests[reads:]]
    assert PREFIX + "/entities" not in asked


# --- finding the Bill Pay bank account --------------------------------------------------------

SHAPES = [
    pytest.param({}, id="spec_accounts_with_a_usage_type"),
    pytest.param(
        {
            "accounts": [],
            "bank_accounts": [{"id": FUNDING_ID, "usage_type": "BILL_PAY_BANK_ACCOUNT"}],
        },
        id="guide_bank_accounts",
    ),
    pytest.param(
        {"accounts": [], "default_bill_pay_payment_account_id": FUNDING_ID}, id="entity_default"
    ),
]


@pytest.mark.parametrize("shape", SHAPES)
async def test_the_bill_pay_account_is_found_in_each_shape_ramp_describes(
    seed_ramp, ramp, fake, shape
):
    fake.entities = [entity(**shape)]
    result = await seed_ramp.seed(ramp, today=TODAY)
    sources = {b["payment_details"]["source_bank_account_id"] for b in fake.posts("/bills")}
    assert sources == {FUNDING_ID} and len(result.bill_ids) == 2


async def test_the_usage_type_wins_over_the_entitys_default(seed_ramp, ramp, fake):
    fake.entities = [entity(default_bill_pay_payment_account_id="bank-default")]
    await seed_ramp.seed(ramp, today=TODAY)
    [body, _] = fake.posts("/bills")
    assert body["payment_details"]["source_bank_account_id"] == FUNDING_ID


async def test_the_primary_entity_pays_and_one_without_bill_pay_is_passed_over(
    seed_ramp, ramp, fake
):
    only_cash = [{"bank_account_id": "bank-other", "usage_type": "CASH_ACCOUNT"}]
    fake.entities = [
        entity("ent-bare", primary=False, accounts=only_cash),
        entity("ent-side", primary=False, accounts=[funding("bank-side")]),
        entity("ent-main", primary=True, accounts=[funding("bank-main")]),
    ]
    await seed_ramp.seed(ramp, today=TODAY)
    [body, _] = fake.posts("/bills")
    assert (body["entity_id"], body["payment_details"]["source_bank_account_id"]) == (
        "ent-main",
        "bank-main",
    )

    fake.bills.clear()  # and when the primary entity cannot pay, the next one that can does
    fake.entities = [
        entity("ent-bare", primary=True, accounts=only_cash),
        entity("ent-side", primary=False, accounts=[funding("bank-side")]),
    ]
    await seed_ramp.seed(ramp, today=TODAY)
    body = fake.posts("/bills")[-1]
    assert (body["entity_id"], body["payment_details"]["source_bank_account_id"]) == (
        "ent-side",
        "bank-side",
    )


async def test_no_bill_pay_account_means_no_bills_and_no_bill_request(
    seed_ramp, ramp, fake, caplog
):
    caplog.set_level(logging.INFO, logger="seed_ramp")
    only_cash = [{"bank_account_id": "bank-other", "usage_type": "CASH_ACCOUNT"}]
    fake.entities = [entity(accounts=only_cash)]
    result = await seed_ramp.seed(ramp, today=TODAY)
    assert result.bill_ids == [] and fake.posts("/bills") == []
    assert "Bill Pay" in caplog.text


async def test_an_entity_answer_that_breaks_the_shape_means_no_bills(seed_ramp, ramp, fake):
    fake.entities = ["not an entity"]  # data should hold objects
    result = await seed_ramp.seed(ramp, today=TODAY)
    assert result.bill_ids == [] and fake.posts("/bills") == []


async def test_an_entity_call_that_fails_means_no_bills_not_a_failed_seed(
    seed_ramp, ramp, fake, pauses
):
    fake.answers[("GET", "/entities")] = [500, 500, 500]  # a read is tried three times
    result = await seed_ramp.seed(ramp, today=TODAY)
    assert result.bill_ids == [] and fake.posts("/bills") == []
    assert (result.vendor_id, result.bank_last4) == ("v-acme", "4417")
    assert pauses == [0.5, 1.0]


# --- a bill that is refused ---------------------------------------------------------------------
# Each bill stands alone, so one that Ramp refuses (a bad date, a duplicate invoice number) does not
# cost the others. A 401 or a 403 is different: the token or the scope is at fault, and every later
# bill would be refused the same way.


@pytest.mark.parametrize("status", [400, 422, 429, 500])
async def test_a_bill_that_is_refused_does_not_stop_the_next_one(
    seed_ramp, ramp, fake, caplog, status
):
    caplog.set_level(logging.INFO, logger="seed_ramp")
    fake.answers[("POST", "/bills")] = [status]  # INV-1001 is refused, INV-1002 goes through
    result = await seed_ramp.seed(ramp, today=TODAY)
    assert len(fake.posts("/bills")) == 2
    assert [bill["invoice_number"] for bill in fake.bills] == ["INV-1002"]
    assert result.bill_ids == [fake.bills[0]["id"]]  # INV-1002's id alone, no stand-in
    assert "INV-1001 not made" in caplog.text


async def test_the_next_run_makes_the_bill_that_was_refused(seed_ramp, ramp, fake):
    fake.answers[("POST", "/bills")] = [422]
    first = await seed_ramp.seed(ramp, today=TODAY)
    second = await seed_ramp.seed(ramp, today=TODAY)
    assert len(first.bill_ids) == 1 and len(second.bill_ids) == 2
    assert first.bill_ids[0] in second.bill_ids  # the bill that was made is not made again
    assert [bill["invoice_number"] for bill in fake.bills] == ["INV-1002", "INV-1001"]
    # The result is in invoice order, whichever Ramp made first.
    assert second.bill_ids == [fake.bills[1]["id"], fake.bills[0]["id"]]


@pytest.mark.parametrize(
    "refusals",
    [
        pytest.param([401, 401], id="401"),  # a 401 is sent again once with a new token: two end it
        pytest.param([403], id="403"),
    ],
)
async def test_a_refused_token_or_scope_ends_the_bills(seed_ramp, ramp, fake, refusals):
    fake.answers[("POST", "/bills")] = list(refusals)  # a copy: the fake uses its list up
    result = await seed_ramp.seed(ramp, today=TODAY)
    assert result.bill_ids == [] and fake.bills == []  # INV-1002 was not tried after INV-1001
    assert len(fake.posts("/bills")) == len(refusals)
    assert (result.vendor_id, result.bank_last4) == ("v-acme", "4417")


# --- the entities scope -----------------------------------------------------------------------
# GET /entities needs entities:read, and the agent's token (RampClient.SCOPES) leaves it out. The
# seed asks for it in a token of its own, so the agent's scopes stay as they are and an app
# without that scope costs the seed its bills and nothing else.


def test_the_entities_client_asks_for_that_scope_and_the_agents_scopes_stay_as_they_are(
    seed_ramp,
):
    assert seed_ramp.EntitiesClient.SCOPES == "entities:read"
    assert RampClient.SCOPES == " ".join(BASE_SCOPES)


async def test_the_agents_token_alone_cannot_read_entities_and_the_seed_says_what_to_do(
    seed_ramp, ramp, fake, caplog
):
    caplog.set_level(logging.INFO, logger="seed_ramp")
    fake.strict_scopes = True  # as in Ramp: `ramp`'s token has no entities:read
    result = await seed_ramp.seed(ramp, today=TODAY)
    assert result.bill_ids == [] and fake.posts("/bills") == []
    assert (result.vendor_id, result.bank_last4) == ("v-acme", "4417")
    assert "entities:read" in caplog.text


async def test_the_entities_are_read_with_the_client_given_for_them(seed_ramp, ramp, fake):
    fake.strict_scopes = True
    entities = seed_ramp.EntitiesClient(BASE, "cid", "secret")
    try:
        result = await seed_ramp.seed(ramp, entities=entities, today=TODAY)
    finally:
        await entities.aclose()
    assert len(result.bill_ids) == 2
    assert fake.token_scopes == [" ".join(BASE_SCOPES), "entities:read"]


async def test_the_entities_token_is_asked_for_only_when_a_bill_is_missing(seed_ramp, ramp, fake):
    entities = seed_ramp.EntitiesClient(BASE, "cid", "secret")
    try:
        await seed_ramp.seed(ramp, entities=entities, today=TODAY)
        asked = len(fake.token_scopes)
        await seed_ramp.seed(ramp, entities=entities, today=TODAY)
    finally:
        await entities.aclose()
    assert len(fake.token_scopes) == asked  # nothing missing the second time, so no new request


# --- when part of it fails --------------------------------------------------------------------


async def test_a_second_bill_that_fails_keeps_the_first(seed_ramp, ramp, fake):
    fake.answers[("POST", "/bills")] = [None, 422]
    result = await seed_ramp.seed(ramp, today=TODAY)
    assert result.bill_ids == ["bill-1"]
    assert len(fake.posts("/bills")) == 2


async def test_a_bank_account_waiting_for_approval_means_no_bills_yet(
    seed_ramp, ramp, fake, caplog
):
    caplog.set_level(logging.INFO, logger="seed_ramp")
    fake.bank_update = "pending_approval"
    result = await seed_ramp.seed(ramp, today=TODAY)
    # Acme has no such account yet, so the result does not claim one.
    assert (result.vendor_id, result.bank_last4, result.bill_ids) == ("v-acme", "", [])
    assert fake.posts("/bills") == []
    assert "approval" in caplog.text


async def test_once_the_account_is_approved_the_next_run_makes_the_bills(seed_ramp, ramp, fake):
    fake.bank_update = "pending_approval"
    await seed_ramp.seed(ramp, today=TODAY)
    fake.accounts["v-acme"].append(account_json())  # someone approves it in the Ramp web app
    result = await seed_ramp.seed(ramp, today=TODAY)
    assert len(fake.posts("/vendors/v-acme/update-bank-accounts")) == 1  # not proposed twice
    assert (result.bank_last4, result.bill_ids) == ("4417", ["bill-1", "bill-2"])


async def test_an_account_that_is_added_but_not_listed_yet_means_no_bills_yet(
    seed_ramp, ramp, fake, caplog
):
    caplog.set_level(logging.INFO, logger="seed_ramp")
    fake.listing_lags = True  # Ramp reads from a replica, which can trail a write
    result = await seed_ramp.seed(ramp, today=TODAY)
    assert (result.bank_last4, result.bill_ids) == ("", [])
    assert fake.posts("/bills") == []
    assert "run the seed again" in caplog.text.lower()


async def test_an_account_that_is_not_the_default_is_left_alone_with_a_warning(
    seed_ramp, ramp, fake, caplog
):
    caplog.set_level(logging.INFO, logger="seed_ramp")
    fake.vendors.append(vendor_json("v-old", "Acme Supplies"))
    fake.accounts["v-old"].append(account_json(default=False))
    result = await seed_ramp.seed(ramp, today=TODAY)
    assert fake.posts("/vendors/v-old/update-bank-accounts") == []
    assert result.bank_last4 == "4417"
    assert "not Acme's default" in caplog.text


async def test_a_vendor_that_cannot_be_created_stops_the_seed(seed_ramp, ramp, fake):
    fake.answers[("POST", "/vendors")] = [422]
    with pytest.raises(RampError) as error:
        await seed_ramp.seed(ramp, today=TODAY)
    assert error.value.status == 422
    assert fake.writes() == [("POST", "/vendors")]  # no account and no bills for a vendor not made


async def test_a_bank_account_that_cannot_be_added_stops_the_seed(seed_ramp, ramp, fake):
    fake.answers[("POST", "/vendors/v-acme/update-bank-accounts")] = [500]
    with pytest.raises(RampError) as error:
        await seed_ramp.seed(ramp, today=TODAY)
    assert error.value.status == 500
    assert fake.posts("/bills") == []


async def test_a_vendor_read_that_fails_stops_the_seed_before_any_write(seed_ramp, ramp, fake):
    fake.answers[("GET", "/vendors")] = [403]
    with pytest.raises(RampError):
        await seed_ramp.seed(ramp, today=TODAY)
    assert fake.writes() == []


# --- the log ----------------------------------------------------------------------------------


async def test_the_log_names_the_failure_but_never_a_bank_number_or_ramps_own_words(
    seed_ramp, ramp, fake, caplog
):
    caplog.set_level(logging.DEBUG)
    fake.answers[("POST", "/bills")] = [422]
    await seed_ramp.seed(ramp, today=TODAY)
    text = "\n".join(record.getMessage() for record in caplog.records)
    for wanted in ("422", "DEVELOPER_7001", "t-seed"):
        assert wanted in text
    for forbidden in (ACCOUNT, ROUTING, ECHO, "tok:"):
        assert forbidden not in text


# --- the command line -------------------------------------------------------------------------

ENV = {"RAMP_CLIENT_ID": "cid", "RAMP_CLIENT_SECRET": "secret", "RAMP_BASE_URL": BASE}


async def test_the_cli_names_the_missing_credentials(seed_ramp, capsys):
    assert await seed_ramp.run({}) == 2
    err = capsys.readouterr().err
    assert "RAMP_CLIENT_ID" in err and "RAMP_CLIENT_SECRET" in err


async def test_a_blank_credential_counts_as_missing(seed_ramp, capsys):
    # A .env copied from .env.example holds `RAMP_CLIENT_ID=` and loads as "".
    assert await seed_ramp.run({**ENV, "RAMP_CLIENT_ID": "  "}) == 2
    assert "RAMP_CLIENT_ID" in capsys.readouterr().err


async def test_the_cli_will_not_run_against_production(seed_ramp, api, fake, capsys):
    # The seed writes made-up bank details, which belong in the sandbox only.
    assert await seed_ramp.run({**ENV, "RAMP_BASE_URL": "https://api.ramp.com"}) == 2
    assert fake.requests == []
    assert "production" in capsys.readouterr().err.lower()


async def test_the_cli_seeds_and_prints_what_it_made(seed_ramp, api, fake, capsys):
    assert await seed_ramp.run(ENV) == 0
    out = capsys.readouterr().out
    for wanted in ("Acme Supplies", "v-acme", "4417", "bill-1", "bill-2"):
        assert wanted in out
    assert ACCOUNT not in out and ROUTING not in out


async def test_the_cli_reads_the_entities_with_a_token_of_their_own(seed_ramp, api, fake, capsys):
    fake.strict_scopes = True
    assert await seed_ramp.run(ENV) == 0
    assert fake.token_scopes == [" ".join(BASE_SCOPES), "entities:read"]  # the agent's, then theirs
    out = capsys.readouterr().out
    assert "bill-1" in out and "bill-2" in out


async def test_the_cli_loses_only_the_bills_when_the_app_does_not_allow_entities_read(
    seed_ramp, api, fake, capsys, caplog
):
    caplog.set_level(logging.INFO, logger="seed_ramp")
    fake.strict_scopes = True
    fake.app_scopes -= {"entities:read"}  # the app was set up with the agent's four scopes
    assert await seed_ramp.run(ENV) == 0  # the vendor and the account still come out right
    out = capsys.readouterr().out
    assert "v-acme" in out and "ending 4417" in out and "no bills" in out.lower()
    assert fake.posts("/bills") == []
    assert "entities:read" in caplog.text
    assert ECHO not in caplog.text


async def test_the_cli_says_so_when_there_are_no_bills(seed_ramp, api, fake, capsys):
    fake.answers[("POST", "/bills")] = [403]
    assert await seed_ramp.run(ENV) == 0  # the demo still runs on vendor holds
    assert "no bills" in capsys.readouterr().out.lower()


async def test_the_cli_reports_a_failed_seed_without_ramps_message(seed_ramp, api, fake, capsys):
    fake.answers[("POST", "/vendors")] = [422]
    assert await seed_ramp.run(ENV) == 1
    err = capsys.readouterr().err
    for wanted in ("422", "DEVELOPER_7001", "t-seed"):
        assert wanted in err
    assert ECHO not in err


async def test_the_cli_closes_both_its_clients_even_when_the_seed_fails(
    seed_ramp, api, fake, monkeypatch
):
    closed: list[bool] = []
    original = RampClient.aclose

    async def spy(self) -> None:
        closed.append(True)
        await original(self)

    monkeypatch.setattr(RampClient, "aclose", spy)
    fake.answers[("POST", "/vendors")] = [500]
    assert await seed_ramp.run(ENV) == 1
    assert closed == [True, True]  # the agent's client and the entities client


def test_it_imports_with_no_environment_and_no_network(seed_ramp):
    # Nothing is read or sent until the CLI runs: a test (or a REPL) can import the module freely.
    assert callable(seed_ramp.seed) and callable(seed_ramp.run)
