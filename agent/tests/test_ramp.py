import asyncio
import base64
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest
import respx
from pydantic import ValidationError

import callback_agent.ramp as ramp_module
from callback_agent.ramp import (
    BankAccountUpdate,
    Bill,
    BillVendor,
    HoldOutcome,
    Money,
    RampClient,
    RampError,
    Vendor,
    VendorBankAccount,
)

BASE = "https://ramp.test"
API = f"{BASE}/developer/v1"
FIRST_PAGE = f"{API}/vendors?page_size=100"
API_PATH = "/developer/v1"
FIXTURES = Path(__file__).parent / "fixtures" / "ramp"
CLIENT_ID = "cid"
CLIENT_SECRET = "secret"
ROUTING = "021000021"  # fake values, the same pair the plan uses
ACCOUNT = "000123456789"


def load(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


def b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


TOKEN = load("token.json")
ACCESS_TOKEN = TOKEN["access_token"]
ACME = load("vendors_page1.json")["data"][0]["id"]


def vendor_json(name: str, *, display_name: str | None = None, id_: str | None = None) -> dict:
    return {
        "id": id_ or name.lower().replace(" ", "-"),
        "name": name,
        "display_name": display_name,
        "is_active": True,
    }


def page_of(*items: Any, next: str | None = None) -> dict:
    return {"data": list(items), "page": {"next": next}}


def issue_tokens(route: respx.Route, *access_tokens: str) -> None:
    """Make the token endpoint hand out these tokens, one per request."""
    route.side_effect = [
        httpx.Response(200, json={**TOKEN, "access_token": t}) for t in access_tokens
    ]


def bearers(route: respx.Route) -> list[str]:
    return [call.request.headers["authorization"] for call in route.calls]


class Clock:
    """Stands in for time.monotonic, so a test can move time."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def pauses(request, monkeypatch):
    """Every wait the client asks for is recorded, not slept, so no test sits through a retry.

    The sandbox test is the exception: it calls the real Ramp, so a retry really waits.
    """
    waited: list[float] = []

    async def record(seconds: float) -> None:
        waited.append(seconds)

    if request.node.get_closest_marker("ramp_sandbox") is None:
        monkeypatch.setattr(ramp_module, "sleep", record)
    return waited


@pytest.fixture
def clock(monkeypatch):
    now = Clock()
    monkeypatch.setattr(ramp_module, "monotonic", now)
    return now


@pytest.fixture
def api():
    # A request that no test expected is an error, so a client that wanders off fails loudly.
    with respx.mock(base_url=BASE, assert_all_called=False) as router:
        yield router


@pytest.fixture
def token_route(api):
    return api.post(f"{API}/token").respond(200, json=TOKEN)


@pytest.fixture
async def ramp(api, token_route):
    client = RampClient(BASE, CLIENT_ID, CLIENT_SECRET)
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def vendor_pages(api):
    """The two pages of vendors: the first one links to the second through page.next."""
    page1, page2 = load("vendors_page1.json"), load("vendors_page2.json")
    first = api.get(FIRST_PAGE).respond(200, json=page1)
    second = api.get(page1["page"]["next"]).respond(200, json=page2)
    return first, second


# --- the token --------------------------------------------------------------------------------


async def test_token_fetched_once_and_cached(ramp, api, token_route):
    vendors = api.get(FIRST_PAGE).respond(200, json=page_of())
    await ramp.list_vendors()
    await ramp.list_vendors()
    assert token_route.call_count == 1
    token_req = token_route.calls.last.request
    assert token_req.headers["authorization"] == "Basic " + b64("cid:secret")
    assert token_req.headers["content-type"] == "application/x-www-form-urlencoded"
    assert parse_qs(token_req.content.decode()) == {
        "grant_type": ["client_credentials"],
        "scope": ["vendors:read vendors:write bills:read bills:write"],
    }
    assert RampClient.SCOPES == "vendors:read vendors:write bills:read bills:write"
    assert bearers(vendors) == [f"Bearer {ACCESS_TOKEN}"] * 2  # the secret goes to /token only


async def test_token_refreshed_once_on_401(ramp, api, token_route):
    issue_tokens(token_route, "tok-1", "tok-2")
    vendors = api.get(FIRST_PAGE)
    vendors.side_effect = [httpx.Response(401), httpx.Response(200, json=page_of(vendor_json("A")))]
    assert [v.name for v in await ramp.list_vendors()] == ["A"]
    assert token_route.call_count == 2
    assert bearers(vendors) == ["Bearer tok-1", "Bearer tok-2"]


async def test_a_second_401_is_an_error_not_another_refresh(ramp, api, token_route):
    vendors = api.get(FIRST_PAGE).respond(401)
    with pytest.raises(RampError) as error:
        await ramp.list_vendors()
    assert error.value.status == 401
    assert (token_route.call_count, vendors.call_count) == (2, 2)


async def test_a_write_refused_with_401_is_sent_once_more_with_a_new_token(ramp, api, token_route):
    # The one case where a write goes out twice: a 401 means Ramp turned the token away before it
    # ran anything, so the first send cannot have applied.
    issue_tokens(token_route, "tok-1", "tok-2")
    hold = api.post(f"{API}/vendors/v1/hold")
    hold.side_effect = [httpx.Response(401), httpx.Response(204)]
    assert await ramp.hold_vendor("v1", "x") is HoldOutcome.APPLIED
    assert bearers(hold) == ["Bearer tok-1", "Bearer tok-2"]


async def test_token_is_renewed_a_minute_before_it_expires(ramp, api, token_route, clock):
    token_route.respond(200, json={**TOKEN, "expires_in": 300})
    api.get(FIRST_PAGE).respond(200, json=page_of())
    await ramp.list_vendors()
    clock.now += 239  # the token is good until 300 s, and renewed from 240 s
    await ramp.list_vendors()
    assert token_route.call_count == 1
    clock.now += 2
    await ramp.list_vendors()
    assert token_route.call_count == 2


async def test_concurrent_calls_share_one_token_fetch(ramp, api, token_route):
    async def slow_token(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0)  # let the other calls reach the token step
        return httpx.Response(200, json=TOKEN)

    token_route.side_effect = slow_token
    api.get(FIRST_PAGE).respond(200, json=page_of())
    api.get(f"{API}/bills?page_size=100&payment_status=OPEN").respond(200, json=page_of())
    await asyncio.gather(ramp.list_vendors(), ramp.list_vendors(), ramp.list_bills())
    assert token_route.call_count == 1


async def test_calls_refused_together_renew_the_token_once(ramp, api, token_route):
    issue_tokens(token_route, "tok-1", "tok-2", "tok-3")
    both_sent = asyncio.Barrier(2)

    async def vendors(request: httpx.Request) -> httpx.Response:
        if request.headers["authorization"] == "Bearer tok-1":
            await both_sent.wait()  # neither call hears back before both have sent tok-1
            return httpx.Response(401)
        return httpx.Response(200, json=page_of())

    api.get(FIRST_PAGE).mock(side_effect=vendors)
    await asyncio.gather(ramp.list_vendors(), ramp.list_vendors())
    assert token_route.call_count == 2  # tok-2 serves both: there is no tok-3


async def test_a_rejected_token_request_is_a_ramp_error_and_nothing_else_is_sent(
    ramp, api, token_route
):
    token_route.respond(
        400,
        json={
            "error_v2": {"error_code": "DEVELOPER_7012", "message": "Client authentication failed."}
        },
        headers={"x-trace-id": "t-3"},
    )
    vendors = api.get(FIRST_PAGE).respond(200, json=page_of())
    with pytest.raises(RampError) as error:
        await ramp.list_vendors()
    assert (error.value.status, error.value.error_code, error.value.trace_id) == (
        400,
        "DEVELOPER_7012",
        "t-3",
    )
    assert (token_route.call_count, vendors.call_count) == (1, 0)
    assert b64("cid:secret") not in str(error.value)


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param({"token_type": "Bearer", "expires_in": 864000}, id="no_token"),
        pytest.param({**TOKEN, "access_token": ""}, id="empty_token"),
        pytest.param({**TOKEN, "expires_in": "soon"}, id="no_lifetime"),
    ],
)
async def test_a_token_answer_that_breaks_the_shape_is_a_ramp_error(ramp, api, token_route, answer):
    token_route.respond(200, json=answer)
    with pytest.raises(RampError) as error:
        await ramp.list_vendors()
    assert error.value.status == 200


async def test_every_request_has_a_ten_second_timeout(ramp, api, token_route):
    vendors = api.get(FIRST_PAGE).respond(200, json=page_of())
    await ramp.list_vendors()
    for route in (token_route, vendors):
        assert set(route.calls.last.request.extensions["timeout"].values()) == {10.0}


async def test_requests_say_what_they_accept_and_who_sends_them(ramp, api):
    vendors = api.get(FIRST_PAGE).respond(200, json=page_of())
    await ramp.list_vendors()
    headers = vendors.calls.last.request.headers
    assert (headers["accept"], headers["user-agent"]) == ("application/json", "callback-agent")


# --- reads ------------------------------------------------------------------------------------


async def test_list_vendors_follows_page_next(ramp, vendor_pages):
    first, second = vendor_pages
    vendors = await ramp.list_vendors()
    assert [v.name for v in vendors] == ["Acme Supplies Inc", "Globex Corporation", "Initech LLC"]
    assert [v.display_name for v in vendors] == ["Acme Supplies", "Globex", None]
    assert all(v.is_active for v in vendors)
    assert (first.call_count, second.call_count) == (1, 1)  # the second URL is page.next, as is


@pytest.mark.parametrize(
    "next_url",
    [
        pytest.param("https://evil.test/developer/v1/vendors?start=x", id="other_host"),
        pytest.param(
            "https://ramp.test.evil.test/developer/v1/vendors?start=x", id="lookalike_host"
        ),
        pytest.param("http://ramp.test/developer/v1/vendors?start=x", id="other_scheme"),
        pytest.param("https://ramp.test:8443/developer/v1/vendors?start=x", id="other_port"),
        pytest.param("/developer/v1/vendors?start=x", id="not_absolute"),
    ],
)
async def test_a_page_next_that_leaves_the_base_url_is_not_followed(ramp, api, next_url):
    # Following it would hand the bearer token to whoever the answer names.
    api.get(FIRST_PAGE).respond(200, json=page_of(vendor_json("A"), next=next_url))
    stray = api.route().respond(200, json=page_of())
    with pytest.raises(RampError):
        await ramp.list_vendors()
    assert stray.call_count == 0


async def test_list_vendor_accounts_reads_last_four_only(ramp, api):
    route = api.get(f"{API}/vendors/{ACME}/accounts?page_size=100")
    route.respond(200, json=load("vendor_accounts.json"))
    accounts = await ramp.list_vendor_accounts(ACME)
    assert [
        (a.account_number_last_four, a.is_default, a.status, a.payment_method) for a in accounts
    ] == [("4417", True, "active", "ACH"), (None, False, "active", "CHECK")]


def test_bank_account_model_has_no_full_numbers():
    assert not {"account_number", "routing_number"} & set(VendorBankAccount.model_fields)
    # Were Ramp ever to send them, the model would drop them rather than hold them.
    account = VendorBankAccount.model_validate(
        {
            **load("vendor_accounts.json")["data"][0],
            "account_number": ACCOUNT,
            "routing_number": ROUTING,
        }
    )
    assert ACCOUNT not in repr(account) + str(account.model_dump())
    assert ROUTING not in repr(account) + str(account.model_dump())


async def test_list_bills_parses_the_bill_and_defaults_to_open(ramp, api):
    route = api.get(f"{API}/bills?page_size=100&payment_status=OPEN")
    route.respond(200, json=load("bills.json"))
    bills = await ramp.list_bills()
    assert [b.invoice_number for b in bills] == ["INV-1001", "INV-1002"]
    first = bills[0]
    assert (first.status, first.status_summary) == ("OPEN", "PAYMENT_SCHEDULED")
    assert first.amount.spoken() == "$1,204.50"
    assert first.due_at == datetime(2026, 11, 20, tzinfo=UTC)
    assert first.vendor == BillVendor(id=ACME, name="Acme Supplies Inc")
    assert bills[1].amount.spoken() == "$3,980.00"


@pytest.mark.parametrize(
    "kwargs, query",
    [
        pytest.param({}, "page_size=100&payment_status=OPEN", id="default_is_open"),
        pytest.param({"payment_status": None}, "page_size=100", id="no_status_filter"),
        pytest.param(
            {"vendor_id": "v1", "payment_status": "PAID"},
            "page_size=100&vendor_id=v1&payment_status=PAID",
            id="vendor_and_status",
        ),
    ],
)
async def test_list_bills_sends_only_the_filters_it_was_given(ramp, api, kwargs, query):
    route = api.get(f"{API}/bills?{query}").respond(200, json=page_of())
    assert await ramp.list_bills(**kwargs) == []
    assert route.call_count == 1


async def test_list_bills_follows_page_next(ramp, api):
    bills = load("bills.json")["data"]
    next_url = f"{API}/bills?page_size=100&payment_status=OPEN&start={bills[0]['id']}"
    api.get(f"{API}/bills?page_size=100&payment_status=OPEN").respond(
        200, json=page_of(bills[0], next=next_url)
    )
    api.get(next_url).respond(200, json=page_of(bills[1]))
    assert [b.invoice_number for b in await ramp.list_bills()] == ["INV-1001", "INV-1002"]


# --- find_vendor ------------------------------------------------------------------------------


async def test_find_vendor_is_case_insensitive_and_fuzzy(ramp, vendor_pages):
    assert (await ramp.find_vendor("acme supplies")).name == "Acme Supplies Inc"
    assert (await ramp.find_vendor("   ACME          Supplies  ")).name == "Acme Supplies Inc"
    assert (await ramp.find_vendor("acme suplies")).name == "Acme Supplies Inc"  # a typo
    assert await ramp.find_vendor("zzz") is None


async def test_find_vendor_also_matches_the_display_name(ramp, api):
    api.get(FIRST_PAGE).respond(
        200, json=page_of(vendor_json("Northwind Traders LLC", display_name="NWT"))
    )
    assert (await ramp.find_vendor("nwt")).name == "Northwind Traders LLC"


async def test_find_vendor_prefers_the_closest_substring_match(ramp, api):
    api.get(FIRST_PAGE).respond(
        200,
        json=page_of(
            vendor_json("Acme Supplies International"),
            vendor_json("Acme Supplies Inc", id_="first"),
            vendor_json("Acme Supplies Inc", id_="second"),
            vendor_json("Acme"),
        ),
    )
    assert (await ramp.find_vendor("acme")).name == "Acme"  # the exact name beats longer ones
    found = await ramp.find_vendor("acme supplies")
    assert (found.name, found.id) == ("Acme Supplies Inc", "first")  # shortest, then Ramp's order


async def test_find_vendor_substring_match_beats_a_closer_spelling(ramp, api):
    # "acme corp" is 0.89 similar to "Acne Corp" but is contained in the other name.
    api.get(FIRST_PAGE).respond(
        200, json=page_of(vendor_json("Acne Corp"), vendor_json("Acme Corp International"))
    )
    assert (await ramp.find_vendor("acme corp")).name == "Acme Corp International"


async def test_find_vendor_accepts_a_similarity_of_exactly_point_eight_and_no_less(ramp, api):
    # "abcde" and "abcdf" share 4 of 5 letters: 2*4/10 = 0.8. "mnopqrsta" and "mnopqrsxy": 14/18.
    api.get(FIRST_PAGE).respond(200, json=page_of(vendor_json("abcdf"), vendor_json("mnopqrsxy")))
    assert (await ramp.find_vendor("abcde")).name == "abcdf"
    assert await ramp.find_vendor("mnopqrsta") is None


async def test_find_vendor_breaks_a_tie_in_spelling_by_ramps_order(ramp, api):
    # "abcdz" is 0.8 similar to both, and neither contains it.
    api.get(FIRST_PAGE).respond(200, json=page_of(vendor_json("abcdx"), vendor_json("abcdy")))
    assert (await ramp.find_vendor("abcdz")).name == "abcdx"


@pytest.mark.parametrize("name", ["", "   ", "\n"])
async def test_find_vendor_with_a_blank_name_finds_nothing(ramp, vendor_pages, name):
    # "" is a substring of every name, so a careless match would pick a vendor at random.
    assert await ramp.find_vendor(name) is None


async def test_find_vendor_with_no_vendors_finds_nothing(ramp, api):
    api.get(FIRST_PAGE).respond(200, json=page_of())
    assert await ramp.find_vendor("acme") is None


# --- vendor holds -----------------------------------------------------------------------------

VENDOR_CALLS = [
    pytest.param(
        lambda r: r.hold_vendor("v1", "Callback hold"),
        "hold",
        {"reason": "Callback hold"},
        id="hold",
    ),
    pytest.param(lambda r: r.release_vendor("v1"), "release", None, id="release"),
]


@pytest.mark.parametrize(
    "status, outcome", [(204, HoldOutcome.APPLIED), (202, HoldOutcome.PENDING_APPROVAL)]
)
@pytest.mark.parametrize("call, action, body", VENDOR_CALLS)
async def test_vendor_hold_204_is_applied_and_202_is_pending(
    ramp, api, call, action, body, status, outcome
):
    route = api.post(f"{API}/vendors/v1/{action}")
    if status == 202:
        route.respond(202, json={"whatever": "Ramp says"})  # untyped, so never read
    else:
        route.respond(204)
    assert await call(ramp) is outcome
    request = route.calls.last.request
    assert request.headers["authorization"] == f"Bearer {ACCESS_TOKEN}"
    assert (json.loads(request.content) if request.content else None) == body


@pytest.mark.parametrize("call, action, body", VENDOR_CALLS)
async def test_a_vendor_hold_answered_with_any_other_2xx_is_not_taken_for_applied(
    ramp, api, call, action, body
):
    api.post(f"{API}/vendors/v1/{action}").respond(200, json={})
    with pytest.raises(RampError) as error:
        await call(ramp)
    assert error.value.status == 200


# --- bill holds -------------------------------------------------------------------------------


async def test_bill_hold_returns_bill_on_hold(ramp, api):
    route = api.post(f"{API}/bills/b1/hold").respond(200, json=load("bill_on_hold.json"))
    bill = await ramp.hold_bill("b1", "Callback hold")
    assert bill.status_summary == "ON_HOLD"
    assert json.loads(route.calls.last.request.content) == {"reason": "Callback hold"}


async def test_bill_release_returns_the_bill(ramp, api):
    released = {**load("bill_on_hold.json"), "status_summary": "PAYMENT_READY"}
    route = api.post(f"{API}/bills/b1/release").respond(200, json=released)
    bill = await ramp.release_bill("b1")
    assert (bill.status_summary, bill.invoice_number) == ("PAYMENT_READY", "INV-1001")
    assert route.calls.last.request.content == b""


# --- changing the bank account ----------------------------------------------------------------


async def test_add_default_bank_account_body_and_pending(ramp, api):
    route = api.post(f"{API}/vendors/v1/update-bank-accounts")
    route.respond(200, json=load("bank_update_pending.json"))
    result = await ramp.add_default_bank_account(
        "v1", routing_number="021000021", account_number="000123456789"
    )
    req = route.calls.last.request
    assert json.loads(req.content) == {
        "is_default": True,
        "ach_details": {
            "routing_number": "021000021",
            "account_number": "000123456789",
            "account_type": "Checking",
            "account_owner_type": "BUSINESS",
        },
    }
    assert result.status == "pending_approval" and result.payment_method_id is None
    assert (result.vendor_id, result.proposal_id) == (ACME, "a2c4e6f8-1357-4b9d-8e0a-2c4e6f8a0b13")


async def test_add_default_bank_account_approved(ramp, api):
    api.post(f"{API}/vendors/v1/update-bank-accounts").respond(
        200, json=load("bank_update_approved.json")
    )
    result = await ramp.add_default_bank_account(
        "v1", routing_number=ROUTING, account_number=ACCOUNT
    )
    assert result == BankAccountUpdate(
        vendor_id=ACME,
        status="approved",
        payment_method_id="d3f5a7c9-0b1e-4d28-9a64-5c7e1f3b8d02",
        proposal_id=None,
    )


async def test_a_bank_update_status_that_is_neither_approved_nor_pending_is_an_error(ramp, api):
    # Reading an unknown status as success would tell the caller a change went through.
    api.post(f"{API}/vendors/v1/update-bank-accounts").respond(
        200, json={"vendor_id": ACME, "status": "declined"}
    )
    with pytest.raises(RampError) as error:
        await ramp.add_default_bank_account("v1", routing_number=ROUTING, account_number=ACCOUNT)
    assert "status" in str(error.value)


async def test_bank_update_errors_never_echo_the_account_or_routing_number(ramp, api):
    api.post(f"{API}/vendors/v1/update-bank-accounts").respond(
        422,
        json={
            "error_v2": {
                "error_code": "DEVELOPER_7001",
                "message": f"Account {ACCOUNT} does not match routing {ROUTING}",
            }
        },
    )
    with pytest.raises(RampError) as error:
        await ramp.add_default_bank_account("v1", routing_number=ROUTING, account_number=ACCOUNT)
    err = error.value
    assert err.error_code == "DEVELOPER_7001"
    for number in (ACCOUNT, ROUTING):
        assert number not in err.message + str(err) + repr(err)


# --- request, for scripts ---------------------------------------------------------------------
# It is the public way in to the private send path, so it keeps every guarantee of the methods
# above. The sandbox seed is its user. WRITES, below, also runs it through the never-retried tests.


async def test_request_sends_the_cached_token_and_the_body_and_returns_the_answer(
    ramp, api, token_route
):
    api.get(FIRST_PAGE).respond(200, json=page_of())
    created = api.post(f"{API}/vendors").respond(201, json={"id": "v9"})
    await ramp.list_vendors()  # this call fetches the token
    response = await ramp.request("POST", "/vendors", json={"name": "Acme"})
    assert (response.status_code, response.json()) == (201, {"id": "v9"})
    sent = created.calls.last.request
    assert json.loads(sent.content) == {"name": "Acme"}
    assert sent.headers["authorization"] == f"Bearer {ACCESS_TOKEN}"
    assert token_route.call_count == 1  # the token was kept, not fetched again


async def test_request_without_a_body_sends_none(ramp, api):
    seen = api.get(f"{API}/entities").respond(200, json={"data": []})
    await ramp.request("GET", "/entities")
    assert seen.calls.last.request.content == b""


@pytest.mark.parametrize("status", [200, 201, 202, 204])
async def test_request_returns_any_2xx_answer(ramp, api, status):
    api.post(f"{API}/vendors/v1/hold").respond(status)
    assert (await ramp.request("POST", "/vendors/v1/hold")).status_code == status


async def test_request_gets_a_new_token_once_on_a_401_for_a_read_and_for_a_write(
    ramp, api, token_route
):
    issue_tokens(token_route, "tok-1", "tok-2", "tok-3")
    entities = api.get(f"{API}/entities")
    entities.side_effect = [httpx.Response(401), httpx.Response(200, json={"data": []})]
    created = api.post(f"{API}/vendors")
    created.side_effect = [httpx.Response(401), httpx.Response(200, json={"id": "v9"})]
    await ramp.request("GET", "/entities")
    await ramp.request("POST", "/vendors", json={"name": "Acme"})
    assert bearers(entities) == ["Bearer tok-1", "Bearer tok-2"]
    assert bearers(created) == ["Bearer tok-2", "Bearer tok-3"]


async def test_request_does_not_refresh_a_second_time_after_a_second_401(ramp, api, token_route):
    created = api.post(f"{API}/vendors").respond(401)
    with pytest.raises(RampError) as error:
        await ramp.request("POST", "/vendors", json={})
    assert error.value.status == 401
    assert (token_route.call_count, created.call_count) == (2, 2)


async def test_request_errors_are_ramp_errors_with_the_status_the_code_and_the_trace(ramp, api):
    api.post(f"{API}/vendors").respond(
        422, json=load("error_422.json"), headers={"x-trace-id": "t-2"}
    )
    with pytest.raises(RampError) as error:
        await ramp.request("POST", "/vendors", json={})
    err = error.value
    assert (err.status, err.error_code, err.trace_id) == (422, "DEVELOPER_7001", "t-2")


async def test_request_retries_a_read_like_every_other_read(ramp, api, pauses):
    route = api.get(f"{API}/entities")
    route.side_effect = [
        httpx.Response(429, headers={"Retry-After": "0"}),
        httpx.Response(200, json={"data": []}),
    ]
    # Lower case on purpose: the method is normalized, or this would count as a write.
    assert (await ramp.request("get", "/entities")).status_code == 200
    assert route.call_count == 2
    assert pauses == [0.0]


async def test_request_logs_the_call_and_the_trace_but_never_a_body_or_the_token(ramp, api, caplog):
    caplog.set_level(logging.DEBUG, logger="callback_agent.ramp")
    api.post(f"{API}/vendors").respond(
        422,
        json={"error_v2": {"error_code": "DEVELOPER_7001", "message": "echo hunter2"}},
        headers={"x-trace-id": "t-7"},
    )
    api.post(f"{API}/bills").respond(201, json={"id": "b9"}, headers={"x-trace-id": "t-8"})
    with pytest.raises(RampError):
        await ramp.request(
            "POST", "/vendors", json={"name": "Acme Private", "email": "owner@acme.test"}
        )
    await ramp.request("POST", "/bills", json={"invoice_number": "INV-PRIVATE"})
    text = "\n".join(r.getMessage() for r in caplog.records if r.name == "callback_agent.ramp")
    for wanted in ("/vendors", "422", "DEVELOPER_7001", "t-7", "/bills", "201", "t-8"):
        assert wanted in text
    for forbidden in (
        "Acme Private",
        "owner@acme.test",
        "INV-PRIVATE",
        "echo",
        "hunter2",
        ACCESS_TOKEN,
    ):
        assert forbidden not in text


@pytest.mark.parametrize("path", ["entities", "", "https://elsewhere.test/entities"])
async def test_request_refuses_a_path_that_does_not_start_with_a_slash(ramp, api, path):
    # "entities" would glue onto the prefix as /developer/v1entities, and a full URL is no path.
    seen = api.route().respond(200)
    with pytest.raises(ValueError):
        await ramp.request("GET", path)
    assert seen.call_count == 0


async def test_request_cannot_be_pointed_at_another_host(ramp, api):
    seen = api.route().respond(200, json={})
    await ramp.request("GET", "//elsewhere.test/entities")
    assert seen.calls.last.request.url.host == "ramp.test"


# --- writes are never retried -----------------------------------------------------------------

WRITES = [
    pytest.param(lambda r: r.request("POST", "/vendors", json={}), "/vendors", id="request"),
    pytest.param(lambda r: r.hold_vendor("v1", "x"), "/vendors/v1/hold", id="hold_vendor"),
    pytest.param(lambda r: r.release_vendor("v1"), "/vendors/v1/release", id="release_vendor"),
    pytest.param(lambda r: r.hold_bill("b1", "x"), "/bills/b1/hold", id="hold_bill"),
    pytest.param(lambda r: r.release_bill("b1"), "/bills/b1/release", id="release_bill"),
    pytest.param(
        lambda r: r.add_default_bank_account("v1", routing_number=ROUTING, account_number=ACCOUNT),
        "/vendors/v1/update-bank-accounts",
        id="add_default_bank_account",
    ),
]


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
@pytest.mark.parametrize("call, path", WRITES)
async def test_writes_are_never_retried(ramp, api, pauses, call, path, status):
    # A 504 does not prove the write failed, so a second send could apply it twice.
    route = api.post(API + path).respond(status, headers={"Retry-After": "0"})
    with pytest.raises(RampError) as error:
        await call(ramp)
    assert error.value.status == status
    assert route.call_count == 1
    assert pauses == []


@pytest.mark.parametrize(
    "failure",
    [httpx.ReadTimeout("slow"), httpx.ConnectError("down")],
    ids=["read_timeout", "connect_error"],
)
@pytest.mark.parametrize("call, path", WRITES)
async def test_a_write_that_gets_no_answer_is_not_sent_again(ramp, api, call, path, failure):
    route = api.post(API + path).mock(side_effect=failure)
    with pytest.raises(RampError) as error:
        await call(ramp)
    assert error.value.status == 0  # no HTTP answer at all
    assert route.call_count == 1


# --- reads are retried ------------------------------------------------------------------------


async def test_reads_retry_429_then_succeed(ramp, api, pauses):
    route = api.get(FIRST_PAGE)
    route.side_effect = [
        httpx.Response(429, headers={"Retry-After": "0"}),
        httpx.Response(200, json=page_of(vendor_json("A"))),
    ]
    assert [v.name for v in await ramp.list_vendors()] == ["A"]
    assert route.call_count == 2
    assert pauses == [0.0]


async def test_reads_retry_a_5xx_twice_waiting_half_a_second_then_one(ramp, api, pauses):
    route = api.get(FIRST_PAGE)
    route.side_effect = [
        httpx.Response(503),
        httpx.Response(502),
        httpx.Response(200, json=page_of(vendor_json("A"))),
    ]
    assert [v.name for v in await ramp.list_vendors()] == ["A"]
    assert route.call_count == 3
    assert pauses == [0.5, 1.0]


async def test_reads_give_up_after_two_retries(ramp, api, pauses):
    route = api.get(FIRST_PAGE).respond(503, headers={"x-trace-id": "t-4"})
    with pytest.raises(RampError) as error:
        await ramp.list_vendors()
    assert (error.value.status, error.value.trace_id) == (503, "t-4")
    assert route.call_count == 3
    assert pauses == [0.5, 1.0]


@pytest.mark.parametrize("status", [400, 403, 404, 422])
async def test_reads_do_not_retry_a_client_error(ramp, api, pauses, status):
    route = api.get(FIRST_PAGE).respond(status)
    with pytest.raises(RampError) as error:
        await ramp.list_vendors()
    assert error.value.status == status
    assert route.call_count == 1
    assert pauses == []


@pytest.mark.parametrize(
    "retry_after, waits",
    [
        pytest.param("0", 0.0, id="zero"),
        pytest.param("2", 2.0, id="seconds"),
        pytest.param("1.5", 1.5, id="fraction"),
        pytest.param("Wed, 21 Oct 2026 07:28:00 GMT", 0.5, id="an_http_date_uses_the_usual_wait"),
        pytest.param("-3", 0.5, id="negative_uses_the_usual_wait"),
        pytest.param("soon", 0.5, id="garbage_uses_the_usual_wait"),
    ],
)
async def test_retry_after_sets_the_wait(ramp, api, pauses, retry_after, waits):
    route = api.get(FIRST_PAGE)
    route.side_effect = [
        httpx.Response(429, headers={"Retry-After": retry_after}),
        httpx.Response(200, json=page_of()),
    ]
    await ramp.list_vendors()
    assert pauses == [waits]


@pytest.mark.parametrize("retry_after", ["60", "5.5", "inf"])
async def test_a_wait_too_long_for_a_phone_call_fails_at_once(ramp, api, pauses, retry_after):
    # Ramp's rate-limit lockout is 60 s. Nobody holds the line that long to find out.
    route = api.get(FIRST_PAGE).respond(429, headers={"Retry-After": retry_after})
    with pytest.raises(RampError) as error:
        await ramp.list_vendors()
    assert error.value.status == 429
    assert (route.call_count, pauses) == (1, [])


async def test_a_wait_of_exactly_five_seconds_is_honored(ramp, api, pauses):
    route = api.get(FIRST_PAGE)
    route.side_effect = [
        httpx.Response(429, headers={"Retry-After": "5"}),
        httpx.Response(200, json=page_of()),
    ]
    await ramp.list_vendors()
    assert pauses == [5.0]


async def test_a_read_that_gets_no_answer_is_a_ramp_error_and_is_not_retried(ramp, api):
    route = api.get(FIRST_PAGE).mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(RampError) as error:
        await ramp.list_vendors()
    assert error.value.status == 0
    assert "ReadTimeout" in str(error.value)
    assert route.call_count == 1


async def test_a_retry_after_a_401_still_gets_its_own_retries(ramp, api, token_route, pauses):
    issue_tokens(token_route, "tok-1", "tok-2")
    route = api.get(FIRST_PAGE)
    route.side_effect = [
        httpx.Response(503),
        httpx.Response(401),
        httpx.Response(503),
        httpx.Response(200, json=page_of(vendor_json("A"))),
    ]
    assert [v.name for v in await ramp.list_vendors()] == ["A"]
    assert bearers(route) == ["Bearer tok-1"] * 2 + ["Bearer tok-2"] * 2
    assert pauses == [0.5, 1.0]


# --- errors -----------------------------------------------------------------------------------


async def test_error_v2_is_parsed(ramp, api):
    api.get(FIRST_PAGE).respond(422, json=load("error_422.json"), headers={"x-trace-id": "t-1"})
    with pytest.raises(RampError) as error:
        await ramp.list_vendors()
    err = error.value
    assert (err.status, err.error_code, err.trace_id) == (422, "DEVELOPER_7001", "t-1")
    assert err.message == "Value must be between 2 and 100"
    assert all(part in str(err) for part in ("422", "DEVELOPER_7001", "t-1", err.message))


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"text": "<html>Bad gateway for account 000123456789</html>"}, id="html"),
        pytest.param(
            {"json": {"error": {"page_size": ["Value must be between 2 and 100"]}}},
            id="legacy_error",
        ),
        pytest.param({"json": {"error_v2": "oops"}}, id="error_v2_is_not_an_object"),
        pytest.param({"content": b""}, id="empty"),
    ],
)
async def test_an_error_without_error_v2_keeps_the_status_and_trace_but_not_the_body(
    ramp, api, body
):
    api.get(FIRST_PAGE).respond(404, headers={"x-trace-id": "t-2"}, **body)
    with pytest.raises(RampError) as error:
        await ramp.list_vendors()
    err = error.value
    assert (err.status, err.error_code, err.trace_id) == (404, None, "t-2")
    assert err.message  # never empty: it says at least what the status means
    assert "<html>" not in str(err) and ACCOUNT not in str(err) and "page_size" not in str(err)


async def test_a_response_without_a_trace_id_has_none(ramp, api):
    api.get(FIRST_PAGE).respond(404)
    with pytest.raises(RampError) as error:
        await ramp.list_vendors()
    assert error.value.trace_id is None


async def test_a_malformed_answer_names_the_fields_but_not_the_values(ramp, api):
    api.get(FIRST_PAGE).respond(
        200, json=page_of({"id": "v1", "name": "Acme", "is_active": "sort-of-secret-value"})
    )
    with pytest.raises(RampError) as error:
        await ramp.list_vendors()
    assert error.value.status == 200
    assert "is_active" in str(error.value)
    assert "sort-of-secret-value" not in str(error.value)


async def test_an_answer_that_is_not_json_is_a_ramp_error(ramp, api):
    api.get(FIRST_PAGE).respond(200, text="not json")
    with pytest.raises(RampError) as error:
        await ramp.list_vendors()
    assert error.value.status == 200


async def test_a_redirect_is_an_error_and_is_not_followed_even_by_a_client_that_would():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path.endswith("/token"):
            return httpx.Response(200, json=TOKEN)
        if request.url.host == "ramp.test":
            return httpx.Response(307, headers={"location": "https://elsewhere.test/vendors"})
        return httpx.Response(200, json=page_of())

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)
    client = RampClient(BASE, CLIENT_ID, CLIENT_SECRET, http=http)
    try:
        with pytest.raises(RampError) as error:
            await client.list_vendors()
    finally:
        await client.aclose()
    assert error.value.status == 307
    assert not any("elsewhere.test" in url for url in seen)


def test_ramp_error_carries_what_the_gate_reads():
    err = RampError(502, "DEVELOPER_7100", "boom", "t-9")
    assert (err.status, err.error_code, err.message, err.trace_id) == (
        502,
        "DEVELOPER_7100",
        "boom",
        "t-9",
    )
    bare = RampError(status=500, message="boom")
    assert (bare.error_code, bare.trace_id) == (None, None)
    assert isinstance(err, Exception)


def test_redacting_an_empty_value_leaves_the_message_alone():
    response = httpx.Response(422, json={"error_v2": {"message": "Account 123 is wrong"}})
    assert RampError.from_response(response, redact=("", "123")).message == (
        "Account [redacted] is wrong"
    )


# --- logging ----------------------------------------------------------------------------------


async def test_logs_name_the_call_and_the_trace_but_never_a_body_a_token_or_the_secret(api, caplog):
    caplog.set_level(logging.DEBUG, logger="callback_agent.ramp")
    api.post(f"{API}/token").respond(200, json=TOKEN)
    api.post(f"{API}/vendors/v1/update-bank-accounts").respond(
        422,
        json={"error_v2": {"error_code": "DEVELOPER_7001", "message": f"echo {ACCOUNT}"}},
        headers={"x-trace-id": "t-5"},
    )
    api.post(f"{API}/vendors/v1/hold").respond(204, headers={"x-trace-id": "t-6"})
    client = RampClient(BASE, CLIENT_ID, "hunter2-s3cr3t")
    try:
        with pytest.raises(RampError):
            await client.add_default_bank_account(
                "v1", routing_number=ROUTING, account_number=ACCOUNT
            )
        await client.hold_vendor("v1", "caller reads 4417 aloud")
    finally:
        await client.aclose()
    text = "\n".join(r.getMessage() for r in caplog.records if r.name == "callback_agent.ramp")
    for wanted in (
        "/vendors/v1/update-bank-accounts",
        "422",
        "DEVELOPER_7001",
        "t-5",
        "/vendors/v1/hold",
        "t-6",
    ):
        assert wanted in text
    for forbidden in (
        ACCOUNT,
        ROUTING,
        "echo",
        "4417",
        ACCESS_TOKEN,
        "hunter2-s3cr3t",
        b64("cid:hunter2-s3cr3t"),
    ):
        assert forbidden not in text


async def test_an_id_with_a_newline_cannot_forge_a_log_line(ramp, api, caplog):
    caplog.set_level(logging.INFO, logger="callback_agent.ramp")
    api.route(method="POST").respond(204)
    await ramp.hold_vendor("v1\nramp POST /forged -> HTTP 204", "x")
    lines = [r.getMessage() for r in caplog.records if r.name == "callback_agent.ramp"]
    assert len(lines) == 1
    assert "\n" not in lines[0] and "%0A" in lines[0]


# --- the HTTP client and the URLs -------------------------------------------------------------


ODD_ID = "a/b?c#d"  # a slash, a question mark and a hash would each end the path segment
QUOTED_ID = "a%2Fb%3Fc%23d"
ID_CALLS = [
    pytest.param(lambda r: r.hold_vendor(ODD_ID, "x"), "/vendors/{}/hold", id="hold_vendor"),
    pytest.param(lambda r: r.release_vendor(ODD_ID), "/vendors/{}/release", id="release_vendor"),
    pytest.param(lambda r: r.hold_bill(ODD_ID, "x"), "/bills/{}/hold", id="hold_bill"),
    pytest.param(lambda r: r.release_bill(ODD_ID), "/bills/{}/release", id="release_bill"),
    pytest.param(
        lambda r: r.add_default_bank_account(
            ODD_ID, routing_number=ROUTING, account_number=ACCOUNT
        ),
        "/vendors/{}/update-bank-accounts",
        id="add_default_bank_account",
    ),
    pytest.param(
        lambda r: r.list_vendor_accounts(ODD_ID), "/vendors/{}/accounts", id="list_vendor_accounts"
    ),
]


@pytest.mark.parametrize("call, path", ID_CALLS)
async def test_ids_are_quoted_so_they_stay_one_path_segment(ramp, api, call, path):
    seen = api.route().respond(404)  # after the token route, so the call under test lands here
    with pytest.raises(RampError):
        await call(ramp)
    raw_path = seen.calls.last.request.url.raw_path.split(b"?")[0]
    assert raw_path == (API_PATH + path.format(QUOTED_ID)).encode()


async def test_a_trailing_slash_on_the_base_url_is_harmless(api, token_route):
    api.get(FIRST_PAGE).respond(200, json=page_of())
    client = RampClient(BASE + "/", CLIENT_ID, CLIENT_SECRET)
    try:
        assert await client.list_vendors() == []
    finally:
        await client.aclose()
    assert token_route.call_count == 1


async def test_an_injected_http_client_is_used_and_closed_with_the_ramp_client():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/token"):
            return httpx.Response(200, json=TOKEN)
        return httpx.Response(200, json=page_of())

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # no base_url, no headers
    client = RampClient(BASE, CLIENT_ID, CLIENT_SECRET, http=http)
    await client.list_vendors()
    await client.aclose()
    assert [(r.method, str(r.url)) for r in seen] == [
        ("POST", f"{API}/token"),
        ("GET", FIRST_PAGE),
    ]
    assert http.is_closed


# --- the models -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "amount, currency, rate, spoken",
    [
        pytest.param(120450, "USD", 100, "$1,204.50", id="the_plans_example"),
        pytest.param(5, "USD", 100, "$0.05", id="cents"),
        pytest.param(0, "USD", 100, "$0.00", id="zero"),
        pytest.param(100000000, "USD", 100, "$1,000,000.00", id="a_million"),
        pytest.param(-4500, "USD", 100, "-$45.00", id="a_credit"),
        pytest.param(500000, "EUR", 100, "5,000.00 EUR", id="another_currency_keeps_its_code"),
        pytest.param(1500, "JPY", 1, "1,500 JPY", id="no_minor_unit"),
        pytest.param(1234567, "KWD", 1000, "1,234.567 KWD", id="three_minor_digits"),
    ],
)
def test_money_spoken(amount, currency, rate, spoken):
    money = Money(amount=amount, currency_code=currency, minor_unit_conversion_rate=rate)
    assert money.spoken() == spoken


def test_money_with_no_conversion_rate_is_rejected_when_parsed():
    with pytest.raises(ValidationError):
        Money(amount=100, currency_code="USD", minor_unit_conversion_rate=0)


@pytest.mark.parametrize(
    "model, payload",
    [
        pytest.param(
            Money,
            {"amount": 1, "currency_code": "USD", "minor_unit_conversion_rate": 100},
            id="Money",
        ),
        pytest.param(Vendor, {"id": "v1", "name": "Acme", "is_active": True}, id="Vendor"),
        pytest.param(
            VendorBankAccount,
            {
                "id": "a1",
                "account_number_last_four": "4417",
                "is_default": True,
                "status": "active",
                "payment_method": "ACH",
            },
            id="VendorBankAccount",
        ),
        pytest.param(BillVendor, {"id": "v1", "name": "Acme"}, id="BillVendor"),
        pytest.param(
            BankAccountUpdate, {"vendor_id": "v1", "status": "approved"}, id="BankAccountUpdate"
        ),
    ],
)
def test_models_ignore_fields_they_do_not_name(model, payload):
    parsed = model.model_validate({**payload, "added_by_ramp_later": {"x": 1}})
    assert parsed == model.model_validate(payload)
    assert not hasattr(parsed, "added_by_ramp_later")


def test_a_bill_needs_its_amount_status_and_vendor_and_nothing_more():
    bill = Bill.model_validate(
        {
            "id": "b1",
            "amount": {"amount": 100, "currency_code": "USD", "minor_unit_conversion_rate": 100},
            "status": "OPEN",
            "status_summary": "ON_HOLD",
            "vendor": {"id": "v1", "name": "Acme"},
        }
    )
    assert (bill.invoice_number, bill.due_at) == (None, None)


def test_hold_outcome_values():
    assert (HoldOutcome.APPLIED.value, HoldOutcome.PENDING_APPROVAL.value) == (
        "applied",
        "pending_approval",
    )


# --- the sandbox (Step 5) ---------------------------------------------------------------------

SANDBOX_ID = os.environ.get("RAMP_CLIENT_ID", "").strip()
SANDBOX_SECRET = os.environ.get("RAMP_CLIENT_SECRET", "").strip()
UNSET = [
    name
    for name, value in (("RAMP_CLIENT_ID", SANDBOX_ID), ("RAMP_CLIENT_SECRET", SANDBOX_SECRET))
    if not value
]


@pytest.mark.ramp_sandbox
@pytest.mark.skipif(
    bool(UNSET), reason=f"{' and '.join(UNSET)} not set: this calls the Ramp sandbox"
)
async def test_sandbox_reads():
    # pytest does not read the repo .env, so export the three variables first. Run with -rP to see
    # the counts: Ramp does not document what the sandbox is seeded with.
    base_url = os.environ.get("RAMP_BASE_URL", "").strip() or "https://demo-api.ramp.com"
    client = RampClient(base_url, SANDBOX_ID, SANDBOX_SECRET)
    try:
        vendors = await client.list_vendors()
        bills = await client.list_bills(payment_status=None)
    finally:
        await client.aclose()
    print(f"the sandbox has {len(vendors)} vendors and {len(bills)} bills")
    assert all(v.id and v.name for v in vendors)
    assert all(b.id and b.vendor.id for b in bills)
