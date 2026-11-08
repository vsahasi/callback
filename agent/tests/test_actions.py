import asyncio
import hashlib
import json
import logging
import re
import subprocess
import sys

import pytest
from callback_shared.contract import ACTION_TIER, Action
from fake_ramp import FakeRamp, bill

from callback_agent import prompts
from callback_agent.actions import (
    ActionOutcome,
    ActionRequest,
    HoldTarget,
    execute,
    hold_vendor_fully,
)
from callback_agent.ramp import HoldOutcome, Money, RampError

ROUTING = "021000021"  # the plan's fake pair; 021000021 passes the ABA checksum
ACCOUNT = "000123456789"
BANK = ActionRequest.update_vendor_bank_account("v1", "Acme Supplies", ROUTING, ACCOUNT)
HOLD = ActionRequest.hold_vendor("v1", "Acme Supplies", "caller asked")
HOLD_BILL = ActionRequest.hold_bill("b1", "v1", "caller asked")
VENDORS = ActionRequest.list_vendors()
ACCOUNTS = ActionRequest.list_vendor_accounts("v1", "Acme Supplies")
BILLS = ActionRequest.list_bills("v1", "Acme Supplies")
RELEASE = ActionRequest.release_vendor_hold("v1", "Acme Supplies")
RELEASE_BILL = ActionRequest.release_bill_hold("b1", "v1")
EVERY_REQUEST = [HOLD, HOLD_BILL, VENDORS, ACCOUNTS, BILLS, BANK, RELEASE, RELEASE_BILL]


@pytest.fixture
def ramp() -> FakeRamp:
    return FakeRamp()


# --- requests ---


def fingerprint_of(routing: str, account: str) -> str:
    req = ActionRequest.update_vendor_bank_account("v1", "Acme Supplies", routing, account)
    return req.detector_params()["account_fingerprint"]


def test_bank_change_params_sent_to_detector_are_redacted():
    params = BANK.detector_params()
    fingerprint = params.pop("account_fingerprint")
    assert params == {
        "vendor_id": "v1",
        "vendor_name": "Acme Supplies",
        "new_account_last4": "6789",
        "new_routing_last4": "0021",
    }
    assert re.fullmatch(r"[0-9a-f]{16}", fingerprint)
    # The same numbers give the same fingerprint, and another account or routing number with the
    # same last four digits gives a different one.
    assert fingerprint_of(ROUTING, ACCOUNT) == fingerprint
    assert fingerprint_of(ROUTING, "999123456789") != fingerprint
    assert fingerprint_of("021110021", ACCOUNT) != fingerprint
    # Keyed: the bare hash could be brute-forced from the last four digits sent beside it.
    assert fingerprint != hashlib.sha256(b"021000021|000123456789").hexdigest()[:16]


def test_the_fingerprint_key_is_new_in_every_process():
    # A key fixed in the source would be public with the repo, and as weak as no key.
    code = (
        "from callback_agent.actions import ActionRequest\n"
        "req = ActionRequest.update_vendor_bank_account(\n"
        "    'v1', 'Acme Supplies', '021000021', '000123456789')\n"
        "print(req.detector_params()['account_fingerprint'])"
    )
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=60
    )
    other = run.stdout.strip()
    assert re.fullmatch(r"[0-9a-f]{16}", other) and other != fingerprint_of(ROUTING, ACCOUNT)


def test_bad_routing_number_is_rejected():
    with pytest.raises(ValueError):
        ActionRequest.update_vendor_bank_account("v1", "Acme Supplies", "021000022", ACCOUNT)


MALFORMED_BANK_NUMBERS = [
    ("02100002", ACCOUNT),  # 8 digits
    ("0210000210", ACCOUNT),  # 10 digits
    ("021-000-021", ACCOUNT),
    ("٠٢١٠٠٠٠٢١", ACCOUNT),  # Arabic-Indic digits pass str.isdigit()
    ("021000022", ACCOUNT),  # 9 digits that fail the ABA checksum
    (ROUTING, "123"),  # 3 digits
    (ROUTING, "1" * 18),
    (ROUTING, "0001-2345"),
    (ROUTING, ""),
]


@pytest.mark.parametrize("routing, account", MALFORMED_BANK_NUMBERS)
def test_malformed_bank_numbers_are_rejected(routing, account):
    with pytest.raises(ValueError):
        ActionRequest.update_vendor_bank_account("v1", "Acme Supplies", routing, account)


@pytest.mark.parametrize("account", ["1234", "1" * 17])
def test_account_numbers_of_4_to_17_digits_are_accepted(account):
    req = ActionRequest.update_vendor_bank_account("v1", "Acme Supplies", ROUTING, account)
    assert req.params["account_number"] == account


@pytest.mark.parametrize("routing, account", [("021000022", ACCOUNT), (ROUTING, "12-3456789")])
def test_a_rejected_number_is_not_repeated_in_the_error(routing, account):
    with pytest.raises(ValueError) as caught:
        ActionRequest.update_vendor_bank_account("v1", "Acme Supplies", routing, account)
    assert routing not in str(caught.value) and account not in str(caught.value)


# --- a request built directly is validated like one built by a constructor ---


def bank_params(routing: str = ROUTING, account: str = ACCOUNT) -> dict[str, str]:
    return {
        "vendor_id": "v1",
        "vendor_name": "Acme Supplies",
        "routing_number": routing,
        "account_number": account,
    }


MISSING_ONE_PARAM = [
    (req.action, {key: value for key, value in req.params.items() if key != missing}, missing)
    for req in EVERY_REQUEST
    for missing in req.params
]


def test_every_action_has_a_request_in_this_file():
    assert {req.action for req in EVERY_REQUEST} == set(Action)


@pytest.mark.parametrize(
    "action, params, missing",
    MISSING_ONE_PARAM,
    ids=[f"{action.value}-without-{missing}" for action, _, missing in MISSING_ONE_PARAM],
)
def test_a_directly_built_request_missing_a_param_is_rejected(action, params, missing):
    with pytest.raises(ValueError, match=missing):
        ActionRequest(action, params)


def test_a_tier2_request_built_directly_without_a_vendor_id_is_rejected():
    # The deny path would have no vendor to hold, and would find out only after the detector
    # had said no.
    with pytest.raises(ValueError, match="vendor_id"):
        ActionRequest(Action.RELEASE_VENDOR_HOLD, {"vendor_name": "Acme Supplies"})


@pytest.mark.parametrize("routing, account", MALFORMED_BANK_NUMBERS)
def test_a_bank_change_built_directly_checks_the_numbers_too(routing, account):
    with pytest.raises(ValueError) as caught:
        ActionRequest(Action.UPDATE_VENDOR_BANK_ACCOUNT, bank_params(routing, account))
    for number in (routing, account, ROUTING, ACCOUNT):
        assert not number or number not in str(caught.value)  # never either number


def test_a_missing_bank_number_does_not_repeat_the_other_one_in_the_error():
    params = bank_params()
    del params["account_number"]
    with pytest.raises(ValueError, match="account_number") as caught:
        ActionRequest(Action.UPDATE_VENDOR_BANK_ACCOUNT, params)
    assert ROUTING not in str(caught.value)


def test_a_directly_built_request_with_a_param_it_does_not_take_is_rejected():
    # detector_params() sends every other action's params as they are, so an extra value would
    # reach the detector unredacted.
    with pytest.raises(ValueError) as caught:
        ActionRequest(Action.LIST_VENDORS, {"note": "routing 021000021"})
    assert "021000021" not in str(caught.value) and "note" not in str(caught.value)
    with pytest.raises(ValueError):
        ActionRequest(Action.UPDATE_VENDOR_BANK_ACCOUNT, {**bank_params(), "note": "x"})


@pytest.mark.parametrize("value", [1, None, b"v1", ["v1"]])
def test_a_param_that_is_not_text_is_rejected(value):
    with pytest.raises(ValueError):
        ActionRequest(Action.LIST_VENDOR_ACCOUNTS, {"vendor_id": value, "vendor_name": "Acme"})


def test_an_action_that_does_not_exist_is_rejected():
    with pytest.raises(ValueError):
        ActionRequest("wire_money", {})  # type: ignore[arg-type]


def test_a_directly_built_request_is_the_same_as_one_from_its_constructor():
    direct = ActionRequest(Action.UPDATE_VENDOR_BANK_ACCOUNT, bank_params())
    assert direct == BANK and direct.detector_params() == BANK.detector_params()
    with pytest.raises(TypeError):
        direct.params["account_number"] = "999999999"  # type: ignore[index]


@pytest.mark.parametrize(
    "req, action, params",
    [
        (
            HOLD,
            Action.HOLD_VENDOR,
            {"vendor_id": "v1", "vendor_name": "Acme Supplies", "reason": "caller asked"},
        ),
        (
            HOLD_BILL,
            Action.HOLD_BILL,
            {"bill_id": "b1", "vendor_id": "v1", "reason": "caller asked"},
        ),
        (VENDORS, Action.LIST_VENDORS, {}),
        (
            ACCOUNTS,
            Action.LIST_VENDOR_ACCOUNTS,
            {"vendor_id": "v1", "vendor_name": "Acme Supplies"},
        ),
        (BILLS, Action.LIST_BILLS, {"vendor_id": "v1", "vendor_name": "Acme Supplies"}),
        (
            BANK,
            Action.UPDATE_VENDOR_BANK_ACCOUNT,
            {
                "vendor_id": "v1",
                "vendor_name": "Acme Supplies",
                "routing_number": ROUTING,
                "account_number": ACCOUNT,
            },
        ),
        (RELEASE, Action.RELEASE_VENDOR_HOLD, {"vendor_id": "v1", "vendor_name": "Acme Supplies"}),
        (RELEASE_BILL, Action.RELEASE_BILL_HOLD, {"bill_id": "b1", "vendor_id": "v1"}),
    ],
)
def test_each_constructor_fixes_its_action_params_and_tier(req, action, params):
    assert (req.action, dict(req.params), req.tier) == (action, params, ACTION_TIER[action])


@pytest.mark.parametrize("req", [r for r in EVERY_REQUEST if r is not BANK])
def test_detector_params_are_the_params_for_every_other_action(req):
    assert req.detector_params() == dict(req.params)


def test_params_cannot_change_once_the_request_is_made():
    # What the detector authorized is what runs: nothing can swap the account number in between.
    with pytest.raises(TypeError):
        BANK.params["account_number"] = "999999999"  # type: ignore[index]


def test_a_request_never_shows_full_numbers_when_printed():
    assert ACCOUNT not in repr(BANK) and ROUTING not in repr(BANK) and "6789" in repr(BANK)


def test_idempotency_key_changes_per_round():
    assert BANK.idempotency_key("s1", 0) != BANK.idempotency_key("s1", 1)


def test_idempotency_key_hashes_session_action_redacted_params_and_round():
    params = json.dumps(BANK.detector_params(), sort_keys=True)
    expected = hashlib.sha256(f"s1|update_vendor_bank_account|{params}|0".encode()).hexdigest()
    assert BANK.idempotency_key("s1", 0) == expected[:32]
    assert BANK.idempotency_key("s1", 0) != BANK.idempotency_key("s2", 0)


def test_hold_targets():
    assert BANK.hold_target() == HoldTarget("vendor", "v1")
    assert ActionRequest.release_bill_hold("b1", "v1").hold_target() == HoldTarget("vendor", "v1")
    # A denied read holds the vendor it names (final ruling, C31). list_vendors names none.
    assert ACCOUNTS.hold_target() == HoldTarget("vendor", "v1")
    assert ActionRequest.list_bills("v1", "Acme Supplies").hold_target() == HoldTarget(
        "vendor", "v1"
    )
    assert ActionRequest.list_vendors().hold_target() is None
    assert HOLD.hold_target() is None
    assert ActionRequest.hold_bill("b1", "v1", "r").hold_target() is None


def test_every_vendor_scoped_tier1_and_tier2_request_holds_its_vendor():
    targets = {req.action: req.hold_target() for req in EVERY_REQUEST}
    assert targets == {
        Action.HOLD_VENDOR: None,
        Action.HOLD_BILL: None,
        Action.LIST_VENDORS: None,
        Action.LIST_VENDOR_ACCOUNTS: HoldTarget("vendor", "v1"),
        Action.LIST_BILLS: HoldTarget("vendor", "v1"),
        Action.UPDATE_VENDOR_BANK_ACCOUNT: HoldTarget("vendor", "v1"),
        Action.RELEASE_VENDOR_HOLD: HoldTarget("vendor", "v1"),
        Action.RELEASE_BILL_HOLD: HoldTarget("vendor", "v1"),
    }


def test_bank_change_context():
    assert BANK.context().pending_action == "change Acme Supplies' bank account to one ending 6789"


@pytest.mark.parametrize(
    "req, pending_action, vendor_name",
    [
        (HOLD, "put Acme Supplies on hold", "Acme Supplies"),
        (HOLD_BILL, "put a bill on hold", None),
        (VENDORS, "list the vendors", None),
        (ACCOUNTS, "read Acme Supplies' bank accounts on file", "Acme Supplies"),
        (BILLS, "list Acme Supplies' open bills", "Acme Supplies"),
        (BANK, "change Acme Supplies' bank account to one ending 6789", "Acme Supplies"),
        (RELEASE, "release the hold on Acme Supplies", "Acme Supplies"),
        (RELEASE_BILL, "release the hold on a bill", None),
        (ActionRequest.hold_vendor("v2", "Globex", "x"), "put Globex on hold", "Globex"),
        (ActionRequest.list_bills("v2", "Globex"), "list Globex's open bills", "Globex"),
    ],
)
def test_context_names_the_pending_action(req, pending_action, vendor_name):
    context = req.context()
    assert (context.pending_action, context.vendor_name, context.vendor_bank_last4) == (
        pending_action,
        vendor_name,
        None,
    )


# --- hold_vendor_fully ---


async def test_pending_vendor_hold_also_holds_open_bills(ramp):
    ramp.vendor_hold = HoldOutcome.PENDING_APPROVAL
    assert await hold_vendor_fully(ramp, "v1", "x") is True
    assert ramp.calls == [
        ("hold_vendor", "v1"),
        ("list_bills", "v1"),
        ("hold_bill", "b1"),
        ("hold_bill", "b2"),
    ]
    assert ramp.reasons == ["x", "x", "x"]


async def test_pending_vendor_hold_with_no_open_bills_is_not_held(ramp):
    ramp.vendor_hold, ramp.bills = HoldOutcome.PENDING_APPROVAL, []
    assert await hold_vendor_fully(ramp, "v1", "x") is False
    assert ramp.calls == [("hold_vendor", "v1"), ("list_bills", "v1")]


async def test_applied_vendor_hold_is_held_without_touching_bills(ramp):
    assert await hold_vendor_fully(ramp, "v1", "x") is True
    assert ramp.calls == [("hold_vendor", "v1")]


async def test_a_bill_counts_as_held_only_if_it_comes_back_on_hold(ramp):
    # hold_bill accepts any 2xx that parses as a Bill, so getting a Bill back proves nothing.
    ramp.vendor_hold = HoldOutcome.PENDING_APPROVAL
    ramp.held_as = {"b1": "PAYMENT_SCHEDULED", "b2": "AWAITING_PAYMENT"}
    assert await hold_vendor_fully(ramp, "v1", "x") is False
    ramp.held_as = {"b1": "PAYMENT_SCHEDULED"}
    assert await hold_vendor_fully(ramp, "v1", "x") is True


async def test_a_bill_that_fails_to_hold_does_not_stop_the_others(ramp):
    ramp.vendor_hold = HoldOutcome.PENDING_APPROVAL
    ramp.fail[("hold_bill", "b1")] = RampError(422, "DEVELOPER_7001")
    assert await hold_vendor_fully(ramp, "v1", "x") is True
    assert ramp.calls[-2:] == [("hold_bill", "b1"), ("hold_bill", "b2")]


# A bill hold can fail with something that is not a RampError: httpx raises a RuntimeError for a
# request on a client that has been closed, and ramp.py does not turn that into a RampError. The
# step is counted as a failed one, like a RampError, and the other bills are still tried.

CLOSED_CLIENT = RuntimeError(
    "Cannot send a request, as the client has been closed. acct 0001-2345-6789"
)


async def test_a_bill_hold_that_raises_something_else_is_not_held_and_the_others_still_are(
    ramp, caplog
):
    caplog.set_level(logging.DEBUG)
    ramp.vendor_hold = HoldOutcome.PENDING_APPROVAL
    ramp.fail[("hold_bill", "b1")] = CLOSED_CLIENT
    assert await hold_vendor_fully(ramp, "v1", "x") is True  # b2 is on hold
    assert ramp.calls[-2:] == [("hold_bill", "b1"), ("hold_bill", "b2")]
    # Only the type is logged: the message could echo what was sent.
    assert "RuntimeError" in caplog.text
    assert "client has been closed" not in caplog.text and "0001-2345-6789" not in caplog.text


async def test_a_bill_hold_that_raises_something_else_counts_as_not_held_in_the_summary(ramp):
    ramp.vendor_hold = HoldOutcome.PENDING_APPROVAL
    ramp.fail[("hold_bill", "b1")] = CLOSED_CLIENT
    assert await execute(ramp, HOLD) == ActionOutcome(
        "pending_approval",
        "Acme Supplies' open bills are on hold; the vendor hold itself is waiting for approval "
        "in Ramp.",
    )  # one bill is on hold; the one that raised is not counted
    ramp.fail[("hold_bill", "b2")] = CLOSED_CLIENT
    assert await execute(ramp, HOLD) == ActionOutcome(
        "pending_approval",
        "The hold on Acme Supplies is waiting for approval in Ramp, so nothing is on hold yet.",
    )


async def test_when_every_bill_hold_raises_something_else_nothing_is_held_and_nothing_raises(ramp):
    ramp.vendor_hold = HoldOutcome.PENDING_APPROVAL
    ramp.fail["hold_bill"] = CLOSED_CLIENT
    assert await hold_vendor_fully(ramp, "v1", "x") is False
    assert [call for call in ramp.calls if call[0] == "hold_bill"] == [
        ("hold_bill", "b1"),
        ("hold_bill", "b2"),
    ]  # and both were tried


async def test_cancellation_still_gets_through_a_bill_hold(ramp):
    ramp.vendor_hold = HoldOutcome.PENDING_APPROVAL
    ramp.fail[("hold_bill", "b1")] = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await hold_vendor_fully(ramp, "v1", "x")
    assert ("hold_bill", "b2") not in ramp.calls  # and nothing went on after it


@pytest.mark.parametrize("failing", ["hold_vendor", "list_bills"])
async def test_a_hold_ramp_fails_is_not_held_and_does_not_raise(ramp, failing):
    ramp.vendor_hold = HoldOutcome.PENDING_APPROVAL
    ramp.fail[failing] = RampError(503)
    assert await hold_vendor_fully(ramp, "v1", "x") is False
    assert not any(method == "hold_bill" for method, *_ in ramp.calls)


async def test_a_failed_hold_logs_status_code_and_trace_but_never_ramps_message(ramp, caplog):
    # Ramp may echo a submitted number back, reformatted so the client's redaction misses it.
    caplog.set_level(logging.DEBUG)
    ramp.fail["hold_vendor"] = RampError(
        422, "DEVELOPER_7001", "acct 0001-2345-6789 bad", "trace-9"
    )
    assert await hold_vendor_fully(ramp, "v1", "x") is False
    assert "422" in caplog.text and "DEVELOPER_7001" in caplog.text and "trace-9" in caplog.text
    assert "0001-2345-6789" not in caplog.text


# --- execute ---


@pytest.mark.parametrize(
    "vendor_hold, bills, held_as, status, summary",
    [
        (HoldOutcome.APPLIED, [bill("b1")], {}, "done", "Acme Supplies is on hold."),
        (
            HoldOutcome.PENDING_APPROVAL,
            [bill("b1")],
            {},
            "pending_approval",
            "Acme Supplies' open bills are on hold; the vendor hold itself is waiting for "
            "approval in Ramp.",
        ),
        (
            HoldOutcome.PENDING_APPROVAL,
            [],
            {},
            "pending_approval",
            "The hold on Acme Supplies is waiting for approval in Ramp, so nothing is on hold yet.",
        ),
        (
            HoldOutcome.PENDING_APPROVAL,
            [bill("b1")],
            {"b1": "PAYMENT_SCHEDULED"},
            "pending_approval",
            "The hold on Acme Supplies is waiting for approval in Ramp, so nothing is on hold yet.",
        ),
    ],
)
async def test_vendor_hold_says_only_what_actually_holds(
    ramp, vendor_hold, bills, held_as, status, summary
):
    ramp.vendor_hold, ramp.bills, ramp.held_as = vendor_hold, bills, held_as
    assert await execute(ramp, HOLD) == ActionOutcome(status, summary)
    assert ramp.calls[0] == ("hold_vendor", "v1") and ramp.reasons[0] == "caller asked"


async def test_bill_hold_is_done_only_when_the_bill_comes_back_on_hold(ramp):
    assert await execute(ramp, HOLD_BILL) == ActionOutcome(
        "done", "Acme Supplies Inc's invoice INV-1001 is on hold."
    )
    assert ramp.calls == [("hold_bill", "b1")] and ramp.reasons == ["caller asked"]
    ramp.held_as = {"b1": "PAYMENT_SCHEDULED"}
    # No approval flow exists for a bill hold, so another status is a failure, not "pending".
    assert await execute(ramp, HOLD_BILL) == ActionOutcome("failed", prompts.RAMP_FAILED)


async def test_bill_without_an_invoice_number_is_named_by_its_vendor(ramp):
    ramp.bills = [bill("b1", invoice=None)]
    outcome = await execute(ramp, HOLD_BILL)
    assert outcome.summary == "A bill from Acme Supplies Inc is on hold."


@pytest.mark.parametrize("currency", ["USD", "EUR"])
@pytest.mark.parametrize("cents", [120450, 987654321, 5])
@pytest.mark.parametrize("invoice", ["INV-1001", "INV-77", None])
async def test_a_bill_hold_never_says_the_amount(ramp, invoice, cents, currency):
    # A hold needs no verification (R56), so what it says can reach a stranger who guessed a vendor
    # and an invoice number. The only digits it may carry are those of the invoice number.
    money = Money(amount=cents, currency_code=currency, minor_unit_conversion_rate=100)
    ramp.bills = [bill("b1", invoice=invoice).model_copy(update={"amount": money})]
    summary = (await execute(ramp, HOLD_BILL)).summary
    assert "$" not in summary and currency not in summary
    assert re.findall(r"\d+", summary) == (re.findall(r"\d+", invoice) if invoice else [])


async def test_bill_release_is_done_only_when_the_bill_is_no_longer_on_hold(ramp):
    assert await execute(ramp, RELEASE_BILL) == ActionOutcome(
        "done", "The hold on invoice INV-1001 from Acme Supplies Inc for $1,204.50 is released."
    )
    ramp.released_as = {"b1": "ON_HOLD"}
    assert await execute(ramp, RELEASE_BILL) == ActionOutcome("failed", prompts.RAMP_FAILED)
    assert ramp.calls == [("release_bill", "b1"), ("release_bill", "b1")]


async def test_vendor_release_waiting_for_approval_is_not_done(ramp):
    outcome = await execute(ramp, RELEASE)
    assert outcome.status == "done" and outcome.summary.startswith(
        "The hold on Acme Supplies is released."
    )
    ramp.vendor_release = HoldOutcome.PENDING_APPROVAL
    assert await execute(ramp, RELEASE) == ActionOutcome(
        "pending_approval", "Releasing the hold on Acme Supplies is waiting for approval in Ramp."
    )


async def test_bank_change_sends_the_full_numbers_to_ramp_and_says_only_the_last_four(ramp):
    assert await execute(ramp, BANK) == ActionOutcome(
        "done", "Acme Supplies now pays to the account ending 6789."
    )
    assert ramp.calls == [("add_default_bank_account", "v1", ROUTING, ACCOUNT)]
    ramp.bank_status = "pending_approval"
    pending = await execute(ramp, BANK)
    assert pending == ActionOutcome(
        "pending_approval",
        "The change to the account ending 6789 for Acme Supplies is waiting for approval in Ramp.",
    )


async def test_reads_name_accounts_by_last_four_and_bills_by_invoice(ramp):
    vendors = await execute(ramp, VENDORS)
    assert vendors == ActionOutcome("done", "Ramp has 2 vendors: Acme Supplies, Globex.")
    accounts = await execute(ramp, ACCOUNTS)
    assert accounts == ActionOutcome(
        "done", "Acme Supplies' payment accounts on file: ACH ending 4417, the default."
    )
    bills = await execute(ramp, BILLS)
    assert bills.status == "done"
    assert bills.summary == (
        "Acme Supplies has 2 open bills: "
        "invoice INV-1001 for $1,204.50, due November 20, payment scheduled (bill id b1); "
        "invoice INV-1002 for $3,980.00, due November 20, payment scheduled (bill id b2)."
    )
    assert ramp.calls == [("list_vendors",), ("list_vendor_accounts", "v1"), ("list_bills", "v1")]


async def test_reads_with_nothing_to_report_say_so(ramp):
    ramp.vendors, ramp.accounts, ramp.bills = [], [], []
    assert (await execute(ramp, VENDORS)).summary == "Ramp has no vendors."
    assert (
        await execute(ramp, ACCOUNTS)
    ).summary == "Ramp has no payment accounts on file for Acme Supplies."
    assert (await execute(ramp, BILLS)).summary == "Acme Supplies has no open bills."


@pytest.mark.parametrize("req", EVERY_REQUEST)
async def test_ramp_errors_reach_the_gate(ramp, req):
    ramp.fail = {
        method: RampError(500)
        for method in (
            "hold_vendor",
            "hold_bill",
            "list_vendors",
            "list_vendor_accounts",
            "list_bills",
            "add_default_bank_account",
            "release_vendor",
            "release_bill",
        )
    }
    with pytest.raises(RampError):
        await execute(ramp, req)
    assert len(ramp.calls) == 1  # never sent twice
