"""A stand-in for RampClient that records each call and answers from settings a test can change."""

from datetime import UTC, datetime

from callback_agent.ramp import (
    BankAccountUpdate,
    Bill,
    BillVendor,
    HoldOutcome,
    Money,
    Vendor,
    VendorBankAccount,
)


def bill(
    bill_id: str,
    *,
    invoice: str | None = "INV-1001",
    cents: int = 120450,
    status_summary: str = "PAYMENT_SCHEDULED",
) -> Bill:
    """An open Acme bill, due Nov 20."""
    return Bill(
        id=bill_id,
        invoice_number=invoice,
        amount=Money(amount=cents, currency_code="USD", minor_unit_conversion_rate=100),
        due_at=datetime(2026, 11, 20, tzinfo=UTC),
        status="OPEN",
        status_summary=status_summary,
        vendor=BillVendor(id="v1", name="Acme Supplies Inc"),
    )


class FakeRamp:
    """Records each call as (method, *ids) in `calls`, and each hold reason in `reasons`.

    `fail` maps a method name, or a whole call such as ("hold_bill", "b1"), to the exception that
    call raises once it is recorded. `held_as` and `released_as` map a bill id to the status_summary
    its hold or release comes back with: ON_HOLD and PAYMENT_READY unless set.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.reasons: list[str] = []
        self.fail: dict[str | tuple[str, ...], Exception] = {}
        self.vendors = [
            Vendor(id="v1", name="Acme Supplies Inc", display_name="Acme Supplies", is_active=True),
            Vendor(id="v2", name="Globex Corporation", display_name="Globex", is_active=True),
        ]
        self.accounts = [
            VendorBankAccount(
                id="a1",
                account_number_last_four="4417",
                is_default=True,
                status="active",
                payment_method="ACH",
            ),
        ]
        self.bills = [bill("b1"), bill("b2", invoice="INV-1002", cents=398000)]
        self.vendor_hold = HoldOutcome.APPLIED
        self.vendor_release = HoldOutcome.APPLIED
        self.held_as: dict[str, str] = {}
        self.released_as: dict[str, str] = {}
        self.bank_status = "approved"

    def _record(self, method: str, *args: str) -> None:
        call = (method, *args)
        self.calls.append(call)
        failure = self.fail.get(call) or self.fail.get(method)
        if failure is not None:
            raise failure

    def _bill(self, bill_id: str) -> Bill:
        return next((b for b in self.bills if b.id == bill_id), None) or bill(bill_id)

    async def list_vendors(self) -> list[Vendor]:
        self._record("list_vendors")
        return list(self.vendors)

    async def list_vendor_accounts(self, vendor_id: str) -> list[VendorBankAccount]:
        self._record("list_vendor_accounts", vendor_id)
        return list(self.accounts)

    async def list_bills(
        self, *, vendor_id: str | None = None, payment_status: str | None = "OPEN"
    ) -> list[Bill]:
        assert payment_status == "OPEN"
        self._record("list_bills", vendor_id)
        return list(self.bills)

    async def hold_vendor(self, vendor_id: str, reason: str) -> HoldOutcome:
        self._record("hold_vendor", vendor_id)
        self.reasons.append(reason)
        return self.vendor_hold

    async def release_vendor(self, vendor_id: str) -> HoldOutcome:
        self._record("release_vendor", vendor_id)
        return self.vendor_release

    async def hold_bill(self, bill_id: str, reason: str) -> Bill:
        self._record("hold_bill", bill_id)
        self.reasons.append(reason)
        status = self.held_as.get(bill_id, "ON_HOLD")
        return self._bill(bill_id).model_copy(update={"status_summary": status})

    async def release_bill(self, bill_id: str) -> Bill:
        self._record("release_bill", bill_id)
        status = self.released_as.get(bill_id, "PAYMENT_READY")
        return self._bill(bill_id).model_copy(update={"status_summary": status})

    async def add_default_bank_account(
        self, vendor_id: str, *, routing_number: str, account_number: str
    ) -> BankAccountUpdate:
        self._record("add_default_bank_account", vendor_id, routing_number, account_number)
        approved = self.bank_status == "approved"
        return BankAccountUpdate(
            vendor_id=vendor_id,
            status=self.bank_status,
            payment_method_id="pm1" if approved else None,
            proposal_id=None if approved else "p1",
        )
