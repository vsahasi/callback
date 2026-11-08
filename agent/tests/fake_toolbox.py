"""A stand-in for Toolbox, for tests of the agent that wraps it.

Each method records its call as (method, *arguments) in `calls`, runs `during` if a test set one,
waits for `hold` if a test set one, and answers `replies[method]`, or a line naming the method (or
raises `fail[method]`). test_structure.py checks that it has exactly the Toolbox's methods.
"""

import asyncio
from collections.abc import Callable


class FakeToolbox:
    def __init__(self, replies: dict[str, str] | None = None) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.replies = replies or {}
        self.fail: dict[str, BaseException] = {}
        self.during: Callable[[], None] | None = None  # runs inside every call
        self.hold: asyncio.Event | None = None  # when set, a call waits until the event is set

    async def _answer(self, method: str, *arguments: str) -> str:
        self.calls.append((method, *arguments))
        if self.during is not None:
            self.during()
        if self.hold is not None:
            await self.hold.wait()
        if method in self.fail:
            raise self.fail[method]
        return self.replies.get(method, f"{method} was done")

    async def identify_caller(self, full_name: str) -> str:
        return await self._answer("identify_caller", full_name)

    async def list_vendors(self) -> str:
        return await self._answer("list_vendors")

    async def get_vendor_bank_account(self, vendor_name: str) -> str:
        return await self._answer("get_vendor_bank_account", vendor_name)

    async def list_open_bills(self, vendor_name: str) -> str:
        return await self._answer("list_open_bills", vendor_name)

    async def hold_vendor(self, vendor_name: str, reason: str) -> str:
        return await self._answer("hold_vendor", vendor_name, reason)

    async def hold_bill(self, vendor_name: str, invoice_number: str, reason: str) -> str:
        return await self._answer("hold_bill", vendor_name, invoice_number, reason)

    async def change_vendor_bank_account(
        self, vendor_name: str, routing_number: str, account_number: str
    ) -> str:
        return await self._answer(
            "change_vendor_bank_account", vendor_name, routing_number, account_number
        )

    async def release_vendor_hold(self, vendor_name: str) -> str:
        return await self._answer("release_vendor_hold", vendor_name)

    async def release_bill_hold(self, bill_id: str) -> str:
        return await self._answer("release_bill_hold", bill_id)

    async def start_voice_enrollment(self) -> str:
        return await self._answer("start_voice_enrollment")
