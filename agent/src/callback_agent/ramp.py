"""The agent's HTTP client for the Ramp Developer API (v1), pointed at the sandbox.

Every failure is a RampError. A write is not retried after a 429, a 5xx or a timeout: none of them
proves the change failed, Ramp has no idempotency key, and a second send could apply it twice. (A
401 is the one resend, since Ramp refused the token before running anything: see RampClient.) Reads
are retried. Response bodies stay out of errors and logs, except the one-line message Ramp puts in
error_v2. The endpoints and response shapes are in
docs/research_notes/Callback agent build research/ramp_developer_api.md.
"""

import logging
from asyncio import Lock, sleep
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from difflib import SequenceMatcher
from enum import StrEnum
from time import monotonic
from typing import Any, Literal
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)

API_PREFIX = "/developer/v1"
PAGE_SIZE = 100  # the most Ramp allows
REQUEST_TIMEOUT_S = 10.0
TOKEN_RENEW_MARGIN_S = 60.0  # a token is replaced this long before it expires
READ_RETRY_WAITS_S = (0.5, 1.0)  # one wait per retry of a read, when Ramp names none
MAX_RETRY_AFTER_S = 5.0  # a longer wait is a rate-limit lockout (Ramp's is 60 s): not worth a call
FUZZY_CUTOFF = 0.8
_HEADERS = {"Accept": "application/json", "User-Agent": "callback-agent"}


class _ErrorDetail(BaseModel):
    error_code: str | None = None
    message: str | None = None


class _ErrorBody(BaseModel):
    error_v2: _ErrorDetail


class RampError(Exception):
    """A call to Ramp failed: a non-2xx status, an answer that doesn't fit, or no answer at all.

    `status` is the HTTP status, or 0 when Ramp gave no answer (a timeout, a refused connection).
    On a write, 0 and any 5xx mean the outcome is unknown: the change may have been applied.
    `message` is Ramp's own error_v2 message when it sent one, and no other part of the body.
    """

    def __init__(
        self,
        status: int,
        error_code: str | None = None,
        message: str = "",
        trace_id: str | None = None,
    ) -> None:
        super().__init__(status, error_code, message, trace_id)
        self.status = status
        self.error_code = error_code
        self.message = message
        self.trace_id = trace_id

    def __str__(self) -> str:
        head = f"Ramp answered HTTP {self.status}" if self.status else "Ramp gave no answer"
        code = f" {self.error_code}" if self.error_code else ""
        detail = f": {self.message}" if self.message else ""
        trace = f" (trace_id {self.trace_id})" if self.trace_id else ""
        return f"{head}{code}{detail}{trace}"

    @classmethod
    def from_response(cls, response: httpx.Response, redact: tuple[str, ...] = ()) -> "RampError":
        """The error for a non-2xx answer. Any `redact` value Ramp echoes back is blanked out."""
        code = message = None
        try:
            detail = _ErrorBody.model_validate_json(response.content).error_v2
            code, message = detail.error_code, detail.message
        except ValidationError:
            pass  # no error_v2: HTML from a gateway, the older {"error": ...} shape, nothing at all
        message = message or response.reason_phrase or f"HTTP {response.status_code}"
        for secret in filter(None, redact):
            message = message.replace(secret, "[redacted]")
        return cls(response.status_code, code, message, response.headers.get("x-trace-id"))


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Money(_Model):
    """Ramp's amount: a whole number of minor units, and what to divide it by."""

    amount: int
    currency_code: str
    minor_unit_conversion_rate: int = Field(gt=0)

    def spoken(self) -> str:
        """The amount as a person says it, "$1,204.50". Another currency keeps its code."""
        places = len(str(self.minor_unit_conversion_rate)) - 1  # a rate of 100 is 2 decimal places
        number = f"{Decimal(abs(self.amount)) / self.minor_unit_conversion_rate:,.{places}f}"
        sign = "-" if self.amount < 0 else ""
        if self.currency_code == "USD":
            return f"{sign}${number}"
        return f"{sign}{number} {self.currency_code}"


class Vendor(_Model):
    id: str
    name: str
    display_name: str | None = None
    is_active: bool


class VendorBankAccount(_Model):
    """A vendor's bank account as Ramp shows it: the last four digits and nothing more."""

    id: str
    account_number_last_four: str | None = None
    is_default: bool
    status: str
    payment_method: str


class BillVendor(_Model):
    id: str
    name: str


class Bill(_Model):
    id: str
    invoice_number: str | None = None
    amount: Money
    due_at: datetime | None = None
    status: str
    status_summary: str
    vendor: BillVendor


class BankAccountUpdate(_Model):
    vendor_id: str
    status: Literal["approved", "pending_approval"]
    payment_method_id: str | None = None  # None while the change waits for approval
    proposal_id: str | None = None


class HoldOutcome(StrEnum):
    APPLIED = "applied"
    PENDING_APPROVAL = "pending_approval"


class _PageLink(BaseModel):
    next: str | None = None


class _Page[T](BaseModel):
    data: list[T]
    page: _PageLink = Field(default_factory=_PageLink)


class _TokenAnswer(BaseModel):
    access_token: str = Field(min_length=1)
    expires_in: int


@dataclass(frozen=True)
class _Token:
    value: str = field(repr=False)
    expires_at: float  # on the monotonic clock, already short of the real expiry


def _problems(error: ValidationError) -> str:
    """The fields that failed and why, without their values."""
    return ", ".join(
        f"{'.'.join(map(str, e['loc'])) or 'body'}: {e['type']}"
        for e in error.errors(include_input=False)
    )


def _parse[M: BaseModel](model: type[M], response: httpx.Response) -> M:
    try:
        return model.model_validate_json(response.content)
    except ValidationError as exc:
        # from None, so the rejected body stays out of tracebacks and logs
        raise RampError(
            response.status_code,
            message=f"the answer breaks the expected shape ({_problems(exc)})",
            trace_id=response.headers.get("x-trace-id"),
        ) from None


def _check(method: str, url: str, response: httpx.Response, redact: tuple[str, ...] = ()) -> None:
    """Raise RampError unless the answer is a 2xx."""
    if response.is_success:
        return
    error = RampError.from_response(response, redact)
    logger.warning(
        "ramp %s %s -> HTTP %s %s trace_id=%s",
        method,
        _log_path(url),
        error.status,
        error.error_code or "-",
        error.trace_id or "-",
    )
    raise error


def _log_path(url: str) -> str:
    """The path of a URL, still percent-encoded, with no query: safe to log."""
    return httpx.URL(url).raw_path.split(b"?", 1)[0].decode("ascii", "replace")


def _seg(identifier: str) -> str:
    """An id as one path segment, whatever it holds."""
    return quote(identifier, safe="")


def _retry_after(response: httpx.Response) -> float | None:
    """The Retry-After header in seconds. An HTTP date, or anything unreadable, counts as absent."""
    try:
        seconds = float(response.headers.get("retry-after", ""))
    except ValueError:
        return None
    return seconds if seconds >= 0 else None  # NaN is not >= 0 either


def _retry_wait(response: httpx.Response, usual: float | None) -> float | None:
    """Seconds to wait before a read's next try, or None to give up.

    `usual` is the next wait of the fixed schedule, None once the retries are used up. A
    Retry-After from Ramp replaces it, unless it is longer than a caller will hold the line.
    """
    if usual is None:
        return None
    asked = _retry_after(response)
    wait = usual if asked is None else asked
    return wait if wait <= MAX_RETRY_AFTER_S else None


def _hold_outcome(response: httpx.Response) -> HoldOutcome:
    """204 means applied and 202 means waiting for approval. No other answer counts as a hold."""
    outcomes = {204: HoldOutcome.APPLIED, 202: HoldOutcome.PENDING_APPROVAL}
    try:
        return outcomes[response.status_code]
    except KeyError:
        raise RampError(
            response.status_code,
            message="Ramp's answer to a hold or release is neither 204 nor 202",
            trace_id=response.headers.get("x-trace-id"),
        ) from None


def _normal(text: str) -> str:
    return " ".join(text.split()).casefold()


def _names(vendor: Vendor) -> list[str]:
    return [n for n in (_normal(vendor.name), _normal(vendor.display_name or "")) if n]


def _containing(wanted: str, vendors: list[Vendor]) -> Vendor | None:
    """The vendor whose shortest matching name contains `wanted`. Ramp's order breaks ties."""
    best: tuple[int, Vendor] | None = None
    for vendor in vendors:
        lengths = [len(n) for n in _names(vendor) if wanted in n]
        if lengths and (best is None or min(lengths) < best[0]):
            best = (min(lengths), vendor)
    return best[1] if best else None


def _spelled_like(wanted: str, vendors: list[Vendor]) -> Vendor | None:
    """The vendor with the closest name, if it is at least FUZZY_CUTOFF similar."""
    best: tuple[float, Vendor] | None = None
    for vendor in vendors:
        for name in _names(vendor):
            ratio = SequenceMatcher(None, wanted, name).ratio()
            if best is None or ratio > best[0]:
                best = (ratio, vendor)
    return best[1] if best and best[0] >= FUZZY_CUTOFF else None


class RampClient:
    """Calls the Ramp Developer API. One instance serves a whole call, from any task.

    A write is sent once, whatever comes back: a timeout or a 5xx does not prove it failed, and a
    second send could apply a change twice. The hold_*, release_* and add_default_bank_account
    methods are meant to be called only through the gate, and request() only by scripts, since it
    can send anything. The one exception is a 401: Ramp turned the token away before it ran
    anything, so the call gets a new token and one more send, for a read or a write. A read is
    also tried again after a 429 or a 5xx, twice at most.

    The token comes from the client-credentials grant, lasts 10 days and is kept until a minute
    before it expires. Pages are followed to the end, but only along URLs on the base URL's own
    origin, because each request carries the bearer token. `http` replaces the client the instance
    would create; aclose() closes it either way.
    """

    SCOPES = "vendors:read vendors:write bills:read bills:write"

    def __init__(
        self,
        base_url: str,
        client_id: str,
        client_secret: str,
        *,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._basic = httpx.BasicAuth(client_id, client_secret)
        self._http = http if http is not None else httpx.AsyncClient()
        self._token: _Token | None = None
        self._token_lock = Lock()

    # --- reads ---

    async def list_vendors(self) -> list[Vendor]:
        return await self._list_all("/vendors", Vendor)

    async def find_vendor(self, name: str) -> Vendor | None:
        """The vendor the caller means by `name`, or None.

        Case and spacing don't matter. A vendor whose name or display name contains `name` comes
        first (the shortest such name, so an exact match beats a longer one). Failing that, the
        vendor with the closest spelling, if it is at least 0.8 similar. A blank `name` finds
        nothing, because "" is inside every name.
        """
        wanted = _normal(name)
        if not wanted:
            return None
        vendors = await self.list_vendors()
        return _containing(wanted, vendors) or _spelled_like(wanted, vendors)

    async def list_vendor_accounts(self, vendor_id: str) -> list[VendorBankAccount]:
        return await self._list_all(f"/vendors/{_seg(vendor_id)}/accounts", VendorBankAccount)

    async def list_bills(
        self, *, vendor_id: str | None = None, payment_status: str | None = "OPEN"
    ) -> list[Bill]:
        """Bills, open ones unless `payment_status` says otherwise. None means every bill."""
        filters = {"vendor_id": vendor_id, "payment_status": payment_status}
        return await self._list_all(
            "/bills", Bill, {k: v for k, v in filters.items() if v is not None}
        )

    # --- writes ---

    async def hold_vendor(self, vendor_id: str, reason: str) -> HoldOutcome:
        response = await self._send(
            "POST", self._url(f"/vendors/{_seg(vendor_id)}/hold"), json={"reason": reason}
        )
        return _hold_outcome(response)

    async def release_vendor(self, vendor_id: str) -> HoldOutcome:
        response = await self._send("POST", self._url(f"/vendors/{_seg(vendor_id)}/release"))
        return _hold_outcome(response)

    async def hold_bill(self, bill_id: str, reason: str) -> Bill:
        response = await self._send(
            "POST", self._url(f"/bills/{_seg(bill_id)}/hold"), json={"reason": reason}
        )
        return _parse(Bill, response)

    async def release_bill(self, bill_id: str) -> Bill:
        response = await self._send("POST", self._url(f"/bills/{_seg(bill_id)}/release"))
        return _parse(Bill, response)

    async def add_default_bank_account(
        self, vendor_id: str, *, routing_number: str, account_number: str
    ) -> BankAccountUpdate:
        """Add an ACH account and make it the vendor's default. Ramp may hold it for approval."""
        body = {
            "is_default": True,
            "ach_details": {
                "routing_number": routing_number,
                "account_number": account_number,
                "account_type": "Checking",
                "account_owner_type": "BUSINESS",
            },
        }
        response = await self._send(
            "POST",
            self._url(f"/vendors/{_seg(vendor_id)}/update-bank-accounts"),
            json=body,
            redact=(routing_number, account_number),
        )
        return _parse(BankAccountUpdate, response)

    # --- anything else: for scripts ---

    async def request(self, method: str, path: str, *, json: dict | None = None) -> httpx.Response:
        """Send one call to an endpoint this client has no method for, and return its 2xx answer.

        For scripts such as scripts/seed_ramp.py. The agent must not call it: it can reach any
        endpoint, so nothing it sends has been through the gate. `path` starts with a slash and
        is relative to /developer/v1, like "/entities"; a caller quotes the ids it puts in it.

        It shares the send path of every method above, so it keeps their guarantees: the cached
        bearer token, one new token and one more send after a 401, a read tried again after a 429
        or a 5xx and a write never, and RampError for anything but a 2xx. Bodies and tokens stay
        out of the log. The caller reads the answer, and RampError.message is not for logs.
        """
        if not path.startswith("/"):
            raise ValueError(f"path must start with a slash and be relative to {API_PREFIX}")
        return await self._send(method.upper(), self._url(path), json=json)

    async def aclose(self) -> None:
        await self._http.aclose()

    # --- plumbing ---

    def _url(self, path: str) -> str:
        return f"{self._base_url}{API_PREFIX}{path}"

    async def _list_all[T: BaseModel](
        self, path: str, item: type[T], filters: dict[str, str] | None = None
    ) -> list[T]:
        """Every item of a list endpoint: page after page, until page.next is null."""
        page_model = _Page[item]
        items: list[T] = []
        url: str | None = self._url(path)
        params: dict[str, str | int] | None = {"page_size": PAGE_SIZE, **(filters or {})}
        while url:
            response = await self._send("GET", url, params=params)
            page = _parse(page_model, response)
            items.extend(page.data)
            url, params = page.page.next, None  # the next URL carries its own query
            if url and not self._same_origin(url):
                # Following it would send the bearer token wherever the answer points.
                logger.warning(
                    "ramp GET %s: page.next leaves the base URL", _log_path(str(response.url))
                )
                raise RampError(
                    response.status_code,
                    message="page.next does not lead to the Ramp base URL",
                    trace_id=response.headers.get("x-trace-id"),
                )
        return items

    def _same_origin(self, url: str) -> bool:
        try:
            ours, theirs = httpx.URL(self._base_url), httpx.URL(url)
        except httpx.InvalidURL:
            return False
        return (theirs.scheme, theirs.host, theirs.port) == (ours.scheme, ours.host, ours.port)

    async def _access_token(self, refused: str | None = None) -> str:
        """The token to send.

        A new one is fetched when there is none, when the kept one is within a minute of expiring,
        and when the kept one is `refused`, the token Ramp has just turned away. If another task
        has already replaced it, that task's token is used.
        """
        async with self._token_lock:
            token = self._token
            if token is None or token.value == refused or monotonic() >= token.expires_at:
                token = self._token = await self._fetch_token()
            return token.value

    async def _fetch_token(self) -> _Token:
        url = self._url("/token")
        response = await self._once(
            "POST",
            url,
            auth=self._basic,
            data={"grant_type": "client_credentials", "scope": self.SCOPES},
        )
        _check("POST", url, response)
        answer = _parse(_TokenAnswer, response)
        return _Token(answer.access_token, monotonic() + answer.expires_in - TOKEN_RENEW_MARGIN_S)

    async def _once(
        self, method: str, url: str, *, headers: dict[str, str] | None = None, **kwargs: Any
    ) -> httpx.Response:
        """One request, with no retry and no status check. httpx errors become RampError(0)."""
        try:
            return await self._http.request(
                method,
                url,
                headers={**_HEADERS, **(headers or {})},
                timeout=REQUEST_TIMEOUT_S,
                follow_redirects=False,
                **kwargs,
            )
        except httpx.HTTPError as exc:
            logger.warning("ramp %s %s: no answer (%s)", method, _log_path(url), type(exc).__name__)
            raise RampError(0, message=type(exc).__name__) from exc

    async def _send(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, str | int] | None = None,
        json: dict | None = None,
        redact: tuple[str, ...] = (),
    ) -> httpx.Response:
        """Send one call and return its 2xx answer. Anything else raises RampError.

        A GET is tried again after a 429 or a 5xx, twice at most, waiting as _retry_wait says. A
        POST is never tried again for those, or for no answer at all. A 401, for either method, gets
        a new token and one more send. `redact` values never appear in the error.
        """
        waits = iter(READ_RETRY_WAITS_S if method == "GET" else ())
        renewed, refused = False, None
        while True:
            token = await self._access_token(refused)
            response = await self._once(
                method, url, headers={"Authorization": f"Bearer {token}"}, params=params, json=json
            )
            refused = None
            status = response.status_code
            if status == 401 and not renewed:
                logger.warning(
                    "ramp %s %s -> 401: the token was refused, getting a new one",
                    method,
                    _log_path(url),
                )
                renewed, refused = True, token
                continue
            if status == 429 or status >= 500:
                wait = _retry_wait(response, next(waits, None))
                if wait is not None:
                    logger.warning(
                        "ramp %s %s -> HTTP %s, trying again in %gs",
                        method,
                        _log_path(url),
                        status,
                        wait,
                    )
                    await sleep(wait)
                    continue
            break
        _check(method, url, response, redact)
        if method != "GET":
            logger.info(
                "ramp %s %s -> HTTP %s trace_id=%s",
                method,
                _log_path(url),
                status,
                response.headers.get("x-trace-id", "-"),
            )
        return response
