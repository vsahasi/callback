"""Where a vendor's reference voice is kept, and the endpoint that puts it there.

Until something is stored here, `voice_match` reports `not_run` on every real
call and its 0.35 weight is redistributed over the four signals that did run.
That redistribution is exactly the condition that makes a diluted FAIL
dangerous, which is documented at length in `detector.scoring`. So this module
is not a feature on the side: it is what lets the heaviest signal in the model
participate at all.

What is stored
--------------
A `VoicePrint`, never a recording. See `detector.analysis.voiceprint` for why
that shape and not the audio. The consequence for this module is that the
store holds a few hundred floats per vendor, which is what makes a hard cap
on identities a sensible bound rather than a memory limit in disguise.

Why a put must be explicit about replacing
------------------------------------------
An attacker holding the service token cannot release a payment: a human
approves that. But if they could overwrite a vendor's enrolment with their own
voice, the strongest signal the detector has would pass for them on every
future call, silently, for as long as nobody re-enrolled that vendor. So
`replace` is a required opt-in rather than a default, and the refusal is a 409
rather than a quiet overwrite.

Why a full store refuses instead of evicting
--------------------------------------------
The session store evicts, because sessions are transient and the alternative
is unbounded growth. Enrolments are not transient. Evicting one to make room
would silently remove a signal from some other vendor's next call, with nobody
watching. Refusing fails in front of the operator who is onboarding right now
and who can raise the cap. The cap is the growth bound either way: a token
holder can fill the store, which stops onboarding, and cannot grow it past
`max_identities` prints.

Backends
--------
    InMemoryVoicePrintStore   the default. One process, lost on restart.
    DictVoicePrintStore       a `modal.Dict` behind the same interface, so
                              enrolments survive a deploy and are shared
                              across replicas.

Neither needs the read-modify-write machinery the session store has, because a
put replaces a whole record and nothing here is built up across requests. Two
operators enrolling the same vendor at the same time is a last-write-wins race
over two recordings of the same voice, which is not a safety problem.
"""

from __future__ import annotations

import asyncio
import hmac
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from detector.analysis.dsp import build_context
from detector.analysis.voiceprint import (
    VoicePrint,
    voiceprint_from,
    voiceprint_from_json,
    voiceprint_to_json,
)
from detector.audio import AudioPolicy, AudioUnavailable, load_span
from detector.config import Settings
from detector.errors import DetectorError
from detector.store import (
    DictBacking,
    StoreFormatError,
    StoreUnavailable,
    open_modal_dict,
)
from shared.contract import AudioRef, ErrorBody, ErrorCode, ErrorResponse
from shared.enrolment import (
    MIN_CONFIDENT_VOICED_SECONDS,
    EnrolmentQuality,
    EnrolmentRecord,
    EnrolmentRequest,
    EnrolmentResponse,
    EnrolmentStatus,
    ErasureResponse,
    IdentityBody,
    normalise_identity,
)

__all__ = [
    "DEFAULT_ENROLMENT_DICT_NAME",
    "DEFAULT_MAX_IDENTITIES",
    "ENROLMENT_FORMAT_VERSION",
    "DictVoicePrintStore",
    "Enrolment",
    "EnrolmentConflict",
    "EnrolmentRefused",
    "EnrolmentStoreFull",
    "InMemoryVoicePrintStore",
    "VoicePrintStore",
    "build_enrolment_router",
    "enrolment_from_json",
    "enrolment_to_json",
    "extract_print",
    "quality_of",
    "store_from_settings",
]

ENROLMENT_FORMAT_VERSION: Final = 1
DEFAULT_MAX_IDENTITIES: Final = 10_000
DEFAULT_ENROLMENT_DICT_NAME: Final = "callback-voiceprints"

_KEY_PREFIX: Final = "vp:"


def _now() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------- errors
#
# All three carry ErrorCode.VALIDATION_ERROR. The contract's ErrorCode list is
# frozen by the schema snapshot and `shared.contract.ErrorBody` forbids values
# outside it, so adding a member here would break a strict client parsing an
# error body. The HTTP status carries the distinction instead, which is what a
# client branches on anyway.


class EnrolmentRefused(DetectorError):
    """The recording cannot support a comparison, so it was not stored."""

    http_status = 422
    code = ErrorCode.VALIDATION_ERROR


class EnrolmentConflict(DetectorError):
    """Something is already enrolled for this identity and replace was not set."""

    http_status = 409
    code = ErrorCode.VALIDATION_ERROR


class EnrolmentStoreFull(DetectorError):
    """At the identity cap. Raise it or erase something."""

    http_status = 503
    code = ErrorCode.VALIDATION_ERROR


# -------------------------------------------------------------- stored record


@dataclass(frozen=True)
class Enrolment:
    """One vendor's reference voice, as the store holds it."""

    identity: str
    voice: VoicePrint
    enrolled_at: datetime
    quality: EnrolmentQuality
    #: Scheme and host the recording came from. Never the uri: a signed
    #: object-store url carries a credential in its query string and this
    #: value reaches the operator's response and the logs.
    source: str
    label: str | None = None

    def record(self) -> EnrolmentRecord:
        """The wire view. Carries no feature of the voice."""

        return EnrolmentRecord(
            identity=self.identity,
            enrolled_at=self.enrolled_at,
            quality=self.quality,
            voiced_seconds=round(self.voice.voiced_seconds, 3),
            duration_s=round(self.voice.duration_s, 3),
            sample_rate_hz=self.voice.sample_rate,
            source=self.source,
            label=self.label,
        )


def quality_of(voice: VoicePrint) -> EnrolmentQuality:
    """How much a comparison against this print will be worth.

    `good` needs both cues measurable and enough voiced speech that
    `detector.analysis.voice` will call the comparison confident. With one cue
    missing the comparison can only ever be inconclusive, which is `thin`: it
    still catches a wildly different talker, so it is worth keeping, and the
    operator is told not to rely on it.
    """

    if not voice.comparable:
        return EnrolmentQuality.UNUSABLE
    if not (voice.has_pitch and voice.has_spectrum):
        return EnrolmentQuality.THIN
    if voice.voiced_seconds < MIN_CONFIDENT_VOICED_SECONDS:
        return EnrolmentQuality.THIN
    return EnrolmentQuality.GOOD


def enrolment_to_json(enrolment: Enrolment) -> dict[str, Any]:
    return {
        "format": ENROLMENT_FORMAT_VERSION,
        "identity": enrolment.identity,
        "voice": voiceprint_to_json(enrolment.voice),
        "enrolled_at": enrolment.enrolled_at.isoformat(),
        "quality": enrolment.quality.value,
        "source": enrolment.source,
        "label": enrolment.label,
    }


def enrolment_from_json(doc: dict[str, Any]) -> Enrolment:
    """Rebuild a record, or raise `StoreFormatError`.

    Guessing at a document written by a different build is how a fail-closed
    service quietly starts failing open, so an unreadable record is a 500 and
    not a silently empty enrolment. Same reasoning as the session codec.
    """

    try:
        version = int(doc["format"])
        if version != ENROLMENT_FORMAT_VERSION:
            raise StoreFormatError(
                f"voice print written in format {version}, this build reads "
                f"{ENROLMENT_FORMAT_VERSION}"
            )
        return Enrolment(
            identity=str(doc["identity"]),
            voice=voiceprint_from_json(doc["voice"]),
            enrolled_at=datetime.fromisoformat(str(doc["enrolled_at"])),
            quality=EnrolmentQuality(str(doc["quality"])),
            source=str(doc["source"]),
            label=None if doc.get("label") is None else str(doc["label"]),
        )
    except StoreFormatError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise StoreFormatError(f"unreadable voice print record: {exc}") from exc


# --------------------------------------------------------------------- stores


class VoicePrintStore(ABC):
    """Where reference prints live.

    Narrow on purpose. There is no `update`, because a put replaces the whole
    record, and no sweep, because nothing here expires on its own: an
    enrolment is valid until the vendor is re-enrolled or erased.
    """

    def __init__(self, *, max_identities: int = DEFAULT_MAX_IDENTITIES) -> None:
        self.max_identities = max(1, max_identities)

    @abstractmethod
    def get(self, identity: str) -> Enrolment | None:
        """The enrolment for this identity, or None. Never raises on a miss."""

    @abstractmethod
    def put(self, enrolment: Enrolment, *, replace: bool) -> None:
        """Store it. Raises `EnrolmentConflict` or `EnrolmentStoreFull`."""

    @abstractmethod
    def delete(self, identity: str) -> bool:
        """Erase it. False when there was nothing stored, which is not an error."""

    @abstractmethod
    def identities(self) -> Iterator[str]:
        """Every stored identity. For the cap check and for tests."""

    @abstractmethod
    def clear(self) -> None: ...

    def __len__(self) -> int:
        return sum(1 for _ in self.identities())

    def _guard_capacity(self, stored: int, *, existing: bool) -> None:
        """Refuse a put that would take the store past its cap.

        `stored` is passed in rather than read from `len(self)` because the
        in-memory backend calls this while already holding its lock, and
        `threading.Lock` is not reentrant.
        """

        if existing:
            # Replacing a record does not grow the store.
            return
        if stored >= self.max_identities:
            raise EnrolmentStoreFull(
                "the voice print store is at its identity limit",
                f"cap is {self.max_identities}; raise "
                "CALLBACK_ENROLMENT_MAX_IDENTITIES or erase an enrolment "
                "that is no longer needed",
            )


class InMemoryVoicePrintStore(VoicePrintStore):
    """The default. One process, lost on restart.

    Fine for a single-replica deployment and for development, and the reason
    the Modal deployment is still pinned to one container. In production the
    loss on restart is the real cost: every vendor would have to be re-enrolled
    after a deploy, which is why `DictVoicePrintStore` exists.
    """

    def __init__(self, *, max_identities: int = DEFAULT_MAX_IDENTITIES) -> None:
        super().__init__(max_identities=max_identities)
        self._records: dict[str, Enrolment] = {}
        self._lock = threading.Lock()

    def get(self, identity: str) -> Enrolment | None:
        with self._lock:
            return self._records.get(normalise_identity(identity))

    def put(self, enrolment: Enrolment, *, replace: bool) -> None:
        key = normalise_identity(enrolment.identity)
        with self._lock:
            existing = key in self._records
            if existing and not replace:
                raise EnrolmentConflict(
                    "an enrolment already exists for this identity",
                    "send replace=true to overwrite it",
                )
            self._guard_capacity(len(self._records), existing=existing)
            self._records[key] = enrolment

    def delete(self, identity: str) -> bool:
        with self._lock:
            return self._records.pop(normalise_identity(identity), None) is not None

    def identities(self) -> Iterator[str]:
        with self._lock:
            return iter(tuple(self._records))

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


class DictVoicePrintStore(VoicePrintStore):
    """A `modal.Dict` behind the same interface.

    Enrolments survive a deploy and are visible to every replica, which is
    what this needs to be useful in production. The session store's lease
    machinery is absent on purpose: there is no read-modify-write here, so the
    one race left is two operators enrolling the same vendor at the same
    moment, and last write wins between two recordings of the same voice.

    `replace=False` is enforced with `put(skip_if_exists=True)`, which is the
    one atomic primitive `modal.Dict` offers, so a concurrent first enrolment
    cannot be clobbered by another one claiming not to replace anything.
    """

    def __init__(
        self,
        backing: DictBacking | None = None,
        *,
        name: str = DEFAULT_ENROLMENT_DICT_NAME,
        create_if_missing: bool = True,
        environment_name: str | None = None,
        max_identities: int = DEFAULT_MAX_IDENTITIES,
    ) -> None:
        super().__init__(max_identities=max_identities)
        self._name = name
        self._create_if_missing = create_if_missing
        self._environment_name = environment_name
        self._backing = backing

    @property
    def backing(self) -> DictBacking:
        if self._backing is None:
            self._backing = open_modal_dict(
                self._name,
                create_if_missing=self._create_if_missing,
                environment_name=self._environment_name,
            )
        return self._backing

    def _key(self, identity: str) -> str:
        return f"{_KEY_PREFIX}{normalise_identity(identity)}"

    def get(self, identity: str) -> Enrolment | None:
        try:
            doc = self.backing.get(self._key(identity))
        except StoreUnavailable:
            raise
        except Exception as exc:
            raise StoreUnavailable(f"voice print read failed: {exc}") from exc
        if doc is None:
            return None
        if not isinstance(doc, dict):
            raise StoreFormatError("voice print record is not a mapping")
        return enrolment_from_json(doc)

    def put(self, enrolment: Enrolment, *, replace: bool) -> None:
        key = self._key(enrolment.identity)
        doc = enrolment_to_json(enrolment)
        try:
            existing = self.backing.get(key) is not None
        except StoreUnavailable:
            raise
        except Exception as exc:
            raise StoreUnavailable(f"voice print read failed: {exc}") from exc
        if existing and not replace:
            raise EnrolmentConflict(
                "an enrolment already exists for this identity",
                "send replace=true to overwrite it",
            )
        self._guard_capacity(len(self), existing=existing)
        try:
            landed = self.backing.put(key, doc, skip_if_exists=not replace)
        except Exception as exc:
            raise StoreUnavailable(f"voice print write failed: {exc}") from exc
        if not landed:
            # Another writer enrolled this identity between the read above and
            # this put. Report the conflict rather than overwriting what they
            # stored, which is what replace=False asked for.
            raise EnrolmentConflict(
                "an enrolment already exists for this identity",
                "send replace=true to overwrite it",
            )

    def delete(self, identity: str) -> bool:
        try:
            return self.backing.pop(self._key(identity), None) is not None
        except Exception as exc:
            raise StoreUnavailable(f"voice print erase failed: {exc}") from exc

    def identities(self) -> Iterator[str]:
        try:
            keys = tuple(self.backing.keys())
        except Exception as exc:
            raise StoreUnavailable(f"voice print listing failed: {exc}") from exc
        return (
            str(key)[len(_KEY_PREFIX) :]
            for key in keys
            if isinstance(key, str) and key.startswith(_KEY_PREFIX)
        )

    def clear(self) -> None:
        for identity in tuple(self.identities()):
            self.delete(identity)


def store_from_settings(settings: Settings) -> VoicePrintStore:
    """The store this deployment is configured for."""

    if settings.enrolment_dict_name:
        return DictVoicePrintStore(
            name=settings.enrolment_dict_name,
            max_identities=settings.enrolment_max_identities,
        )
    return InMemoryVoicePrintStore(max_identities=settings.enrolment_max_identities)


# ------------------------------------------------------------------ extraction


def _measure(samples: Any, sample_rate: int) -> VoicePrint:
    return voiceprint_from(build_context(samples, sample_rate))


async def extract_print(
    ref: AudioRef,
    policy: AudioPolicy,
    *,
    client: httpx.AsyncClient | None = None,
) -> tuple[VoicePrint, str]:
    """Fetch an enrolment recording and reduce it to a print.

    Raises `AudioUnavailable` on anything that stops it, which the router
    turns into a 422 with the reason. Unlike the call path, an enrolment is
    not happening inside a live conversation, so a failure here should be loud
    rather than swallowed into an absent signal.

    Returns the print and the source description, which is a scheme and host
    and never the uri.
    """

    loaded = await load_span(ref, policy, client=client)
    voice = await asyncio.to_thread(_measure, loaded.samples, loaded.sample_rate)
    return voice, loaded.source


# ------------------------------------------------------------------ the router


def _error(
    status: int, code: ErrorCode, message: str, detail: str | None
) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message, detail=detail))
    return JSONResponse(status_code=status, content=body.model_dump(mode="json"))


def _authorised(header: str | None, settings: Settings | None) -> bool:
    if settings is None:
        # Nothing to compare against means nobody is authorised. The
        # alternative is an open endpoint that writes to the store.
        return False
    if not header or not header.lower().startswith("bearer "):
        return False
    presented = header.split(" ", 1)[1].strip()
    return hmac.compare_digest(
        presented.encode("utf-8"), settings.api_token.encode("utf-8")
    )


def _detail_for(quality: EnrolmentQuality, voice: VoicePrint) -> str:
    if quality is EnrolmentQuality.GOOD:
        return (
            f"Enrolled on {voice.voiced_seconds:.1f}s of voiced speech. Voice "
            "comparison will run on this vendor's next call."
        )
    return (
        f"Enrolled, but only {voice.voiced_seconds:.1f}s of voiced speech was "
        f"found, under the {MIN_CONFIDENT_VOICED_SECONDS:.1f}s a confident "
        "comparison needs. This will catch an obviously different talker and "
        "little else. A longer recording would be worth taking."
    )


def _refusal_for(voice: VoicePrint) -> str:
    if not voice.usable:
        reason = "; ".join(voice.notes) or "too short or too low a sample rate"
        return f"the recording is not analysable ({reason})"
    if not voice.has_pitch and not voice.has_spectrum:
        return (
            "no voiced speech and no usable spectrum were found, so there is "
            "nothing to compare a caller against"
        )
    return "the recording produced nothing measurable"


def build_enrolment_router(
    settings: Settings | None = None,
    *,
    store: VoicePrintStore | None = None,
    policy: AudioPolicy | None = None,
    client_of: Callable[[], httpx.AsyncClient] | None = None,
) -> APIRouter:
    """Build the router.

    The identity travels in the request body on all three routes, including
    the two that only read. A vendor's user id in a path or a query string
    ends up in every access log and proxy log between the caller and here,
    which is the same class of leak this service already refuses for audio
    uris and rejected request bodies. So `status` and `erase` are POSTs that
    look a little unusual and keep identities out of the request line.

    Pass `store` to pin one, or leave it out and the store is read from
    `app.state.voiceprints`, which is what `detector.app` puts there.

    `client_of` is a getter rather than a client so that mounting this router
    does not open a connection pool. Only an https enrolment needs one, and
    `detector.app` points it at the audio pipeline's client so that the two
    share a pool instead of holding two.
    """

    router = APIRouter(prefix="/enrol", tags=["enrolment"])

    def _resolve(request: Request) -> tuple[Settings | None, VoicePrintStore | None]:
        resolved = settings or getattr(request.app.state, "settings", None)
        chosen = store or getattr(request.app.state, "voiceprints", None)
        return resolved, chosen

    def _policy_for(resolved: Settings) -> AudioPolicy:
        if policy is not None:
            return policy
        base = AudioPolicy.from_settings(resolved)
        # An enrolment is not inside a live call, so the 400 ms that keeps the
        # event path within the agent's budget is needlessly tight for pulling
        # a whole recording out of an object store.
        return AudioPolicy(
            enabled=base.enabled,
            allowed_hosts=base.allowed_hosts,
            allow_file_uris=base.allow_file_uris,
            file_root=base.file_root,
            allow_data_uris=base.allow_data_uris,
            max_bytes=base.max_bytes,
            fetch_timeout_ms=max(
                base.fetch_timeout_ms, resolved.enrolment_fetch_timeout_ms
            ),
            max_seconds=base.max_seconds,
        )

    @router.post("/voice", response_model=EnrolmentResponse)
    async def enrol_voice(payload: EnrolmentRequest, request: Request) -> Response:
        """Store a reference voice for an identity."""

        resolved, chosen = _resolve(request)
        if resolved is None or not _authorised(
            request.headers.get("authorization"), resolved
        ):
            return _error(
                401, ErrorCode.UNAUTHORIZED, "missing or invalid bearer token", None
            )
        if resolved.mock_mode:
            return _error(
                409,
                ErrorCode.VALIDATION_ERROR,
                "this deployment is in mock mode",
                "mock mode decides from scripted scenarios and never reads an "
                "enrolment, so storing one here would have no effect. Restart "
                "with CALLBACK_MOCK_MODE=0.",
            )
        if chosen is None:
            return _error(
                503,
                ErrorCode.VALIDATION_ERROR,
                "no voice print store is configured",
                None,
            )

        identity = normalise_identity(payload.identity)
        if not identity:
            return _error(
                422,
                ErrorCode.VALIDATION_ERROR,
                "identity is blank once trimmed",
                None,
            )

        try:
            voice, source = await extract_print(
                payload.audio,
                _policy_for(resolved),
                client=client_of() if client_of is not None else None,
            )
        except AudioUnavailable as exc:
            return _error(
                422,
                exc.code,
                "the enrolment recording could not be read",
                str(exc),
            )

        quality = quality_of(voice)
        if quality is EnrolmentQuality.UNUSABLE:
            # Refuse rather than store. A stored unusable print would make
            # every later comparison inconclusive while the operator believed
            # the vendor was covered.
            return _error(
                422,
                ErrorCode.VALIDATION_ERROR,
                "the enrolment recording cannot support a comparison",
                _refusal_for(voice),
            )

        enrolment = Enrolment(
            identity=identity,
            voice=voice,
            enrolled_at=_now(),
            quality=quality,
            source=source,
            label=payload.label,
        )
        try:
            chosen.put(enrolment, replace=payload.replace)
        except DetectorError as exc:
            return _error(exc.http_status, exc.code, exc.message, exc.detail)

        answer = EnrolmentResponse(
            stored=True,
            record=enrolment.record(),
            detail=_detail_for(quality, voice),
        )
        return JSONResponse(content=answer.model_dump(mode="json"))

    @router.post("/voice/status", response_model=EnrolmentStatus)
    async def enrolment_status(payload: IdentityBody, request: Request) -> Response:
        """Whether this identity has a reference voice stored."""

        resolved, chosen = _resolve(request)
        if not _authorised(request.headers.get("authorization"), resolved):
            return _error(
                401, ErrorCode.UNAUTHORIZED, "missing or invalid bearer token", None
            )
        identity = normalise_identity(payload.identity)
        found = chosen.get(identity) if chosen is not None else None
        answer = EnrolmentStatus(
            identity=identity,
            enrolled=found is not None,
            record=found.record() if found is not None else None,
        )
        return JSONResponse(content=answer.model_dump(mode="json"))

    @router.post("/voice/erase", response_model=ErasureResponse)
    async def erase_enrolment(payload: IdentityBody, request: Request) -> Response:
        """Erase this identity's reference voice."""

        resolved, chosen = _resolve(request)
        if not _authorised(request.headers.get("authorization"), resolved):
            return _error(
                401, ErrorCode.UNAUTHORIZED, "missing or invalid bearer token", None
            )
        identity = normalise_identity(payload.identity)
        deleted = chosen.delete(identity) if chosen is not None else False
        answer = ErasureResponse(identity=identity, deleted=deleted)
        return JSONResponse(content=answer.model_dump(mode="json"))

    return router


# Mountable without arguments; settings and store then come from app.state.
router = build_enrolment_router()
