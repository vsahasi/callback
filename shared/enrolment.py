"""Wire models for enrolling a vendor's reference voice.

The detector's heaviest signal, `voice_match` at 0.35, needs something to
compare the caller against. This is the surface that puts it there: one
recording of the vendor, from a call that is already trusted, reduced to a
`VoicePrint` and stored under the identity the caller will later claim.

A separate surface, deliberately
--------------------------------
This is not in `shared.contract`. The contract is frozen by a schema snapshot
and carries a version the agent pins, and enrolment is not something the
agent does during a call: it happens once per vendor, out of band, from
whatever tool onboards them. `shared.verification` is split out for the same
reason, and this file follows it so that adding enrolment cannot move
`CONTRACT_VERSION` or invalidate the contract snapshot.

What a response may say
-----------------------
Never the print. Not the median pitch, not the spectrum, not a similarity
against anything. A pitch register is a weak biometric and a service that
hands it back on request is a service that leaks one, so the response carries
only what an operator needs in order to judge whether the enrolment is good
enough to rely on: how much voiced speech was in it, how long it was, where
it came from, and a quality verdict. The one number that matters for the
decision, how close the next caller sounds, is reported during the call as a
`Factor` and nowhere else.

Identity is opaque
------------------
The identity is matched byte for byte after trimming surrounding whitespace.
No case folding, no unicode normalisation, no stripping of punctuation. The
tempting alternative, canonicalising so that a caller who claims
"CFO@Northwind.example" finds an enrolment stored as "cfo@northwind.example",
trades a missed lookup for a possible collision. A missed lookup leaves
`voice_match` at `not_run`, which removes a signal and denies more often. A
collision compares one vendor against another vendor's voice. Those two
mistakes are not the same size, so the matching stays strict and the caller
is expected to use one spelling.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Final, Literal

from pydantic import BaseModel, Field

from shared.contract import AudioRef, StrictModel

__all__ = [
    "CHANNELS",
    "ENROLMENT_VERSION",
    "MAX_IDENTITY_LENGTH",
    "MIN_CONFIDENT_VOICED_SECONDS",
    "EnrolmentQuality",
    "EnrolmentRecord",
    "EnrolmentRequest",
    "EnrolmentResponse",
    "EnrolmentStatus",
    "ErasureResponse",
    "IdentityBody",
    "enrolment_schema_bundle",
    "normalise_identity",
]

# Bumped whenever any model here changes shape. Independent of
# CONTRACT_VERSION on purpose: the agent pins that one.
ENROLMENT_VERSION: Final = "0.2.0-provisional"

MAX_IDENTITY_LENGTH: Final = 128

# The channels a print can be stored against. These are the members of
# `callback_shared.contract.Channel`, written out rather than imported so this
# module stays independent of the agent contract; a test asserts the two lists
# agree, so adding a channel there fails here rather than drifting.
#
# Restricted rather than free text on purpose. A print written under a channel
# nobody reads is invisible, and an invisible print is the exact failure this
# whole surface exists to remove: `voice_match` quietly not running.
CHANNELS: Final = ("phone", "browser")

# Below this much voiced speech, `detector.analysis.voice` refuses to call any
# comparison confident, so every call against such a reference comes back
# inconclusive. Stored anyway, flagged `thin`, because a weak reference is
# still better than none for catching a wildly different talker.
MIN_CONFIDENT_VOICED_SECONDS: Final = 1.5


class EnrolmentQuality(StrEnum):
    """How much a comparison against this reference will be worth."""

    #: Enough voiced speech for a confident verdict.
    GOOD = "good"
    #: Stored, but short on voiced speech. Comparisons will tend to come back
    #: inconclusive rather than pass or fail.
    THIN = "thin"
    #: Refused. Nothing measurable to compare against, so storing it would
    #: create the appearance of coverage without any of it.
    UNUSABLE = "unusable"


def normalise_identity(raw: str) -> str:
    """Trim surrounding whitespace. Nothing else. See the module docstring."""

    return raw.strip()


class EnrolmentRequest(StrictModel):
    """POST /enrol/voice body."""

    identity: str = Field(
        min_length=1,
        max_length=MAX_IDENTITY_LENGTH,
        description="The claimed_user_id this reference belongs to. Matched "
        "byte for byte after trimming whitespace.",
    )
    channel: Literal["phone", "browser"] | None = Field(
        default=None,
        description="Which channel this recording came over. A print is "
        "stored per channel, because a handset and a browser microphone put "
        "a different response in front of the same voice. Omit it and the "
        "print is stored unscoped, which the phone path falls back to.",
    )
    audio: AudioRef = Field(
        description="Where the enrolment recording is. Fetched under the same "
        "policy as call audio, so the host must be allowlisted."
    )
    replace: bool = Field(
        default=False,
        description="Overwrite an existing enrolment for this identity. "
        "Required, because silently replacing a reference would let anyone "
        "holding the service token swap in their own voice.",
    )
    label: str | None = Field(
        default=None,
        max_length=200,
        description="Operator note about where the recording came from. "
        "Returned verbatim; never used for matching.",
    )


class IdentityBody(StrictModel):
    """Body of the two routes that only name an identity.

    A body rather than a path segment or a query parameter, because a vendor's
    user id in the request line is logged by every proxy on the way in. See
    the note on `build_enrolment_router`.
    """

    identity: str = Field(min_length=1, max_length=MAX_IDENTITY_LENGTH)
    channel: Literal["phone", "browser"] | None = Field(
        default=None,
        description="Which channel's print to read or erase. Omit it for the "
        "unscoped one.",
    )


class EnrolmentRecord(StrictModel):
    """What the service will say about a stored enrolment.

    Carries no feature of the voice itself. See the module docstring.
    """

    identity: str
    channel: Literal["phone", "browser"] | None = None
    enrolled_at: datetime
    quality: EnrolmentQuality
    voiced_seconds: float = Field(
        ge=0.0, description="Voiced speech found in the recording."
    )
    duration_s: float = Field(ge=0.0, description="Length of the recording.")
    sample_rate_hz: int = Field(
        ge=0, description="Rate the analysis ran at, after any decimation."
    )
    source: str = Field(
        description="Scheme and host the recording came from. Never the full "
        "uri, which may carry a credential in its query string."
    )
    label: str | None = None


class EnrolmentResponse(StrictModel):
    """POST /enrol/voice 200 response."""

    enrolment_version: str = ENROLMENT_VERSION
    stored: bool
    record: EnrolmentRecord
    detail: str = Field(
        max_length=500, description="One sentence an operator can act on."
    )


class EnrolmentStatus(StrictModel):
    """GET /enrol/voice/{identity} 200 response.

    Answers one question, so the onboarding tool can tell whether a vendor
    still needs enrolling without guessing from a 404.
    """

    enrolment_version: str = ENROLMENT_VERSION
    identity: str
    channel: Literal["phone", "browser"] | None = None
    enrolled: bool
    record: EnrolmentRecord | None = None


class ErasureResponse(StrictModel):
    """DELETE /enrol/voice/{identity} 200 response.

    Present because the thing being stored is derived from biometric data and
    a vendor may ask for it to be gone. `deleted` false means there was
    nothing stored, which is not an error.
    """

    enrolment_version: str = ENROLMENT_VERSION
    identity: str
    channel: Literal["phone", "browser"] | None = None
    deleted: bool


# Frozen by tests/contract/test_enrolment.py, in a stable order.
SCHEMA_MODELS: tuple[type[BaseModel], ...] = (
    EnrolmentRequest,
    IdentityBody,
    EnrolmentRecord,
    EnrolmentResponse,
    EnrolmentStatus,
    ErasureResponse,
)


def enrolment_schema_bundle() -> dict[str, Any]:
    """Deterministic JSON-schema bundle of the enrolment surface."""

    return {
        "enrolment_version": ENROLMENT_VERSION,
        "enums": {
            EnrolmentQuality.__name__: [member.value for member in EnrolmentQuality],
        },
        "limits": {
            "max_identity_length": MAX_IDENTITY_LENGTH,
            "min_confident_voiced_seconds": MIN_CONFIDENT_VOICED_SECONDS,
        },
        "models": {
            model.__name__: model.model_json_schema(mode="serialization")
            for model in SCHEMA_MODELS
        },
    }
