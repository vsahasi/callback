"""How a person's voice print is keyed, and where to look for it.

Thirty lines in a module of their own because three layers need to agree on
them and none of the three should have to import the other two. The enrolment
router writes keys, the audio pipeline reads them, and `detector.v1` does
both; putting the key shape in any one of those would drag its dependencies
into the others, and the audio pipeline in particular has no business
importing a web framework to find out how a dictionary key is spelled.

A print is stored per person **per channel**. Ruling section 2.15: "Each
channel enrolls separately. Real-mode prints go into P3's VoicePrintStore
under the identity `{user_id}:{channel}`." That is not bookkeeping. A phone
handset and a browser microphone put a different frequency response in front
of the same larynx, and the spectral half of the comparison is confounded by
the channel about as much as by the talker, so a browser print judging a
phone call would be measuring the handset. One print cannot serve both.

Keys are plain strings here, not the contract's `Channel` enum, so that this
layer does not need the agent contract either. `detector.v1.enrollments`
keeps the typed shorthand over the top.
"""

from __future__ import annotations

from typing import Final

from shared.enrolment import normalise_identity

__all__ = [
    "EVENTS_CHANNEL",
    "candidate_identities",
    "store_identity",
]

#: The channel the root `/events` API is about. Callback telephones the
#: vendor, so a call arriving there is a phone call.
EVENTS_CHANNEL: Final = "phone"


def store_identity(user_id: str, channel: str | None = None) -> str:
    """The store key of one person's print, optionally on one channel.

    With no channel this is the bare identity, which is what the operator
    route writes when it is not told one. Those unscoped prints are the
    development and demo path; see `candidate_identities`.
    """

    base = normalise_identity(user_id)
    return f"{base}:{channel}" if channel else base


def candidate_identities(
    user_id: str, channel: str | None = EVENTS_CHANNEL
) -> tuple[str, ...]:
    """Where to look for this person's print, best first.

    A reader that knows its channel wants that channel's print, and failing
    that an unscoped one.

    Deliberately not every channel. Falling back to a browser print to judge
    a phone call would compare across exactly the thing the two keys exist to
    keep apart, and a confident wrong answer from the heaviest signal in the
    model is worse than no answer at all: with no print `voice_match` reports
    `not_run`, which removes a signal and denies more often. Missing a print
    costs a step-up. Using the wrong one costs the property the signal is
    there to provide.
    """

    bare = normalise_identity(user_id)
    if not bare:
        return ()
    if not channel:
        return (bare,)
    return (f"{bare}:{channel}", bare)
