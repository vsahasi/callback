"""The two voices on a callback, and whether they are the same person.

A callback session carries audio from two legs. The agent names the leg in
`AudioRef.track_id`:

    caller      whoever rang in and claimed to be the vendor
    callback    whoever answered when the service rang the number on file

The callback leg is trustworthy by construction: the service dialled a number
it already had, and the person who picked up is whoever holds that phone. The
caller leg is the claim being tested. If the two are different voices, nothing
else about the call matters. Someone claiming to be the vendor rang in, and
the real vendor answered the callback and said yes. No enrolment is needed to
catch this, and no one-time code can rescue it, because the code would go to
the phone the real vendor is holding.

What is kept
------------
One `VoicePrint` per leg, combined from each span as it arrives with
`combine_prints`, plus how many spans fed it. Never audio: a span's samples
are reduced to a print inside the analysis engine and dropped. The print is a
pitch register and a long-term spectrum, which cannot reproduce what was said;
see `detector.analysis.voiceprint` for why that shape.

When the check runs
-------------------
Only when both legs have a print and each carries at least
`MIN_LEG_SPEECH_SECONDS` of speech. Below that the comparison would be
tempered toward 0.5 anyway (`detector.analysis.voice` needs 0.6 s voiced on
each side to call itself confident), and reporting a half-measured
"inconclusive" on the heaviest factor in the model is worse than waiting one
more span. A leg that is still short leaves `voice_match` exactly where the
enrolment comparison left it.

How it scores
-------------
`compare_prints`, the same function that judges a caller against an enrolled
reference, with the same thresholds (`detector.analysis.cues`: pass at or
above 0.62, fail at or below 0.32, inconclusive between or when either side
is too short to be confident). The result is a `voice_match` factor, because
that is what it is, and because the wire contract is frozen by a schema
snapshot. A failing comparison is marked with `SPEAKER_MISMATCH_PREFIX` in its
detail, which `detector.scoring.fatal_factor` reads as unambiguous attack
evidence. `combine_factors` then keeps whichever `voice_match` scored lowest,
so a cross-leg mismatch is never averaged away by a matching enrolment, and a
mismatching enrolment is never hidden by consistent legs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final

from detector.analysis.engine import DEFAULT_WEIGHTS
from detector.analysis.voice import compare_prints
from detector.analysis.voiceprint import (
    VoicePrint,
    combine_prints,
    voiceprint_from_json,
    voiceprint_to_json,
)
from detector.scoring import SPEAKER_MISMATCH_PREFIX
from shared.contract import Factor, FactorName, FactorVerdict

__all__ = [
    "CALLBACK",
    "CALLER",
    "MIN_LEG_SPEECH_SECONDS",
    "LegPrint",
    "absorb_span",
    "cross_leg_factor",
    "leg_of",
    "legs_from_json",
    "legs_to_json",
]

CALLER: Final = "caller"
CALLBACK: Final = "callback"

#: Speech a leg needs before it is compared. Two seconds of speech is around
#: 1.5 s voiced on the synthetic corpus, which is `MIN_CONFIDENT_VOICED_SECONDS`
#: in `shared.enrolment`: the same bar an enrolment has to clear to be called
#: good.
MIN_LEG_SPEECH_SECONDS: Final = 2.0

# "caller", "Caller-0", "callback:1", "CALLBACK_7". The leg name, then either
# the end of the string or a separator. "callers" is not a leg.
_TRACK = re.compile(r"^(caller|callback)(?:$|[^a-z0-9])", re.IGNORECASE)


def leg_of(track_id: str | None) -> str | None:
    """Which leg a track belongs to, or None for a track that names neither.

    Tolerant of a numeric or instance suffix because that is how LiveKit
    track ids tend to be spelled; strict about the stem because a wrong guess
    here would compare a voice against itself and call the legs consistent.
    """

    if not track_id:
        return None
    found = _TRACK.match(track_id.strip())
    if found is None:
        return None
    return found.group(1).lower()


@dataclass(frozen=True)
class LegPrint:
    """One leg's accumulated voice, and how much went into it."""

    voice: VoicePrint
    spans: int = 1

    @classmethod
    def from_print(cls, voice: VoicePrint) -> LegPrint:
        return cls(voice=voice, spans=1)

    @property
    def speech_seconds(self) -> float:
        return float(self.voice.speech_seconds)

    @property
    def enough(self) -> bool:
        return self.voice.comparable and self.speech_seconds >= MIN_LEG_SPEECH_SECONDS

    def absorb(self, incoming: VoicePrint) -> LegPrint:
        """This leg with one more span folded in.

        Raises `ValueError` if the span was analysed on a different frequency
        grid; the caller decides whether to drop the span. Combining would
        line up different frequencies and call the result a spectrum.
        """

        combined = combine_prints((self.voice, incoming))
        assert combined is not None  # two inputs never combine to nothing
        return LegPrint(voice=combined, spans=self.spans + 1)


def absorb_span(
    legs: dict[str, LegPrint], track_id: str | None, voice: VoicePrint | None
) -> str | None:
    """Fold one analysed span into the leg its track names. Returns the leg.

    Mutates `legs` in place; it is the session's dictionary. Returns None,
    and changes nothing, for a track that is not a leg, a span that produced
    no print, a print from unusable audio (which would poison the combined
    print: `combine_prints` makes the whole leg unusable), or a print on a
    different frequency grid from what the leg already holds.
    """

    leg = leg_of(track_id)
    if leg is None or voice is None or not voice.usable:
        return None
    held = legs.get(leg)
    if held is None:
        legs[leg] = LegPrint.from_print(voice)
        return leg
    try:
        legs[leg] = held.absorb(voice)
    except ValueError:
        return None
    return leg


def cross_leg_factor(legs: dict[str, LegPrint]) -> Factor | None:
    """Compare the caller leg against the callback leg, if both are ready.

    None when either leg is missing or still too short, which leaves the
    `voice_match` factor to the enrolment comparison. Otherwise a factor on
    the 0..1 "consistent with the claimed identity" scale, failing at or
    below the threshold the enrolment comparison fails at, with a detail that
    says in words that this was the two legs of the call being compared.
    """

    caller = legs.get(CALLER)
    callback = legs.get(CALLBACK)
    if caller is None or callback is None:
        return None
    if not caller.enough or not callback.enough:
        return None

    result = compare_prints(caller.voice, callback.voice)
    score = max(0.0, min(1.0, result.score))
    verdict = result.verdict
    scope = (
        f"caller leg {caller.speech_seconds:.1f}s of speech over {caller.spans} "
        f"span{'s' if caller.spans != 1 else ''}, callback leg "
        f"{callback.speech_seconds:.1f}s over {callback.spans} "
        f"span{'s' if callback.spans != 1 else ''}"
    )

    # `verdict_for` in detector.analysis.cues: FAIL at or below
    # FAIL_AT_OR_BELOW, PASS at or above PASS_AT_OR_ABOVE, INCONCLUSIVE in
    # between or whenever the comparison was not confident.
    if verdict is FactorVerdict.FAIL:
        detail = (
            f"{SPEAKER_MISMATCH_PREFIX} caller and callback legs differ. The "
            "person who rang in is not the voice that answered the callback to "
            f"the number on file. {result.detail} ({scope})"
        )
    elif verdict is FactorVerdict.PASS:
        detail = (
            "Cross-leg check: caller and callback legs are consistent, the "
            "voice that rang in matches the voice that answered the callback. "
            f"{result.detail} ({scope})"
        )
    else:
        verdict = FactorVerdict.INCONCLUSIVE
        detail = (
            "Cross-leg check inconclusive: caller and callback legs could not "
            f"be told apart or together with confidence. {result.detail} ({scope})"
        )

    return Factor(
        name=FactorName.VOICE_MATCH,
        score=score,
        weight=DEFAULT_WEIGHTS[FactorName.VOICE_MATCH],
        verdict=verdict,
        detail=detail[:500],
    )


# ------------------------------------------------------------ serialisation
#
# Lossless, like the voice print codec it wraps, so a leg that round-trips
# through the session store compares identically to one that did not.


def legs_to_json(legs: dict[str, LegPrint]) -> dict[str, Any]:
    return {
        leg: {"spans": held.spans, "voice": voiceprint_to_json(held.voice)}
        for leg, held in legs.items()
    }


def legs_from_json(doc: dict[str, Any] | None) -> dict[str, LegPrint]:
    """Rebuild the legs. Raises `KeyError`, `TypeError` or `ValueError` on a
    malformed document, which the session codec turns into a format error."""

    if not doc:
        return {}
    return {
        str(leg): LegPrint(
            voice=voiceprint_from_json(entry["voice"]), spans=int(entry["spans"])
        )
        for leg, entry in doc.items()
    }
