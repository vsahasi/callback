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

And the latest comparison of the two, as a `CrossLegCheck` on the session
(`Session.cross_leg`): the score, the comparator's verdict and its confidence,
and how much speech each leg carried. That record, not the factor below, is
what `detector.scoring.fatal_factor` and `detector.evidence` read. The factor
competes with the enrolment comparison in worst-score-wins merging, so an
enrolment FAIL scoring lower would replace it there and a prefix on its detail
would vanish with it. The session's own field cannot be merged away.

Two gates, in seconds of speech per leg
---------------------------------------
`MIN_LEG_SPEECH_SECONDS` (2.0 s) is when the legs are compared at all. Below
it the comparison would be tempered toward 0.5 anyway (`detector.analysis.voice`
needs 0.6 s voiced on each side to call itself confident), and reporting a
half-measured "inconclusive" on the heaviest factor in the model is worse than
waiting one more span. A leg that is still short leaves `voice_match` exactly
where the enrolment comparison left it.

`MIN_CONFIRMED_LEG_SPEECH_SECONDS` (3.0 s) is when a mismatch may refuse the
call. A mismatch is *confirmed*, and therefore fatal and unrescuable, only when
the comparator said FAIL, said it was confident (its own flag, read from
`compare_prints`, not re-derived here), and both legs carry at least this much
speech. A FAIL that clears the first gate but not the second, like an
inconclusive result, is recorded on the session and goes no further: it adds
no `voice_match` factor (`CrossLegCheck.merge_factor` is None), so the factor
stays exactly where the enrolment comparison left it until the legs carry
enough speech to confirm the mismatch or to pass. It used to be merged as a
FAIL, which `/v1` read through `detector.bridge` as a voiceprint under its
0.45 fail line, one voice fail against a caller the comparator had not yet
made its mind up about.

The margin, measured
--------------------
On the synthetic corpus (`detector.analysis.synthetic`), 8 kHz, seeds 1 and 2
unless stated, caller against callback, `compare_prints` thresholds pass at or
above 0.62 and fail at or below 0.32:

    132 Hz against 215 Hz (9 st apart)    0.16  FAIL  at 4, 6 and 8 s, both rates
    same talker, +3 st, 4 s clips         0.85  PASS
    same talker, +4 st, 4 s clips         0.3216  inconclusive  (0.0016 above the line)
    same talker, +4 st, 6 s clips         0.3786  inconclusive; seeds 3/4: 0.2891 FAIL
    same talker, +4.5 st, 4 s clips       0.2031  FAIL
    same talker, +5 st, 4 s clips         0.1965  FAIL

So the within-speaker margin of the comparator itself is narrow: the same
talker four to five semitones up can already produce a confident FAIL, and the
confirmation gate does not widen that margin. What it buys is time. A
four-second turn carries 2.3 to 2.8 s of speech on this corpus (1.8 to 2.6 s
at 16 kHz), so one turn per leg can be compared but never refuses anyone;
refusal needs roughly five seconds of audio on each leg, or a second turn.
Two clearly different voices are confirmed as soon as the second gate is
crossed, at either sample rate (the 2.0 s and 3.0 s gates fall on the same
clip counts at 8 and 16 kHz). The inconclusive +4 st cases above never are,
and never touch the factor list.

How it scores
-------------
`compare_prints`, the same function that judges a caller against an enrolled
reference, with the same thresholds (`detector.analysis.cues`). A settled
result (a PASS, or a confirmed FAIL) is a `voice_match` factor, because that
is what it is, and because the wire contract is frozen by a schema snapshot.
`combine_factors` then keeps whichever `voice_match` scored lowest, so a
confirmed cross-leg mismatch is never averaged away by a matching enrolment,
and a mismatching enrolment is never hidden by consistent legs.

`/v1` learns of a confirmed mismatch by name: `detector.bridge` reads
`Session.cross_leg` beside the factor list and carries it as a fatal deny code
(`EngineSignals.fatal`), so the policy denies `speaker_mismatch` and locks the
session rather than reading the low voiceprint as one voice fail of two and
stepping the caller up for the code and callback that cannot rescue it.
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
from shared.contract import Factor, FactorName, FactorVerdict

__all__ = [
    "CALLBACK",
    "CALLER",
    "MIN_CONFIRMED_LEG_SPEECH_SECONDS",
    "MIN_LEG_SPEECH_SECONDS",
    "MISMATCH_HEADLINE",
    "CrossLegCheck",
    "LegPrint",
    "absorb_span",
    "compare_legs",
    "cross_leg_from_json",
    "cross_leg_to_json",
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

#: Speech both legs need before a confident FAIL may refuse the call. See
#: "The margin, measured" in the module docstring for what this buys.
MIN_CONFIRMED_LEG_SPEECH_SECONDS: Final = 3.0

#: How a confirmed mismatch's detail opens. A label for humans, not a marker
#: anything reads: the mechanism is `Session.cross_leg`.
MISMATCH_HEADLINE: Final = "Cross-leg speaker mismatch"

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


@dataclass(frozen=True)
class CrossLegCheck:
    """The latest comparison of the caller leg against the callback leg.

    Lives on the session as `Session.cross_leg`. Everything a decision needs
    is here in numbers, so the fatal rule reads this and never a detail
    string: the comparator's score and verdict, whether the comparator called
    itself confident, and how much speech each leg had when it ran.
    """

    score: float
    verdict: FactorVerdict
    confident: bool
    #: The comparator's own words: the measurements behind the verdict.
    detail: str
    caller_seconds: float
    caller_spans: int
    callback_seconds: float
    callback_spans: int

    @property
    def differ(self) -> bool:
        """The comparator said the two legs are different voices."""

        return self.verdict is FactorVerdict.FAIL

    @property
    def full(self) -> bool:
        """Both legs cleared the confirmation gate."""

        return (
            min(self.caller_seconds, self.callback_seconds)
            >= MIN_CONFIRMED_LEG_SPEECH_SECONDS
        )

    @property
    def confirmed(self) -> bool:
        """A mismatch solid enough to refuse the call on: FAIL, confident, full.

        This is the only thing that makes the cross-leg check fatal and
        unrescuable. Anything less is a step-up.
        """

        return self.differ and self.confident and self.full

    @property
    def scope(self) -> str:
        return (
            f"caller leg {self.caller_seconds:.1f}s of speech over "
            f"{self.caller_spans} span{'s' if self.caller_spans != 1 else ''}, "
            f"callback leg {self.callback_seconds:.1f}s over "
            f"{self.callback_spans} span{'s' if self.callback_spans != 1 else ''}"
        )

    def factor(self) -> Factor:
        """The `voice_match` factor for the risk blend.

        On the 0..1 "consistent with the claimed identity" scale, with the
        comparator's own verdict, so a mismatch that is not yet confirmed
        still fails the factor and steps the caller up. The detail says in
        words that this was the two legs of the call being compared, and
        whether the mismatch was confirmed.
        """

        if self.confirmed:
            detail = (
                f"{MISMATCH_HEADLINE}: caller and callback legs differ. The person "
                "who rang in is not the voice that answered the callback to the "
                f"number on file. {self.detail} ({self.scope})"
            )
        elif self.differ:
            detail = (
                "Cross-leg check: caller and callback legs differ, but this is not "
                "confident enough to refuse the call outright: a deny needs a "
                "confident comparison and at least "
                f"{MIN_CONFIRMED_LEG_SPEECH_SECONDS:.1f} s of speech on each leg. "
                f"A step-up is required. {self.detail} ({self.scope})"
            )
        elif self.verdict is FactorVerdict.PASS:
            detail = (
                "Cross-leg check: caller and callback legs are consistent, the "
                "voice that rang in matches the voice that answered the callback. "
                f"{self.detail} ({self.scope})"
            )
        else:
            detail = (
                "Cross-leg check inconclusive: caller and callback legs could not "
                f"be told apart or together with confidence. {self.detail} "
                f"({self.scope})"
            )

        return Factor(
            name=FactorName.VOICE_MATCH,
            score=self.score,
            weight=DEFAULT_WEIGHTS[FactorName.VOICE_MATCH],
            verdict=self.verdict,
            detail=detail[:500],
        )

    @property
    def settled(self) -> bool:
        """The comparison is evidence: the legs match, or they are confirmed
        not to. Anything else is a measurement still waiting for speech."""

        return self.verdict is FactorVerdict.PASS or self.confirmed

    def merge_factor(self) -> Factor | None:
        """The factor to merge into the span's list, or None.

        Only a settled comparison goes on the wire. An inconclusive result,
        or a FAIL on legs too thin to confirm, is not evidence against the
        caller: the same talker four semitones up lands there. Merged, it
        reached `/v1` as a voiceprint under its 0.45 fail line and counted
        as a voice fail, so it stays on the session instead.
        """

        return self.factor() if self.settled else None

    def fatal_factor(self) -> Factor | None:
        """The factor to hand `scoring.fatal_factor`, or None if not confirmed."""

        return self.factor() if self.confirmed else None


def compare_legs(legs: dict[str, LegPrint]) -> CrossLegCheck | None:
    """Compare the caller leg against the callback leg, if both are ready.

    None when either leg is missing or still under `MIN_LEG_SPEECH_SECONDS`,
    which leaves the `voice_match` factor to the enrolment comparison and the
    session's `cross_leg` as it was.
    """

    caller = legs.get(CALLER)
    callback = legs.get(CALLBACK)
    if caller is None or callback is None:
        return None
    if not caller.enough or not callback.enough:
        return None

    result = compare_prints(caller.voice, callback.voice)
    # `verdict_for` in detector.analysis.cues: FAIL at or below
    # FAIL_AT_OR_BELOW, PASS at or above PASS_AT_OR_ABOVE, INCONCLUSIVE in
    # between or whenever the comparison was not confident. The confidence
    # flag is the comparator's own, recorded in its measurements.
    verdict = result.verdict
    if verdict not in (FactorVerdict.PASS, FactorVerdict.FAIL):
        verdict = FactorVerdict.INCONCLUSIVE
    return CrossLegCheck(
        score=max(0.0, min(1.0, float(result.score))),
        verdict=verdict,
        confident=result.measurements.get("confident", 0.0) >= 1.0,
        detail=result.detail,
        caller_seconds=caller.speech_seconds,
        caller_spans=caller.spans,
        callback_seconds=callback.speech_seconds,
        callback_spans=callback.spans,
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


def cross_leg_to_json(check: CrossLegCheck | None) -> dict[str, Any] | None:
    if check is None:
        return None
    return {
        "score": check.score,
        "verdict": check.verdict.value,
        "confident": check.confident,
        "detail": check.detail,
        "caller_seconds": check.caller_seconds,
        "caller_spans": check.caller_spans,
        "callback_seconds": check.callback_seconds,
        "callback_spans": check.callback_spans,
    }


def cross_leg_from_json(doc: dict[str, Any] | None) -> CrossLegCheck | None:
    """Rebuild the check, or None for a document without one. Raises on a
    malformed document, as `legs_from_json` does."""

    if not doc:
        return None
    return CrossLegCheck(
        score=float(doc["score"]),
        verdict=FactorVerdict(doc["verdict"]),
        confident=bool(doc["confident"]),
        detail=str(doc["detail"]),
        caller_seconds=float(doc["caller_seconds"]),
        caller_spans=int(doc["caller_spans"]),
        callback_seconds=float(doc["callback_seconds"]),
        callback_spans=int(doc["callback_spans"]),
    )
