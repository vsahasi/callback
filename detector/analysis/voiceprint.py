"""What the voice comparison needs from an enrolled recording, and nothing more.

`analyse_voice_match` compares a caller against a reference. The obvious way
to hold that reference is to keep the enrolment recording and rebuild it on
every comparison. This module exists so that the service does not have to.

Three reasons, in the order they matter.

Privacy
-------
An enrolment recording is a vendor's voice, held indefinitely, in a service
that sits in a payments flow. That is biometric data, and keeping a corpus of
it is a liability that grows with every vendor onboarded. What is kept here
instead is a median pitch, two durations and a long-term average log
spectrum: a few hundred numbers, averaged over the whole recording, with no
phase and no temporal structure. Nothing that was said can be recovered from
it, because everything that carries what was said has been averaged away.
That is not a side effect of compressing the reference, it is the reason for
the shape.

Cost
----
`AnalysisEngine.analyse_report` rebuilds the reference context on every call
to it, which means on every speech segment of every call. The reference never
changes. Extracting once at enrolment moves that work out of the 800 ms
budget the agent gives the whole decision.

Size
----
Thirty seconds at 8 kHz is a quarter of a megabyte of samples. A print is
around 1.6 KB, which matters because the store behind it is a dictionary with
a bound on it, and in the deployed configuration a `modal.Dict` with a bound
on each value.

The frequency grid, and why it travels with the print
-----------------------------------------------------
The spectral half of the comparison correlates two long-term spectra bin by
bin, which is only meaningful if a bin means the same frequency on both
sides. `FrameGrid.n_fft` doubles the window, which lands the bin spacing at
15.625 Hz at 8 kHz and again at 16 kHz, so a print taken from an 8 kHz
enrolment compares correctly against a 16 kHz call. That is a property of the
grid, not a coincidence worth relying on silently: a print carries the
spacing and the first bin's frequency it was built on, and a comparison
across two grids that do not line up declines to produce a number rather than
correlating the wrong frequencies against each other.

A print built from unusable audio is still a print. It records that the
enrolment was unusable and why, which is what lets the enrolment endpoint
refuse it with a reason instead of storing something that would make every
later comparison inconclusive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from detector.analysis.dsp import EPS, AudioContext

__all__ = [
    "LTAS_BAND_HI",
    "LTAS_BAND_LO",
    "LTAS_MIN_BINS",
    "VoicePrint",
    "voiceprint_from",
    "voiceprint_from_json",
    "voiceprint_to_json",
]

# The band the long-term spectrum is taken over. Inside the 3.4 kHz telephone
# passband, so the same bins exist whether the call arrived at 8 kHz or 16.
LTAS_BAND_LO = 300.0
LTAS_BAND_HI = 3400.0

# Below this there is not enough of a spectrum to correlate against anything.
LTAS_MIN_BINS = 16

# Two bin spacings are the same grid if they agree to this, in Hz. Generous
# enough to absorb the rounding in FrameGrid.win, tight enough that an
# 11.025 kHz print (10.8 Hz bins) is not mistaken for an 8 kHz one (15.6 Hz).
GRID_TOLERANCE_HZ = 0.05


@dataclass(frozen=True)
class VoicePrint:
    """A reference voice, reduced to what the comparison actually reads.

    Deliberately not a speaker embedding. `detector.analysis.voice` is explicit
    that it is not doing speaker verification, and this is the matching
    reference: a pitch register and a spectral envelope, enough to notice a
    different talker and not enough to identify one.
    """

    #: Median fundamental frequency over voiced frames, Hz. 0.0 when no voiced
    #: speech was found, which makes the pitch comparison unmeasurable rather
    #: than wrong.
    median_f0_hz: float
    voiced_seconds: float
    speech_seconds: float
    duration_s: float
    #: The rate the analysis ran at, after any decimation. Provenance for a
    #: reviewer, not an input to the comparison.
    sample_rate: int
    #: Spacing between adjacent `ltas_db` bins, Hz.
    bin_hz: float
    #: Frequency of `ltas_db[0]`, Hz.
    lo_hz: float
    #: Mean log power spectrum over speech frames, in dB, shifted to zero mean
    #: so that overall level drops out and only shape remains. Empty when the
    #: audio had too little bandwidth or no spectral variation to compare.
    ltas_db: tuple[float, ...]
    #: Whether the source audio was long enough and high-rate enough to
    #: analyse at all. From `AudioBlock.usable`.
    usable: bool
    #: The audio-conditioning caveats from the source context, verbatim.
    notes: tuple[str, ...] = ()

    @property
    def has_pitch(self) -> bool:
        return self.median_f0_hz > 0.0

    @property
    def has_spectrum(self) -> bool:
        return len(self.ltas_db) >= LTAS_MIN_BINS

    @property
    def comparable(self) -> bool:
        """Whether this print can support any comparison at all.

        Either half is enough. A reference with pitch but no usable spectrum
        still catches a talker in a different register, and the spectral cue
        reports itself unmeasurable.
        """

        return self.usable and (self.has_pitch or self.has_spectrum)

    def shares_grid_with(self, other: VoicePrint) -> bool:
        """Whether the two spectra are on the same frequency grid.

        Same spacing, and the same first bin to within half a bin. Both hold
        by construction for 8 kHz and 16 kHz audio; a print from some other
        rate will not line up and says so here instead of being correlated
        against the wrong frequencies.
        """

        if abs(self.bin_hz - other.bin_hz) > GRID_TOLERANCE_HZ:
            return False
        if self.bin_hz <= 0.0:
            return False
        return abs(self.lo_hz - other.lo_hz) <= self.bin_hz / 2.0

    def spectrum(self) -> np.ndarray:
        return np.asarray(self.ltas_db, dtype=np.float64)


def voiceprint_from(ctx: AudioContext) -> VoicePrint:
    """Reduce an analysed recording to its reference print.

    Never raises. Degenerate audio produces a print that reports itself
    uncomparable, which is the honest answer and the one the enrolment
    endpoint turns into a refusal.
    """

    freqs = np.asarray(ctx.spec.freqs, dtype=np.float64)
    bin_hz = float(freqs[1] - freqs[0]) if freqs.size >= 2 else 0.0

    ltas, lo_hz = _ltas(ctx, freqs)

    return VoicePrint(
        median_f0_hz=float(ctx.pitch.median_f0),
        voiced_seconds=float(ctx.pitch.voiced_seconds),
        speech_seconds=float(ctx.activity.speech_seconds),
        duration_s=float(ctx.block.duration_s),
        sample_rate=int(ctx.block.sample_rate),
        bin_hz=bin_hz,
        lo_hz=lo_hz,
        ltas_db=tuple(float(value) for value in ltas),
        usable=bool(ctx.block.usable),
        notes=tuple(ctx.notes),
    )


def _ltas(ctx: AudioContext, freqs: np.ndarray) -> tuple[np.ndarray, float]:
    """Zero-mean mean log power spectrum over speech frames, in dB.

    Averaging the log spectrum over frames is what separates a fixed channel
    response from voiced harmonics: the harmonic comb moves with F0 across the
    utterance and smears out, while a room or a handset response is the same
    in every frame and survives. See `AudioContext.mean_log_spectrum`.
    """

    band = (freqs >= LTAS_BAND_LO) & (freqs <= min(LTAS_BAND_HI, ctx.nyquist))
    if int(np.count_nonzero(band)) < LTAS_MIN_BINS:
        return np.zeros(0), 0.0

    spectrum = ctx.mean_log_spectrum(ctx.activity.speech)[band] * 10.0
    if float(np.std(spectrum)) <= EPS:
        # A tone, silence or stationary noise. Correlating a flat line against
        # anything produces a meaningless number, so produce nothing.
        return np.zeros(0), 0.0

    centred = spectrum - float(np.mean(spectrum))
    return centred, float(freqs[band][0])


# ------------------------------------------------------------ serialisation
#
# Lossless, so a print that round-trips through the store compares identically
# to one that did not. Floats go out as floats: the store's JSON codec is the
# same one the session store uses and it does not quantise.


def voiceprint_to_json(print_: VoicePrint) -> dict[str, Any]:
    return {
        "median_f0_hz": print_.median_f0_hz,
        "voiced_seconds": print_.voiced_seconds,
        "speech_seconds": print_.speech_seconds,
        "duration_s": print_.duration_s,
        "sample_rate": print_.sample_rate,
        "bin_hz": print_.bin_hz,
        "lo_hz": print_.lo_hz,
        "ltas_db": list(print_.ltas_db),
        "usable": print_.usable,
        "notes": list(print_.notes),
    }


def voiceprint_from_json(doc: dict[str, Any]) -> VoicePrint:
    """Rebuild a print. Raises `KeyError` or `TypeError` on a malformed doc.

    The caller turns that into a store format error, the same way the session
    codec does, so a value written by an older version of this module fails
    loudly instead of comparing against partially-default numbers.
    """

    return VoicePrint(
        median_f0_hz=float(doc["median_f0_hz"]),
        voiced_seconds=float(doc["voiced_seconds"]),
        speech_seconds=float(doc["speech_seconds"]),
        duration_s=float(doc["duration_s"]),
        sample_rate=int(doc["sample_rate"]),
        bin_hz=float(doc["bin_hz"]),
        lo_hz=float(doc["lo_hz"]),
        ltas_db=tuple(float(value) for value in doc["ltas_db"]),
        usable=bool(doc["usable"]),
        notes=tuple(str(note) for note in doc.get("notes", ())),
    )
