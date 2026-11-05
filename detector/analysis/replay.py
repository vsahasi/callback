"""Replay detection: was this audio played out of a loudspeaker into a phone?

This is the strongest classical signal in the package, because a replay attack
necessarily adds a second acoustic path that live speech does not have. The
attacker's recording already contains one room and one microphone; playing it
back adds a loudspeaker, a second room and a second microphone. That leaves
four kinds of trace, and three of them survive an 8 kHz telephone codec:

1. A fixed spectral comb. Any early reflection at delay tau multiplies the
   spectrum by a ripple of period 1/tau Hz. Averaged over an utterance the
   talker's harmonics smear (F0 moves) while the room's ripple does not, so
   it stands out in the cepstrum at quefrency tau.
2. A reverberant tail. Energy keeps arriving after the talker stops, so the
   gaps between words fill with speech-shaped energy and the decay at each
   offset is bounded by the room's RT60 rather than by the talker.
3. Playback-chain band-limiting. A small loudspeaker cannot reproduce the
   low end, and a second codec pass narrows the top.
4. Requantisation. Decoding and re-encoding can leave the samples on a
   coarser lattice than the capture path would.

Honest limits, which the `detail` string also carries:

* Cues 3 and 4 are weak on telephone audio. The network already high-passes
  at ~300 Hz and low-passes at ~3.4 kHz, so "band-limited" describes every
  genuine call too, and any lossy codec in the path erases the sample
  lattice. They are weighted low and used only as corroboration.
* Cues 1 and 2 are confounded by a genuine caller on speakerphone in a hard
  room. A hands-free vendor in a warehouse will look like a replay here. That
  is a false accusation this code cannot rule out from audio alone, which is
  why a low score should hold a payment for review rather than silently deny
  it.
* `detector.scoring` treats replay as a fatal factor at or below 0.10. This
  module therefore floors its score at 0.12 unless at least two of the three
  strong cues agree, so no single noisy measurement can deny a payment.
"""

from __future__ import annotations

import numpy as np

from detector.analysis.cues import (
    Cue,
    DetectorResult,
    blend,
    inconclusive,
    ramp,
    summarise,
    temper,
    verdict_for,
)
from detector.analysis.dsp import (
    EPS,
    AudioContext,
    band_energy,
    interior_gap_mask,
    linear_slope,
    pearson,
    quantisation_step,
    real_cepstrum,
    rolloff_hz,
    true_runs,
)

__all__ = ["analyse_replay"]

# Early reflections from a phone held near a laptop or desk speaker land
# between about 1 and 5 ms. Below 1 ms the cepstrum is the vocal tract; above
# 5 ms it collides with the glottal period for most adult voices.
COMB_QUEFRENCY_MIN_S = 0.0012
COMB_QUEFRENCY_MAX_S = 0.0045
COMB_REFERENCE_MAX_S = 0.025

# Score floors, against the fatal-factor rule in detector.scoring, which
# denies outright at or below 0.10.
#
# The strong cues are four independent measurements of one claim: there is a
# playback chain between the talker and the microphone. A fixed comb, a
# speech-shaped tail, a room-limited decay and a coarse sample lattice have
# nothing in common except that cause. One firing alone is not allowed to
# deny a payment, two may deny but not reach the fatal band on the floor
# alone, and three agreeing is treated as decisive.
#
# The residual false-positive risk is a genuine caller on speakerphone in a
# hard room, who can produce the three acoustic ones. That is stated in the
# `detail` string so a reviewer sees it rather than only seeing a denial.
FLOOR_BY_AGREEMENT = {0: 0.25, 1: 0.25, 2: 0.12}
CORROBORATED_FLOOR = 0.02
STRONG_CUES = (
    "fixed_comb",
    "speech_shaped_tail",
    "offset_decay",
    "requantisation",
)
STRONG_CUE_PENALTY = 0.50

# Independent cues compound; see cues.blend.
SHARPNESS = 4.0

# The fixed drop each speech offset is measured over. Every offset uses the
# same span so the slope is a property of the channel rather than of the
# signal-to-noise ratio.
DECAY_DROP_DB = 15.0

# Above this fraction of samples sitting on the peak, hard clipping is
# generating its own harmonic distortion, which fakes a spectral comb.
CLIPPING_LIMIT = 0.02


def analyse_replay(ctx: AudioContext) -> DetectorResult:
    """Score how consistent the recording is with a live handset (1.0) or a
    loudspeaker replay (0.0)."""

    block = ctx.block
    if not block.usable:
        reason = "; ".join(ctx.notes) or "no usable audio"
        return inconclusive(f"Replay analysis not possible: {reason}.")

    cues: list[Cue] = []
    measurements: dict[str, float] = {}

    comb_cue, comb_measures = _comb_cue(ctx)
    cues.append(comb_cue)
    measurements.update(comb_measures)

    tail_cue, tail_measures = _tail_cue(ctx)
    cues.append(tail_cue)
    measurements.update(tail_measures)

    decay_cue, decay_measures = _decay_cue(ctx)
    cues.append(decay_cue)
    measurements.update(decay_measures)

    band_cues, band_measures = _band_cues(ctx)
    cues.extend(band_cues)
    measurements.update(band_measures)

    lattice_cue, lattice_measures = _lattice_cue(ctx)
    cues.append(lattice_cue)
    measurements.update(lattice_measures)

    strong_agreement = sum(
        1
        for cue in cues
        if cue.name in STRONG_CUES
        and cue.measurable
        and cue.penalty >= STRONG_CUE_PENALTY
    )
    floor = FLOOR_BY_AGREEMENT.get(strong_agreement, CORROBORATED_FLOOR)
    raw_score = blend(cues, floor=floor, sharpness=SHARPNESS)

    measurable = sum(1 for cue in cues if cue.measurable)
    strong_measurable = sum(
        1 for cue in cues if cue.name in STRONG_CUES and cue.measurable
    )
    confident = (
        ctx.activity.has_structure
        and ctx.activity.speech_seconds >= 0.8
        and strong_measurable >= 2
        and measurable >= 3
        and block.sample_rate >= 8000
    )
    score = temper(raw_score, confident=confident)

    measurements["strong_cues_agreeing"] = float(strong_agreement)
    measurements["cues_measurable"] = float(measurable)
    measurements["score_floor"] = floor
    measurements["score_before_tempering"] = raw_score

    detail = summarise("Replay channel:", cues)
    if not confident:
        detail = f"{detail} Low confidence: {_confidence_note(ctx, strong_measurable)}"
    elif strong_agreement >= 3:
        detail = (
            f"{detail} {strong_agreement} independent replay cues agree. A genuine "
            "caller on speakerphone in a hard room can also produce the acoustic ones."
        )

    return DetectorResult(
        score=score,
        verdict=verdict_for(score, confident=confident),
        detail=detail[:500],
        measurements=measurements,
        cues=tuple(cues),
    )


def _confidence_note(ctx: AudioContext, strong_measurable: int) -> str:
    reasons: list[str] = []
    if not ctx.activity.has_structure:
        reasons.append("no speech/silence structure to measure a tail against")
    if ctx.activity.speech_seconds < 0.8:
        reasons.append(f"only {ctx.activity.speech_seconds:.2f}s of speech")
    if strong_measurable < 2:
        reasons.append("fewer than two of the three strong cues were measurable")
    if ctx.block.sample_rate < 8000:
        reasons.append(f"{ctx.block.sample_rate} Hz is below telephone bandwidth")
    return (", ".join(reasons) or "insufficient evidence") + "."


# ---------------------------------------------------------------------------
# Cue 1: a fixed spectral comb from an early reflection
# ---------------------------------------------------------------------------


def _comb_cue(ctx: AudioContext) -> tuple[Cue, dict[str, float]]:
    """Cepstral peak in the early-reflection quefrency band.

    Measured on the log power spectrum averaged over speech frames, which is
    what separates a fixed channel from the talker: harmonics move with F0
    and average away, a reflection does not.
    """

    spec = ctx.spec
    rate = spec.sample_rate
    if ctx.block.clipped_fraction > CLIPPING_LIMIT:
        note = (
            f"{ctx.block.clipped_fraction:.0%} of samples are clipped; the resulting "
            "harmonic distortion is indistinguishable from a spectral comb"
        )
        return _unmeasurable("fixed_comb", 0.34, note), {}
    log_spectrum = ctx.mean_log_spectrum(ctx.activity.speech)
    if log_spectrum.size < 16:
        return _unmeasurable(
            "fixed_comb", 0.34, "spectrum too short for a cepstrum"
        ), {}

    cepstrum = np.abs(real_cepstrum(log_spectrum))
    half = cepstrum.size // 2
    if half < 8:
        return _unmeasurable("fixed_comb", 0.34, "cepstrum too short"), {}
    cepstrum = cepstrum[:half]
    quefrency = np.arange(half, dtype=np.float64) / rate

    target = (quefrency >= COMB_QUEFRENCY_MIN_S) & (quefrency <= COMB_QUEFRENCY_MAX_S)
    reference = (quefrency >= COMB_QUEFRENCY_MIN_S) & (
        quefrency <= COMB_REFERENCE_MAX_S
    )

    # The glottal period and its first multiple also produce cepstral peaks.
    # Exclude them, or a high-pitched voice is scored as a room reflection.
    period = ctx.pitch.median_period_s
    if period > 0.0:
        for multiple in (1.0, 2.0, 3.0):
            centre = period * multiple
            keep = np.abs(quefrency - centre) > 0.25 * centre
            target &= keep
            reference &= keep

    if int(np.count_nonzero(target)) < 6 or int(np.count_nonzero(reference)) < 24:
        note = "early-reflection band collides with the glottal period"
        return _unmeasurable("fixed_comb", 0.34, note), {"comb_peak_ratio": 0.0}

    peak_ratio, peak_index = _local_prominence(cepstrum, target, reference)
    if peak_ratio <= 0.0:
        return _unmeasurable("fixed_comb", 0.34, "flat cepstrum"), {
            "comb_peak_ratio": 0.0
        }
    delay_ms = float(quefrency[peak_index] * 1000.0)

    penalty = ramp(peak_ratio, 4.6, 9.5)
    if penalty >= 0.25:
        note = (
            f"fixed {delay_ms:.1f} ms spectral comb, {peak_ratio:.1f}x the cepstral "
            "baseline (a reflection in the playback path)"
        )
    else:
        note = f"no fixed spectral comb (peak {peak_ratio:.1f}x its neighbourhood)"

    return (
        Cue(
            name="fixed_comb",
            value=peak_ratio,
            penalty=penalty,
            weight=0.34,
            note=note,
        ),
        {"comb_peak_ratio": peak_ratio, "comb_delay_ms": delay_ms},
    )


def _local_prominence(
    cepstrum: np.ndarray,
    target: np.ndarray,
    reference: np.ndarray,
    half_window: int = 20,
) -> tuple[float, int]:
    """Largest ratio of a target bin to the median of its neighbourhood.

    A *local* baseline, not a global one, because the cepstrum of speech
    decays steeply with quefrency: the spectral envelope dominates the first
    millisecond or two and the curve falls by an order of magnitude by
    10 ms. Dividing by a global median therefore measures how early in the
    band the peak sits, which is why a high-pitched talker with widely
    spaced harmonics scored as a room reflection before this existed.

    Comparing each bin against roughly 2.5 ms either side of itself tracks
    the decay out and leaves only genuine prominence: a fixed reflection
    stands above its neighbours, a smooth envelope does not.
    """

    size = cepstrum.size
    best_ratio = 0.0
    best_index = 0
    for index in np.flatnonzero(target):
        low = max(int(index) - half_window, 0)
        high = min(int(index) + half_window + 1, size)
        neighbourhood = np.zeros(size, dtype=bool)
        neighbourhood[low:high] = True
        # Exclude the peak and its immediate skirt, keep only the quefrency
        # range this cue is allowed to look at.
        neighbourhood[max(int(index) - 2, 0) : int(index) + 3] = False
        neighbourhood &= reference
        if int(np.count_nonzero(neighbourhood)) < 8:
            continue
        baseline = float(np.median(cepstrum[neighbourhood]))
        if baseline <= EPS:
            continue
        ratio = float(cepstrum[index] / baseline)
        if ratio > best_ratio:
            best_ratio = ratio
            best_index = int(index)
    return best_ratio, best_index


# ---------------------------------------------------------------------------
# Cue 2: a speech-shaped tail filling the gaps between words
# ---------------------------------------------------------------------------


def _tail_cue(ctx: AudioContext) -> tuple[Cue, dict[str, float]]:
    """Level and spectral shape of the inter-word gaps.

    Two measurements, deliberately combined into one cue: how far the gaps
    sit below the speech, and how closely the gap spectrum resembles the
    speech spectrum. A loud gap on its own is just a noisy line. A loud gap
    whose spectrum matches the talker is a reverberant tail.
    """

    activity = ctx.activity
    speech = np.asarray(activity.speech, dtype=bool)
    if speech.size == 0:
        return _unmeasurable("speech_shaped_tail", 0.30, "no frames"), {}

    gap = interior_gap_mask(speech, erode=2)
    if int(np.count_nonzero(gap)) < 4:
        note = "no inter-word gap long enough to look for a reverberant tail"
        return _unmeasurable("speech_shaped_tail", 0.30, note), {}

    gap_db = float(np.median(activity.energy_db[gap]))
    gap_minus_speech = gap_db - activity.speech_db

    speech_spectrum = ctx.mean_log_spectrum(speech)
    gap_spectrum = ctx.mean_log_spectrum(gap)
    band = (ctx.spec.freqs >= 300.0) & (ctx.spec.freqs <= min(3400.0, ctx.nyquist))
    if int(np.count_nonzero(band)) >= 8:
        shape_match = pearson(speech_spectrum[band], gap_spectrum[band])
    else:
        shape_match = 0.0

    level_penalty = ramp(gap_minus_speech, -34.0, -14.0)
    shape_gate = ramp(shape_match, 0.25, 0.65)
    penalty = level_penalty * shape_gate

    if penalty >= 0.25:
        note = (
            f"gaps between words only {abs(gap_minus_speech):.0f} dB below speech and "
            f"spectrally matched to it (r={shape_match:.2f}): a reverberant tail"
        )
    else:
        note = (
            f"gaps {abs(gap_minus_speech):.0f} dB below speech, tail shape r="
            f"{shape_match:.2f}"
        )

    return (
        Cue(
            name="speech_shaped_tail",
            value=gap_minus_speech,
            penalty=penalty,
            weight=0.30,
            note=note,
        ),
        {
            "gap_minus_speech_db": gap_minus_speech,
            "tail_shape_correlation": shape_match,
        },
    )


# ---------------------------------------------------------------------------
# Cue 3: how fast the energy falls at a speech offset
# ---------------------------------------------------------------------------


def _decay_cue(ctx: AudioContext) -> tuple[Cue, dict[str, float]]:
    """Median energy decay slope at speech offsets, in dB per second.

    A talker into a handset stops abruptly: the measured slope is limited
    only by the analysis window, so it is steep. A room enforces a ceiling of
    60 / RT60 dB per second, so a replayed recording decays visibly slower.

    Every offset is measured over the same fixed drop, `DECAY_DROP_DB`, and
    the cue refuses to run unless the recording has that much range below
    the speech level. Without that rule the measurement is really an SNR
    measurement: on a noisy line the energy cannot fall far before it hits
    the floor, the fitted slope flattens, and a perfectly live caller on a
    hissy connection reads as a reverberant room. That false accusation
    showed up the first time this was tested against added line noise.
    """

    activity = ctx.activity
    hop = max(activity.hop_s, 1e-6)
    energy = np.asarray(activity.energy_db, dtype=np.float64)
    speech = np.asarray(activity.speech, dtype=bool)
    if energy.size < 6:
        return _unmeasurable("offset_decay", 0.18, "too few frames"), {}

    range_db = activity.speech_db - activity.floor_db
    if range_db < DECAY_DROP_DB + 2.0:
        note = (
            f"only {range_db:.0f} dB between speech and the noise floor: not enough "
            f"range to measure a {DECAY_DROP_DB:.0f} dB decay"
        )
        return _unmeasurable("offset_decay", 0.18, note), {
            "speech_to_floor_db": range_db
        }

    slopes: list[float] = []
    for _, stop in true_runs(speech):
        if stop >= energy.size - 1:
            continue
        start = stop - 1
        target_db = energy[start] - DECAY_DROP_DB
        end = stop
        while end < energy.size and energy[end] > target_db and (end - start) < 30:
            end += 1
        if end >= energy.size:
            continue  # the recording ended before the decay did
        end = min(end + 1, energy.size)
        if (end - start) < 2:
            continue
        times = np.arange(end - start, dtype=np.float64) * hop
        slopes.append(linear_slope(times, energy[start:end]))

    if not slopes:
        note = "no clean speech offset to measure a decay against"
        return _unmeasurable("offset_decay", 0.18, note), {
            "speech_to_floor_db": range_db
        }

    slope = float(np.median(slopes))
    penalty = ramp(slope, -700.0, -150.0)
    if penalty >= 0.25:
        rt60 = 60.0 / max(abs(slope), 1.0)
        note = (
            f"speech offsets decay at only {abs(slope):.0f} dB/s "
            f"(RT60 ~ {rt60 * 1000:.0f} ms), consistent with a room in the path"
        )
    else:
        note = f"speech offsets decay at {abs(slope):.0f} dB/s, handset-like"

    return (
        Cue(
            name="offset_decay",
            value=slope,
            penalty=penalty,
            weight=0.18,
            note=note,
        ),
        {"offset_decay_db_per_s": slope, "offsets_measured": float(len(slopes))},
    )


# ---------------------------------------------------------------------------
# Cues 4 and 5: playback-chain band-limiting
# ---------------------------------------------------------------------------


def _band_cues(ctx: AudioContext) -> tuple[list[Cue], dict[str, float]]:
    """Band energy at the top of the channel, and in-band tilt at the bottom.

    Both are weak on telephone audio, for the reason in the module
    docstring, and are weighted accordingly. They corroborate; they never
    accuse on their own.

    The top-of-band measure is a ratio between two bands inside the
    telephone passband rather than a roll-off frequency. A roll-off
    percentile mostly reports the talker's own spectral tilt: 95% of the
    energy of ordinary speech sits below about 1.1 kHz whatever the channel
    does, so that statistic cannot see the channel at all.
    """

    freqs = ctx.spec.freqs
    mean_power = _mean_speech_power(ctx)
    if mean_power is None:
        note = "no speech frames to measure a channel response"
        return (
            [
                _unmeasurable("playback_highband", 0.10, note),
                _unmeasurable("playback_tilt", 0.08, note),
            ],
            {},
        )

    top = min(3400.0, ctx.nyquist)
    high = band_energy(freqs, mean_power, 2200.0, top)
    mid = band_energy(freqs, mean_power, 700.0, 2200.0)
    high_db = 10.0 * np.log10(max(high, EPS) / max(mid, EPS))

    low = band_energy(freqs, mean_power, 300.0, 700.0)
    tilt_db = 10.0 * np.log10(max(low, EPS) / max(mid, EPS))

    high_measurable = ctx.nyquist >= 3000.0
    high_penalty = ramp(float(high_db), -24.0, -34.0) if high_measurable else 0.0
    if high_penalty >= 0.25:
        high_note = (
            f"2.2-3.4 kHz band {abs(float(high_db)):.0f} dB below the mid band: the top "
            "of the channel is attenuated beyond what a phone line does"
        )
    elif high_measurable:
        high_note = f"top of the channel {abs(float(high_db)):.0f} dB down, phone-like"
    else:
        high_note = f"Nyquist is only {ctx.nyquist / 1000:.1f} kHz"

    tilt_penalty = ramp(float(tilt_db), -2.0, -14.0)
    if tilt_penalty >= 0.25:
        tilt_note = (
            f"300-700 Hz band {abs(float(tilt_db)):.0f} dB below 700-2200 Hz, the tilt a "
            "small loudspeaker leaves"
        )
    else:
        tilt_note = f"in-band tilt {float(tilt_db):+.0f} dB, unremarkable"

    cues = [
        # Both are guilt-only. A full band and a flat tilt are what every
        # genuine call has AND what a good playback chain has, so their
        # absence is no evidence of a live handset and must not dilute the
        # room cues. On the synthetic corpus the high-band ratio does not
        # separate replay from live at all (-21 dB either way), which is
        # exactly the case this flag is for.
        Cue(
            name="playback_highband",
            value=float(high_db),
            penalty=high_penalty,
            weight=0.10,
            note=high_note,
            measurable=high_measurable,
            one_sided=True,
        ),
        Cue(
            name="playback_tilt",
            value=float(tilt_db),
            penalty=tilt_penalty,
            weight=0.08,
            note=tilt_note,
            one_sided=True,
        ),
    ]
    return cues, {
        "high_band_db": float(high_db),
        "in_band_tilt_db": float(tilt_db),
        "rolloff_hz": rolloff_hz(freqs, mean_power, fraction=0.95),
    }


def _mean_speech_power(ctx: AudioContext) -> np.ndarray | None:
    power = ctx.spec.power
    if power.size == 0:
        return None
    speech = np.asarray(ctx.activity.speech, dtype=bool)
    selected = power[speech] if np.any(speech) else power
    return np.mean(selected, axis=0)


# ---------------------------------------------------------------------------
# Cue 6: a coarse sample lattice left by requantisation
# ---------------------------------------------------------------------------


def _lattice_cue(ctx: AudioContext) -> tuple[Cue, dict[str, float]]:
    """Effective bit depth of the sample lattice.

    Only meaningful on uncompressed PCM. When the samples do not sit on a
    lattice at all the cue is unmeasurable, not innocent: a lossy codec
    anywhere in the path produces exactly that, and it is no evidence either
    way.
    """

    relative, on_lattice, bits = quantisation_step(ctx.block.raw_samples)
    measures = {
        "lattice_relative_step": relative,
        "lattice_fraction": on_lattice,
        "effective_bits": bits,
    }
    if on_lattice < 0.995 or not 4.0 <= bits <= 20.0:
        note = "samples are not on a detectable lattice (lossy codec in the path)"
        return _unmeasurable("requantisation", 0.10, note), measures

    penalty = ramp(bits, 14.0, 9.0)
    if penalty >= 0.25:
        note = (
            f"samples quantised to about {bits:.0f} effective bits, coarser than a "
            "capture path should be"
        )
    else:
        note = f"sample lattice is {bits:.0f}-bit, as captured"

    return (
        Cue(
            name="requantisation",
            value=bits,
            penalty=penalty,
            weight=0.10,
            note=note,
            # Guilt-only: a 16-bit lattice is what a single clean capture
            # leaves and also what a replay re-recorded at 16 bits leaves.
            one_sided=True,
        ),
        measures,
    )


def _unmeasurable(name: str, weight: float, note: str) -> Cue:
    return Cue(
        name=name, value=0.0, penalty=0.0, weight=weight, note=note, measurable=False
    )
