"""Draw the README banner from the audio of a real call and a real attack.

    uv run python showcase/hero.py

Writes docs/assets/hero-light.svg and docs/assets/hero-dark.svg.

The banner is not decoration. It is two envelopes drawn from the same voice:
the live caller on top, and underneath the same recording played back down a
phone line. The difference is what the replay detector keys on and it is
visible without reading a number. Between the live caller's words the signal
drops to the noise floor. On the replay it does not, because the room the
recording was played into is still ringing after each word.

Text uses a plain font stack rather than a web font, because GitHub renders a
README image in an `img` tag where no external font loads.
"""

from __future__ import annotations

import pathlib

import numpy as np

from detector.analysis.synthetic import live_speech, replay_of

HERE = pathlib.Path(__file__).resolve().parent
ASSETS = HERE.parent / "docs" / "assets"

RATE = 8000
SECONDS = 6.0
WIDTH = 1280
HEIGHT = 320

PANEL_X = 448
PANEL_W = WIDTH - PANEL_X - 56
ROW_H = 58
TOP_MID = 128
BOTTOM_MID = 236

FONTS = "Helvetica Neue, Helvetica, Arial, sans-serif"
MONO = "SFMono-Regular, Menlo, Consolas, monospace"

THEMES = {
    "light": {
        "bg": "#e9e7e1",
        "sheet": "#f6f5f1",
        "ink": "#191714",
        "soft": "#6a665e",
        "rule": "#cbc7bd",
        "red": "#a8322a",
        "green": "#3d6b45",
    },
    "dark": {
        "bg": "#121316",
        "sheet": "#191b1f",
        "ink": "#ecebe7",
        "soft": "#8d8980",
        "rule": "#32353b",
        "red": "#e0655c",
        "green": "#5cc08c",
    },
}


def envelope(samples: np.ndarray, columns: int) -> list[tuple[float, float]]:
    """Min and max per column, the shape a scope would draw."""

    step = len(samples) / columns
    out: list[tuple[float, float]] = []
    for x in range(columns):
        start = int(x * step)
        end = min(int((x + 1) * step), len(samples))
        window = samples[start:end]
        out.append(
            (0.0, 0.0)
            if window.size == 0
            else (float(window.min()), float(window.max()))
        )
    return out


def trace(samples: np.ndarray, mid: int, gain: float, colour: str) -> str:
    bars = []
    for x, (lo, hi) in enumerate(envelope(samples, PANEL_W)):
        top = mid - hi * gain
        bars.append(
            f'<rect x="{PANEL_X + x}" y="{top:.1f}" width="1" '
            f'height="{max(0.8, (hi - lo) * gain):.1f}"/>'
        )
    return f'<g fill="{colour}">{"".join(bars)}</g>'


def render(theme: str) -> str:
    c = THEMES[theme]
    live = live_speech(SECONDS, RATE, seed=1)
    played_back = replay_of(live, sample_rate=RATE)

    # One gain for both rows, so the comparison is honest rather than flattering.
    gain = ROW_H / max(
        float(np.max(np.abs(live))), float(np.max(np.abs(played_back))), 1e-6
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" \
width="{WIDTH}" height="{HEIGHT}" role="img" \
aria-label="Callback. Two waveforms of the same voice: a live caller, whose \
signal falls to the noise floor between words, and the same voice played back \
down a phone line, where it does not because the room is still ringing.">
  <rect width="{WIDTH}" height="{HEIGHT}" fill="{c["bg"]}"/>
  <rect x="24" y="24" width="{WIDTH - 48}" height="{HEIGHT - 48}" fill="{c["sheet"]}" \
stroke="{c["rule"]}" stroke-width="1"/>

  <text x="64" y="120" font-family="{FONTS}" font-size="56" font-weight="700" \
letter-spacing="-1.6" fill="{c["ink"]}">Callback</text>
  <text x="66" y="154" font-family="{FONTS}" font-size="17" fill="{c["soft"]}">\
Who is actually on the phone?</text>
  <text x="66" y="206" font-family="{MONO}" font-size="12" fill="{c["soft"]}">\
A vendor emails new bank details. We hold</text>
  <text x="66" y="226" font-family="{MONO}" font-size="12" fill="{c["soft"]}">\
the payment and ring the number on the</text>
  <text x="66" y="246" font-family="{MONO}" font-size="12" fill="{c["soft"]}">\
oldest invoice, not the one in the email.</text>

  <text x="{PANEL_X}" y="68" font-family="{MONO}" font-size="11" fill="{c["soft"]}">\
the same six seconds of one voice</text>

  {trace(live, TOP_MID, gain, c["ink"])}
  <text x="{PANEL_X}" y="100" font-family="{MONO}" font-size="11" fill="{c["soft"]}">\
live caller</text>
  <text x="{PANEL_X + PANEL_W}" y="100" text-anchor="end" font-family="{MONO}" font-size="11" \
fill="{c["green"]}">allow 0.04</text>

  {trace(played_back, BOTTOM_MID, gain, c["ink"])}
  <text x="{PANEL_X}" y="188" font-family="{MONO}" font-size="11" fill="{c["soft"]}">\
played back down the line</text>
  <text x="{PANEL_X + PANEL_W}" y="188" text-anchor="end" font-family="{MONO}" font-size="11" \
fill="{c["red"]}">deny 0.17</text>
  <text x="{PANEL_X}" y="284" font-family="{MONO}" font-size="11" fill="{c["red"]}">\
between words the signal never falls back: a room ringing after each one</text>
</svg>
"""


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        out = ASSETS / f"hero-{theme}.svg"
        out.write_text(render(theme), encoding="utf-8")
        print(f"{out.relative_to(HERE.parent)}  {out.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
