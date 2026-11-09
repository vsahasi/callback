# Showcase page

One self-contained HTML file that shows what the detector hears. Four callers,
the real audio, the waveform, and the verdict the service actually returns for
each, with and without a reference voice on file for the vendor.

```sh
uv run python showcase/build.py
```

That writes `showcase/dist/showcase.html`. Open it in a browser. There is no
server, no detector to run and nothing to configure: the audio and the
verdicts are inside the file.

## Why it exists

The dashboard in `web/` answers "what did the detector decide about call X"
for someone who already knows what Callback is. This answers "what is this and
why is it hard" for someone who does not, and it is the only place the real
measured output is visible without a terminal.

## Where the numbers come from

`build.py` posts events to the real app over an in-process ASGI transport and
records what comes back. Not `AnalysisEngine` directly: that would give the
five factor scores and leave the page to work out the decision itself, which
is a second implementation of the scoring rules and a second thing to be
wrong. So the decision, the blended risk, the confidence, the next action and
the sentence under each signal are the service's own, exactly as the agent
receives them.

A threshold that moves moves the page, next time this runs.

## The two states

Every caller is measured twice, because the difference is the point.

| caller | no reference | enrolled |
| --- | --- | --- |
| the real CFO | allow 0.07 | allow 0.04 |
| a recording played down the line | deny 0.30 | deny 0.17 |
| a synthesised clone | deny 0.59 | challenge 0.35 |
| a different person | **allow 0.05** | challenge 0.37 |

The fourth row is why enrolment matters. With nothing on file, a live human
reading the attacker's script passes synthesis, replay and liveness, because a
live human at a handset is exactly what those three look for. The voice
comparison is the only signal that would notice, and it cannot run.

The third row goes the other way, and that is the intended policy rather than
a regression: a clone of the vendor is supposed to match on voice, so the two
signals that did catch it become a smaller share of the total. What stops a
clone is the code texted to the number on the invoice. `docs/DETECTOR.md` has
the long version.

## Files

| file | what it is |
| --- | --- |
| `build.py` | measures the audio and writes the page |
| `page.template.html` | the page, with a `__DATA__` placeholder for the measurements |
| `hero.py` | draws the README banner from the same audio |
| `dist/showcase.html` | the built file, gitignored because it rebuilds in seconds |

## The README banner

```sh
uv run python showcase/hero.py
```

Writes `docs/assets/hero-light.svg` and `hero-dark.svg`, which the top-level
README swaps between with a `<picture>` element. The banner is two envelopes
drawn from the same voice at the same gain: a live caller, and the same
recording played back down a phone line. Between the live caller's words the
signal drops to the noise floor. On the replay it does not, because the room
the recording was played into is still ringing. That is the thing the replay
detector keys on, and it is visible without reading a number.

To change the wording or the design, edit `page.template.html` and run the
build again. To change what is measured, edit `callers()` in `build.py`.

## Publishing it

The built file is one document with everything inline, so anywhere that serves
a static file will do. It is around half a megabyte, nearly all of it the
base64 audio.
