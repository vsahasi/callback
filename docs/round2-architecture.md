# Round 2: the architecture answer

For the submission form: "What does your architecture look like, and why did
you choose the tools, models, and approach you did? What tradeoffs did you
make, and what would you change with more time?"

Everything below is checked against the code. The receipts are at the bottom.

---

Callback is two services and a contract between them. A LiveKit voice agent
runs the call: Deepgram `flux-general-en`, GPT-4.1, Cartesia `sonic-3`,
Silero VAD. A FastAPI detection service judges how much to trust the caller.
The two speak a versioned package, `callback_shared`, frozen by schema
snapshot tests so neither side can change the wire alone.

The model never authorises anything. It can request an action; a Python gate
decides, on the detector's answer. The gate asks for allow, step_up or deny
before any read or money move, and calls Ramp only on allow. A detector
error, an answer for the wrong session, or anything slower than three seconds
counts as a deny. On a denied money move it holds the vendor anyway, because
holding needs no verification. A prompt injection reaches the request, never
the decision.

The detector measures five signals: voice match against an enrolled print,
synthesis artefacts, replay channel, turn-taking behaviour and liveness.
These are classical signal processing, numpy only, with no trained model in
the serving path. That was deliberate. We had no labelled corpus, and an
untrained model would have been a guess wearing a confidence score. Classical
DSP is deterministic, so a disputed decision is reproducible from the stored
audio.

Scoring is where the care went. Audio that could not be measured blends to
maximum risk rather than zero. One failing signal beats a low average. A
correct SMS code cannot rescue a replay.

The honest gap: the detector cannot yet hear the room. The agent sends words
and timings; the audio field on the wire is still unfilled, so the acoustic
signals run on uploaded clips, not live calls.

With more time: build that listener, and recalibrate the thresholds against
ASVspoof instead of audio this repo generated itself.

---

## Checked on 2026-11-08

| claim | where |
| --- | --- |
| Deepgram `flux-general-en`, GPT-4.1, Cartesia `sonic-3`, Silero VAD | `agent/src/callback_agent/worker.py:100` |
| three second budget, a timeout counts as deny | `AUTHORIZE_TIMEOUT_S = 3.0`, `shared/src/callback_shared/contract.py:30` |
| the detector answers deny past 2.2 s | `AUTHORIZE_SERVER_DEADLINE_S`, same file |
| every other path is a deny | `agent/src/callback_agent/gate.py:9` |
| a denied money move still holds the vendor | `DENY_HOLD_REASON`, `gate.py:46` |
| step-up factors: voice challenge, SMS code, callback on file | `agent/src/callback_agent/stepups.py:154` |
| nothing measured blends to 1.0, not 0.0 | `blend_risk` in `detector/scoring.py` |
| the listener is not built | no `ClipRef` or `Turn.audio` is set anywhere in `agent/src/` |

The earlier draft of this file said the detection service "joins the LiveKit
room as a hidden listener and hears only the caller's audio track." It does
not. `detector/v1/clips.py` is the seam that a listener would plug into, and
the producer on the agent side was never written.

That draft also cited a figure of 43 to 83 percent for cloned voices passing
speaker verification. No source for it exists in this repo, so it is left out
above. If Veer has the citation, it belongs back in with a link.
