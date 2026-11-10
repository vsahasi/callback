# Round 2: the architecture answer

For the submission form: "What does your architecture look like, and why did
you choose the tools, models, and approach you did? What tradeoffs did you
make, and what would you change with more time?"

Everything below is checked against the code. The receipts are at the bottom.

---

Callback is two services and a contract between them. A LiveKit voice agent
runs the call: Deepgram `flux-general-en`, gpt-5.6-terra, Cartesia `sonic-3`,
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
the serving path. It hears the caller because the agent sends the speech:
frames already tapped for STT go out as one segment per stretch of caller
speech, inline 16 kHz mono WAV, 20 seconds at most, tagged `caller` or
`callback` for the leg it was heard on, to the detector's `/events`. `/v1`
reads what `/events` measured through one seam, `detector/bridge.py`: voice
match passes through, synthesis and bot-likeness invert into risks, and the
worst of synthesis, replay and liveness is the spoof number the policy gates
on. Nothing heard is still a deny. The audio field on the `/v1` wire stays
unfilled; the hidden listener an earlier draft described was never built,
and this replaces it.

The leg tag is for the one attack no other signal can see: a live human
impostor, whose voice is real and whose code goes to the number on file.
Callback rings that number back, so whoever answers is the in-call
reference, and the two legs' prints are compared. That check is in review;
main carries the tag, not the comparison.

Scoring is where the care went. Audio that could not be measured blends to
maximum risk rather than zero. One failing signal beats a low average. A
correct SMS code cannot rescue a replay.

The honest tradeoff: synthetic speech is caught by a DSP detector, not a
trained one. That was deliberate. We had no labelled corpus; the trained
countermeasure in `training/` has run once, a smoke test on random audio
with a dev EER of 0.5, which is chance, and an untrained model would have
been a guess wearing a confidence score. Classical DSP is deterministic, so a
disputed decision is reproducible from the stored audio. The cost: a
commercial clone that fakes micro-variation is where this detector is
weakest.

With more time: train that model on phone-channel data, put it behind the
same `/events` seam, and recalibrate the thresholds against ASVspoof instead
of audio this repo generated itself.

---

## Checked on 2026-11-09

| claim | where |
| --- | --- |
| Deepgram `flux-general-en`, gpt-5.6-terra, Cartesia `sonic-3`, Silero VAD | `AgentSession` at `agent/src/callback_agent/worker.py:119`, the vendors at `:120` to `:122` |
| three second budget, a timeout counts as deny | `AUTHORIZE_TIMEOUT_S = 3.0`, `shared/src/callback_shared/contract.py:30` |
| the detector answers deny past 2.2 s | `AUTHORIZE_SERVER_DEADLINE_S`, same file, `:33` |
| every other path is a deny | `agent/src/callback_agent/gate.py:10` |
| a denied money move still holds the vendor | `DENY_HOLD_REASON`, `gate.py:46` |
| step-up factors: voice challenge, SMS code, callback on file, and a probe question | `Factor`, `contract.py:83`; `agent/src/callback_agent/stepups.py:153` |
| nothing measured blends to 1.0, not 0.0 | `blend_risk`, `detector/scoring.py:30` |
| segments: 20 s at most, 16 kHz mono WAV as a `data:` URI, leg-tagged | `MAX_SEGMENT_S`, `agent/src/callback_agent/audio_segments.py:31`; `WIRE_SAMPLE_RATE_HZ`, `:34`; the URI, `:168`; the leg stamp, `session_wiring.py:334`; the POST to `/events`, `detector.py:143` |
| the `/v1` audio field is still unfilled | no `ClipRef` or `Turn.audio` is set anywhere in `agent/src/` |
| `/v1` reads `/events`: polarity, worst-of, `no_audio` | `signals_from_factors`, `detector/bridge.py:73`, worst-of at `:93`; `_engine_signals`, `detector/v1/service.py:684`; `NO_AUDIO`, `detector/v1/policy.py:198` |
| the cross-leg check is not on main | the only `track_id` in `detector/` is the placeholder `enrolment.py:562`; `detector/legs.py` is not in the tree |
| DSP synthesis; torch only in the `train` extra; one smoke test, EER 0.5 | `detector/analysis/synthesis.py:1`; `pyproject.toml:12` and `:37`; `training/train.py:4`; `README.md:309` |

The earlier draft of this file said the detection service "joins the LiveKit
room as a hidden listener and hears only the caller's audio track." It does
not. The agent posts the caller's speech to `/events` itself, and the
detector's `/v1` side reads what that measured.

That draft also cited a figure of 43 to 83 percent for cloned voices passing
speaker verification. No source for it exists in this repo, so it is left out
above. If Veer has the citation, it belongs back in with a link.
