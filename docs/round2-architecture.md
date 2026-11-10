# Round 2: the architecture answer

For the submission form: "What does your architecture look like, and why did
you choose the tools, models, and approach you did? What tradeoffs did you
make, and what would you change with more time?"

Everything below is checked against the code; receipts at the bottom.

---

Callback is two services and a contract between them. A LiveKit voice agent
runs the call: Deepgram `flux-general-en`, `gpt-5.6-terra`, Cartesia
`sonic-3`, Silero VAD. A FastAPI detection service judges how much to trust
the caller. The two speak a versioned package, `callback_shared`, frozen by
schema snapshot tests so neither side can change the wire alone.

The model never authorises anything. It can request an action; a Python gate
decides, on the detector's answer. The gate asks for allow, step_up or deny
before any read or money move, and calls Ramp only on allow. A detector
error, a wrong-session answer, or anything slower than three seconds counts
as a deny. On a denied money move it holds the vendor anyway. A prompt
injection reaches the request, never the decision.

The detector measures five signals: voice match against an enrolled print,
synthesis artefacts, replay channel, turn-taking behaviour and liveness,
classical signal processing. It hears the caller because the agent sends the
speech: frames already tapped for STT go out as one segment per stretch of
caller speech, inline 16 kHz mono WAV, 20 seconds at most, tagged `caller` or
`callback` for its leg, to the detector's `/events`. `/v1` reads
what `/events` measured through one seam, `detector/bridge.py`: voice match
passes through, synthesis and bot-likeness invert into risks, and the worst
of synthesis, replay and liveness is the spoof number the policy gates on.
Nothing heard is still a deny. The hidden listener was never built; this
replaces it.

The leg tag is for the one attack no other signal can see: a live human
impostor, whose voice is real and whose code goes to the number on file. Callback rings that number back, so whoever answers is the in-call
reference, and the two legs' prints are compared. That check is in review;
main carries the tag, not the comparison.

Scoring is where the care went. Audio that could not be measured blends to
maximum risk rather than zero. One failing signal beats a low average. A
correct SMS code cannot rescue a replay.

The honest tradeoff: synthetic speech is caught by a DSP detector. The
trained countermeasure, `training/`, has run once, a smoke test on random
audio with a dev EER of 0.5, chance. It is kept out of the serving path: an
untrained model is a guess wearing a confidence score, and DSP is
deterministic, so a disputed decision is reproducible. The cost: a commercial
clone that fakes micro-variation is where this detector is weakest.

With more time: train it on phone-channel data, put it behind the same
`/events` seam, and recalibrate the thresholds against ASVspoof instead of
this repo's own synthetic audio.

---

## Checked on 2026-11-09

| claim | where |
| --- | --- |
| the four vendors and models | `agent/src/callback_agent/worker.py:119` |
| 3 s budget, timeout is deny; 2.2 s deadline | `shared/src/callback_shared/contract.py:30` |
| deny path; a denied money move holds; the three step-up factors | `agent/src/callback_agent/gate.py:10`, `:46`; `stepups.py:153` |
| unmeasured blends to 1.0 | `detector/scoring.py:30` |
| segments: 20 s, 16 kHz mono WAV `data:` URI, leg-tagged | `audio_segments.py:31`, `:96`; `session_wiring.py:484`; `detector.py:143` |
| `/v1` reads `/events`: polarity, worst-of, `no_audio` | `detector/bridge.py:88`; `detector/v1/service.py:593`; `policy.py:198` |
| the cross-leg check is not on main | no `track_id` read in `detector/`; `detector/legs.py` is on a branch |
| DSP synthesis, torch only in the `train` extra; one smoke test, EER 0.5 | `detector/analysis/synthesis.py:1`; `pyproject.toml:12`; `training/train.py:4`; `README.md:309` |

The agent lines are from the audio-pipe branch, not yet on main, where
`worker.py:101` still reads `gpt-4.1`. The 43 to 83 percent figure for cloned
voices passing speaker verification, from an earlier draft, has no source in
this repo and stays out. Veer may have the citation.
