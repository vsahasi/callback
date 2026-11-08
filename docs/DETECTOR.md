# The detector service

Caller authenticity detection for the Callback agent. Answers one narrow
question on a live call: is the voice actually the vendor, or is it a clone, a
recording, an automated caller, or a different person?

The agent owns the conversation and decides when to ask for the account
number. This service owns two narrower things: scoring the authenticity of the
voice and the step-up code, and, on `POST /verify/bank-details`, comparing the
digits the caller spoke against the expected number without ever repeating it
back.

## Status

The service runs in two modes. Mock mode drives five scripted scenarios
through the real state machine and needs no audio, which is what the agent is
built against. Real mode fetches each speech span, decodes it and measures it
with the classical detectors in `detector/analysis`, which is covered below.
All five signals run in real mode once a vendor has a reference voice stored;
until then `voice_match` reports `not_run` and the call is judged on four.

The contract is frozen by a schema snapshot. Nothing is deployed, and no
trained model is in the serving path: real mode is signal processing, not
inference.

## Two open questions for the contract

The wire contract is `0.1.0-provisional`. It was written from the Day-1 build
brief, not from the Notion Contract tab, which was unreachable at the time. Two
things in particular need settling before the agent is built against it.

**1. Decision vocabulary.** The project README describes outcomes as *verified*,
*mismatch* and *needs review*. This service currently returns `allow`, `deny`
and `challenge`, which came from the build brief. A likely mapping is:

| This service | Project vocabulary |
|---|---|
| `allow` | verified |
| `deny` | mismatch, on the voice dimension |
| `challenge` | needs review, pending the texted code |

That is a guess. The mapping matters because the agent has to combine this
voice verdict with the separate bank-details check before reaching a final
verified / mismatch / needs review. Confirm which vocabulary crosses the wire.

**2. Who owns the texted code.** This service models an OTP step-up and the
project README says Callback texts a code to the same line. If the agent owns
Twilio and the code, then `otp_submit` here is the agent reporting an outcome
rather than this service adjudicating it, and the OTP factor should probably
carry less weight. Confirm the division.

Both are in [`CONTRACT.md`](CONTRACT.md). The Contract tab wins on any
disagreement.

## Quickstart

```bash
uv sync --extra dev
uv run pytest
```

57 tests, under a second. Then run the service:

```bash
export CALLBACK_API_TOKEN=any-long-random-string
uv run uvicorn detector.app:app --port 8000
```

```bash
curl -s localhost:8000/healthz
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/events` | Push a call event, get a decision |
| GET | `/factors/{session_id}` | Current factor breakdown for a session |
| POST | `/enrol/voice` | Store a vendor's reference voice |
| POST | `/enrol/voice/status` | Whether an identity has one stored |
| POST | `/enrol/voice/erase` | Remove one |
| GET | `/healthz` | Liveness, contract version, mock flag |

Everything except `/healthz` needs `Authorization: Bearer
<CALLBACK_API_TOKEN>`.

The two enrolment routes that only read are POSTs with the identity in the
body, which looks wrong and is deliberate: a vendor's user id in a path or a
query string is written into every access log between the caller and here.

## The five scenarios

Mock mode runs scripted factor readings through the real state machine and the
real scoring code, so a fixture that passes here exercises the same decision
path a trained model will. Select one with the `X-Callback-Scenario` header, or
by setting `claimed_user_id` to the scenario name.

Measured, not estimated. Thresholds are allow below 0.30, deny at or above
0.55, challenge in between.

| Scenario | After speech | Correct OTP | Three wrong OTPs |
|---|---|---|---|
| `real_cfo` | allow, risk 0.045 | not offered | not offered |
| `cloned_voice` | challenge, risk 0.437 | allow, risk 0.262 | deny, risk 0.662 |
| `human_impostor` | challenge, risk 0.331 | allow, risk 0.199 | deny, risk 0.599 |
| `replay` | deny, fatal factor (risk 0.387) | not offered | not offered |
| `ai_agent` | deny, risk 0.690 | not offered | not offered |

`replay` is the interesting one: its blended risk of 0.387 sits in the challenge
band, but a detected replay is unambiguous attack evidence, so the fatal-factor
rule denies it outright. A good voice match should never rescue a recording.

The mock OTP code is `424242`. Three attempts, 120 second expiry. Mock mode adds
150 to 400 ms of simulated latency per request, so client timeout handling gets
exercised rather than assumed.

## Fail closed, everywhere

A verification service that fails open is worse than no service. Every failure
path resolves to DENY:

- client timeout, transport error, 5xx or 4xx (see `shared/client.py`)
- the call ending before a decision was reached
- session TTL expiry
- OTP expired or all attempts used
- the middle risk band when OTP step-up is not available

`shared/client.py` is the reference implementation of the client half.

## Layout

```
shared/contract.py     Pydantic v2 wire models, CONTRACT_VERSION, client policy
shared/client.py       Fail-closed reference client for the agent side
shared/verification.py Spoken digits against the expected number
shared/enrolment.py    Wire models for storing a reference voice
detector/app.py        FastAPI surface, bearer auth, error shaping
detector/state.py      Session state machine, OTP policy, idempotency
detector/evidence.py   The seam: scripted or measured, one shape either way
detector/scoring.py    Risk blending, thresholds, fatal-factor rule
detector/mock.py       The five scripted scenarios
detector/audio.py      Fetching and decoding a span, and what it refuses
detector/enrolment.py  Reference voices: the store and the endpoint
detector/pipeline.py   Span to factors, off the event loop
detector/analysis/     The classical detectors, numpy only
detector/store.py      Session store: in memory, or modal.Dict
detector/observability.py  Logs, metrics, the per-session journal
detector/review.py     Evidence bundles and the human review dashboard
detector/modal_app.py  Modal serving (not deployed)
training/download.py   Resumable dataset downloads with recorded checksums
tests/contract/        Scenario, policy, transport and edge-case suites
schemas/               Frozen JSON-schema snapshot of the contract
```

## Real mode

Mock mode answers from a script. Real mode fetches the audio each
`speech_segment` points at, decodes it, and measures it.

```bash
CALLBACK_API_TOKEN=kit-test \
CALLBACK_MOCK_MODE=0 \
CALLBACK_AUDIO_ENABLED=1 \
uv run uvicorn detector.app:app --port 8021

uv run python examples/drive_real.py --base-url http://127.0.0.1:8021 --token kit-test
```

That runs every case twice, once with nothing enrolled for the vendor and
again after storing a reference voice, because the difference is the point:

```
what the caller sent                             not enrolled            enrolled  voice_match
a live caller                                      allow 0.07          allow 0.04  pass
a different person reading the script              allow 0.05      challenge 0.37  fail
the same voice, played back down the line           deny 0.30           deny 0.17  pass
a synthesiser reading the same script               deny 0.59      challenge 0.35  pass
a recording first, then a live voice                deny 0.34           deny 0.20  pass
no audio at all                                     deny 1.00           deny 1.00  absent
a span this deployment will not fetch               deny 1.00           deny 1.00  absent
```

Read the second row first. Without a reference voice, a different person
reading the attacker's script is **allowed** at 0.05. Synthesis, replay and
liveness all pass it, because a live human at a handset is exactly what they
are looking for. `voice_match` is the only signal that can catch it and it
cannot run without something to compare against. That is what enrolment
fixes, and it is why `voice_match` carries the largest weight in the first
place.

The fourth row goes the other way, and the section on enrolment below
explains why that is the intended policy rather than a regression.

### Where the audio comes from

`AudioRef.uri` on the event. Three schemes:

| Scheme  | Default | Notes |
|---|---|---|
| `data:` | on | The span inline, base64. No network, so nothing to forge. |
| `https:` | off | Needs `CALLBACK_AUDIO_ALLOWED_HOSTS`. The production path. |
| `file:` | off | Needs `CALLBACK_AUDIO_FILE_ROOT`. Development only. |

Plain `http:` is never fetched.

### Why the allowlist is not optional

`uri` arrives in a request body, so anyone who can reach `POST /events` with a
valid token chooses which URL this service connects to. That is a server-side
request forgery primitive pointed at whatever the container can reach: cloud
metadata on 169.254.169.254, the service itself, anything inside the VPC. The
response body never goes back to the caller, but the factors and the latency
are both side channels.

So the allowlist is the control, and behind it: every address the host
resolves to must be globally routable, redirects are an error rather than a
hop, the body is abandoned past `CALLBACK_AUDIO_MAX_BYTES`, and the whole
fetch runs inside `CALLBACK_AUDIO_FETCH_TIMEOUT_MS`. The residual weakness, a
name that resolves differently between the check and the connection, is
written down in the module docstring of `detector/audio.py` rather than
hidden.

An empty allowlist fetches nothing. A deployment that enables real mode and
forgets the hosts denies every call and says why in the reason, which is the
safe direction to fail in.

### What it decodes

`pcm_s16le` raw or in a WAV container, and `flac` through soundfile. A WAV is
detected by its magic rather than by the declared encoding, because agents
write `pcm_s16le` in the contract and hand over a `.wav`, and reading the
44-byte header as samples puts a click in front of the speech that the
liveness detector would then score.

`opus` is refused with a clear message. There is no opus decoder in the
serving image, and a decoder that guessed would feed noise to the analysis
engine, which would score the noise.

### How spans combine

A call arrives as several segments. The analysis engine is stateless and
scores one span at a time, so the running factors take the lowest score seen
per signal. One clearly bad span is evidence, and averaging it against three
clean ones is how a short burst of a replayed recording would get smoothed
away. The cost is that a single unlucky span pulls the whole call down with no
way back up, which is the right trade for a thirty-second verification call
and would not be for a long one.

### Enrolment: the reference voice

`voice_match` compares the caller against a recording of the vendor from a
call that was already trusted. One POST per vendor, once, out of band:

```bash
curl -X POST http://127.0.0.1:8021/enrol/voice \
  -H "Authorization: Bearer kit-test" -H "Content-Type: application/json" \
  -d '{"identity": "cfo@northwind.example",
       "label": "invoice call 2026-03-02",
       "audio": {"track_id": "enrol", "start_ms": 0, "end_ms": 6000,
                 "sample_rate_hz": 8000, "encoding": "pcm_s16le",
                 "uri": "https://your-bucket.example/vendor-123.wav"}}'
```

The recording is fetched through the same transport and the same allowlist as
call audio, so enrolment is off whenever `CALLBACK_AUDIO_ENABLED` is off. One
switch, no second way in.

**What is stored is not the recording.** A `VoicePrint` holds a median pitch,
two durations, and a long-term average log spectrum: a few hundred numbers,
averaged over every speech frame, with no phase and no time axis. Nothing
that was said can be recovered from it, because everything carrying what was
said has been averaged away. That shape was chosen for three reasons and the
privacy one came first: an enrolment recording is a vendor's voice, held
indefinitely, in a service inside a payments flow, and that is biometric data
whose liability grows with every vendor onboarded. The other two are that a
print is about 1.6 KB against a quarter of a megabyte, and that the reference
stops being re-measured on every speech segment of every call.

The response says nothing about the voice either. Not the pitch, not the
spectrum, not a similarity against anything: only how much voiced speech was
found, how long the recording was, where it came from, and a quality verdict.
`POST /enrol/voice/erase` removes a print, because the thing stored is
derived from biometric data and a vendor may ask for it to be gone.

A recording with nothing measurable in it is refused with a reason rather
than stored. Storing it would leave an operator believing a vendor was
covered while every comparison came back inconclusive. A recording with under
1.5 s of voiced speech is stored and flagged `thin`, because it still catches
a wildly different talker.

**`replace` must be explicit.** Someone holding the service token cannot
release a payment, since a human approves that. But overwriting a vendor's
reference with their own voice would pass the heaviest signal in the model on
every future call, silently, until somebody re-enrolled that vendor. So a
second enrolment for an identity is a 409 unless the request asks to replace.

**A full store refuses rather than evicting.** The session store evicts
because sessions are transient; enrolments are not. Evicting one would
quietly remove a signal from some other vendor's next call. Refusing fails in
front of the operator onboarding right now, who can raise
`CALLBACK_ENROLMENT_MAX_IDENTITIES`.

**Identity is matched byte for byte** after trimming whitespace. No case
folding, no unicode normalisation. Canonicalising would trade a missed lookup
for a possible collision, and those are not the same size: a missed lookup
leaves `voice_match` unrun and denies more often, while a collision compares
one vendor against another vendor's voice.

By default prints live in process memory and a restart loses them all. Set
`CALLBACK_ENROLMENT_DICT_NAME` to keep them in a `modal.Dict` across deploys
and replicas.

### What this does not do

It is not speaker verification. `detector/analysis/voice.py` says so at the
top and the `detail` string says so on every passing comparison. Two adults of
the same sex with similar pitch will match here even if a listener could tell
them apart instantly. What it reliably catches is an obviously different
talker, which is the `human_impostor` attack and the second row of the table
above. What it cannot do is confirm identity, so a passing `voice_match` is
evidence that nothing is obviously wrong and not evidence that the caller is
the vendor.

### Why a synthesiser is challenged rather than denied

Fourth row of the table: enrolling the vendor moves the synthesiser from
`deny 0.59` to `challenge 0.35`. Nothing about the synthesiser changed. What
changed is that `voice_match` now runs, passes, and carries 0.35 of the
weight, so the two failing signals are a smaller share of the total.

A clone of the vendor's voice is supposed to match on `voice_match`. That is
not the detector being fooled, it is the factor answering its own question
correctly, and this product has said so from the start: the `cloned_voice`
scenario in `detector/mock.py` scores `voice_match` 0.88 PASS with the note
"Embedding matches, which a high-quality clone is expected to do", fails
`synthetic_speech`, and gives a reason ending "Step-up required" with
`otp_can_rescue` true. Real mode now agrees with that policy. The old deny
was an artefact of `voice_match` never running, which inflated every
real-mode risk score by removing a passing factor from the renormalisation.

So what catches a clone is the one-time code sent to the number on the oldest
invoice, not the blended score. Whether a cloned voice should instead be
unrescuable, as `replay` and `agent_behaviour` are, is a product decision and
is not made in the code.

The replay row is the other half of the same machinery and is worth
understanding because three separate rules have to hold for it:

* the replay scores 0.12, **above** the 0.10 fatal line, so the fatal rule
  does not fire;
* the blend lands at 0.17, **below** the 0.30 allow threshold, so the blend
  would allow it;
* `first_failing_factor` blocks the allow, and `replay` being in
  `UNRESCUABLE` means no step-up is offered, so the middle band falls to deny.

Remove any one of those and a replayed recording is allowed or merely
challenged.

## Testing against a deployment

The contract suite can point at a running service instead of the in-process app,
which is how the agent side validates without matching this Python environment:

```bash
CALLBACK_BASE_URL=https://your-mock-url CALLBACK_API_TOKEN=the-token uv run pytest tests/contract -v
```

## Changing the contract

The schema snapshot fails the build on any drift, including a renamed enum
member. To make a deliberate change:

```bash
CALLBACK_UPDATE_SNAPSHOT=1 uv run pytest tests/contract/test_schema_snapshot.py
```

Then bump `CONTRACT_VERSION` in `shared/contract.py` in the same commit. A test
checks the two against each other.

## Datasets

```bash
uv run python -m training.download --dry-run
```

About 35 GB of archives into `~/data/callback`, never into the repo. Downloads
resume, so interrupting is safe. Run the real thing detached:

```bash
tmux new-window -t callback -n downloads 'cd ~/projects/callback && uv run python -m training.download'
```

Checksums are trust-on-first-use: the official SHA-256 digests for these
archives are not all published machine-readably, so the first successful fetch
records a digest in `~/data/callback/checksums.json` and later runs verify
against it. Paste official digests into the registry in `training/download.py`
to have them enforced from the first byte.

## Configuration

See `.env.example`. `CALLBACK_API_TOKEN` is the only value with no usable
default.

## GPU

Verified on an RTX 5080 (Blackwell, `sm_120`) under WSL2:

```
torch 2.15.0+cu132, CUDA 13.2, cuDNN 92600
sm_120 present in torch.cuda.get_arch_list()
fp16 4096x4096 matmul executed, 14.3 of 15.9 GiB free
```

Those are the versions to pin in the training image. Anything older than CUDA
12.8 will not have `sm_120` kernels.
