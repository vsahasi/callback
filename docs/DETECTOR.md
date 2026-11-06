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
| GET | `/healthz` | Liveness, contract version, mock flag |

`/events` and `/factors` need `Authorization: Bearer <CALLBACK_API_TOKEN>`.
`/healthz` does not.

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

Note that the scenario names come from the build brief and read as impersonation
of a company officer. In this product the impersonated party is the vendor, so
`real_cfo` means the genuine expected caller. Worth renaming once the contract
is settled.

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
detector/app.py        FastAPI surface, bearer auth, error shaping
detector/state.py      Session state machine, OTP policy, idempotency
detector/evidence.py   The seam: scripted or measured, one shape either way
detector/scoring.py    Risk blending, thresholds, fatal-factor rule
detector/mock.py       The five scripted scenarios
detector/audio.py      Fetching and decoding a span, and what it refuses
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

That prints live, replayed and synthetic audio going through the real path:

```
what the caller sent                       decision    risk  signals
a live caller                              allow       0.07  3 measured, none failed
the same voice, played back down the line  deny        0.30  fail: replay
a synthesiser reading the same script      deny        0.59  fail: synthetic_speech, liveness
a recording first, then a live voice       deny        0.34  fail: replay
no audio at all                            deny        1.00  none measured
a span this deployment will not fetch      deny        1.00  none measured
```

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

### What real mode cannot do yet

`voice_match` needs an enrolment recording of the claimed identity. There is
no enrolment store, so it reports `not_run` rather than guessing. It carries
the largest weight at 0.35, so a real-mode call is judged on the remaining
four signals with the weights renormalised over them. That renormalisation is
exactly the condition that makes a diluted FAIL dangerous, which is why the
guard in `detector/scoring.py` matters more here than it ever did in mock
mode. The replay case in the table above is that guard firing: the replay
scores 0.12, just above the 0.10 fatal line, and the blend lands at 0.30,
which without the guard would have been an allow.

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
