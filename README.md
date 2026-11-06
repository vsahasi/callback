# Callback

An agent that makes the phone call to confirm a vendor's bank change.

When a vendor emails new bank details, Callback holds the payment in Ramp, takes the vendor's phone number from the oldest invoice or contract on file from before the change, and calls that number. The vendor says the new account out loud, and Callback never reads it to them. Callback then texts a code to the same line, decides verified, mismatch, or needs review, and attaches the recording and transcript to the bill. A person approves before any money moves.

Built by Veer Sahasi and Eric Catalano for the BBS x Ramp Hackathon.

## Components

| Directory | Owner | What it does |
|---|---|---|
| `agent/` | Veer | The LiveKit agent that places and runs the call |
| `detector/` | Eric | Caller authenticity detection service, HTTP |
| `shared/` | Eric | The wire contract between agent and detector, plus a reference client |
| `training/` | Eric | Datasets and model training for the detector |
| `examples/` | Eric | A runnable agent skeleton and a script that drives the mock |
| `docs/` | Eric | The contract, how to run the detector, how to integrate |
| `ci/` | Eric | The GitHub Actions workflow, parked until the token has `workflow` scope |

## What the detector answers

During the call the detector answers one narrow question: is the voice on the
line actually the vendor, or is it a clone, a recording, an automated caller or
a different person? It returns a risk score, the five signals behind it, and one
of allow, challenge or deny.

Catching "a different person" needs something to compare against, so there is
a third endpoint: `POST /enrol/voice` stores a reference voice for a vendor,
taken from a call that was already trusted. It stores a pitch and a spectral
envelope rather than the recording, so the service never holds a library of
vendors' voices.

It also offers a second endpoint the agent can call: `POST /verify/bank-details`
compares the digits the vendor spoke against the expected account number and
combines that with the voice verdict into verified, mismatch or needs review.
The expected number never appears in a response, a log line or an error body.
The agent still owns the conversation and decides when to ask; the detector owns
the comparison and the never-reveal guarantee.

Every failure path in the detector resolves to deny. A service that cannot reach
a confident answer refuses the call rather than waving it through.

## Running it

The detector runs in mock mode with no model and no audio, so the agent can be
built against it today. Serve it:

```bash
CALLBACK_API_TOKEN=kit-test uv run uvicorn detector.app:app --port 8021
```

Then, in another shell, drive all five scenarios through it end to end:

```bash
uv run python examples/drive_mock.py --base-url http://127.0.0.1:8021 --token kit-test
```

The five are `real_cfo`, `cloned_voice`, `replay`, `ai_agent` and
`human_impostor`. Pick one per call with the `X-Callback-Scenario` header. They
run through the real state machine, so the decisions, the step-ups and the
timings are the ones the agent will see.

### Real mode

The same endpoints, with the audio actually measured instead of scripted.
Restart the detector with two more variables and drive it with waveforms:

```bash
CALLBACK_API_TOKEN=kit-test CALLBACK_MOCK_MODE=0 CALLBACK_AUDIO_ENABLED=1 uv run uvicorn detector.app:app --port 8021
```

```bash
uv run python examples/drive_real.py --base-url http://127.0.0.1:8021 --token kit-test
```

That runs every case twice, once with nothing enrolled for the vendor and
again after storing a reference voice:

```
what the caller sent                             not enrolled            enrolled
a live caller                                      allow 0.07          allow 0.04
a different person reading the script              allow 0.05      challenge 0.37
the same voice, played back down the line           deny 0.30           deny 0.17
a synthesiser reading the same script               deny 0.59      challenge 0.35
a recording first, then a live voice                deny 0.34           deny 0.20
no audio at all                                     deny 1.00           deny 1.00
```

A live caller allows. A replayed recording denies on its room acoustics. A
call with no readable audio denies too, because a caller nothing could be
measured against is not a caller who passed.

The second row is the one to read. A different person reading the attacker's
script is allowed until the vendor has a reference voice, because synthesis,
replay and liveness all pass a live human at a handset and `voice_match` is
the only signal that notices. `POST /enrol/voice` stores that reference, once
per vendor. What it keeps is a pitch and a spectral envelope, never the
recording: a vendor's voice held indefinitely is biometric data, and nothing
said in the enrolment call can be recovered from what is stored.

The fourth row goes the other way, and the reason is worth knowing before you
rely on either number. `docs/DETECTOR.md` has it, along with the threat model
for the fetcher: each span is an https url in your object store and the bucket
host has to be named in `CALLBACK_AUDIO_ALLOWED_HOSTS`, because the uri comes
from the request body and an unrestricted fetcher would be a way into your
network.

```bash
make test                                       # the suite
make lint                                       # ruff check and a format check
uv run python -m detector.analysis.calibration  # the analysis, on synthetic audio
```

- [`docs/DETECTOR.md`](docs/DETECTOR.md) to run the detector locally
- [`docs/CONTRACT.md`](docs/CONTRACT.md) for the agent-to-detector API
- [`docs/AGENT-INTEGRATION.md`](docs/AGENT-INTEGRATION.md) for wiring the agent to it

## Reviewing a call

`GET /review` is a dashboard a person can open. It shows the decision, every
signal with its number and the reason behind it, the request timeline, the code
attempt history, and the recording and transcript once the agent attaches them.
`GET /metrics` is Prometheus text. Both need the service token. The metrics
labels carry the decision mix, so an open scrape would publish the deny rate;
that is why it is not open by default.

## Secrets

Never commit API keys. Put them in `.env`, which is gitignored, and list the variable names in `.env.example`.
