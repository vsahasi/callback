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

```bash
make test                                       # the suite
make lint                                       # ruff check and a format check
uv run python -m detector.analysis.calibration  # the real analysis, on synthetic audio
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
