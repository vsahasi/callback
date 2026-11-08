# Callback

Callback is a voice agent for Ramp. You phone it, or talk to it in the browser. It reads a company's Ramp account (vendors, bills, the last four digits of a bank account) and acts on it (hold a vendor, release a hold, change a vendor's bank account). Before any sensitive action, it checks that the caller is the real person, not a scammer or an AI-cloned voice.

Voice biometrics alone are broken, because a cloned voice passes them. In one 2026 study, cloned voices got past a tuned speaker-verification system 43 to 83 percent of the time, depending on the cloning tool ([arXiv 2601.02914](https://arxiv.org/html/2601.02914)). So we check that the voice matches the enrolled person and that it comes from a real human. We also add an SMS code and a callback, both to the number on file, because a clone can't fake those.

Callback runs against Ramp's sandbox, which can't move real money.

## How a call works

Callback has two halves. The agent (Veer) runs the call and talks to Ramp. The detection service (Eric) listens to the caller and decides whether the agent may go ahead. The agent owns the conversation. The detector owns the judgment about who is on the line.

1. A caller, say the company's CFO, joins a LiveKit room by phone or from the browser. The agent opens with a recording disclosure and asks who they are.
2. For the rest of the call, the detection service scores the caller's voice.
3. When the caller asks for something sensitive, the agent's tool for it goes through the gate. The gate asks the detection service for a decision.
4. The service can ask for more proof first: a spoken challenge phrase, an SMS code read back, or a callback to the number on file. The agent runs those and asks again.
5. The agent makes the Ramp call only when the service allows it. When the service denies a money move, the agent puts the vendor on hold first. Then it tells the caller we will follow up through the usual channel, and ends the call.

What each request needs, roughly:

- Holding a vendor or a bill needs nothing. A hold is always allowed.
- Reading vendors, bills or the last four digits of a bank account needs the voice checks to pass.
- Changing a vendor's bank account or releasing a hold needs all of it: the voice checks, the SMS code and a callback to the number on file. The callback needs a phone line, so these two need a phone call. From the browser the agent refuses them and tells the caller to call in by phone. Holds and reads work on both channels.

## What the detector checks

The detector answers one narrow question: is the voice on the line really the enrolled person, or is it a clone, a recording, an automated caller or a different person? It measures five signals (voice match, synthetic speech, replay, automated-caller behaviour and liveness) and combines them, with the result of the SMS code, into a risk score and a decision.

It runs in two modes:

- Mock mode needs no model and no audio. It plays one of five scripted calls through the real state machine and the real scoring code, so you can run against it without the models.
- Real mode fetches each speech span, decodes it and measures it with classical signal processing in `detector/analysis`. No trained model is in the serving path yet. Real mode can't match a voice to an enrolled person yet either: there is no enrolment store, so that signal reports `not_run` and the call is judged on the other four.

The detector also has a bank-detail check. It compares spoken digits against an expected account number, and the expected number never appears in a response, a log line or an error body. `docs/DETECTOR.md` covers it. It is off unless `CALLBACK_BANK_VERIFY_ENABLED=1`, and the agent never calls it.

## Rules we don't break

- The LLM never decides authorization. The gate is Python code, not a prompt. It acts only on the decision the detection service sends back.
- Fail closed, on both sides. If the detection service errors or times out, the agent denies the request, and a denied money move also puts the vendor on hold. Inside the detector, every failure path resolves to deny. A service that can't reach a confident answer refuses the call rather than waving it through.
- Holding is always allowed. A hold protects the company, so it never needs verification.
- A voiceprint match never authorizes anything alone. A high deepfake score overrides any voiceprint score, and the detector never averages a failing signal into an allow. A good voice match doesn't rescue a recording.
- We never use a number the caller gives. SMS codes and callbacks go only to the number already on file.
- Never commit API keys. Secrets go in `.env`, which git ignores. `.env.example` lists the variable names and holds no values.

## The contract

<!-- CONTRACT-SECTION -->

The agent and the detection service talk over `callback_shared` 1.1.0 (`shared/src/callback_shared/contract.py`), served by the detector under `/v1`. `docs/CONTRACT.md` describes the detector's internal `/events` API, which the agent does not use.

## Repo layout

| Path | Owner | What it does |
|---|---|---|
| `agent/` | Veer | The voice agent, built on LiveKit Agents. It has the detector client, the Ramp client, the gate, the tools the LLM can call, and the step-ups. |
| `detector/` | Eric | The detection service, over HTTP: the state machine, the signal analysis, the scoring and the review dashboard. |
| `shared/` | Both | The contract between the agent and the detector, plus a reference client. We change it only together. |
| `training/` | Eric | Dataset downloads, features, model training and metrics for the detector. No audio goes in git. |
| `examples/` | Eric | An agent skeleton, and scripts that drive the detector in mock and real mode. |
| `tests/` | Eric | The detector's tests: contract, detector and training suites. The agent's tests are in `agent/tests/`. |
| `schemas/` | Eric | Frozen JSON-schema snapshots of the detector's request and response models. |
| `scripts/` | Eric | A check that a Modal deploy of the detector would work, without deploying it. |
| `docs/` | Both | Veer's spec (a copy of our Notion page), demo runbook, build plan and research notes. Eric's contract, detector guide and agent integration guide. |
| `ci/` | Eric | The GitHub Actions workflow, parked until the token has `workflow` scope. |
| `Dockerfile` | Veer | Builds a cloud copy of the agent worker, kept as a backup. |
| `Makefile` | Eric | Install, test, lint and run targets for the detector. |
| `.env.example` | Both | Every environment variable name, grouped by component, with no values. |

## Running it

You need [uv](https://docs.astral.sh/uv/). The agent uses Python 3.13 and the detector uses Python 3.11. uv installs either one if you don't have it.

Copy `.env.example` to `.env` in the repo root. It lists every variable name. The agent loads `.env` itself. The detector reads only its process environment, so pass its variables on the command line, as below.

### The agent

1. In `.env`, fill in the LiveKit, Deepgram, OpenAI, Cartesia and Ramp sandbox values. A variable already set in your shell wins over the file.
2. Set `DETECTOR_URL` and `DETECTOR_TOKEN` to the detection service's URL and token. If the detector runs in mock mode, set `DETECTOR_SCENARIO` to the call you want to play.
3. Start the worker:

```bash
cd agent
uv sync
uv run -m livekit.agents download-files
uv run agent.py dev
```

`DETECTOR_SCENARIO` is one of `real_cfo`, `cloned_voice`, `replay`, `ai_agent` or `human_impostor`. The worker reads its settings when it starts, so restart it after you change the scenario. Leave the scenario empty when you point at a detector in real mode. In mock mode the SMS code is always `424242`.

The worker registers as `callback`. Phone calls reach it through the dispatch rule in `agent/dispatch-rule.json`, which has to be added to your LiveKit project first. A browser session has to name the agent `callback` too. `docs/demo-runbook.md` walks through the five calls and what each should do.

### The detector

Mock mode runs with no model and no audio. Serve it from the repo root:

```bash
CALLBACK_API_TOKEN=kit-test uv run uvicorn detector.app:app --port 8021
```

Then, in another shell, drive all five scenarios through it end to end:

```bash
uv run python examples/drive_mock.py --base-url http://127.0.0.1:8021 --token kit-test
```

The five scenarios are the same five names as above. Pick one per call with the `X-Callback-Scenario` header. Mock mode adds 150 to 400 ms of latency to each request on purpose, so client timeout handling gets exercised.

Real mode uses the same endpoints, with the audio measured instead of scripted. Restart the detector with two more variables and drive it with waveforms:

```bash
CALLBACK_API_TOKEN=kit-test CALLBACK_MOCK_MODE=0 CALLBACK_AUDIO_ENABLED=1 uv run uvicorn detector.app:app --port 8021
```

```bash
uv run python examples/drive_real.py --base-url http://127.0.0.1:8021 --token kit-test
```

A live caller allows at risk 0.07. The same voice played back down the line denies on its room acoustics. A synthesiser reading the same script denies on the absence of micro-variation. A call with no readable audio denies too, because a caller nothing could be measured against is not a caller who passed.

In production the detector fetches audio over https, and only from hosts named in `CALLBACK_AUDIO_ALLOWED_HOSTS`. The address comes from the request, so an unrestricted fetcher would be a way into your network. `docs/DETECTOR.md` has the threat model and every limit.

More on the detector:

- [`docs/DETECTOR.md`](docs/DETECTOR.md) to run the detector locally
- [`docs/CONTRACT.md`](docs/CONTRACT.md) for the API between the agent and the detector
- [`docs/AGENT-INTEGRATION.md`](docs/AGENT-INTEGRATION.md) for wiring an agent to the detector

### Tests

The agent and the shared package:

```bash
(cd agent && uv run pytest)
(cd shared && uv run --group dev pytest)
```

Most of the agent tests run against fakes, so they need no keys. Two of them call a live service, a running detection service and the Ramp sandbox. They are skipped unless you export `DETECTOR_URL` and `DETECTOR_TOKEN`, or `RAMP_CLIENT_ID` and `RAMP_CLIENT_SECRET`, in your shell. The tests marked `llm` call GPT-4.1, so they need `OPENAI_API_KEY` and are skipped without it. pytest doesn't read `.env`.

The detector, from the repo root:

```bash
make test                                       # the suite
make lint                                       # ruff check and a format check
uv run python -m detector.analysis.calibration  # the analysis, on synthetic audio
```

## Reviewing a call

`GET /review` on the detector is a dashboard a person can open. It shows the decision, every signal with its number and the reason behind it, the request timeline, and the code attempt history. `GET /metrics` is Prometheus text. Both need the service token. The metrics labels carry the decision mix, so an open scrape would publish the deny rate. That is why it is not open by default.

## Privacy

- We don't keep raw call audio. Session recording is fully off (`record=False`): LiveKit records no audio, uploads no transcript, and exports no traces or logs.
- A voiceprint is a set of numbers derived from a person's voice. We make one only after the person agrees out loud, and we delete it by November 30, 2026, or sooner if they ask.
- Before it enrolls a voiceprint, the agent reads this consent text, and we store its version and the person's spoken "yes":

  > To protect your company's account, Callback creates a voiceprint, a set of numbers derived from your voice, and analyzes call audio and transcripts for fraud. Raw audio is not kept. Your data is deleted by November 30, 2026, or sooner if you ask. Do you agree?

- Every call opens with a spoken disclosure that the call is recorded and analyzed by automated fraud checks. California requires consent from every party, so we say it even though we don't keep the audio.
- The caller does say the full bank numbers aloud. Deepgram transcribes them, and GPT-4.1 receives them as the arguments of the bank-change tool. The agent never forwards a full bank number: forwarded turns have every run of 4 or more digits replaced by [NUM] (a 4-6 digit run is kept only in the caller's answer to a voice challenge), and the authorize call carries the last four digits and a keyed fingerprint. The detector never puts an expected account number in a response, a log line or an error body.
- The exceptions are on Eric's side: training recordings and team test calls. Training speakers sign a separate written form, and we delete those recordings by December 31, 2026. A test call is recorded only with a consent row, and deleted after 7 days. No audio is in this repo, and we don't release raw audio.

## Team

Veer Sahasi built the voice agent. Eric Catalano built the detection service. We made Callback for the BBS x Ramp Hackathon.
