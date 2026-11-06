# Callback

An agent that makes the phone call to confirm a vendor's bank change.

When a vendor emails new bank details, Callback holds the payment in Ramp, takes the vendor's phone number from the oldest invoice or contract on file from before the change, and calls that number. The vendor says the new account out loud, and Callback never reads it to them. Callback then texts a code to the same line, decides verified, mismatch, or needs review, and attaches the recording and transcript to the bill. A person approves before any money moves.

Built by Veer Sahasi and Eric Catalano for the BBS x Ramp Hackathon.

## Components

| Directory | Owner | What it does |
|---|---|---|
| `agent/` | Veer | The LiveKit agent that places and runs the call |
| `detector/` | Eric | Caller authenticity detection service, HTTP |
| `training/` | Eric | Datasets and model training for the detector |
| `shared/` | Eric | The wire contract between agent and detector, plus a reference client |

The detector answers one narrow question during the call: is the voice on the
line actually the vendor, or is it a clone, a recording, an automated caller or
a different person? It does not decide whether the bank details match. That
belongs to the agent.

- [`docs/DETECTOR.md`](docs/DETECTOR.md) to run the detector locally
- [`docs/CONTRACT.md`](docs/CONTRACT.md) for the agent-to-detector API

## Secrets

Never commit API keys. Put them in `.env`, which is gitignored, and list the variable names in `.env.example`.
