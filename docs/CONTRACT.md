# The detector contract

For Veer, so the LiveKit agent can be written against this without reading the
service code.

**Version `0.1.0-provisional`.** Read the warning below before you build on it.

## Read this first

This contract was derived from the Day-1 build brief, not from the Notion
Contract tab, because that tab was not reachable when the service was written
(the Notion MCP server is registered but not authenticated yet). The endpoints,
the header, the five scenarios and the OTP policy come straight from the brief.
Everything else, field names and types in particular, is a reasonable guess.

So: if the Contract tab says something different, **the Contract tab wins**.
Tell me what differs and I will reconcile and bump the version. A schema
snapshot test makes any change show up as an explicit diff, so this will not
drift silently.

Two specific things need your answer, both found by reading the project README
after this service was written.

**1. Decision vocabulary.** The project README describes outcomes as *verified*,
*mismatch* and *needs review*. This service returns `allow`, `deny` and
`challenge`, which is what the build brief said. The likely mapping is `allow`
to verified, `deny` to mismatch on the voice dimension, and `challenge` to
needs review pending the code. That is a guess. It matters because you have to
combine this voice verdict with the separate bank-details check before reaching
a final verdict, so tell me which vocabulary should cross the wire.

**2. Who owns the texted code.** This service models an OTP step-up, and the
project README says Callback texts a code to the same line. If you own Twilio
and the code, then `otp_submit` here is you reporting an outcome rather than me
adjudicating it, and the OTP factor should carry less weight in the blend.
Confirm the division and I will adjust.

Also note the scenario names (`real_cfo` and friends) came from the build brief
and read as impersonating a company officer. In this product the impersonated
party is the vendor, so `real_cfo` means the genuine expected caller. Worth
renaming once the rest is settled.

The machine-readable version of everything below is
[`schemas/contract.schema.json`](../schemas/contract.schema.json).

## Auth

Every request to `/events` and `/factors` carries:

```
Authorization: Bearer <token>
```

Compared in constant time. A missing or wrong token is `401` with
`error.code = "unauthorized"`. `/healthz` needs no auth.

## The one rule that matters most

**If the detector does not answer, deny the call.**

Recommended client timeout is 800 ms (`CLIENT_TIMEOUT_MS`). On timeout,
transport error, 5xx or 4xx, treat the result as `deny`. Never allow a call
through because verification failed to respond. `shared/client.py` implements
this and is tested for it; import it or copy the handling.

## POST /events

Push one call event, get the current decision back.

### Request

```json
{
  "session_id": "sess-7f3a9c21b0e4d8a5",
  "event": "speech_segment",
  "sequence": 3,
  "occurred_at": "2026-11-04T21:00:00+00:00",
  "claimed_user_id": "real_cfo",
  "audio": {
    "track_id": "TR_abc123",
    "start_ms": 1200,
    "end_ms": 4800,
    "sample_rate_hz": 16000,
    "encoding": "pcm_s16le",
    "uri": null
  },
  "otp_code": null,
  "metadata": {}
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `session_id` | string, 8 to 128 chars | yes | One per call. You generate it. |
| `event` | event type enum | yes | See the table below. |
| `sequence` | integer >= 0 | yes | Must strictly increase per session. |
| `occurred_at` | RFC 3339 datetime | yes | When it happened on your side. |
| `claimed_user_id` | string, max 128 | no | Who the caller says they are. Also selects a mock scenario. |
| `audio` | AudioRef | no | A span of call audio, referenced not inlined. |
| `otp_code` | string, 4 to 10 chars | no | Only on `otp_submit`. |
| `metadata` | object | no | Free-form passthrough. |

Unknown fields are **rejected** with `422`. That is deliberate, so a typo fails
loudly instead of being silently ignored.

### Event types

| `event` | Meaning |
|---|---|
| `session_start` | Call connected. Send this first. |
| `caller_claim` | The caller stated an identity. |
| `speech_segment` | A span of speech is available to analyse. |
| `otp_request` | You want a step-up challenge issued or reissued. |
| `otp_submit` | The caller gave a code. Put it in `otp_code`. |
| `session_end` | Call finished. |

Two `speech_segment` events are needed before a decision is reached. Before
that you get `pending`.

### Response

```json
{
  "contract_version": "0.1.0-provisional",
  "session_id": "sess-7f3a9c21b0e4d8a5",
  "state": "challenging",
  "decision": "challenge",
  "risk_score": 0.4365,
  "confidence": 0.86,
  "factors": [
    {
      "name": "synthetic_speech",
      "score": 0.08,
      "weight": 0.25,
      "verdict": "fail",
      "detail": "Vocoder phase artefacts and unnaturally flat spectral tilt."
    }
  ],
  "next_action": "request_otp",
  "reason": "Voice matches but the audio shows synthesis artefacts. Step-up required.",
  "scenario": "cloned_voice",
  "sequence": 3,
  "otp": {
    "code_length": 6,
    "attempts_used": 0,
    "attempts_remaining": 3,
    "expires_at": "2026-11-04T21:02:00+00:00"
  },
  "latency_ms": 212,
  "idempotent_replay": false
}
```

**Act on `next_action`.** It is there so you never have to infer behaviour from
the score.

| `next_action` | Do this |
|---|---|
| `continue_listening` | Keep the call going, keep sending speech. |
| `request_otp` | Ask the caller for their code, send `otp_submit`. |
| `complete_allow` | Verification passed. Proceed. |
| `complete_deny` | Refuse and hang up. |

`decision` is one of `pending`, `challenge`, `allow`, `deny`. `state` is one of
`created`, `listening`, `challenging`, `allowed`, `denied`, `expired`; the last
three are terminal. `risk_score` runs 0.0 safest to 1.0 riskiest. `scenario` is
populated in mock mode only.

## GET /factors/{session_id}

The same decision payload, plus `created_at`, `updated_at` and `events_seen`.
Use it for debugging and for showing an operator why a call was refused. It
never advances the state machine.

`404` with `session_not_found` if the id is unknown.

## State machine

```
created ──session_start──> listening
listening ──speech_segment x2──> allow ──> allowed      (terminal)
                               ├─> deny  ──> denied     (terminal)
                               └─> challenge ──> challenging
challenging ──otp_submit, correct──> allowed             (terminal)
challenging ──otp_submit, 3 wrong──> denied              (terminal)
any non-terminal ──session_end──> denied                 (terminal, fails closed)
any non-terminal ──TTL exceeded──> expired               (terminal, fails closed)
```

Once a session is terminal, any event other than `session_end` gets `409`
`session_terminal`. A call that just stops, without a decision, is denied.

That includes expiry: once a session passes its TTL, `/events` answers `409`
`session_terminal` rather than reporting the expiry in a 200 body. To see why a
session ended, read `/factors`, which reports `state: "expired"` and
`decision: "deny"`.

## Idempotency and ordering

Retries are safe if the payload is byte-identical. Send `Idempotency-Key` to
make the intent explicit, otherwise the key defaults to the sequence number.

| Situation | Result |
|---|---|
| Same key, identical payload | `200`, the original response, `idempotent_replay: true` |
| Same key, different payload | `409` `sequence_conflict` |
| `sequence` not greater than the last accepted | `409` `sequence_conflict` |

## OTP

Three attempts, 120 second expiry, six digits. In mock mode the code is
`424242`. The expected code is never returned in any response.

A wrong code with attempts left keeps `decision: "challenge"` and decrements
`otp.attempts_remaining`. The third wrong code is terminal `deny`.

**A correct code is not a guaranteed `allow`.** It moves the risk score; it
does not override policy. A correct code returns `deny` when any of these
holds, so do not treat `otp_submit` as an approval step:

- a fatal factor has tripped (a detected replay, an automated caller)
- the scenario is one that cannot be rescued by step-up
- no voice analysis has happened yet, so there is nothing for the code to
  corroborate
- the blended risk is still above the allow threshold after the code is
  credited

That last case is the normal one to design for: the code is one factor among
several, weighted at 0.40, not a master key. Always act on `next_action` and
`decision`, never on the fact that a code was accepted.

Relatedly, sending `otp_request` before any speech no longer manufactures a
usable gate. Voice analysis continues while a challenge is open, so a fatal
signal arriving then ends the session, and the next event gets `409`
`session_terminal`.

Two scenarios cannot be rescued by OTP at all: `replay` and `ai_agent` are
denied outright and no challenge is issued. Check `otp` for `null` rather than
assuming a challenge is always available.

## Errors

Every non-2xx body looks like this:

```json
{
  "contract_version": "0.1.0-provisional",
  "error": {
    "code": "sequence_conflict",
    "message": "sequence 3 is not greater than the last accepted sequence 5",
    "detail": null
  }
}
```

| `error.code` | HTTP | Cause |
|---|---|---|
| `unauthorized` | 401 | Missing or wrong bearer token |
| `invalid_scenario` | 400 | `X-Callback-Scenario` names something unknown |
| `validation_error` | 422 | Body violates the contract, unknown field included |
| `session_not_found` | 404 | Unknown `session_id` on `/factors` |
| `session_terminal` | 409 | Event sent to a finished session, or `otp_submit` with no challenge |
| `sequence_conflict` | 409 | Replayed key with a changed payload, or a non-increasing sequence |
| `otp_expired` | 410 | Challenge expired before submission |
| `otp_exhausted` | 429 | Defined but not currently emitted, see below |

One wrinkle worth knowing, since it is not what you would guess. Submitting a
fourth wrong code returns `409 session_terminal`, not `429 otp_exhausted`,
because the third wrong code already moved the session to `denied` and the
terminal guard runs first. `otp_exhausted` stays in the enum as a reserved code
for when OTP state outlives the session. Both behaviours are pinned by tests,
so do not code against `429` here.

## Driving the five scenarios

Either set the header:

```
X-Callback-Scenario: cloned_voice
```

Or set `claimed_user_id` to the scenario name, which needs no custom header.
The header wins when both are present. With neither, you get `real_cfo`.

Valid values: `real_cfo`, `cloned_voice`, `replay`, `ai_agent`,
`human_impostor`.

Mock mode adds 150 to 400 ms of latency per request on purpose, so your timeout
handling gets exercised rather than assumed.

## Running the fixtures against your own deployment

The contract suite can point at a URL instead of running the app in process, so
you do not need this repository's Python environment to match yours:

```bash
CALLBACK_BASE_URL=https://the-mock-url CALLBACK_API_TOKEN=the-token uv run pytest tests/contract -v
```

That covers all five scenarios, both OTP outcomes, auth, idempotency,
sequencing and the client-side timeout rule. If those pass against your
deployment, the two halves agree.
