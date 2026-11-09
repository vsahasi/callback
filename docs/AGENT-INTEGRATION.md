# Integrating the voice agent with the detector

> This describes the detector's internal `/events` dev API. The agent contract is
> `callback_shared` 1.2.0 (`shared/src/callback_shared/contract.py`), served under
> `/v1`; see `docs/DETECTOR.md`.
>
> **1.2.0 adds one optional field and asks one thing of the agent.** `Turn.audio`
> is a `ClipRef` pointing at the audio of that turn. It is what makes a `/v1`
> voiceprint enrollment possible at all, since nothing else on the wire carries
> audio and the hidden listener is not in Round 2. A minor bump: the version
> check compares majors, so sending no clip keeps working exactly as before and
> simply enrols nothing. See "Enrolling a voice" below.

For Veer. Copy-paste oriented. This is the "how do I use it" guide;
[`docs/CONTRACT.md`](CONTRACT.md) is the "what does the wire look like"
reference. Where they overlap, CONTRACT.md wins on field names and types, and
this file wins on sequencing and error handling.

Contract version `0.1.0-provisional`. CONTRACT.md has two open questions for
you at the top (decision vocabulary, and who owns the texted code). Neither
blocks you from building against this today.

---

## The rule that outranks everything else

**If you do not have a decision, you have a DENY.**

This is a verification service. A detector that is slow, unreachable, broken,
misconfigured or returning nonsense must stop the call, never let it through.
Timeout, connection refused, 500, 401, 409, a 200 whose body does not parse:
all of them are `deny`.

`shared/client.py` does this on every path and the tests in
`tests/contract/test_client_kit.py` pin it. If you import the client you get
the guarantee for free, and the only way to lose it is to write a branch that
overrides it. Do not write that branch.

```python
outcome = await call.submit_speech(span)

if outcome.failed_closed:
    # The detector did not answer. This is an outage, not an acquittal.
    await hang_up()
    return
```

There is no `allow_on_error` flag and there will not be one.

---

## Quickstart

### 1. Run a detector

```bash
cp .env.example .env            # set CALLBACK_API_TOKEN
CALLBACK_API_TOKEN=kit-test uv run uvicorn detector.app:app --port 8021
```

Mock mode is on by default and injects 150 to 400 ms of latency per request on
purpose, so your timeout handling gets exercised instead of assumed.

### 2. See what it does

```bash
uv run python examples/drive_mock.py --base-url http://127.0.0.1:8021 --token kit-test
```

```
scenario        decision  state    risk   factors                                          otp                      verdict
--------------  --------  -------  -----  -----------------------------------------------  -----------------------  -------
real_cfo        allow     allowed  0.045  5 factors, all pass                              not required             ok
cloned_voice    allow     allowed  0.262  6 factors, fail: synthetic+liveness              correct code accepted    ok
replay          deny      denied   0.386  5 factors, fail: replay+liveness                 not offered (hard deny)  ok
ai_agent        deny      denied   0.690  5 factors, fail: voice+synthetic+agent+liveness  not offered (hard deny)  ok
human_impostor  allow     allowed  0.199  6 factors, fail: voice                           correct code accepted    ok
```

It grades itself against the documented behaviour and exits non-zero on a
mismatch, so it works as a smoke test too. Useful flags:

| Flag | Effect |
|---|---|
| `--wrong-otp` | Fail every step-up instead of passing it |
| `--detail` | Full factor breakdown under the table |
| `--scenario replay` | One scenario, repeatable |
| `--show-fail-closed` | Also drive a session against a detector that is not there |
| `--no-retry` | One attempt per event |

### 3. Copy the integration

[`examples/agent_skeleton.py`](../examples/agent_skeleton.py) is a commented
sketch of where your agent calls the detector. LiveKit is stubbed out, so it
runs as is:

```bash
uv run python examples/agent_skeleton.py cloned_voice --token kit-test
uv run python examples/agent_skeleton.py --detector-down     # the fail-closed path
```

Replace the `FakeCall` methods with the LiveKit equivalents. The
`verify_caller` function above them stays as it is.

---

## Auth

Bearer token on `/events` and `/factors`. `/healthz` needs none.

```
Authorization: Bearer <CALLBACK_API_TOKEN>
```

Compared in constant time. Wrong or missing is `401 unauthorized`, which the
client turns into a fail-closed deny rather than an exception, so a bad token
shows up as a denied call and not as a crash mid-conversation.

### Preflight: two checks, not one

`/healthz` deliberately carries **no auth**, so a platform liveness probe
needs no secret. The consequence catches people out: a 200 from `/healthz`
says nothing at all about your token. A completely wrong token still gets
`{"status":"ok"}`, and you find out at the first real event, in front of a
vendor.

So run both:

```python
async with DetectorClient(base_url=..., token=...) as detector:
    if await detector.health() is None:
        raise SystemExit("detector unreachable, or not the detector")

    match await detector.verify_credentials():
        case False:
            raise SystemExit("detector rejected the token")
        case None:
            raise SystemExit("could not check the token; detector is unwell")
        case True:
            pass  # good to take calls
```

`verify_credentials()` reads `/factors` for a session id that cannot exist, so
it is side-effect free: the auth dependency runs before the handler, and a
good token gets `404 session_not_found` while a bad one gets `401`. It returns
`True` / `False` / `None`, where `None` means the detector could not be
reached to ask.

---

## The client, in two layers

```python
from shared.client import CallSession, DetectorClient, RetryPolicy
```

**`DetectorClient`** is the transport. One method per endpoint. You own the
sequence numbers and the idempotency keys.

**`CallSession`** owns one call and is the one you want. It handles:

- monotonic `sequence` numbering, starting at 1
- a deterministic `Idempotency-Key` per logical event
- one method per step of the real flow
- a terminal guard, so a finished session cannot be driven further
- bounded retry, inside the latency budget

```python
async with DetectorClient(base_url=BASE_URL, token=TOKEN) as detector:
    call = CallSession(detector, claimed_user_id="vendor-northwind-7781")

    outcome = await call.start()
    outcome = await call.claim_identity("vendor-northwind-7781")

    while outcome.should_continue_listening:
        outcome = await call.submit_speech(next_audio_span())

    if outcome.should_request_otp:
        outcome = await call.submit_otp(await ask_caller_for_code())

    if outcome.is_allow:
        release_hold_for_human_approval()
    else:
        refuse(outcome.reason)

    await call.end_call()
```

### Every method returns an `Outcome`

`Outcome.decision` is always populated. These properties read the same whether
the detector answered or not, which is the point: one branch, not two.

| Property | Meaning |
|---|---|
| `decision` | `pending` / `challenge` / `allow` / `deny` |
| `next_action` | What to do. `complete_deny` when we failed closed |
| `state` | Server session state. `denied` when we failed closed |
| `is_allow` / `is_deny` | Shorthand |
| `should_continue_listening` | Keep the call going, send more speech |
| `should_request_otp` | Ask for the code. **False on a hard deny** |
| `is_final` | Nothing further can change the answer |
| `otp` | The live challenge, or `None`. `None` is normal |
| `risk_score` | Diagnostics only. Never branch on it |
| `reason` | Human-readable, for the operator screen |
| `failed_closed` | True when the detector did not answer |
| `failure_reason` | Why, when it did not |
| `status_code` / `error_code` | The HTTP status and the contract error code |
| `attempts` | HTTP attempts this event cost, retries included |

### CallSession methods

| Method | Event | Notes |
|---|---|---|
| `start()` | `session_start` | First, when the call connects |
| `claim_identity(id)` | `caller_claim` | Sticks for the session |
| `submit_speech(audio=None)` | `speech_segment` | Two needed before a decision |
| `request_otp()` | `otp_request` | Rarely needed, see below |
| `submit_otp(code)` | `otp_submit` | Three attempts |
| `end_call()` | `session_end` | Always. Idempotent on the client side |
| `factors()` | `GET /factors` | Read-only, for the operator screen |
| `run_speech_until_decision()` | several | Bounded loop, fails closed if it runs out |

Properties: `session_id`, `sequence`, `closed`, `decision`, `last_outcome`.

On `DetectorClient` directly: `push_event()`, `get_factors()`, `health()`,
`verify_credentials()`.

### None of them raise for a detector problem

`push_event()` and `get_factors()` return a fail-closed outcome for every
failure mode, including the ones that are easy to miss:

- a timeout, a refused connection, a reset mid-body
- any 5xx, any 4xx
- a 200 whose body is not JSON, or is JSON of the wrong shape
- a body that will not **decode** at all, for example a proxy that sets
  `Content-Encoding: gzip` on something it did not gzip. httpx raises
  `DecodingError` for this, and it is a sibling of `TransportError` rather
  than a subclass, so catching only transport errors lets it escape
- too many redirects, for the same reason
- anything else, via a final blind catch, so a future httpx exception class
  cannot quietly turn a deny into a raise

That last point matters more than it looks. An exception is not a deny: your
`if outcome.is_allow: ... else: hang_up()` never runs, and an agent with a
broad `except` around its turn handler would carry on with the call. Every one
of these is pinned by a test in `tests/contract/test_client_kit.py`.

---

## The event flow, in order

```
1. session_start     call connected
2. caller_claim      the vendor said who they are
3. speech_segment    x2 minimum, until next_action stops saying continue_listening
4. otp_request       only if you need a challenge reissued
5. otp_submit        the code they read back, up to 3 attempts
6. session_end       always, including after a deny
```

Rules the client enforces for you, and that you have to enforce yourself if
you hand-roll it:

- **`sequence` must strictly increase per session.** It is per session, not
  global, and it does not have to be gap-free. A non-increasing sequence is
  `409 sequence_conflict`.
- **A retry must reuse its idempotency key and resend an identical payload.**
  Same key plus identical bytes is a safe replay (`idempotent_replay: true`).
  Same key plus a changed payload is `409 sequence_conflict`. The payload
  includes `occurred_at`, so a retry that regenerates the timestamp is a
  conflict, not a replay. This is the single easiest way to break idempotency
  by accident.
- **`session_end` is the one event a terminal session still accepts.** Send it
  anyway. A session that just stops without a decision is denied server-side,
  and a session that passes its TTL expires to `deny`.

### Two speech segments

The detector will not commit until it has heard enough. Before then you get
`decision: pending` and `next_action: continue_listening`. Do not treat
`pending` as a soft allow, and do not put a deadline on it yourself: loop on
`should_continue_listening` with a bound, and treat running out as a deny.

```python
for _ in range(4):
    outcome = await call.submit_speech(span())
    if outcome.failed_closed or not outcome.should_continue_listening:
        break
else:
    # Never decided. That is a deny, not a retry.
    await call.end_call()
    return DENY
```

`call.run_speech_until_decision(max_segments=4)` does exactly this, including
failing closed when the bound is hit.

### Step-up

A `challenge` decision already carries a live `otp`, so you usually do not
need `otp_request` at all. Send it only to reissue an expired challenge.

Three attempts, 120 second expiry, six digits. In mock mode the code is
`424242`; it is never returned in any response. A wrong code with attempts
left comes back as another `challenge` with `otp.attempts_remaining`
decremented. The third wrong code is a terminal `deny`.

---

## The trap: `otp` can be `null`

`replay` and `ai_agent` are refused outright and **never issue a challenge**.
They come back as:

```json
{ "decision": "deny", "next_action": "complete_deny", "otp": null }
```

So:

- do not assume step-up is available
- do not read `otp == null` as "no challenge *yet*"
- branch on `next_action`, which says `complete_deny` and is unambiguous

```python
# Wrong. Treats a hard deny as "keep waiting".
if outcome.otp is None:
    await asyncio.sleep(1)
    continue

# Right.
if outcome.should_request_otp:
    outcome = await call.submit_otp(code)
```

Asking for a challenge on a hard-denied session is `409 session_terminal`.
You cannot talk your way out of a detected replay.

---

## Act on `next_action`, not on `risk_score`

`next_action` exists so you never have to infer behaviour from a float. The
thresholds are the detector's business and they will move as real models
replace the scripts; `next_action` will not.

| `next_action` | Do this |
|---|---|
| `continue_listening` | Keep the call going, keep sending speech |
| `request_otp` | Ask for the code, send `otp_submit` |
| `complete_allow` | Verification passed. Release the hold for human approval |
| `complete_deny` | Refuse and hang up. The hold stays |

`risk_score` and `factors` are for the operator screen and for explaining a
refusal after the fact. `risk_score` is not monotonic with severity across
scenarios either: `replay` denies at `0.386` while `human_impostor` reaches
`allow` at `0.199` only because a correct code rescued it, and a single fatal
factor denies regardless of the blend. Another reason not to threshold it
yourself.

---

## Retry policy

`CallSession` opts into bounded retry by default. `DetectorClient` on its own
does not, so nothing changed for existing callers.

```python
RetryPolicy(
    max_attempts=2,  # total, the first one included
    total_budget_ms=800,  # the whole logical event, retries and backoff
    backoff_initial_ms=25,  # doubling, capped at backoff_max_ms
    backoff_max_ms=100,
    jitter_ms=25,
    min_attempt_ms=100,  # do not start an attempt with less left than this
)
```

Three properties make it safe, and all three are tested:

1. **A retry resends the same idempotency key and byte-identical content.**
   The body, `occurred_at` included, is built once per logical event. So a
   retry can only land as a fresh application or as a replay of the first
   attempt, and the state machine sees the event exactly once.
   (`test_a_retry_replays_rather_than_re_driving_the_transition` proves this
   against the real app: it loses the first 200 in transit, retries, gets
   `idempotent_replay: true`, and `/factors` still reports `events_seen: 1`.)

2. **Only failures that carry no decision are retried.** Timeouts, transport
   errors, 5xx. Never a 4xx: those are deterministic answers and a retry would
   only hide a real 401 or 409. Never a malformed 200 either, since the cached
   reply would be just as unreadable. Note that `429` here means
   `otp_exhausted`, not rate limiting, so it is *not* retried.

3. **Retries live inside the latency budget, they do not extend it.** The
   whole event, backoff included, stays inside `total_budget_ms`. Per-attempt
   timeouts shrink to fit, and an attempt is skipped when too little budget
   remains to be worth it.

Rule 3 has a consequence worth stating: a read timeout usually consumes the
whole budget on its own, so **a timeout is usually not retried**. That is
correct. Out of time means deny now, not deny later, because there is a human
on the phone. Retries exist for the cheap failures, a refused connection or a
cold-start 503, which fail in milliseconds. You can watch both happen:

```
$ uv run python examples/drive_mock.py --show-fail-closed ...
  refused     2 attempt(s) inside the 800ms budget: detector unreachable: ConnectError
  blackholed  1 attempt(s) inside the 800ms budget: detector unreachable: ConnectTimeout
```

To turn retries off: `CallSession(detector, retry=RetryPolicy.none())`.

---

## Error codes and what to do about each

Every non-2xx body is:

```json
{
  "contract_version": "0.1.0-provisional",
  "error": { "code": "sequence_conflict", "message": "...", "detail": null }
}
```

The client surfaces this as `outcome.error_code` and `outcome.status_code`.
**All of these are already a deny**; the column says what to do beyond that.

| `error.code` | HTTP | Cause | What to do |
|---|---|---|---|
| `unauthorized` | 401 | Missing or wrong bearer token | Configuration bug. Page yourself, do not retry. `verify_credentials()` at start-up catches it; `health()` does **not** |
| `invalid_scenario` | 400 | `X-Callback-Scenario` names something unknown | Typo. Valid: `real_cfo`, `cloned_voice`, `replay`, `ai_agent`, `human_impostor` |
| `validation_error` | 422 | Body violates the contract, **including an unknown field** | Your payload is wrong. `extra="forbid"` is deliberate so a typo fails loudly |
| `session_not_found` | 404 | Unknown `session_id` on `/factors` | Wrong id, or the process holding it restarted. Sessions are in-memory |
| `session_terminal` | 409 | Event sent to a finished session, or `otp_submit` with no challenge | You drove a decided session. `CallSession`'s guard stops this client-side |
| `sequence_conflict` | 409 | Reused key with a changed payload, or a non-increasing sequence | Almost always a hand-rolled retry that regenerated `occurred_at`. Let `CallSession` own the keys |
| `otp_expired` | 410 | Challenge expired before submission | Send `otp_request` to reissue, if the session is still alive |
| `otp_exhausted` | 429 | Reserved. Not currently emitted | Nothing. See the wrinkle below |

### The 409-not-429 wrinkle

A fourth wrong code returns **`409 session_terminal`, not `429
otp_exhausted`**, because the third wrong code already moved the session to
`denied` and the terminal guard runs first. `otp_exhausted` stays in the enum
as a reserved code. Both behaviours are pinned by tests. Do not code against
`429` here.

With `CallSession` you will not see either: the guard raises `SessionClosed`
locally before the request is made.

### The terminal guard

```python
outcome = await call.submit_speech(span)  # -> allow, state allowed

call.closed  # True
await call.end_call()  # fine, the one event a terminal session accepts
await call.submit_speech()  # raises SessionClosed
```

`SessionClosed` is a programming error, not a detector failure, so it raises
rather than returning a decision. It carries `.outcome`, the last real
decision, so a caller that catches it still has something to act on. Check
`call.closed` if you would rather not catch it.

---

## curl

Set these once:

```bash
BASE=http://127.0.0.1:8021
TOKEN=kit-test
SID=sess-7f3a9c21b0e4d8a5
```

### Health, no auth

```bash
curl -s $BASE/healthz
# {"status":"ok","contract_version":"0.1.0-provisional","mock_mode":true}
```

### A full allow

```bash
curl -s -X POST $BASE/events \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "X-Callback-Scenario: real_cfo" \
  -H "Idempotency-Key: $SID:000001" \
  -d '{"session_id":"'$SID'","event":"session_start","sequence":1,
       "occurred_at":"2026-11-04T21:00:00+00:00","metadata":{}}'

# two speech segments, sequences 2 and 3
for SEQ in 2 3; do
  curl -s -X POST $BASE/events \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -H "X-Callback-Scenario: real_cfo" \
    -H "Idempotency-Key: $SID:00000$SEQ" \
    -d '{"session_id":"'$SID'","event":"speech_segment","sequence":'$SEQ',
         "occurred_at":"2026-11-04T21:00:0'$SEQ'+00:00","metadata":{}}'
  echo
done
# the second one: "decision":"allow","next_action":"complete_allow","state":"allowed"
```

### Step-up

```bash
SID=sess-clone0123456789
# ... session_start + 2 speech_segment with -H "X-Callback-Scenario: cloned_voice"
# the second returns "decision":"challenge","next_action":"request_otp",
#   "otp":{"code_length":6,"attempts_used":0,"attempts_remaining":3,...}

curl -s -X POST $BASE/events \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "X-Callback-Scenario: cloned_voice" \
  -d '{"session_id":"'$SID'","event":"otp_submit","sequence":4,"otp_code":"424242",
       "occurred_at":"2026-11-04T21:00:10+00:00","metadata":{}}'
# "decision":"allow","next_action":"complete_allow"
```

### Factors, for the operator screen

```bash
curl -s $BASE/factors/$SID -H "Authorization: Bearer $TOKEN"
# the same decision payload plus created_at, updated_at, events_seen
```

### The error cases

```bash
# 401 unauthorized
curl -s -o /dev/null -w '%{http_code} ' $BASE/events -X POST \
  -H "Content-Type: application/json" -d '{}'
curl -s $BASE/events -X POST -H "Authorization: Bearer wrong" \
  -H "Content-Type: application/json" -d '{}'
# {"contract_version":"...","error":{"code":"unauthorized",...}}

# 422 validation_error: unknown field
curl -s -X POST $BASE/events \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"session_id":"sess-typo0123456789","event":"session_start","sequence":1,
       "occurred_at":"2026-11-04T21:00:00+00:00","claimed_user_idd":"typo"}'

# 400 invalid_scenario
curl -s -X POST $BASE/events \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -H "X-Callback-Scenario: not_a_real_scenario" \
  -d '{"session_id":"sess-bad01234567890","event":"session_start","sequence":1,
       "occurred_at":"2026-11-04T21:00:00+00:00","metadata":{}}'

# 409 sequence_conflict: same key, different payload
KEY=demo-conflict-1
curl -s -X POST $BASE/events -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -H "Idempotency-Key: $KEY" \
  -d '{"session_id":"sess-conflict000001","event":"session_start","sequence":1,
       "occurred_at":"2026-11-04T21:00:00+00:00","metadata":{}}' > /dev/null
curl -s -X POST $BASE/events -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -H "Idempotency-Key: $KEY" \
  -d '{"session_id":"sess-conflict000001","event":"speech_segment","sequence":2,
       "occurred_at":"2026-11-04T21:00:01+00:00","metadata":{}}'
# {"error":{"code":"sequence_conflict",...}}

# 200 idempotent_replay: same key, identical payload
curl -s -X POST $BASE/events -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -H "Idempotency-Key: $KEY" \
  -d '{"session_id":"sess-conflict000001","event":"session_start","sequence":1,
       "occurred_at":"2026-11-04T21:00:00+00:00","metadata":{}}'
# "idempotent_replay":true

# 409 session_terminal: drive a decided session
# (after the real_cfo allow above)
curl -s -X POST $BASE/events -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -H "X-Callback-Scenario: real_cfo" \
  -d '{"session_id":"'$SID'","event":"speech_segment","sequence":9,
       "occurred_at":"2026-11-04T21:00:20+00:00","metadata":{}}'

# 404 session_not_found
curl -s $BASE/factors/sess-does-not-exist-0 -H "Authorization: Bearer $TOKEN"
```

### Driving a scenario without the custom header

Set `claimed_user_id` to the scenario name. The header wins when both are
present; with neither you get `real_cfo`.

```bash
curl -s -X POST $BASE/events -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"sess-noheader0000001","event":"session_start","sequence":1,
       "claimed_user_id":"ai_agent",
       "occurred_at":"2026-11-04T21:00:00+00:00","metadata":{}}'
```

---

## Checking your agent against the real thing

The contract suite can point at a URL instead of running the app in process,
so your Python environment does not have to match this repository's:

```bash
CALLBACK_BASE_URL=https://the-mock-url \
CALLBACK_API_TOKEN=the-token \
uv run pytest tests/contract -v
```

That covers all five scenarios, both OTP outcomes, auth, idempotency,
sequencing and the client-side rules. If it passes against your deployment,
the two halves agree.

`examples/drive_mock.py` is the lighter version of the same check and needs
nothing but the two flags.

---

## Things that will bite you

1. **`pending` is not a soft allow.** It means keep listening.
2. **`otp` is `null` on a hard deny.** Branch on `next_action`.
3. **A hand-rolled retry that regenerates `occurred_at`** turns a safe replay
   into `409 sequence_conflict`. Let `CallSession` own the keys.
4. **Unknown fields are rejected with 422.** Deliberate, so a typo fails
   loudly. Put anything extra in `metadata`.
5. **A fourth wrong code is `409`, not `429`.**
6. **Sessions are in-memory.** One process owns them today
   (`min_containers=1, max_containers=1`). A restart means `404` on
   `/factors` and a `sequence_conflict`-free fresh start; it does not mean a
   session survived.
7. **Mock mode adds 150 to 400 ms per request** on purpose. Do not tune your
   timeout to the mock.
8. **`risk_score` is not comparable across scenarios.** It is for the
   operator screen.
9. **A 200 from `/healthz` does not mean your token works.** `/healthz` has
   no auth. Use `verify_credentials()`.
10. **A decision you did not receive is a DENY.** Still the only rule that
    matters.


---

## Enrolling a voice (contract 1.2.0)

This is the one new thing asked of the agent, and it is optional.

`voice_match` is the heaviest signal in the detector's model, 0.35 of the
weight. It compares the caller against a stored reference, and with no
reference it does not run at all, which is how a different person reading an
attacker's script used to be allowed. Giving it a reference needs the
caller's audio, and in 1.1.0 nothing on the `/v1` wire carried any: the design
was a hidden listener in the room, and that is not in Round 2.

So `Turn` gained an optional `audio`:

```json
POST /v1/sessions/{session_id}/turns
{
  "role": "user",
  "text": "four seven two",
  "start_ms": 1000400,
  "end_ms": 1003400,
  "words": [],
  "interrupted": false,
  "audio": {
    "uri": "https://your-bucket.example/sessions/s1/turn-42.pcm",
    "sample_rate_hz": 8000,
    "encoding": "pcm_s16le"
  }
}
```

| Field | Notes |
|---|---|
| `uri` | An https url, at most 8192 characters. The host must be named in the detector's `CALLBACK_AUDIO_ALLOWED_HOSTS` or the fetch is refused. |
| `sample_rate_hz` | 8000 to 48000. |
| `encoding` | `pcm_s16le` or `flac`. A WAV container is detected from its magic bytes, so `pcm_s16le` is right for a `.wav` too. |

Send it or do not. Omitting it is exactly the old behaviour, and the version
check compares major versions only, so a 1.1.0 agent is served unchanged.

### What the detector does with it

It fetches the clip once, reduces it to a voice print (a median pitch and a
spectral envelope, a few hundred numbers), and drops the audio. It stores no
recording, ever. The print lands under `{user_id}:{channel}`, one per channel,
because a handset and a browser microphone put a different response in front
of the same voice.

### When it matters

Only on a user turn that answers an open `enroll` challenge. That is the
whole of it:

* a clip on an **agent** turn is ignored;
* a clip on a **`digits`, `phrase` or `liveness`** challenge is ignored, and
  deliberately not even measured. Those challenges ask what was said and by
  whom. If uploading audio passed one, anyone with the service token could
  clear a step-up with any recording, so the detector will not do it. An
  `enroll` prompt is different: it records a voice rather than verifying one;
* a clip with no open enrollment is ignored. An enrollment needs consent and
  a verified code first, which has not changed.

So in practice: run the enrollment flow as before, and put `audio` on the
user turn that answers each of the seven `enroll` prompts. Five that the
detector can measure, and `complete` answers `enrolled` with a print stored.

### What it costs you if the clip is bad

Nothing that denies a call. An unreachable uri, a host that is not
allowlisted, a clip with no speech in it: the prompt resolves `failed` with
reason `no_audio`, which the enrollment counts as a failed try, and a
`no_audio` failure is never a voice fail. You get the same three tries per
prompt you already had.

One timing note. The detector fetches the clip before answering your 204, and
gives the fetch 400 ms. A challenge resolves 700 ms after the turn's receive
time, and a prompt that has already been read as `failed` must never later
read as `passed`, so a fetch that outruns that window is dropped. Keep the
clips small, and keep the bucket near the detector.
