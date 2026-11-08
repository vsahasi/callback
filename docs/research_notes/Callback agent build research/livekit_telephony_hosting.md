# LiveKit telephony, Cloud free-tier limits, and agent hosting for "Callback" (as of 2026-11-04)

Scope note: every docs.livekit.io page below was fetched on 2026-11-05 UTC (the pages carry no "last updated" date; the render timestamp is the only date). Prices are from livekit.com/pricing on the same day. Items marked **(unverified)** or **(from memory)** were not confirmed by a fetched source and should be checked with `lk <cmd> --help` or in the dashboard. Items marked **(search summary only)** come from a search engine's summary of a page I could not fetch directly.

Quick orientation for Veer:
- LiveKit Phone Numbers are **inbound-only** today ("Support for outbound calls is coming soon"), so the "callback to the number on file" step-up needs a third-party trunk (Telnyx or Twilio) no matter what.
- The Build (free) plan's **50 US-local inbound minutes/month is a hard cap** (requests fail, no overage billing) and is shared across all of a user's free projects. A Telnyx number instead burns the separate **1,000 third-party SIP minutes** allowance and gives HD voice (G.722) plus outbound.
- HD voice in LiveKit = **G.722 (16 kHz, 7 kHz audio bandwidth)**; Telnyx is the only supported SIP provider for it, and LiveKit's own numbers support it "out of the box". Whether a given Verizon/AT&T/T-Mobile caller actually gets G.722 is carrier-dependent and **nothing in LiveKit exposes the negotiated codec**; plan to measure it.
- Free-plan cloud agents **scale to zero and cold-start in 10-20 s**; paid plans keep them warm. One production deployment on Build.

---

## KQ1. LiveKit Phone Numbers: getting one, what is free, minutes, price, outbound

### Takeaway
Every LiveKit Cloud plan includes 1 free US local number; Build adds 50 free US-local inbound minutes per month as a hard cap. Numbers are US-only and inbound-only as of Nov 2026, so they cannot place the callback.

### Cited Findings
- "LiveKit Phone Numbers currently only supports inbound calling. Support for outbound calls is coming soon." Numbers are available only in the US; other countries planned. — [LiveKit Phone Numbers docs](https://docs.livekit.io/telephony/start/phone-numbers/)
- All plans include 1 free US local number; "The Build plan also includes 50 free inbound minutes." Releasing a number early still bills the full month. — [LiveKit Phone Numbers docs](https://docs.livekit.io/telephony/start/phone-numbers/)
- Pricing table (telephony rows): US local number: Build 1 free; Ship/Scale 1 free then $1.00/month per number. US local inbound minutes: Build 50 included (no overage listed); Ship 100 included then $0.01/min; Scale 1,000 included then $0.01/min. US toll-free number: not on Build; $2.00/month on Ship/Scale; toll-free inbound $0.02/min. Third-party SIP minutes (inbound and outbound through your own Twilio/Telnyx trunk): Build 1,000 included; Ship 5,000 then $0.004/min; Scale 50,000 then $0.003/min. — [livekit.com/pricing](https://livekit.com/pricing)
- Quotas page: Build = 1 active US local number, 0 toll-free, 50 US local inbound minutes; Ship up to 100 local numbers; Scale up to 300 (increases only for Scale). — [Quotas and limits](https://docs.livekit.io/deploy/admin/quotas-and-limits/)
- Build-plan allowances are hard caps: "After you exceed it, new requests fail rather than incurring overage charges." Free allowances and limits are shared across all of a user's free projects. Allowances reset on the first of each calendar month and do not roll over. — [Quotas and limits](https://docs.livekit.io/deploy/admin/quotas-and-limits/)
- Dashboard purchase flow: Telephony → Phone Numbers → Rent a number → search by area code (toll-free via 800/833/888 area codes) → Rent → Confirm rental. — [LiveKit Phone Numbers docs](https://docs.livekit.io/telephony/start/phone-numbers/)
- CLI purchase flow:
  ```
  lk number search --country-code US --area-code 415
  lk number purchase --numbers +14155550100
  lk number list
  lk number update --id <PHONE_NUMBER_ID> --sip-dispatch-rule-id <DISPATCH_RULE_ID>
  ```
  (commands are under `lk number`, not `lk sip number`). — [LiveKit Phone Numbers docs](https://docs.livekit.io/telephony/start/phone-numbers/)
- Attaching to an agent: assign the number to a dispatch rule (dashboard: number's ⋮ menu → Assign dispatch rule → Save, or the `lk number update` command above). LiveKit recommends explicit dispatch for inbound-call agents. — [LiveKit Phone Numbers docs](https://docs.livekit.io/telephony/start/phone-numbers/)
- "If you're using LiveKit Phone Numbers, you **do not** need to create an inbound trunk." — [Inbound trunk docs](https://docs.livekit.io/telephony/accepting-calls/inbound-trunk/)
- Limitation: forwarding with `TransferSIPParticipant` is not yet supported on LiveKit Phone Numbers. — [LiveKit Phone Numbers docs](https://docs.livekit.io/telephony/start/phone-numbers/)
- Launch post dated 2025-11-18: US local and toll-free only; outbound calling and international numbers listed as roadmap; LiveKit "connects directly to upstream carriers" but does not name the carrier. — [Introducing LiveKit Phone Numbers](https://livekit.com/blog/introducing-livekit-phone-numbers-zero-to-ringing-in-60-seconds)
- The telephony testing page repeats that outbound "requires a third-party SIP provider, since LiveKit Phone Numbers don't support outbound calling." — [Telephony testing](https://docs.livekit.io/telephony/testing/)

### Inferences
- The team's "50 inbound minutes" belief is correct for Build, but 50 minutes per month across all free projects is very little for a week of integration testing plus a live demo; budget it (e.g. keep test calls under 60 s) or route test traffic through a Telnyx number, which draws on the separate 1,000-minute third-party SIP allowance.
- No KYC/address requirements are mentioned anywhere for LiveKit numbers; the purchase appears to be one click. (The page does not discuss verification, so absence is not proof.)

### Gaps
- Overage price for US-local inbound minutes on Build: none listed; the hard-cap rule implies calls fail at 50 minutes, but the docs do not say what the caller hears.
- Underlying carrier behind LiveKit Phone Numbers: not disclosed in docs or blog.
- Exact outbound ETA: "coming soon" with no date.

---

## KQ2. HD voice: what it is, which paths deliver it, what a US mobile caller gets, how to see the codec

### Takeaway
"HD voice" in LiveKit means G.722 wideband (16 kHz sampling, ~7 kHz audio) negotiated on the SIP leg; it is supported with LiveKit Phone Numbers and with Telnyx, and with no other SIP provider. Whether a Verizon/AT&T/T-Mobile VoLTE caller lands on G.722 depends on carrier interconnects that neither LiveKit nor Telnyx guarantees, and LiveKit exposes no participant attribute or dashboard field for the negotiated codec.

### Cited Findings
- HD voice page: wideband audio at a higher sample rate than PSTN's 8 kHz; "LiveKit SIP supports wideband codecs such as G.722 out of the box"; "Currently, Telnyx is the only SIP provider that supports HD voice"; LiveKit Phone Numbers also support HD voice out of the box. No mention of Opus, EVS or AMR-WB on the SIP leg. — [HD voice](https://docs.livekit.io/telephony/features/hd-voice/)
- Codec reference: supported SIP codecs are PCMU (G.711 µ-law, 8 kHz), PCMA (G.711 A-law, 8 kHz) and G.722 ("delivers 16 kHz audio (HD voice) at a bitrate similar to G.711"; its SDP rate shows 8000 as a known SDP quirk). Default codec list is PCMU, PCMA, G722. The caller's INVITE lists codecs in preference order; the callee picks one in the 200 OK. — [Codecs negotiation](https://docs.livekit.io/reference/telephony/codecs-negotiation/)
- Codec configuration lives in a `media` (SIPMediaConfig) field on inbound trunks, outbound trunks, dispatch rules and `CreateSIPParticipant`: `codecs: [{name, rate?}]`, `only_listed_codecs: true` to restrict (recommended for UDP), unset to add to defaults (recommended for TCP/TLS). Example: `{"media": {"only_listed_codecs": true, "codecs": [{"name": "PCMU"}]}}`. — [Codecs negotiation](https://docs.livekit.io/reference/telephony/codecs-negotiation/)
- Telnyx HD-voice setup in LiveKit's Telnyx guide ("US customers only"): Telnyx portal → Manage Numbers → edit number → Voice tab → Services → enable HD Voice → save; then SIP Trunking → edit trunk → Inbound tab → Codecs → enable **G.722** and leave **G.711U** selected for compatibility. No LiveKit-side change is listed. — [Telnyx provider guide](https://docs.livekit.io/telephony/start/providers/telnyx/)
- Telnyx (2023-10-15 release note): HD Voice "Compatible with top US carriers including Verizon, AT&T, US Cellular, and T-Mobile"; wideband codec choices "OPUS, G.722, or AMR-WB"; at that time "$1/month on specific US numbers". — [Telnyx: Experience true HD voice](https://telnyx.com/release-notes/experience-true-hd-voice-with-telnyx)
- Telnyx says HD voice became automatically enabled on all US numbers at no additional cost from July 2025, and that HD Voice requires both ends (the SIP trunk and the caller's line) to support G.722. **(search summary only)** — [Telnyx: free HD voice](https://telnyx.com/release-notes/free-hd-voice); [Telnyx: HD voice Australia](https://telnyx.com/release-notes/hd-voice-australia-release)
- Telnyx (2026-04-14): "HD Voice for LiveKit agents on Telnyx" in Telnyx regions SFO3, NYC1, ATL1, SYD1: G.722 negotiated automatically when the SIP endpoint advertises it; Opus (full-band, up to 48 kHz) only when SRTP is enabled on the trunk. The note contrasts this with LiveKit Cloud, where HD voice "requires routing through a third-party SIP carrier." This describes Telnyx's hosted "LiveKit on Telnyx" product, not LiveKit Cloud. — [Telnyx: HD Voice for LiveKit](https://telnyx.com/release-notes/hd-voice-livekit)
- Informal forum view (quoting Telnyx support): wideband works on calls with T-Mobile, AT&T, and "a portion of Verizon Wireless customers"; "the PSTN is still where wideband audio goes to die." **(search summary only, informal source)** — [DSLReports thread](https://www.dslreports.com/forum/r33974548-Voip-and-Codec-G-722)
- No codec/media participant attribute exists: the SIP participant attribute list is `sip.callID`, `sip.callIDFull`, `sip.callStatus`, `sip.phoneNumber`, `sip.ruleID`, `sip.trunkID`, `sip.trunkPhoneNumber`, `sip.twilio.accountSid`, `sip.twilio.callSid`, plus custom attributes from `headers_to_attributes`. — [SIP participant reference](https://docs.livekit.io/reference/telephony/sip-participant/); the protocol's `sip.go` likewise defines no codec, media or sample-rate attribute — [livekit/protocol sip.go](https://github.com/livekit/protocol/blob/main/livekit/sip.go)
- The testing page describes "call logs in the LiveKit Cloud Telephony dashboard" and the Sessions page (join/leave events, published tracks, timing) but says nothing about negotiated codec or media parameters. — [Telephony testing](https://docs.livekit.io/telephony/testing/)
- Media path: "Media is bridged between the caller and LiveKit's internal media pipeline"; if no RTP arrives within 30 s (or 15 s after the first packet) the call drops with a `media timeout` error (`media_timeout` setting raises it). — [SIP primer](https://docs.livekit.io/reference/telephony/sip-primer/)
- The SIP bridge is built against libopus (plus opusfile, soxr). — [livekit/sip README](https://github.com/livekit/sip)

### Inferences
- Inside the room the SIP participant's track is ordinary WebRTC audio (Opus, as implied by the libopus dependency and WebRTC), so the detection service will always receive a "48 kHz-looking" stream; the real information content is capped at ~3.4 kHz (G.711) or ~7 kHz (G.722). The only practical way to know which codec a call negotiated, on LiveKit Cloud, is to measure spectral energy above ~4 kHz in the detection service (or read the provider's call-detail record if Telnyx exposes the codec; not verified).
- Expectation for the demo: with a Telnyx number with HD Voice on and G.722 enabled on the trunk, a modern VoLTE phone on AT&T or T-Mobile has a reasonable chance of negotiating G.722; Verizon is "partial" per the informal source; any call that traverses a legacy TDM interconnect falls back to G.711. Treat G.722 as opportunistic, not guaranteed, and test from each carrier before the final.
- The LiveKit Phone Number path claims HD voice "out of the box" but the upstream carrier is unnamed, so its real-world carrier interop is unknown; test it the same way.
- Keep `krisp_enabled` **off** on the inbound trunk/dispatch rule and on `CreateSIPParticipant` for Callback: trunk-level noise cancellation rewrites the caller's audio before the detection participant hears it.

### Gaps
- No LiveKit source states which codec LiveKit Phone Numbers use for HD voice (presumably G.722, the only wideband codec in the default list) or which carrier interconnects support it.
- No official Verizon/AT&T/T-Mobile documentation on wideband interworking to third-party SIP providers was found in this research; the carrier list rests on Telnyx marketing (2023) and one forum post.
- Whether the Telnyx call-detail record or LiveKit's Telephony call logs show the negotiated codec: not verified.
- Twilio's codec support with LiveKit: LiveKit says only Telnyx supports HD voice, so assume G.711 on Twilio, but Twilio's own codec list was not fetched.

---

## KQ3. Inbound setup end to end (trunk, dispatch rule, agent_name, caller identity, hangup)

### Takeaway
With a LiveKit number you skip the trunk entirely: create a dispatch rule with `room_config.agents[].agent_name`, assign the number to it, and run a worker whose `@server.rtc_session(agent_name=...)` matches. The caller arrives as a `PARTICIPANT_KIND_SIP` participant with `sip.phoneNumber`; hang up by deleting the room (`await ctx.delete_room()` or the prebuilt `EndCallTool`).

### Cited Findings
- Inbound trunk (only needed for Twilio/Telnyx numbers, not LiveKit numbers). CLI file `inbound-trunk.json` then `lk sip inbound create inbound-trunk.json`:
  ```json
  {"trunk": {"name": "My trunk", "numbers": ["+15105550100"]}}
  ```
  Optional `allowed_numbers` (caller allowlist). Dashboard: Telephony → SIP trunks → Create new trunk → JSON editor → Trunk direction Inbound → paste JSON **without** the `trunk` wrapper (`krisp_enabled` and `allowed_numbers` only via JSON editor). — [Inbound trunk](https://docs.livekit.io/telephony/accepting-calls/inbound-trunk/)
- Inbound-trunk security notes: `allowed_addresses` is **not enforced until LiveKit support enables it** for the project (the list is saved but the trunk accepts calls from any address); empty `numbers` requires username/password auth or `allowed_addresses`; Twilio Elastic SIP Trunking does not support username/password auth (TwiML path does); `ringing_timeout` rejects unanswered INVITEs with `486 Busy Here` by default (`ringing_timeout_status` to change). — [Inbound trunk](https://docs.livekit.io/telephony/accepting-calls/inbound-trunk/)
- Dispatch rule. CLI: `lk sip dispatch create dispatch-rule.json` with
  ```json
  {
    "dispatch_rule": {
      "rule": {"dispatch_rule_individual": {"room_prefix": "call-"}},
      "name": "My dispatch rule",
      "room_config": {
        "agents": [{"agent_name": "inbound-agent", "metadata": "job dispatch metadata"}]
      }
    }
  }
  ```
  `agents[]` entries take `agent_name` (required), `metadata` (optional string, read via `ctx.job.metadata`), `deployment` (optional; omitted = production). — [Dispatch rule](https://docs.livekit.io/telephony/accepting-calls/dispatch-rule/)
- Rule types: `dispatch_rule_individual` creates one room per caller named caller-number + random suffix with optional `room_prefix` (`no_randomness: true` drops the suffix; the docs warn these room names contain PII and are not redacted); `dispatch_rule_direct` sends everyone to one `room_name` (optionally `pin`-protected, caller enters `12345#`); `dispatch_rule_callee` names rooms after the dialed number (`randomize` default true). `trunk_ids` binds a rule to trunks (omit = all inbound trunks). `metadata` and `attributes` on the rule are inherited by SIP participants. — [Dispatch rule](https://docs.livekit.io/telephony/accepting-calls/dispatch-rule/)
- Rule management: `lk sip dispatch list`; `lk sip dispatch update --id <id> --name ... --callee number- --trunks <t1>,<t2>` (`--trunks` replaces the whole list); full replace with `lk sip dispatch update --id <id> dispatch-rule.json` (omitted fields are cleared). Delete not documented on that page. — [Dispatch rule](https://docs.livekit.io/telephony/accepting-calls/dispatch-rule/)
- Worker side: `@server.rtc_session(agent_name="test-agent")` on the entrypoint; "With `agent_name` set, the agent is only assigned to rooms when explicitly dispatched"; without it, the agent auto-joins every new room. SIP dispatch rules "can define one or more agents using the `room_config.agents` field." — [Agent dispatch](https://docs.livekit.io/agents/server/agent-dispatch/)
- Cloud deployments keep "the agent's dispatch name from the source code"; that name is what you use for explicit dispatch. — [Deploy quickstart](https://docs.livekit.io/deploy/agents/quickstart/)
- Caller in the room: identified by `Participant.Kind == SIP` (`rtc.ParticipantKind.PARTICIPANT_KIND_SIP` in Python). Attributes: `sip.phoneNumber` = caller's number (inbound) / dialed number (outbound), absent if `HidePhoneNumber` is set on the rule; `sip.trunkPhoneNumber` = the number dialed (inbound) / the from-number (outbound); `sip.callStatus` ∈ {`dialing`, `ringing`, `automation`, `active`, `hangup`} (`ringing` → `active` once the SIP participant subscribes to remote audio); `sip.callID`, `sip.callIDFull`, `sip.ruleID`, `sip.trunkID`; Twilio adds `sip.twilio.accountSid` / `sip.twilio.callSid`. Header-derived attributes (`headers_to_attributes`) arrive asynchronously; watch `participant_attributes_changed` or call the `lk.sip.GetRemoteHeaders` RPC. — [SIP participant reference](https://docs.livekit.io/reference/telephony/sip-participant/)
- Hangup: the outbound-calls page's Python example defines `hangup_call()` which obtains the job context via `get_job_context()` and calls `await ctx.delete_room()`; an `end_call` function tool calls it after `ctx.wait_for_playout()`; LiveKit also recommends the prebuilt `EndCallTool`. — [Outbound calls](https://docs.livekit.io/telephony/making-calls/outbound-calls/)
- Disconnect reasons on the SIP participant: `CLIENT_INITIATED` (either side hung up after connect; inbound BYE maps here; auto-closes `AgentSession`), `ROOM_DELETED` (auto-closes), `USER_REJECTED` (outbound, SIP 486/603, auto-closes), `USER_UNAVAILABLE` (outbound, 408/480, **not** auto-closed, call `ctx.shutdown()`), `SIP_TRUNK_FAILURE` (outbound, 5xx, not auto-closed). — [SIP participant reference](https://docs.livekit.io/reference/telephony/sip-participant/)
- Verification commands: `lk number list`, `lk sip inbound list`, `lk sip outbound list`, `lk sip dispatch list`, `lk room list`, `lk room participants list <ROOM_NAME>`, `lk room participants get --room <ROOM_NAME> <PARTICIPANT_ID>` (full SIP participant details incl. attributes). — [Telephony testing](https://docs.livekit.io/telephony/testing/)
- Global vs regional SIP endpoint: by default LiveKit uses a global SIP endpoint and routes to the nearest region; the regional form seen in the Telnyx guide is `<subdomain>.us.sip.livekit.cloud`; find the project's URI in LiveKit Cloud Project settings or generate it with the CLI. — [Inbound trunk](https://docs.livekit.io/telephony/accepting-calls/inbound-trunk/); [Telnyx provider guide](https://docs.livekit.io/telephony/start/providers/telnyx/); [Twilio provider guide](https://docs.livekit.io/telephony/start/providers/twilio/)

### Inferences
- Minimal LiveKit-number inbound path: (1) `lk number purchase`, (2) `lk sip dispatch create` with `dispatch_rule_individual` + `room_config.agents[{agent_name:"callback"}]`, (3) `lk number update --id ... --sip-dispatch-rule-id ...`, (4) worker with `@server.rtc_session(agent_name="callback")`. The hidden detection participant can be dispatched the same way by listing a second agent in `room_config.agents` (the field is documented as "one or more agents"), or by having the Callback agent create it via the API.
- For the "number on file" comparison, read `participant.attributes["sip.phoneNumber"]` from the SIP participant (`ctx.wait_for_participant(kind=SIP)` pattern), not the room name.

### Gaps
- Default SIP participant `identity`/`name` format is not stated on the reference page. **(from memory)** it is `sip_<E.164 number>` with name `Phone <number>`; confirm with `lk room participants list` on the first test call.
- `krisp_enabled`, `media_encryption`, `room_preset` on dispatch rules are referenced only via the SIP API reference, which I did not fetch.
- Whether two agents in one `room_config.agents` array both get dispatched from a free-plan project with only one cloud deployment (the second would have to be a self-hosted worker) was not tested.

---

## KQ4. Outbound calls (trunks, CreateSIPParticipant from Python, status, cost, trial limits)

### Takeaway
Outbound needs a Twilio or Telnyx outbound trunk (address + numbers + auth), then `lkapi.sip.create_sip_participant(CreateSIPParticipantRequest(sip_trunk_id, sip_call_to, room_name, participant_identity, wait_until_answered=True))`; failures surface as `api.SipCallError` with the SIP status code. LiveKit charges third-party SIP minutes (1,000 free on Build); the carrier charges its own per-minute rate.

### Cited Findings
- Outbound trunk. `outbound-trunk.json`:
  ```json
  {"trunk": {"name": "My outbound trunk", "address": "<my-trunk>.pstn.twilio.com", "numbers": ["+15105550100"]}}
  ```
  then `lk sip outbound create outbound-trunk.json --auth-user "$SIP_AUTH_USERNAME" --auth-pass "$SIP_AUTH_PASSWORD"`; output includes `SIPTrunkID`. Telnyx uses `"address": "sip.telnyx.com"` (regional signaling addresses available; `+E.164` destination format assumed); Plivo `<trunk-id>.zt.plivo.com`. Dashboard: Telephony → SIP trunks → Create new trunk → JSON editor → Outbound; dashboard JSON has no `trunk` wrapper and takes `auth_username`/`auth_password` inline. "Create one outbound trunk and reuse it for every call"; vary caller ID with `sip_number` on the request. — [Outbound trunk](https://docs.livekit.io/telephony/making-calls/outbound-trunk/)
- Telnyx outbound requires the custom header `X-Telnyx-Username` in the trunk's `headers_to_attributes`; otherwise Telnyx may match an unrelated connection from the same source IP instead of challenging with 407. Example trunk JSON fields: `name`, `address: sip.telnyx.com`, `numbers`, `auth_username`, `auth_password`, `headers_to_attributes`. — [Telnyx provider guide](https://docs.livekit.io/telephony/start/providers/telnyx/)
  - Correction (Task 14b): the header goes under `headers`, which LiveKit sends on the INVITE, not only under `headers_to_attributes`, which maps headers from the 200 OK into participant attributes and does not send it. `agent/outbound-trunk.json` sets both.
- `CreateSIPParticipantRequest` fields shown: `sip_trunk_id` (or inline `trunk: SIPOutboundConfig{hostname, destination_country, auth_username, auth_password}`, in which case `sip_number` is required), `sip_call_to`, `sip_number` (from-number), `room_name`, `participant_identity`, `participant_name`, `wait_until_answered` (blocks until pickup), `display_name` (custom caller-ID name; empty triggers CNAM lookup at most providers; provider must support custom caller ID), `to_user_override`, `dtmf` (e.g. `"*123#ww456"`, `w` = 0.5 s pause), `play_dialtone`. Python: `await lkapi.sip.create_sip_participant(request)` inside `async with api.LiveKitAPI()`. CLI equivalent: `lk sip participant create sip-participant.json`. — [Outbound calls](https://docs.livekit.io/telephony/making-calls/outbound-calls/)
- From inside an agent: dispatch the agent with the number in metadata, e.g. `lk dispatch create --new-room --agent-name my-telephony-agent --metadata '{"phone_number": "+15105550123"}'` (or `lkapi.agent_dispatch.create_dispatch(api.CreateAgentDispatchRequest(agent_name=..., room=..., metadata=...))`); in the entrypoint `json.loads(ctx.job.metadata or "{}")`, then `ctx.api.sip.create_sip_participant(... sip_trunk_id=..., sip_call_to=..., room_name=ctx.room.name, wait_until_answered=True)`; on `api.SipCallError` call `ctx.shutdown()`; then `ctx.wait_for_participant(identity=...)` before starting the session. — [Outbound calls](https://docs.livekit.io/telephony/making-calls/outbound-calls/); [Agent dispatch](https://docs.livekit.io/agents/server/agent-dispatch/)
- Outcome mapping: 486/603 → rejected (`USER_REJECTED`); 408/480 → no answer (`USER_UNAVAILABLE`); 5xx → `SIP_TRUNK_FAILURE`; 200 OK with voicemail counts as answered (use answering-machine detection). `SipCallError` exposes `sip_status_code` and `sip_status`. `sip.callStatus == "active"` means answered. — [Outbound calls](https://docs.livekit.io/telephony/making-calls/outbound-calls/); [SIP participant reference](https://docs.livekit.io/reference/telephony/sip-participant/)
- LiveKit-side cost: third-party SIP minutes (inbound and outbound) Build 1,000 included; Ship 5,000 then $0.004/min; Scale 50,000 then $0.003/min. — [livekit.com/pricing](https://livekit.com/pricing)
- Telnyx US pay-as-you-go: inbound local from $0.0032/min, outbound local from $0.005/min, local or toll-free number $1.00/month, $0 platform fee, card on file, no contract; "Using Telnyx with LiveKit requires a paid Telnyx account." — [Telnyx elastic SIP pricing](https://telnyx.com/pricing/elastic-sip); [Telnyx provider guide](https://docs.livekit.io/telephony/start/providers/telnyx/)
- Twilio US Elastic SIP Trunking (page says current as of August 2026): origination local $0.0034/min, toll-free $0.0130/min; termination Zone 1 (48 states) $0.0100/min; local number $1.15/month, toll-free $2.15/month; emergency calling $0.75/month per number. — [Twilio SIP trunking pricing](https://www.twilio.com/en-us/sip-trunking/pricing/us)
- Twilio trial, primary sources: "You can call or message only verified recipients from a trial account" (up to 5 verified numbers); calls restricted to the sign-up country; upgrading removes the verified-recipient restriction. — [Twilio free trial tutorial](https://www.twilio.com/docs/usage/tutorials/how-to-use-your-free-trial-account). Since 2023-10-03, **inbound** calls to trial-account numbers, including Elastic SIP Trunking, must come from numbers verified in that trial account, until the account is upgraded. — [Twilio changelog](https://www.twilio.com/en-us/changelog/inbound-calls-to-trial-accounts-must-use-verified-callerid). The SIP trunking limits page states "SIP trunking is only available after you upgrade your account", default termination 1 CPS (raise to 5 in Console), unlimited concurrent calls subject to carrier support, and "New accounts without an approved Business Profile will have limited concurrent calls". — [Twilio SIP trunking scale and limits](https://www.twilio.com/docs/sip-trunking/scale-and-limits)
- Twilio trial, **search summary only** of a help-center page I could not fetch (JS shell): trial trunks limited to one trunk and one number, trial calls play a short trial message, From must be your Twilio number or a Verified Caller ID (403 Invalid Caller ID / 400 unverified-number otherwise), max 4 concurrent calls, and the help center says a trunk can be configured on trial while the limits page says upgrade first (conflict). — [Twilio help center: trial restrictions](https://help.twilio.com/articles/360036052753)

### Inferences
- For the callback step-up, the caller's own phone is the callee, so on a Twilio trial every judge/demo phone would need to be pre-verified (5 max) and would hear the trial announcement; Telnyx (paid, pay-as-you-go) avoids both. Expect roughly $1/month for the number and well under $1 of per-minute charges for the whole hackathon on either provider.
- Caller ID of the callback should be the same number the user originally dialed (`sip_number` = the Telnyx number in the trunk's `numbers`), which also makes the "we are calling you back from the number you know" story coherent.
- Callback flow with the documented primitives: agent says goodbye → `await ctx.delete_room()` → the backend creates a dispatch with `metadata={"phone_number": <stored number>, "ticket": ...}` → new job dials with `wait_until_answered=True` → on `SipCallError` log the status and `ctx.shutdown()`.

### Gaps
- `krisp_enabled`, `max_call_duration`, `ringing_timeout`, `media_encryption`, `attributes` on `CreateSIPParticipant`: not on the fetched page (they are in the SIP API reference, not fetched).
- Telnyx minimum initial top-up / account-verification time for a new paid account: not found; worth checking on signup day since "paid account required" could add a verification delay.
- Twilio's exact upgrade minimum (dollar amount) was not fetched.

---

## KQ5. Alternatives for the phone path: Twilio Elastic SIP Trunking vs Telnyx into LiveKit

### Takeaway
Both are documented one-page setups (~30-60 minutes each in my estimate) with similar prices; Telnyx is the only one that gives HD voice with LiveKit and the only one without trial-mode caller restrictions, so it is the better fit for Callback.

### Cited Findings
- Twilio steps: buy a number; install Twilio CLI; create a trunk with a domain ending `pstn.twilio.com`; Origination URI `sip:<your LiveKit SIP endpoint>;transport=tcp` (regional endpoint optional, fallback via `priority`); for outbound create a Credential List (Voice → Credential lists), attach under Termination → Authentication, set a unique termination SIP domain, and use the same username/password in the LiveKit outbound trunk; associate the number with `twilio api trunking v1 trunks phone-numbers create`. For inbound only, TwiML/Programmable Voice is an alternative to SIP trunking. — [Twilio provider guide](https://docs.livekit.io/telephony/start/providers/twilio/)
- Telnyx steps (portal): Real-time Communications → Voice → SIP Trunking → Create SIP connection → FQDN → add your LiveKit SIP endpoint and outbound auth username/password → Inbound settings: destination and origination formats `+E.164`, TCP recommended, pick a SIP region → Outbound settings: select/create an Outbound Voice Profile (Voice → Settings) → associate the number → Complete. API equivalents (curl, Telnyx API v2) are also given: FQDN connection with `ani_number_format: +E.164`, `dnis_number_format: +e164`, transport TCP, `anchorsite_override: Latency`; FQDN record pointing at the LiveKit endpoint, port 5060, `dns_record_type: a`; PATCH the number's `connection_id`. — [Telnyx provider guide](https://docs.livekit.io/telephony/start/providers/telnyx/)
- HD voice support matrix: Telnyx only (plus LiveKit's own numbers). — [HD voice](https://docs.livekit.io/telephony/features/hd-voice/)
- Prices: see KQ4 (Telnyx $1.00/number, $0.0032 in / $0.005 out per minute; Twilio $1.15/number, $0.0034 in / $0.0100 out per minute). — [Telnyx pricing](https://telnyx.com/pricing/elastic-sip); [Twilio pricing](https://www.twilio.com/en-us/sip-trunking/pricing/us)
- Other documented providers: Plivo, Wavix, Sinch, didlogic, DIDWW; plus a Twilio WebSocket "connector" (Build: 5 concurrent calls; Ship 20 concurrent at $0.004/min). — [Telephony docs index](https://docs.livekit.io/telephony/llms.txt); [livekit.com/pricing](https://livekit.com/pricing)
- LiveKit Cloud already fails over between regions automatically; provider-side fallback URIs are an advanced option. — [Twilio provider guide](https://docs.livekit.io/telephony/start/providers/twilio/); [Telnyx provider guide](https://docs.livekit.io/telephony/start/providers/telnyx/)

### Inferences
- Rough setup time (my estimate): Telnyx account + payment + number + SIP connection + LiveKit inbound/outbound trunks + dispatch rule: 45-90 minutes if the Telnyx account is approved instantly. Twilio similar, plus the upgrade step to escape trial limits.
- The Twilio WebSocket connector was not researched in depth; Twilio Media Streams are narrowband in my experience **(unverified here)**, so it is unlikely to help the HD-audio goal.

### Gaps
- Whether Telnyx exposes HD Voice as a per-number toggle for all US numbers today (docs still show a toggle; the July 2025 note says auto-enabled) is a minor conflict; check the number's Voice tab after purchase.
- Option D (Pipecat) was outside what I could verify in this pass; no sources gathered.

---

## KQ6. LiveKit Cloud Build (free) tier quotas as of Nov 2026, exceed behaviour, next tier

### Takeaway
Build: 1,000 agent session minutes, 5,000 WebRTC participant minutes, 50 GB downstream, 100 concurrent connections, 5 concurrent agent sessions, 1 agent deployment, 1 free US number with 50 inbound minutes, 1,000 third-party SIP minutes, $2.50 inference credits; all hard caps that reset monthly. Ship is $50/month.

### Cited Findings
- Plan prices: Build $0 (no credit card), Ship from $50/month, Scale from $500/month, Enterprise custom. — [livekit.com/pricing](https://livekit.com/pricing)
- Agent platform: agent session minutes Build 1,000 / Ship 5,000 then $0.01/min / Scale 50,000 then $0.01/min; concurrent agent sessions 5 / 20 / up to 600; agent deployments 1 / 2 / 4; non-production deployments per agent 0 / 2 / 5. — [livekit.com/pricing](https://livekit.com/pricing)
- Participants and bandwidth: WebRTC minutes 5,000 / 150,000 then $0.0005 / 1.5M then $0.0004; concurrent connections (users + agents) 100 / 1,000 / 5,000; downstream transfer 50 GB / 250 GB then $0.12/GB / 3 TB then $0.10/GB. — [livekit.com/pricing](https://livekit.com/pricing)
- Inference credits $2.50 (~50 min) / $5 / $50; inference concurrency 5 / 20 / up to 600; session recordings 1,000 min on Build; observability events 100,000 on Build; text simulations 350 turns, audio simulations 60 turns on Build; voice isolation 100 min on Build. — [livekit.com/pricing](https://livekit.com/pricing)
- Telephony on Build: 1 free US local number, 50 US local inbound minutes, 1,000 third-party SIP minutes, Twilio/WhatsApp connectors 5 concurrent calls; no toll-free. — [livekit.com/pricing](https://livekit.com/pricing); [Quotas and limits](https://docs.livekit.io/deploy/admin/quotas-and-limits/)
- Build rate limits: 100 concurrent participants across all rooms; 2 concurrent ingress and 2 concurrent egress; 100 video and 100 audio track subscriptions per participant; 5 concurrent agent sessions; 5 concurrent STT / TTS / Realtime inference connections; LLM 100 requests/min and 600,000 tokens/min; 1,000 server API requests/min; agent cold-start delay 10-20 s; `lk agent deploy` build context upload 1 GB. — [Quotas and limits](https://docs.livekit.io/deploy/admin/quotas-and-limits/)
- Exceeding limits: Build allowances are hard caps ("new requests fail rather than incurring overage charges"); concurrency/rate limits make new operations fail until a session closes or the window resets; Ship/Scale bill overage at the published rates; allowances reset on the 1st of the month and do not roll over; free allowances are shared across all of a user's free projects. — [Quotas and limits](https://docs.livekit.io/deploy/admin/quotas-and-limits/)
- Per-minute phone-call estimate from LiveKit's calculator (Build/Ship): agent session $0.0100 + telephony $0.0100 + LLM $0.0014 + STT $0.0075 + TTS $0.0090 + observability $0.0100 = $0.0479/min. — [livekit.com/pricing](https://livekit.com/pricing)

### Inferences
- Callback's room has 3 participants (caller, agent, hidden detector); every minute of call burns ~1 agent session minute, ~3 WebRTC participant minutes, and 1 phone minute. The binding constraint is the 50 LiveKit-number inbound minutes, not the 1,000 agent minutes.
- Creating a second free project does not double the allowances (they are shared per user).

### Gaps
- Per-replica CPU/memory for cloud agents on Build is not published on the pricing or quotas pages.
- Concurrent rooms and max participants per room are not listed.

---

## KQ7. Cloud agent deployment (`lk agent create/deploy/logs`), Dockerfile, secrets, regions, cold start, free limits

### Takeaway
`lk agent create` (first deploy) and `lk agent deploy` (updates) build your Dockerfile in LiveKit's build service (1 GB context, 10-minute build limit), inject `LIVEKIT_URL/API_KEY/API_SECRET` automatically, and take other secrets from a `.env` file. On Build you get one production deployment that scales to zero and cold-starts in 10-20 s; only paid plans stay warm and can roll back instantly.

### Cited Findings
- Commands: `lk cloud auth`; `lk project list`; `lk project set-default "<project-name>"`; `lk agent create` ("registers the agent, uploads the code to the build service, builds the image from the Dockerfile, and deploys it"); `lk agent status` (status, replica count); `lk agent logs` (live tail of the most recently started instance); `lk agent deploy` for subsequent versions; `lk agent config` generates `livekit.toml`; `lk agent rollback` (paid plans; free users revert code and redeploy). Generated files: `livekit.toml` (agent id, project subdomain) and a `Dockerfile` if none exists. — [Deploy quickstart](https://docs.livekit.io/deploy/agents/quickstart/); [Managing deployments](https://docs.livekit.io/deploy/agents/managing-deployments/); [LiveKit CLI reference](https://docs.livekit.io/reference/developer-tools/livekit-cli/)
- Scaffold path: `lk agent init my-agent --template agent-starter-python` → `cd my-agent` → `lk agent dev` (prints an Agent Console link; "Start a session" to talk) or `lk agent console` → `lk agent create`; Agents dashboard at `https://cloud.livekit.io/projects/p_/agents` for the deployed agent's Agent Console. — [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai/); [LiveKit CLI reference](https://docs.livekit.io/reference/developer-tools/livekit-cli/)
- Dockerfile templates: uv variant based on `ghcr.io/astral-sh/uv:python3.14-bookworm-slim`, build stage installs gcc/g++/python3-dev, `WORKDIR /app`, copies `pyproject.toml` + `uv.lock`, `uv sync --locked`, then `COPY . .`; production stage runs as non-root `appuser` (UID 10001), sets `PYTHONUNBUFFERED=1` and `UV_COMPILE_BYTECODE=1`; `CMD ["uv", "run", "agent.py", "start"]` (pip variant: `python:3.14-slim`, `CMD ["python", "agent.py", "start"]`). Requirements: fixed ENTRYPOINT/CMD that launches with `start`, no backgrounded processes or wrapper scripts, do not copy `.env*` or secrets into the image. — [Builds](https://docs.livekit.io/deploy/agents/builds/)
- Build limits: 1 GB context upload (exclude model weights, datasets, venvs; download large assets during the build); 10-minute build maximum; build logs stream to the CLI and `lk agent logs --log-type=build` prints them; bring-your-own image (`--image`, `--image-tar`) is Enterprise-only. — [Builds](https://docs.livekit.io/deploy/agents/builds/); [Logs](https://docs.livekit.io/deploy/agents/logs/)
- Secrets: `lk agent create --secrets-file=path/to/secrets.env` (KEY=value lines); with no flag the CLI finds `.env`, `.env.local` or `.env.production` and offers to load it; `lk agent update-secrets --secrets "SECRET_A=foo"` (repeatable) or `--secrets-file=new-secrets.env`, `--overwrite` to replace all; updating triggers a rolling restart; `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` are injected automatically and cannot be set as secrets (the CLI skips them when copying a file); values max 16 KB, names letters/digits/underscore max 70 chars; `lk agent secrets` lists names only. — [Secrets](https://docs.livekit.io/deploy/agents/secrets/)
- Logs: `lk agent logs` streams the most recently started instance; session logs in the Sessions page under Agent insights (project admins only); build logs for the latest version kept indefinitely; log level default `info`, change with `--log-level=debug` in the Dockerfile CMD or the `LIVEKIT_LOG_LEVEL` secret; log drains (Datadog, CloudWatch, Sentry, New Relic) for crashes/startup failures outside sessions. — [Logs](https://docs.livekit.io/deploy/agents/logs/); [Deploy agents overview](https://docs.livekit.io/deploy/agents/)
- Cold start and scaling: "On the Build (free) plan, production agents can be scaled down to zero replicas after all active sessions end"; cold start adds 10-20 s before the agent joins; "Paid plans (Ship and Scale) keep production agents warm"; non-production deployments scale to zero on every plan. Rollouts: new instances take sessions after passing a health check (5 minutes allowed; a long `prewarm` can block routing), old instances stay up to 1 hour to drain. — [Managing deployments](https://docs.livekit.io/deploy/agents/managing-deployments/); [Quotas and limits](https://docs.livekit.io/deploy/admin/quotas-and-limits/)
- Deployment counts: Build 1 agent deployment, 0 non-production; Ship 2 and 2; Scale 4 and 5. A non-production deployment counts whether awake or asleep; free a slot with `lk agent delete --deployment <name>`. — [livekit.com/pricing](https://livekit.com/pricing); [Non-production deployments](https://docs.livekit.io/deploy/agents/deployments/)
- Regions: the GitHub Action `livekit/deploy-action` has a `REGION` input defaulting to the nearest region; the CLI docs fetched do not document a region flag. — [Managing deployments](https://docs.livekit.io/deploy/agents/managing-deployments/)

### Inferences
- Demo risk with option A on Build: if no session is active for a while before a judge calls, the first ring is followed by a 10-20 s silent cold start. Untested mitigations: keep one session alive (e.g. a browser tab connected to a dummy room with the agent) since scale-down happens only "after all active sessions end"; or pay $50 for Ship for the month to keep it warm and gain instant rollback and a second deployment.
- The detection service cannot be a second cloud deployment on Build (one deployment); it would have to run inside the same agent process/container or on a laptop/VM (option B style) regardless of where the main agent runs.
- Apple Silicon: builds happen in LiveKit's service, so local arch does not matter for option A (the docs do not discuss architecture at all).

### Gaps
- `lk agent create` flags such as `--region`, `--name`, `--subdomain`: not documented on the pages fetched; check `lk agent create --help`.
- Idle timeout before a free-plan agent scales to zero: not published.
- Replica CPU/memory and whether a GPU is available for the detection model: not published; assume CPU-only.
- Whether the starter template's Dockerfile includes a `download-files` step (for turn-detector/VAD weights) was not confirmed from the builds page, which shows no such step; check the generated Dockerfile.

---

## KQ8. Self-hosting the agent process (option B): laptop, Railway, Fly.io, Modal

### Takeaway
A worker only needs outbound connectivity: it registers with LiveKit Cloud over a WebSocket and needs no public ports, so `uv run agent.py start` on a laptop or a $5-7/month always-on container is enough; the demo-day risks are the laptop's network and sleep, not LiveKit.

### Cited Findings
- Agent servers "do not need to expose any inbound hosts or ports to the public internet"; the worker registers with LiveKit server over a WebSocket and accepts jobs on that connection. Required env: `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` plus provider keys. — [Self-hosted deployments](https://docs.livekit.io/deploy/custom/deployments/)
- Sizing: start at 4 cores / 8 GB per agent server for 10-25 concurrent jobs; a load test ran 30 voice agents on 4 cores / 8 GB peaking at ~3.8 cores and 2.8 GB; noise cancellation and the turn detector need more CPU/memory; ~10 GB ephemeral disk. — [Self-hosted deployments](https://docs.livekit.io/deploy/custom/deployments/)
- Job distribution: LiveKit assigns each job to one agent server round-robin (Cloud also prefers geographically closer servers); if a server does not accept within a timeout the job goes elsewhere; availability is gated by `load_fnc` (default CPU utilisation) against `load_threshold` (default 0.7). — [Self-hosted deployments](https://docs.livekit.io/deploy/custom/deployments/)
- Graceful shutdown: on SIGINT/SIGTERM the server stops taking jobs and lets running ones finish; voice apps may need 10+ minutes of grace; optional private health endpoint default `http://0.0.0.0:8081/`. — [Self-hosted deployments](https://docs.livekit.io/deploy/custom/deployments/)
- Local run commands from the quickstart: `lk agent dev` (dev mode with Agent Console link) and the Dockerfile `CMD` uses `uv run agent.py start` for production mode. — [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai/); [Builds](https://docs.livekit.io/deploy/agents/builds/)
- Railway: Free trial $5 one-time credit (30 days, no card); Free plan $1/month credit; Hobby $5/month incl. $5 usage; Pro $20/month; usage billed per second at $0.00000386/GB-s (~$10/GB-month) and $0.00000772/vCPU-s (~$20/vCPU-month); "stopped services cost nothing". — [Railway pricing](https://railway.com/pricing)
- Fly.io: no free tier (new accounts get $1 credit for 7 days); `shared-cpu-1x` 256 MB $2.19/month, 512 MB $3.69, 1 GB $6.70; `performance-1x` 2 GB $33.00 (iad/ewr prices); no platform fee. — [Fly.io pricing](https://docs.fly.io/about/pricing)
- Modal: Starter plan $30/month free credits; CPU $0.0000131 per physical core-second (min 0.125 cores), memory $0.00000222 per GiB-second; containers shut down 60 s after the last input by default (configurable); Team $250/month. — [Modal pricing](https://modal.com/pricing)

### Inferences
- Cost for one always-on worker: Fly `shared-cpu-1x` 1 GB ≈ $6.70/month; Railway Hobby ≈ $5/month plus usage beyond the credit (an idle-ish 1 vCPU/1 GB worker lands around $5-15/month at Railway's rates, my arithmetic); Modal's $30 credit covers it but its scale-to-zero design fights a long-lived WebSocket worker, so I would not pick it for this. A laptop is $0.
- Running the same `agent_name` on a laptop and in the cloud at the same time splits jobs round-robin between them; for the demo run exactly one worker per `agent_name` (or use distinct names such as `callback-cloud` / `callback-laptop` and point the dispatch rule at one).
- Laptop reliability checklist for Nov 19 (my recommendations): `caffeinate -dims uv run agent.py start`, phone hotspot as a fallback to office Wi-Fi (captive portals can break WebSockets), pre-download model files, and keep a Fly/Railway copy deployed but stopped as a backup.

### Gaps
- No LiveKit documentation specific to Railway, Fly or Modal; the self-hosting page only says to consult the platform for grace-period settings.
- Modal's maximum function duration for an always-on worker was not verified.

---

## KQ9. Option C: self-hosting LiveKit server + SIP bridge

### Takeaway
Self-hosting adds Redis, a LiveKit server, a public IP, UDP 5060 plus a 10,000-port RTP range, and host networking, and you still need a carrier trunk for numbers; it buys nothing for HD voice that Cloud+Telnyx does not already give, so it is not worth the week.

### Cited Findings
- livekit/sip requirements: Redis (shared with the LiveKit server; also stores SIP session state), a LiveKit server (`ws_url`), "The SIP service must also expose a public IP address for remote SIP peers to connect to"; `config.yaml` needs `api_key`, `api_secret`, `ws_url`, `redis`, optional `sip_port` (default 5060), `rtp_port` (default 10000-20000), `health_port`, `prometheus_port`, `log_level`. Run with `docker run --rm -e SIP_CONFIG_BODY="$(cat config.yaml)" --network host livekit/sip` (host networking required because of the RTP range). Native builds need Go 1.18+ and libopus/opusfile/soxr. Apache-2.0. — [livekit/sip README](https://github.com/livekit/sip)
- Cloud-only conveniences you would lose: LiveKit Phone Numbers (Cloud feature), automatic multi-region failover, the Telephony dashboard call logs and Sessions view. — [LiveKit Phone Numbers docs](https://docs.livekit.io/telephony/start/phone-numbers/); [Twilio provider guide](https://docs.livekit.io/telephony/start/providers/twilio/); [Telephony testing](https://docs.livekit.io/telephony/testing/)

### Inferences
- The one thing self-hosting would give the detection work is raw SIP server logs showing the negotiated codec per call; that is not worth a VM with a public IP, firewall rules and two extra services during hackathon week. A cheaper way to learn the codec mix is to spot-check spectral bandwidth in the detection service.

### Gaps
- I did not fetch the LiveKit server self-hosting guide (transport section); port/NAT details for the SFU itself are not captured here.

---

## KQ10. Installing and authenticating the `lk` CLI on macOS, project, credentials

### Takeaway
`brew install livekit-cli`, then `lk cloud auth` opens the browser to link a Cloud project; `lk project list` / `lk project set-default` manage projects; `LIVEKIT_*` credentials come from the project's settings and are auto-injected for cloud agents.

### Cited Findings
- Install: macOS `brew install livekit-cli`; Linux `curl -sSL https://get.livekit.io/cli | bash`; Windows `winget install LiveKit.LiveKitCLI`; from source `git clone github.com/livekit/livekit-cli && make install` (needs Git LFS). Update: `brew update && brew upgrade livekit-cli`. — [LiveKit CLI reference](https://docs.livekit.io/reference/developer-tools/livekit-cli/)
- Auth: `lk cloud auth` ("Link a LiveKit Cloud project, then follow the instructions in your browser"); sign up at https://cloud.livekit.io/ if needed; `lk project list`; `lk project set-default "<project-name>"`. — [LiveKit CLI reference](https://docs.livekit.io/reference/developer-tools/livekit-cli/); [Deploy quickstart](https://docs.livekit.io/deploy/agents/quickstart/); [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai/)
- Command groups: `lk cloud`, `lk project`, `lk app`, `lk agent`, `lk room`, `lk token`, `lk dispatch`, `lk egress`, `lk ingress`, `lk sip`, `lk number`, `lk perf`, `lk docs`, `lk skills`; `lk <command> --help` for options; `lk docs search "voice agent quickstart"`; `lk skills install --global` to give a coding agent LiveKit skills. — [LiveKit CLI reference](https://docs.livekit.io/reference/developer-tools/livekit-cli/)
- Templates: `lk agent init --template agent-starter-python|agent-starter-node`; `lk app create` templates include `agent-starter-react`, `agent-starter-android`, `agent-starter-swift`, `agent-starter-flutter`, `agent-starter-react-native`, `agent-starter-embed`, `token-server`, `meet`, `multi-agent-python`, `outbound-caller-python`. — [LiveKit CLI reference](https://docs.livekit.io/reference/developer-tools/livekit-cli/)
- Credentials: cloud-deployed agents get `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` injected; self-hosted workers need them as env vars. The SIP endpoint URI is shown in LiveKit Cloud Project settings. — [Secrets](https://docs.livekit.io/deploy/agents/secrets/); [Self-hosted deployments](https://docs.livekit.io/deploy/custom/deployments/); [Twilio provider guide](https://docs.livekit.io/telephony/start/providers/twilio/)

### Inferences
- `outbound-caller-python` is a first-party template worth scaffolding to copy the callback pattern (`lk app create --template outbound-caller-python`).

### Gaps
- `lk app env -w .env.local` (writing project credentials to a file) is **(from memory)**; it did not appear on the fetched pages. Confirm with `lk app env --help`. The dashboard path for API keys (Settings → Keys) is likewise **(from memory)**.

---

## KQ11. Browser channel: simplest way to put a browser caller on the same agent

### Takeaway
Fastest: `lk agent dev` prints a hosted Agent Console link (and the Agents dashboard offers one for the deployed agent). For a demo UI: turn on the project's Development token server in Settings, scaffold `agent-starter-react` (Next.js), and request tokens with `agentName` so the same `callback` agent is dispatched.

### Cited Findings
- `lk agent dev` prints an Agent Console link; click "Start a session" and speak. For a deployed agent, open the Agents dashboard (`https://cloud.livekit.io/projects/p_/agents`) to launch its Agent Console. — [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai/); [Deploy quickstart](https://docs.livekit.io/deploy/agents/quickstart/)
- Development token server: Settings page (`cloud.livekit.io/projects/p_/settings/project`) → toggle "Development token server" → copy the Token server ID; JS usage `TokenSource.developmentTokenServer("<id>")` then `tokenSource.fetch({ roomName, agentName, deployment })` (empty deployment = production). "The token server is for development and testing only" (any frontend can request any permissions). "LiveKit Sandbox is no longer available, but the token server remains available as a standalone project setting" (some SDK APIs still say "Sandbox"). — [Development token server](https://docs.livekit.io/frontends/build/authentication/development-token-server/)
- React starter: repo `livekit-examples/agent-starter-react` (Next.js + Agents UI); copy `.env.example` to `.env.local` with LiveKit credentials; `pnpm dev`; open http://localhost:3000; lists token-server support. Scaffold via `lk app create --template agent-starter-react`. — [React starter app](https://docs.livekit.io/frontends/start/starter-apps/react/); [LiveKit CLI reference](https://docs.livekit.io/reference/developer-tools/livekit-cli/)
- Dispatching the named agent from a browser token: embed `RoomConfiguration(agents=[RoomAgentDispatch(agent_name="test-agent", metadata=...)])` in the access token (`AccessToken().with_room_config(...)`); token-based dispatch only fires when the room is first created. — [Agent dispatch](https://docs.livekit.io/agents/server/agent-dispatch/)
- The older open-source Agents Playground (KITT demo) lives at a Vercel URL and its README warns layouts break on small screens; current docs do not reference an `agents-playground.livekit.io` host. **(search summary only)** — [livekit/agent-playground](https://github.com/livekit/agent-playground)

### Inferences
- The browser caller gives the detection service full-band 48 kHz Opus microphone audio, which makes it the natural "clean baseline" channel to contrast against the phone leg in the demo.
- For a one-screen judge demo, the Agent Console is enough; build the React starter only if you want to show the detector's verdict in the UI.

### Gaps
- `agent-starter-react` configuration for `agentName` (env var vs `app-config.ts`) was not on the docs page; read the repo README.

---

## Recommendation: option A vs option B for the Nov 19 live demo (my recommendation)

### Takeaway
Run the agent and the detection service as self-hosted workers (option B) for the live final, with the Callback agent also deployed to LiveKit Cloud (option A) as a backup under a different `agent_name`; use a Telnyx number rather than the free LiveKit number for the phone leg.

### Cited Findings
- Build-plan cloud agents scale to zero and cold-start in 10-20 s; paid plans keep them warm; Build has 1 deployment. — [Managing deployments](https://docs.livekit.io/deploy/agents/managing-deployments/); [Quotas and limits](https://docs.livekit.io/deploy/admin/quotas-and-limits/); [livekit.com/pricing](https://livekit.com/pricing)
- Self-hosted workers need no inbound ports and are load-balanced round-robin with cloud workers by `agent_name`. — [Self-hosted deployments](https://docs.livekit.io/deploy/custom/deployments/)
- LiveKit Phone Numbers: inbound-only, 50 free minutes hard cap; Telnyx: HD voice (G.722) supported, outbound supported, pay-as-you-go. — [LiveKit Phone Numbers docs](https://docs.livekit.io/telephony/start/phone-numbers/); [HD voice](https://docs.livekit.io/telephony/features/hd-voice/); [Telnyx pricing](https://telnyx.com/pricing/elastic-sip)

### Inferences
- Why B for the demo: no cold start, both processes (agent + hidden detector) run side by side with shared local models and logs, and the free plan's single deployment cannot host the detector anyway. Why keep A: `lk agent create` is a 5-minute task once the Dockerfile exists, it demonstrates a production path for the Round 2 write-up, and it is a one-command fallback if the laptop dies (switch the dispatch rule's `agent_name` or stop the laptop worker so round-robin sends everything to the cloud replica).
- If the team prefers A as primary, pay $50 for one month of Ship: warm replicas, 2 deployments (agent + detector), 100 LiveKit-number inbound minutes, instant rollback.
- Phone plan: Telnyx number with HD Voice on and G.722 enabled on the trunk's inbound codecs, one LiveKit inbound trunk + one outbound trunk, dispatch rule with `room_config.agents[{agent_name:"callback"}]`; keep the free LiveKit number as a second inbound path for comparison testing of the HD-voice claim.

### Gaps
- None of the carrier-interop or cold-start mitigations above were tested; schedule a test call from an AT&T, T-Mobile and Verizon phone to both numbers in week 1 and log the measured audio bandwidth per call.
