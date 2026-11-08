# Callback: Twilio Verify on a trial account, Supabase logging + live dashboard, recording and voiceprint consent (state as of 2026-11-04)

Scope note. Researched 2026-11-04 for Veer's slice of Callback (BBS x Ramp hackathon; Round 2 due 2026-11-10, final 2026-11-19). Primary sources were fetched directly where possible (Twilio docs, Supabase docs, LiveKit docs, statute mirrors). `leginfo.legislature.ca.gov`, `law.justia.com`, `legiscan.com` and `ilga.gov` all refused automated fetches (HTTP 403 / certificate error), so California statute text comes from `california.public.law` (which stamps a "verified Nov 2, 2026" date on each section) and Illinois text from FindLaw ("current as of January 1, 2025"). The team's internal Notion page "BBS x Ramp: Callback" was read only in keyword slices (not 100%); anything from it is marked "internal Notion". Compliance notes are practical pointers for a student demo, not legal advice.

---

## 1. Twilio trial account (Nov 2026): credit, verified-recipient cap, verification, upgrade, trial prefix

### Takeaway
New Twilio trials no longer carry a dollar balance: they give product-specific free units (100 SMS, 75 voice minutes, 40 Verify verifications, ...) that expire 30 days after sign-up, and a trial can only message or call up to 5 verified recipients (the sign-up number is verified automatically). Upgrading (adding payment details) removes the verified-recipient restriction and the trial prefix; Twilio's public sign-up page still advertises "$15 free credits" and an "unlimited trial period", which contradicts the current docs and should be treated as stale.

### Cited Findings
- Trial model: "Instead of a dollar-based balance, your trial includes transparent, product-specific free units." Free units per trial account: SMS 100, WhatsApp 100, Voice 75 call minutes, Verify 40 verifications, Email 3,000, Lookup 100; text-to-speech and speech recognition are "Included" within call minutes. "Trial accounts expire 30 days after sign-up." "When your trial expires, you can no longer use trial resources." Page metadata shows `dateModified: 2026-11-04`. — [Twilio: Twilio trial account](https://www.twilio.com/docs/usage/trials)
- Verified recipients: "Each account can verify up to 5 recipients." "The phone number you signed up with is automatically verified." "SMS messages and voice calls are restricted to your sign-up country"; email is limited to the sign-up address. "You must use Twilio-provided templates or message examples." "Custom message bodies, email content, and WhatsApp templates aren't available during the trial." — [Twilio: Twilio trial account](https://www.twilio.com/docs/usage/trials)
- "You can call or message only verified recipients from a trial account." Each non-Twilio number that should receive SMS/MMS or calls from your Twilio number, or call your Twilio number, must be verified. — [Twilio: Get started with your free trial account](https://www.twilio.com/docs/usage/tutorials/how-to-use-your-free-trial-account)
- How to verify (console): in the current Console you add and verify recipients from the "Try out" page of the product you are testing (Twilio texts a code you type back); in the Legacy Console it is the "Verified Caller IDs" page, where "you must select SMS as the verification method and have access to the device that receives the code." — [Twilio: Get started with your free trial account](https://www.twilio.com/docs/usage/tutorials/how-to-use-your-free-trial-account)
- How to verify (API, voice validation call): `POST https://api.twilio.com/2010-04-01/Accounts/{AccountSid}/OutgoingCallerIds.json` with `PhoneNumber` (E.164); optional `FriendlyName` (max 64 chars), `CallDelay` (0–60 s), `Extension`, `StatusCallback`. Twilio calls the number; "the person who answers the call must enter the validation code." The response's `validation_code` is "The 6-digit validation code that must be entered via the phone to validate this phone number for Caller ID." The verification call is English only. Python: `validation_request = client.validation_requests.create(friendly_name="My Home Phone Number", phone_number="+15555550100")`. The page does not mention trial accounts. — [Twilio: Outgoing Caller IDs API](https://www.twilio.com/docs/voice/api/outgoing-caller-ids)
- Upgrade lifts the cap: "Twilio removes this security restriction when you upgrade your account." Upgrading = click the Trial banner / "Upgrade your account", create a profile, add an address, load a starting balance and payment details. After upgrade you "receive a fresh set of post-upgrade free units for eligible products" (same amounts: SMS 100, WhatsApp 100, calls 75 min, email 3,000, Verify 40, Lookup 100; "Sign-up upgrades get a 30% bonus on PUFU"). "A2P 10DLC registration requires a paid account." — [Twilio: Get started with your free trial account](https://www.twilio.com/docs/usage/tutorials/how-to-use-your-free-trial-account); [Twilio: Twilio trial account](https://www.twilio.com/docs/usage/trials)
- Trial voice behaviour (relevant to the "callback" leg): "You can only make or receive calls from verified numbers." "Your call is placed from a Twilio trial phone number." Per-call cap of 10 minutes, 75 total minutes, "Calls are terminated automatically when exhausted", up to 5 concurrent calls, calls only within your country. Programmable Voice `Create a Call` on trial only lets you set `to`, the sample TwiML webhook URL and `statusCallback`. Blocked TwiML on trial: `<Stream>`, `<Dial><Sip>`, `<Dial><Number>`, `<Record>`, `<ConversationRelay>`, `<VirtualAgent>`, `<Siprec>`, `<Refer>`, `<Pay>`. — [Twilio: Try out Twilio Voice](https://www.twilio.com/docs/usage/trials/try-out-voice)
- Warning 13805: "Trial calls are limited to 10 minutes for both inbound and outbound calls"; "Twilio automatically disconnects calls that hit that limit and flags them with Warning 13805." "The trial lasts 30 days or until you exhaust your free product units (for example, 100 SMS), whichever comes first." — [Twilio: Error 13805](https://www.twilio.com/docs/api/errors/13805)
- Trial prefix: Twilio's sign-up page says texts begin with "Sent from a Twilio trial account." and "Calls play a short trial message before your TwiML runs." The same page still says "$15 free credits preloaded to your account", "No credit card required" and "Unlimited trial period - use it until you upgrade." — [Twilio: sign up / credits page](https://twilio.com/en-us/signup-credits). Contradicted on credit and expiry by the trial docs above (free units, 30-day expiry).
- Twilio support: the trial prefix is removed once the project is upgraded; upgrading keeps your trial number ("your trial number will be carried over and charged monthly") but trial credits do not carry over. — [Twilio Support article 360036052753](https://support.twilio.com/hc/en-us/articles/360036052753) (as summarised in search results; article body not fetched)
- Since 2023-10-03, inbound calls to trial accounts must come from a verified caller ID. — [Twilio changelog, Oct 3 2023](https://static1.twilio.com/en-us/changelog/inbound-calls-to-trial-accounts-must-use-verified-callerid)
- Verify's trial page adds: a single phone number can be verified on up to three Twilio accounts (including subaccounts); "If you sign up for a fourth Twilio account with the same number, your number isn't added as a verified number." The 5-number cap is shared across Verify, Messaging and Voice. — [Twilio: Try out Twilio Verify](https://www.twilio.com/docs/usage/trials/try-out-verify)

### Inferences
- For Callback the practical cap is 5 demo phones per trial account (sign-up phone included). With two team members you could run two trial accounts (Veer's and Eric's) to reach ~10 verified numbers, and a given friend's phone can appear on up to three accounts. Register the demo phones on day one (the internal Notion plan already lists "Verify demo phones in Twilio" for Nov 4).
- The trial restrictions make Programmable Voice unusable for a custom callback leg (`Create a Call` cannot point at your own TwiML; `<Stream>`, `<Dial><Sip>` and `<Record>` are blocked). The Verify `call` channel (section 2) is the trial-safe way to do "we will call the number on file": Twilio places the call from its own Verify number and reads a code after a keypad challenge.
- Connecting a Twilio number to LiveKit via SIP trunking is also off the table on a trial (`<Dial><Sip>` blocked; Elastic SIP trunking is a paid-account feature per the trial pages). This matches the team's plan (internal Notion) to use a LiveKit phone number or Telnyx for the voice path and use Twilio only for Verify.
- If you upgrade: minimum spend is whatever starting balance you load (Twilio does not publish a minimum on the pages fetched); you also get 40 more Verify verifications and 75 more voice minutes as post-upgrade free units.

### Gaps
- Exact dollar credit on a brand-new account today could not be verified: docs say "no dollar-based balance", the marketing page says "$15 free credits" (older trials were commonly reported as $15.50). Check the Console banner after sign-up.
- Whether the API validation call (`OutgoingCallerIds`) is allowed on a trial account, or whether trial verification is console-only via SMS, is not stated in the docs fetched. Use the Console "Try out Verify" page for the demo phones.
- The exact wording of the trial preamble played on trial voice calls was not found in current docs.

---

## 2. Twilio Verify: service creation, API calls, Python, trial quota, price, limits, 10DLC, voice fallback

### Takeaway
Verify works on a trial without buying a number (it sends from a Twilio-owned "Verify trial phone number" with a fixed template), gives 40 free successful verifications on trial, only to the 5 verified numbers, and after that costs $0.05 per successful verification plus $0.0083 per US SMS segment. Built-in limits: codes expire after 10 minutes, 5 check attempts and 5 send attempts per verification, 429 on excess. Verify using Twilio's pooled senders is exempt from A2P 10DLC registration; the `call` channel is the voice fallback and does a keypad challenge before reading the code.

### Cited Findings
- Trial quota: "40 free verifications". "One unit is deducted from your 40 verification quota only after you have successfully checked a verification code." Each SMS verification sent also deducts one unit from the 100 SMS units. "You can only send verification codes to verified numbers." "Each account can verify up to five numbers." Trial sends use "the Verify trial phone number and test template"; you cannot write a custom message body or customise code templates during the trial; Fraud Guard is available after upgrade. Console trial channels: SMS and Voice; the API also exposes Push and TOTP. After upgrading "you can create new Verify Services and assign the Twilio phone numbers you purchase." "To send SMS and Voice OTPs to any number, you must upgrade and get Twilio approval for a Primary Customer Profile (PCP)." — [Twilio: Try out Twilio Verify](https://www.twilio.com/docs/usage/trials/try-out-verify)
- Pricing: "$0.05 per successful verification, plus channel fees." US SMS channel fee $0.0083 per SMS. Voice: $0.05 per successful verification plus per-call Voice pricing (on a separate page). Email: $0.05 (+ email pricing). Push/TOTP: no separate channel fee. The base $0.05 "isn't charged for failed, expired, incorrect, or uncompleted verifications" but "SMS and Voice attempts are always billed" (the channel fee). Volume discounts via sales. — [Twilio Verify pricing](https://www.twilio.com/en-us/verify/pricing)
- Create a Verify Service: `POST https://verify.twilio.com/v2/Services`; required `FriendlyName` ("up to 32 characters", "should not contain PII"); optional `CodeLength` ("integer value between 4 and 10, inclusive"), `LookupEnabled`, `SkipSmsToLandlines` (requires lookup), `DtmfInputRequired` ("Whether to ask the user to press a number before delivering the verify code in a phone call"), `DoNotShareWarningEnabled` (adds a security warning to the SMS body; off by default), `CustomCodeEnabled`, `TtsName`. Python: `service = client.verify.v2.services.create(friendly_name="My Verify Service")` then `print(service.sid)`. Sample responses show `"code_length": 4`. — [Twilio: Verify Service resource](https://www.twilio.com/docs/verify/api/service)
- Start a verification: `POST https://verify.twilio.com/v2/Services/{ServiceSid}/Verifications`; required `To` (E.164) and `Channel` ("One of: email, sms, whatsapp, call, sna or auto"; Twilio "does not accept a Channel value of voice"). Optional: `CustomFriendlyName`, `SendDigits`, `Locale`, `CustomCode`, `RateLimits`, `ChannelConfiguration`, `AppHash`, `TemplateSid`, `RiskCheck`, `Tags`, etc. Status: "pending, approved, canceled, max_attempts_reached, deleted, failed or expired". Python:
  ```python
  verification = client.verify.v2.services(
      "VAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  ).verifications.create(to="+15555550101", channel="sms")
  print(verification.sid)
  ```
  — [Twilio: Verification resource](https://www.twilio.com/docs/verify/api/verification)
- Check a code: `POST https://verify.twilio.com/v2/Services/{ServiceSid}/VerificationCheck`; `Code` ("The 4-10 character string being verified"); either `To` or `VerificationSid`. Twilio deletes the verification SID once it is "expired (after 10 minutes)", approved, or "max check attempts reached"; a check against a deleted verification "returns a 404 Not Found error." Python:
  ```python
  verification_check = client.verify.v2.services(
      "VAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  ).verification_checks.create(to="+15555550101", code="1234")
  print(verification_check.status)  # "approved" on success, "pending" on a wrong code
  ```
  — [Twilio: Verification Check resource](https://www.twilio.com/docs/verify/api/verification-check)
- Check-attempt limit: error 60202 "Max (5) verification check attempts reached." (HTTP 429); you "have 5 attempts to check a verification code"; resolution is to wait for the verification to expire (10 minutes) and create a new one. — [Twilio: Error 60202](https://www.twilio.com/docs/api/errors/60202)
- Send-attempt limit: error 60203 (HTTP 429) is triggered when "You have attempted to send the verification code more than 5 times and have reached the limit"; "This rate limit is triggered when the verification lifecycle (sending and checking) is not completed." Resolve by completing the check, waiting 10 minutes for expiry, or cancelling via the Verification Update endpoint (useful in tests). — [Twilio: Error 60203](https://www.twilio.com/docs/api/errors/60203)
- Other 429s: 20429 is the generic too-many-requests error (listed causes include multiple verification starts for the same number in a short period and exceeding your configured Service Rate Limits); 60245 is returned when "a single phone number has received too many verification attempts." Custom per-number / per-IP buckets are configured with Verify Service Rate Limits (key + max count + interval in seconds). — [Twilio: Programmable Rate Limits](https://www.twilio.com/docs/verify/api/programmable-rate-limits); [Error 20429](https://www.twilio.com/docs/api/errors/20429); [Error 60245](https://www.twilio.com/docs/api/errors/60245)
- A2P 10DLC: "Verify is exempt from A2P 10DLC registration when using a provided pooled sender." (post dated 2026-08-13) — [Twilio blog: Migrate from Programmable Messaging to Verify](https://www.twilio.com/en-us/blog/migrate-programmable-messaging-to-verify). Twilio's Sole Proprietor registration page says Verify removes the need for A2P registration (brand or campaign) or any other sender provisioning. — [Twilio: Direct Sole Proprietor Registration Overview](https://www.twilio.com/docs/messaging/compliance/a2p-10dlc/direct-sole-proprietor-registration-overview) (search summary). Separately, "A2P 10DLC registration requires a paid account." — [Twilio trial tutorial](https://www.twilio.com/docs/usage/tutorials/how-to-use-your-free-trial-account)
- Voice channel: set `Channel` to `call` ("`voice` is not a valid Channel value"); "the Verify API will challenge a user with a random keypad digit before reading them the token"; "Voice supports localization for dozens of languages"; "Voice is Twilio's primary backup to SMS for non-smartphone authentication"; "Voice is prioritized on carrier networks and gives the greatest reliability." Verify will also auto-upgrade SMS OTPs to RCS where it performs as well or better. — [Twilio: Verification Channels](https://www.twilio.com/docs/verify/authentication-channels)
- Consent reminder in the SMS channel guide: "Before you send your first OTP, obtain and document recipient consent following the Twilio Messaging Policy." — [Twilio: Verify SMS](https://www.twilio.com/docs/verify/sms)

### Inferences
- Budget: the demo plus rehearsals (say 60 successful OTPs) costs nothing on trial up to 40 successes; past that, each success is $0.05 + $0.0083 ≈ $0.058 (SMS), and every SMS attempt costs $0.0083 even if the user never enters the code. Failed checks are free. A Verify `call` costs $0.05 + Twilio's US outbound voice per-minute rate (not on the Verify pricing page; check Voice pricing).
- Minimal wiring for Callback (my composition from the snippets above; `pip install twilio`):
  ```python
  import os
  from twilio.rest import Client
  from twilio.base.exceptions import TwilioRestException

  client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
  SVC = os.environ[
      "TWILIO_VERIFY_SERVICE_SID"
  ]  # create once: client.verify.v2.services.create(friendly_name="Callback")


  def send_otp(phone_e164: str, channel: str = "sms") -> str:
      # channel="call" = callback-to-number-on-file fallback (keypad challenge, then code read aloud)
      v = client.verify.v2.services(SVC).verifications.create(
          to=phone_e164, channel=channel
      )
      return v.sid  # status "pending"


  def check_otp(phone_e164: str, code: str) -> str:
      try:
          r = client.verify.v2.services(SVC).verification_checks.create(
              to=phone_e164, code=code
          )
          return r.status  # "approved" | "pending" (wrong code)
      except TwilioRestException as e:
          if (
              e.status == 404
          ):  # expired (10 min), already approved, or max 5 checks -> deleted
              return "expired"
          if e.status == 429:  # 60202 max checks / 60203 max sends
              return "locked"
          raise


  def cancel_otp(verification_sid: str) -> None:  # avoid 60203 during tests
      client.verify.v2.services(SVC).verifications(verification_sid).update(
          status="canceled"
      )
  ```
  The Twilio SDK is synchronous; call it via `asyncio.to_thread(...)` from the LiveKit agent so you do not block the audio loop.
- Design hooks: the code is only ever sent to the number on file (never a number spoken by the caller), so the `To` value should come from `users_on_file`, not from the transcript. `DoNotShareWarningEnabled=true` on the Service adds Twilio's "do not share this code" line to the SMS for free, which fits the anti-social-engineering story (not available on trial since templates are fixed; untested).
- Because the trial Verify sender is a Twilio-owned pooled number and 10DLC is handled by Twilio, nothing needs registering for the demo. If the account is upgraded later and you keep using the pooled sender, still no 10DLC registration; only attaching your own purchased numbers to the Service would re-open that question (not documented on the pages fetched).

### Gaps
- Whether a `call`-channel verification on trial consumes trial voice minutes as well as a Verify unit is not stated.
- Default `CodeLength` for a new Service is not stated explicitly (examples show 4; the Console today typically creates 6-digit services, unverified). Set it explicitly.
- Twilio does not publish a fixed default "N verifications per number per 10 minutes" cap beyond the 5 sends / 5 checks per verification; per-number throttling (60245) exists but the threshold is not documented. If the demo needs repeated OTPs to the same phone in quick succession, cancel pending verifications between runs.
- The PCP (Primary Customer Profile) approval step required "to send SMS and Voice OTPs to any number" after upgrade: processing time not stated. Plan on verified numbers only.

---

## 3. Supabase: free tier, Python connections (asyncpg + pooler), supabase-py async, Realtime, RLS for a public read-only dashboard, tables via SQL editor and CLI

### Takeaway
Free tier is enough for the demo (2 active projects, 500 MB Postgres, 200 concurrent Realtime connections, 2M Realtime messages/month) but projects pause after 7 days without activity, so wake it before the final. For a long-running agent process use the session-mode pooler on port 5432 (IPv4) or the direct connection (IPv6 only on free); if you use transaction mode on 6543, asyncpg must be created with `statement_cache_size=0`. Realtime for the dashboard: enable Postgres Changes on the table (`alter publication supabase_realtime add table ...`), enable RLS and add a `select`-only policy for `anon`, and give the browser only the publishable key.

### Cited Findings
- Free plan limits: "Limit of 2 active projects"; 500 MB database per project; 5 GB egress; 1 GB file storage; 50,000 MAU; Realtime "200 concurrent peak connections" and "2 million messages per month"; 500,000 Edge Function invocations; shared CPU with 500 MB RAM; "Free projects are paused after 1 week of inactivity." Pro from $25/month. — [Supabase pricing](https://supabase.com/pricing)
- Connection methods: Direct `postgresql://postgres:[YOUR-PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres` ("Use it for persistent backends, such as virtual machines (VMs) and long-running containers"); shared pooler session mode `postgresql://postgres.[PROJECT-REF]:[YOUR-PASSWORD]@[POOLER-HOST]:5432/postgres` (for persistent servers on IPv4-only networks); shared pooler transaction mode `...@[POOLER-HOST]:6543/postgres` for serverless/edge functions that "open many short-lived connections"; dedicated pooler (Pro+) on 6543. "The shared pooler doesn't support prepared statements in transaction mode." Driver settings: asyncpg `statement_cache_size=0`; Prisma `pgbouncer=true`; Postgres.js/Drizzle `prepare: false`; JDBC `prepareThreshold=0`. "Direct connections use IPv6 on the free plan and on the Pro Plan"; "The shared pooler is IPv4-only on every plan." Use the direct connection for migrations and `pg_dump`. — [Supabase: Connect to your database](https://supabase.com/docs/guides/database/connecting-to-postgres)
- supabase-py async: `from supabase import acreate_client, AsyncClient`; `supabase: AsyncClient = await acreate_client(url, key)`; "Realtime in Python only works with the asynchronous client." Async work from a sync callback (such as the `.subscribe()` status callback) should be scheduled with `asyncio.create_task()`. — [Supabase Python reference: Realtime](https://supabase.com/docs/reference/python/realtime-api)
- supabase-py subscribe examples: `channel = supabase.channel("room1")`; `.on_postgres_changes("INSERT", schema="public", table="countries", callback=handle_record_inserted)`; `.on_postgres_changes("UPDATE", schema="public", table="countries", filter="id=eq.200", callback=handle_record_updated)`; `.on_postgres_changes("*", schema="*", callback=...)`; broadcast `await channel.on_broadcast(event="cursor-pos", callback=handle_broadcast).subscribe(on_subscribe)` with `def on_subscribe(status, err):`. — [Supabase Python reference: Subscribe to channel](https://supabase.com/docs/reference/python/subscribe)
- Postgres Changes: enable per table with `alter publication supabase_realtime add table your_table_name;` or Dashboard > Publications > `supabase_realtime` toggle. JS: `.on('postgres_changes', { event: 'UPDATE', schema: 'public', table: 'todos', filter: 'id=eq.1' }, (payload) => ...).subscribe()`. "Postgres Changes authorizes every event against each subscriber" (RLS applies per subscriber); "RLS policies are not applied to DELETE statements." Old row values on UPDATE/DELETE require `alter table messages replica identity full;`. Changes are processed on a single thread; Supabase recommends Broadcast above roughly 3,000 concurrent subscribers to the same changes. The quick start enables RLS and creates a policy "Allow anonymous access" granting `select` to `anon` with `using (true)`. — [Supabase: Postgres Changes](https://supabase.com/docs/guides/realtime/postgres-changes)
- Broadcast: client `myChannel.send({ type: 'broadcast', event: 'shout', payload: { message: 'Hi' } })`; server-side without subscribing: REST `POST /realtime/v1/api/broadcast` (batch, `messages` array) or `/realtime/v1/api/broadcast/{topic}/events/{event}`; SQL `select realtime.send(payload => '{}'::jsonb, event => 'event', topic => 'topic', private => true);`; "Broadcast from Database" via a trigger calling `realtime.broadcast_changes('topic:' || NEW.id::text, ...)` `AFTER INSERT OR UPDATE OR DELETE ON public.your_table`. "By default, all database broadcasts are private, meaning clients must authenticate to receive them"; "Realtime Authorization is required and enabled by default"; receiving requires a policy on `realtime.messages` (example: `CREATE POLICY "authenticated can receive broadcasts" ON "realtime"."messages" FOR SELECT TO authenticated USING ( true );`) and `await supabase.realtime.setAuth()` in the client. "Subscribing to a channel only to send messages from a server can multiply your Realtime bill." — [Supabase: Broadcast](https://supabase.com/docs/guides/realtime/broadcast)
- RLS: "Enable RLS on every table in an exposed schema." With RLS enabled and no policies, "no data is accessible through the API with a publishable key until you create policies." A grant is required before any policy runs or the request fails with `42501`. Reference pattern:
  ```sql
  alter table public.announcements enable row level security;
  revoke all on table public.announcements from anon, authenticated;
  grant select on table public.announcements to anon, authenticated;
  create policy "Anyone can read announcements"
  on public.announcements for select
  to anon, authenticated
  using ( true );
  ```
  "to anon using ( true )" gives every unauthenticated visitor read access: "Use it only for data that is meant to be public." The secret key "authorizes access through the service_role Postgres role, which has the bypassrls attribute"; "Never use a secret key in the browser or expose it to customers." — [Supabase: Row Level Security](https://supabase.com/docs/guides/database/postgres/row-level-security)
- RLS defaults: RLS is enabled by default on tables created with the dashboard Table Editor; tables created with raw SQL / the SQL Editor are not, so enable it yourself. — [Supabase: Row Level Security (auth guide)](https://www.supabase.com/docs/guides/auth/row-level-security) (search summary; page not fetched directly)
- API keys: new keys are `sb_publishable_...` (safe to expose: "web page, mobile or desktop app, GitHub actions, CLIs, source code") and `sb_secret_...` ("Never put one in a browser, a shipped application, or source control"); legacy `anon`/`service_role` JWTs still work ("Both key systems work at the same time") but "Supabase is deprecating the anon and service_role keys by the end of 2026." Suggested env names: `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY` (browser), `SUPABASE_URL`, `SUPABASE_SECRET_KEY` (server only). — [Supabase: API keys](https://supabase.com/docs/guides/api/api-keys)
- CLI migrations: `supabase migration new create_employees_table` creates `supabase/migrations/<timestamp>_create_employees_table.sql`; `supabase migration up` applies locally; `supabase db reset` reapplies migrations and seed locally; for a hosted project `supabase login`, `supabase link` (pick the project from the prompt), then `supabase db push --remote` (as written on the page). — [Supabase: Database migrations](https://supabase.com/docs/guides/deployment/database-migrations)

### Inferences
- Which pooler for Callback: the agent worker and detection service are long-lived processes (LiveKit Cloud agent / Modal), so session mode on 5432 (`postgres.[ref]@aws-0-us-west-1.pooler.supabase.com:5432`) is the safer default: prepared statements work, and it is IPv4, which most hosts need. The direct `db.[ref].supabase.co:5432` host is IPv6-only on the free plan and will fail from IPv4-only containers. The internal Notion plan says "asyncpg, pooler port 6543"; that works only with `statement_cache_size=0`, and Vercel serverless routes (if any write from the dashboard) should use 6543.
- asyncpg setup (my composition from the documented settings):
  ```python
  import asyncpg, os

  pool = await asyncpg.create_pool(
      os.environ["SUPABASE_DB_URL"],  # session mode: ...pooler.supabase.com:5432/postgres
      min_size=1,
      max_size=5,
      statement_cache_size=0,  # required only for transaction mode (6543); harmless on 5432
  )
  async with pool.acquire() as con:
      await con.execute(
          "insert into decisions(session_id, action, decision, risk_score, signals) values($1,$2,$3,$4,$5::jsonb)",
          session_id,
          action,
          decision,
          risk,
          json.dumps(signals),
      )
  ```
  Keep `max_size` small: the free plan's shared pooler has a limited pool per project (exact number not verified here).
- Minimal schema for the dashboard (public-readable, no PII): keep `users_on_file`, `consent_log`, `enrollments` (embeddings) with RLS enabled and no anon policy (server writes with the secret key / direct Postgres role, which bypasses RLS), and expose only `sessions` and `decisions` with a `select`-only anon policy. Store `phone_last4`, never the full number, in the exposed tables; the Ramp vendor-account `account_number_last_four` field fits the same pattern.
- Enabling realtime on exactly the two exposed tables: `alter publication supabase_realtime add table public.sessions, public.decisions;` plus `alter table public.decisions replica identity full;` only if you want `payload.old` in UPDATE events (not needed if the dashboard just re-renders `payload.new`).
- Wake the free project before Nov 10 and Nov 19 (open the dashboard or run a query); a paused project returns connection errors and takes a minute or two to restore.
- Table creation: fastest is pasting the SQL into the Dashboard SQL Editor; commit the same SQL under `supabase/migrations/` via `supabase migration new` so Eric can `supabase db push` the identical schema.

### Gaps
- Free-plan pooler pool size / max client connections for Supavisor were not on the pages fetched.
- Supabase's own "Realtime with Next.js" guide exists but its body did not come through the fetch (only the intro); the pattern in section 4 relies on a third-party tutorial and the Postgres Changes docs instead.
- Whether `anon` (publishable key, unauthenticated) can receive private "Broadcast from Database" messages given a `TO anon` policy on `realtime.messages` was not verified; Postgres Changes with an anon `select` policy is the documented anonymous path.

---

## 4. Live dashboard: Next.js (App Router) + Supabase Realtime on Vercel, versus LiveKit data packets

### Takeaway
For a one-day stretch build, write decisions/scores to Supabase from the agent, enable Postgres Changes on `sessions`/`decisions`, and render a single `"use client"` component that loads rows once and then patches state from `postgres_changes` events; deploy to Vercel with only the publishable key. Subscribing the browser directly to LiveKit data packets gives sub-second per-turn updates but requires minting a room token per viewer and keeps no history, so use it only as a second, optional feed for the live risk ticker.

### Cited Findings
- Supabase realtime in Next.js (client pattern): browser client created with `createClient(process.env.NEXT_PUBLIC_SUPABASE_URL!, process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!)`; a hook file starting with `"use client";` that in `useEffect` does `channel = supabase.channel("messages").on("postgres_changes", { event: "INSERT", schema: "public", table: "messages" }, (payload) => { const newMessage = payload.new as Message; setMessages((prev) => [...prev, newMessage]); }).subscribe();` with cleanup `return () => { if (channel) { supabase.removeChannel(channel); } };`; UPDATE handled by checking `payload.eventType === "UPDATE"` and replacing the matching row. — [DEV: Building Real-Time Magic: Supabase Subscriptions in Next.js 15](https://dev.to/lra8dev/building-real-time-magic-supabase-subscriptions-in-nextjs-15-2kmp) (third-party tutorial); Supabase's own guide lives at [Realtime with Next.js](https://supabase.com/docs/guides/realtime/realtime-with-nextjs) (body not retrievable in this run).
- Postgres Changes event shape and filters (`event`, `schema`, `table`, `filter: 'id=eq.1'`), and per-subscriber RLS authorization. — [Supabase: Postgres Changes](https://supabase.com/docs/guides/realtime/postgres-changes)
- LiveKit data packets (Python, from the agent or a hidden participant): `await room.local_participant.publish_data("my payload", reliable=True, destination_identities=["identity1", "identity2"], topic="topic1")`; strings are UTF-8 encoded; omit `destination_identities` to send to the whole room. Reliable packets are retransmitted and ordered ("best-effort, not guaranteed"; nothing is buffered server-side) and capped at 15 KiB of user payload; lossy packets should stay under 1300 bytes. Browser: `room.on(RoomEvent.DataReceived, (payload: Uint8Array, participant, kind) => { const strData = decoder.decode(payload) ... })`. — [LiveKit: Data packets](https://docs.livekit.io/home/client/data/packets/)
- Internal Notion (team plan): the detection service "publishes a risk update on the LiveKit data topic `risk` every ~2 s (`{session_id, spoof_prob, text_risk, flags, agent_likelihood}`)" and the Decision object carries `signals` per layer (voiceprint score/threshold/passed/phrase_match, ...); the dashboard is listed as "Next.js on Vercel, live from Supabase" with "LiveKit Playground + logs" as the fallback.

### Inferences
- Recommended one-day shape:
  1. Agent writes one `sessions` row at call start (`id`, `started_at`, `channel` phone/browser, `phone_last4`, `status`) and upserts a `decisions` row per authorize/step-up/deny with the full `signals` JSON; also upsert a `live_risk` JSON column on `sessions` whenever a `risk` data packet arrives (every ~2 s), so the dashboard only needs Supabase.
  2. `app/page.tsx` (server component) fetches the last 20 sessions with their latest decision using the publishable key (RLS-safe) and passes them as props.
  3. `components/LiveBoard.tsx` (`"use client"`) subscribes to `postgres_changes` on `sessions` (`event: '*'`) and `decisions` (`event: 'INSERT'`), keyed by `session_id`, and renders one card per session with a bar per layer (voiceprint, deepfake, text rubric, OTP, callback) plus the reasons list. Re-render from `payload.new`; no need for `replica identity full`.
  4. Vercel: set `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY` only; never the secret key. No server actions needed; the page is read-only.
  Sketch of the client piece (my composition):
  ```tsx
  "use client";
  import { useEffect, useState } from "react";
  import { createClient } from "@supabase/supabase-js";
  const supabase = createClient(process.env.NEXT_PUBLIC_SUPABASE_URL!, process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY!);
  export function LiveBoard({ initial }: { initial: Session[] }) {
    const [rows, setRows] = useState<Record<string, Session>>(Object.fromEntries(initial.map(r => [r.id, r])));
    useEffect(() => {
      const ch = supabase.channel("callback-live")
        .on("postgres_changes", { event: "*", schema: "public", table: "sessions" },
            p => setRows(r => ({ ...r, [p.new.id]: { ...r[p.new.id], ...(p.new as Session) } })))
        .on("postgres_changes", { event: "INSERT", schema: "public", table: "decisions" },
            p => setRows(r => ({ ...r, [p.new.session_id]: { ...r[p.new.session_id], decision: p.new as Decision } })))
        .subscribe();
      return () => { supabase.removeChannel(ch); };
    }, []);
    /* render cards from Object.values(rows) sorted by started_at desc */
  }
  ```
- Latency: Postgres Changes adds the DB round trip plus Realtime fan-out (typically well under a second at demo scale); for the 2-second `risk` ticker that is fine. If you want the browser to show per-turn scores with no DB hop, add a second feed: mint a viewer-only LiveKit token server-side (`canSubscribe` only, hidden participant) and decode `RoomEvent.DataReceived` on topic `risk`. That costs a token endpoint and a LiveKit connection per viewer and gives no history, so treat it as optional polish.
- Do not subscribe a server to a Supabase channel just to send; the agent should write rows (or call `realtime.send()` / the REST broadcast endpoint) rather than hold a Realtime socket.

### Gaps
- No Supabase-authored Next.js App Router realtime example could be quoted verbatim in this run (the official guide page returned only its intro); the pattern above is standard and consistent with the Postgres Changes docs but came from a third-party tutorial.
- Vercel-specific pricing/limits were not researched (not needed for a read-only static page on the Hobby plan).

---

## 5. Consent and recording compliance (California two-party consent, AI disclosure, BIPA, LiveKit recording facts)

### Takeaway
California's CIPA (Penal Code 632 and 632.7) requires all-party consent before recording a phone call, with $2,500 per violation exposure and active class-action use against AI call vendors in 2025, so every Callback call should open with a short "this call is recorded and analysed by automated systems" line before anything else is said. No California or federal rule in force today requires a customer-service voice agent to announce it is an AI (SB 1001 covers "online" bots with deceptive commercial intent; AB 2905 covers prerecorded robocalls; SB 243 covers companion chatbots; AB 1609, chaptered 2026-09-28, covers businesses over $500M revenue), but saying "I'm an automated assistant" costs one sentence and removes the question. Voiceprints are biometric identifiers under Illinois BIPA, which requires written (electronic signature counts) informed consent and a published retention/destruction schedule; collect the spoken consent plus a tap-to-agree record, keep embeddings only, and delete them after the final. LiveKit does not record a room unless you start an Egress or turn on Agent observability (which uploads session audio to LiveKit Cloud with 30-day retention); streaming audio to your detector or to Modulate is "receiving/analysing" by a third party and should be covered by the same disclosure.

### Cited Findings
- Penal Code 632(a): it is an offence to record "without the consent of all parties to a confidential communication"; punishable by "a fine not exceeding two thousand five hundred dollars ($2,500) per violation" and/or up to one year in county jail (prior conviction: up to $10,000). 632(c): a "confidential communication" is one carried on in circumstances that "reasonably indicate" a party wants it "confined to the parties thereto", excluding communications where "the parties to the communication may reasonably expect that the communication may be overheard or recorded." Mirror shows text updated Jan. 1, 2017; verified Nov. 2, 2026. — [Cal. Penal Code 632 (public.law)](https://california.public.law/codes/ca_penal_code_section_632)
- Penal Code 632.7(a): "Every person who, without the consent of all of the parties to a communication, intercepts or receives and intentionally records, or assists in the interception or reception and intentional recordation of" a communication between cellular/cordless/landline phones is punishable by a fine up to $2,500 (up to $10,000 with a prior) and/or imprisonment. "Communication" includes "communications transmitted by voice, data, or image, including facsimile." Mirror shows text updated Jan. 1, 2023. — [Cal. Penal Code 632.7 (public.law)](https://california.public.law/codes/ca_penal_code_section_632.7)
- AI vendor as third party (CIPA 631/632): in *Taylor v. ConverseNow Technologies, Inc.*, No. 25-cv-00990-SI (N.D. Cal., Judge Illston, Aug. 11, 2025), the court denied dismissal of CIPA 631 and 632 claims against an AI phone-ordering vendor whose system answered Domino's calls; it "adopted the 'capability' over the 'extension' approach" to third-party status because the vendor allegedly used call data to improve its own product, and held that calls containing name, address and card details could be confidential under 632. The alert calls disclosures and consent mechanisms "critical defensive measures". — [Duane Morris alert, Aug. 19 2025](https://www.duanemorris.com/alerts/northern_district_california_allows_cipa_claims_against_ai_pizza_ordering_assistant_0825.html). Similar: *Ambriz v. Google* (N.D. Cal., Feb. 10, 2025) let CIPA claims proceed over Google's Contact Center AI transcribing customer calls. — [ZwillGen summary](https://www.zwillgen.com/?p=24025) (search summary)
- Bot disclosure law (SB 1001, in force since July 1, 2019): B&P 17941(a) makes it unlawful "to use a bot to communicate or interact with another person in California online, with the intent to mislead the other person about its artificial identity for the purpose of knowingly deceiving the person about the content of the communication in order to incentivize a purchase or sale of goods or services in a commercial transaction or to influence a vote in an election"; no liability if the disclosure is "clear, conspicuous, and reasonably designed to inform" people that they are dealing with a bot. — [Cal. B&P 17941 (public.law)](https://california.public.law/codes/ca_bus_and_prof_code_section_17941). 17940 defines "bot" as "an automated online account where all or substantially all of the actions or posts" are not by a person and "online" as "appearing on any public-facing Internet Web site, Web application, or digital application"; "online platform" needs "10,000,000 or more unique monthly United States visitors or users". Telephone calls are not mentioned. — [Cal. B&P 17940 (public.law)](https://california.public.law/codes/ca_bus_and_prof_code_section_17940)
- AB 2905 (effective Jan. 1, 2025) amended Public Utilities Code 2874 for automatic dialing-announcing devices: the unrecorded natural-voice announcement before a prerecorded message must "Inform the person called if the prerecorded message uses an artificial voice", where "Artificial voice" means "a voice that is generated or significantly altered using artificial intelligence." Section updated Jan. 1, 2025. — [Cal. Pub. Util. Code 2874 (public.law)](https://california.public.law/codes/ca_pub_util_code_section_2874)
- SB 243 (companion chatbots): signed Oct. 13, 2025, effective Jan. 1, 2026; operators of "companion chatbots" must disclose the user is talking to an AI, with a private right of action; scope is companion-style social chatbots, not all chatbots. — [Skadden, Oct. 2025](https://skadden.com/insights/publications/2025/10/new-california-companion-chatbot-law); [FPF overview](https://fpf.org/?p=247288)
- AB 1609 "Customer service chatbots" (Right to Human Customer Service Act): "Chaptered by Secretary of State, Chapter 733, Statutes of 2026" on Sept. 28, 2026 (enrolled Sept. 4, 2026). Applies to businesses with over $500 million in annual gross revenue; bars representing a customer-service chatbot as human, requires disclosure where a reasonable person could be misled, requires a path to a human agent (15-minute good-faith connection or an appointment), adds hold-time limits for telephone service; enforced by public prosecutors, up to $5,000 first violation and $10,000 each subsequent, no private right of action. — [CalMatters Digital Democracy: AB 1609](https://calmatters.digitaldemocracy.org/bills/ca_202520260ab1609); [Law Commentary, Sept. 29 2026](https://www.lawcommentary.com/articles/california-ai-customer-service-chatbots-real-person-law). (One legislative tracker still showed "in committee" as of May 6, 2026; the chaptering record above is the more recent and authoritative one.)
- AB 410 (would extend bot disclosure to any bot communicating with a person): held under submission in committee on 2025-08-29; not enacted. — [Digital Democracy: AB 410](https://calmatters.digitaldemocracy.org/bills/ca_202520260ab410) (search summary)
- FCC, Feb. 8, 2024 (FCC 24-17): AI-generated voices in robocalls are "artificial or prerecorded" voices under the TCPA, so outbound calls using them need prior express consent and must follow the identification, disclosure and opt-out rules; the ruling does not ban AI voices. — [FCC: FCC Makes AI-Generated Voices in Robocalls Illegal](https://www.fcc.gov/document/fcc-makes-ai-generated-voices-robocalls-illegal); [Wiley alert](https://www.wiley.law/alert-FCC-Extends-Regulatory-Reach-Over-AI-Announces-TCPA-Restrictions-Cover-AI-Generated-Voices-in-Outbound-Calls)
- FCC NPRM FCC 24-84 (adopted Aug. 7, 2024; Federal Register Sept. 10, 2024, CG Docket 23-362): proposed a definition of "AI-generated call", AI-specific consent disclosures and an in-call disclosure that AI is being used. — [Federal Register, 89 FR 2024-19028](https://www.govinfo.gov/content/pkg/FR-2024-09-10/html/2024-19028.htm). As of Jan. 27, 2025 NCLC reported no final FCC action on it; the Eleventh Circuit vacated the separate one-to-one consent rule on Jan. 24, 2025 (*Insurance Marketing Coalition v. FCC*). — [NCLC: Top six TCPA developments 2024-2025](https://library.nclc.org/article/top-six-tcparobocall-developments-20242025). A 2026 vendor guide states the AI-disclosure NPRM was still not finalised as of May 2026. — [CallSphere, 2026](https://callsphere.ai/blog/vw2d-fcc-ai-call-disclosure-rules-2026) (vendor blog; unverified against the FCC docket)
- BIPA definitions (740 ILCS 14/10): "Biometric identifier" means "a retina or iris scan, fingerprint, voiceprint, or scan of hand or face geometry"; "biometric information" is "any information, regardless of how it is captured, converted, stored, or shared" based on an identifier and used to identify a person; "written release" means "informed written consent, electronic signature", or an employment release; "electronic signature" is "an electronic sound, symbol, or process attached to or logically associated with a record" and adopted with intent to sign. FindLaw text current as of Jan. 1, 2025. — [740 ILCS 14/10 (FindLaw)](https://codes.findlaw.com/il/chapter-740-civil-liabilities/il-st-sect-740-14-10/)
- BIPA duties (740 ILCS 14/15): (a) a publicly available written retention schedule and destruction guidelines; destroy when the purpose is satisfied or "within 3 years of the individual's last interaction with the private entity", whichever is first; (b) before collecting, (1) inform in writing that a biometric identifier is being collected or stored, (2) inform in writing of the "specific purpose and length of term", (3) obtain a written release; (c) no selling or profiting; (d) no disclosure without consent except listed cases; (e) protect with "the reasonable standard of care within the private entity's industry". — [740 ILCS 14/15 (FindLaw)](https://codes.findlaw.com/il/chapter-740-civil-liabilities/il-st-sect-740-14-15/)
- BIPA remedies (740 ILCS 14/20): private right of action; $1,000 liquidated or actual damages for negligent violations, $5,000 for intentional or reckless, plus attorneys' fees; repeated collection of the same identifier from the same person by the same method is "a single violation of subsection (b) of Section 15" with at most one recovery (same for repeated disclosures under 15(d)). — [740 ILCS 14/20 (FindLaw)](https://codes.findlaw.com/il/chapter-740-civil-liabilities/il-st-sect-740-14-20/). These per-person accrual and electronic-signature changes came from SB 2979, signed Aug. 2, 2024. — [Fisher Phillips](https://www.fisherphillips.com/en/news-insights/breaking-bipa-news-illinois-curtails-massive-damage-awards-in-biometric-cases.html); [King & Spalding](https://www.kslaw.com/news-and-insights/illinois-bipa-reform-takes-effect)
- LiveKit recording facts: Egress is started explicitly (`StartEgress`) or via auto egress ("It starts recording automatically when a room is created" only when the `egress` field is set in `CreateRoom`); room composite egress "stops automatically when the room ends"; audio-only egress "is billed at a lower rate than video". — [LiveKit: Egress overview](https://docs.livekit.io/home/egress/overview/). Python: `req = api.RoomCompositeEgressRequest(room_name="my-room", audio_only=False, segment_outputs=[api.SegmentedFileOutput(filename_prefix="my-output", s3=api.S3Upload(bucket="my-bucket", ...))])`; `async with api.LiveKitAPI() as lkapi: res = await lkapi.egress.start_room_composite_egress(req)`. — [LiveKit: Egress examples](https://docs.livekit.io/reference/other/egress/examples.md)
- LiveKit Agent observability: once "Agent observability" is turned on in project settings, LiveKit Cloud "collects transcripts, traces, logs, and audio recordings for each session"; "The project setting governs every session"; per session, "pass the `record` parameter to `AgentSession.start()`", e.g. `record={"audio": False}`; "By default, agent observability data is deleted 30 days after a session ends"; PII redaction "is off by default"; recordings "include both the agent and the user audio, on separate channels" and are uploaded after the session ends. — [LiveKit: Agent insights](https://docs.livekit.io/testing/observability/insights.md); the recording page states "Audio recordings are automatically collected and uploaded to LiveKit Cloud for each session" when observability is on. — [LiveKit: Session recording](https://docs.livekit.io/testing/observability/recording.md)
- Internal Notion: the plan lists a `consent_log` (append-only) table, `enrollments` holding "embeddings only", a "Draft the consent form" task for Nov 4, and "Never cut: fail-closed behavior, consent, the word [check]".

### Inferences
- Opening line (covers CIPA all-party consent, the third-party analysis issue raised by *ConverseNow*, and the AI-identity question in one breath; keep it before any question): "Hi, this is Callback, Ramp's automated voice assistant. This call is recorded and analysed by automated fraud-detection systems, including our service providers, to verify your identity. If that's not okay, you can hang up now or say 'human' to stop." If the caller stays on and speaks, that is implied consent under 632(c)'s "may reasonably expect ... recorded" carve-out; log the timestamp and the exact disclosure version in `consent_log`.
- Voiceprint enrollment script (spoken consent, recorded, plus an electronic record to satisfy BIPA's "written release" via "electronic sound ... or process"): "To protect your account we can create a voiceprint, a mathematical model of your voice. We'll store only that model, not the recording, use it only to confirm it's you on future calls, and delete it within 30 days or sooner if you ask. Say 'I agree' to create your voiceprint, or 'no' to skip." Store `consent_id`, user, timestamp, script version, channel, and the ASR text of the answer; link each `enrollments` row to a `consent_id` (the internal plan already does this). Publish the retention text (the sentence above) in the repo README or a `/privacy` route; that satisfies BIPA 15(a)'s "publicly available" policy for a demo.
- For demo users in Illinois (or Texas/Washington, which also regulate voiceprints), use the same flow plus a tap-to-agree checkbox on the enrollment web page so there is an unambiguous electronic signature; honestly, the easiest hackathon answer is to enroll only California-based teammates and friends.
- "Does sending audio to a detection service count as recording?" Under 632/632.7 the operative acts are recording and (for mobile calls) intercepting/receiving plus recording; under 631 it is a third party learning contents in transit. Your own detector buffering windows of audio for inference is a recording even if it keeps nothing afterwards, and shipping clips to Modulate's deepfake API is disclosure to a third-party processor. The disclosure above covers both; also keep the detector from persisting raw audio (embeddings and scores only), which is both the privacy-minimising choice and what the plan already says.
- Decide explicitly whether the demo records at all: (1) LiveKit Egress off and Agent observability off means no persisted audio except what your detector holds in memory; (2) if you turn on Agent observability for debugging, you are recording to LiveKit Cloud (30-day retention) and should say "recorded", or pass `record={"audio": False}` per session. Saying "recorded and analysed" in all cases is the simplest safe choice.
- AI identity: nothing in force today obliges a hackathon voice agent to say it is an AI for an inbound support call in California (17941 is "online" plus deceptive commercial intent; 2874 is prerecorded ADAD calls; AB 1609 starts at $500M revenue). The FCC's Feb. 2024 ruling matters only for outbound calls with an AI voice: your Verify `call` fallback is Twilio's TTS reading a code to a number the user just asked to be called on, which is a consented, user-initiated call, and the "callback" leg should always be user-triggered for the same reason. Still open with "automated voice assistant"; it is free and judges will ask.

### Gaps
- Could not fetch the official leginfo text for Penal Code 632/632.7, B&P 17940-17943, or AB 1609's chaptered text; mirrors were used. The effective date of AB 1609 (likely Jan. 1, 2027, as for most 2026 chaptered bills) was not confirmed.
- Current status of the FCC AI-disclosure NPRM after May 2026 is unverified (only a vendor blog covers 2026); no FCC Report and Order adopting it was found.
- No source was found on whether California's CCPA biometric provisions or any 2025–2026 California biometric-specific statute would apply to a student demo (CCPA has revenue/volume thresholds a hackathon project will not meet; not verified here).
- LiveKit Cloud egress pricing and whether the free/Build tier includes egress minutes were not retrieved (not needed if you do not record).

---

## 6. Hackathon logistics (BBS x Ramp): Round 2 requirements and the Nov 19 final

### Takeaway
Publicly, only the Luma info-session page is findable: a three-round format (video pitch; "walk us through the product or show us the code"; a final "judged live in person by Ramp engineers"), $2,500 in prizes and guaranteed Ramp interviews for finalists. The specific Round 2 requirements the team believes (public repo, no API keys, demo video) and the Nov 10 11:59 PM PT / Nov 19 at Ramp SF dates appear only in the team's internal Notion, not on any public page found.

### Cited Findings
- Public: "Berkeley Business Society and Ramp are hosting a three-round hackathon. The challenge: pick an outdated financial workflow and rebuild it for 2026 with the newest AI tools." Lanes: "Design it: pitch a product people would actually use." / "Build it: ship working code." Rounds: "Video pitch: Sell us your idea." → "Go deeper: Walk us through the product or show us the code." → "The final: Judged live in person by Ramp engineers." "$2500 in prizes." "Finalists present directly to Ramp's team, and final-round participants get guaranteed interviews with Ramp." Open to "All Berkeley students, any major, any experience level." Hosts: Richard Noguchi and Cornellius Suhartono. No dates, submission rules, judging criteria or team size are listed. — [Luma: BBS x Ramp Hackathon Info Session](https://luma.com/1gj6jag5)
- Not this event: the other Ramp Luma page is "Builders Cup 2026 - A Hackathon by Ramp", a one-day in-person hackathon in New York (Flatiron) with crowd-voted tracks. — [Luma: Builders Cup 2026](https://luma.com/nrletiv9)
- Internal Notion ("BBS x Ramp: Callback", last edited 2026-11-05): "Round 2 due: Nov 10, 11:59 PM PT · Live final: Nov 19 at Ramp SF · Repo: vsahasi/callback (private)"; schedule row "Nov 10: Freeze by 8 PM. One pass on the test set. Results table and the clone figure. Record the submission. ... Submitted by 11:59 PM PT."; task "Record demo video and submit Round 2" with Day = 2026-11-10, Notes "Due 11:59 PM PT.", Owner "Both". The page also notes "No new features after Nov 17; dry runs Nov 17–18."

### Inferences
- The repo is currently private; if the Round 2 rule really is "public repo, no API keys", plan a scrub (git history included) before flipping it public: `.env.example` only, rotate any key that ever touched a commit, and keep training audio out of git (the plan already says "no audio in git").
- Nothing public describes the Nov 19 final's format beyond "judged live in person by Ramp engineers", so assume a live demo plus Q&A and rehearse the five scripted calls against the mock-detector fallback.

### Gaps
- No public source for the Round 2 submission requirements (public repo / no API keys / demo video length) or for the final's venue, time and format; confirm with the organizers (Luma "Contact the Host") or the BBS brief shared at the info session.
- The internal Notion page was read in keyword slices only (about 71k characters total); the rules may be quoted elsewhere on it.
