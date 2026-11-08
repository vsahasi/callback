# Ramp Developer API (v1) and Sandbox — notes for the Callback Ramp tools

Researched 2026-11-04/05 (US Pacific). Primary sources are Ramp's own machine-readable docs: the plain-text guides under `https://docs.ramp.com/llms-guides/<slug>.txt`, the plain-text API reference `https://docs.ramp.com/llms-api.txt`, and the OpenAPI spec `https://docs.ramp.com/openapi/developer-api.json` (OpenAPI 3.0.2, title "Ramp Developer API", version v1, 2.6 MB, downloaded 2026-11-05). The HTML pages at docs.ramp.com are JavaScript shells; their source comments say to use those text files instead. Every schema fact below marked "(spec)" was read directly from that OpenAPI file; guide facts cite the guide URL. "Inference" and "Gap" sections are my reasoning, not Ramp's statements.

Quick map from Callback tools to endpoints (all paths relative to `https://demo-api.ramp.com` in sandbox, `https://api.ramp.com` in production):

| Callback tool | Endpoint | Scope | Notes |
| --- | --- | --- | --- |
| read vendors | `GET /developer/v1/vendors`, `GET /developer/v1/vendors/{vendor_id}` | `vendors:read` | `name` query filter exists; vendor object has no hold-state field |
| read a vendor's bank accounts (last four) | `GET /developer/v1/vendors/{vendor_id}/accounts` | `vendors:read` | returns `account_number_last_four` only; no full account or routing number anywhere in the response |
| read bills | `GET /developer/v1/bills?vendor_id=...`, `GET /developer/v1/bills/{bill_id}` | `bills:read` | `status_summary` includes `ON_HOLD` |
| hold vendor (protective) | `POST /developer/v1/vendors/{vendor_id}/hold` | `vendors:write` | 204 applied, 202 pending approval; beta per changelog |
| hold bill (protective) | `POST /developer/v1/bills/{bill_id}/hold` | `bills:write` | 200 returns the Bill; `x-beta: true` in spec |
| change vendor bank account (gated) | `POST /developer/v1/vendors/{vendor_id}/update-bank-accounts` | `vendors:write` | adds a new account; response `status` is `approved` or `pending_approval` |
| release vendor hold (gated) | `POST /developer/v1/vendors/{vendor_id}/release` | `vendors:write` | 204 released, 202 pending approval |
| release bill hold (gated) | `POST /developer/v1/bills/{bill_id}/release` | `bills:write` | 200 returns the Bill |

## OAuth: token endpoint, client-credentials request, lifetime, scopes

### Takeaway
Client credentials against `POST {base}/developer/v1/token` with HTTP Basic auth and a form body; the token lasts 10 days, carries only the scopes you request (which must be a subset of the scopes configured on the app), and the Callback tools need `vendors:read vendors:write bills:read bills:write` (plus `business:read` only if you want `GET /business`). The authorization-code flow is not needed for Bill Pay on your own sandbox business.

### Cited Findings
- Token endpoint and hosts: "Replace `{environment-url}` with: `https://demo-api.ramp.com` for Sandbox, `https://api.ramp.com` for Production", and "The API base URL is not the same as the URL for the Ramp customer app" (Production app `https://app.ramp.com`, Sandbox app `https://demo.ramp.com`) — [Quickstart](https://docs.ramp.com/developer-api/v1/getting-started) ([text](https://docs.ramp.com/llms-guides/getting-started.txt))
- The documented client-credentials request, copied from the Quickstart:
  ```bash
  curl -X POST '{environment-url}/developer/v1/token' \
    -H "Authorization: Basic $(echo -n "$CLIENT_ID:$CLIENT_SECRET" | base64)" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "grant_type=client_credentials" \
    -d "scope=transactions:read"
  ```
  Response:
  ```json
  {
    "access_token": "ramp_business_tok_abc123...",
    "token_type": "Bearer",
    "expires_in": 864000,
    "scope": "transactions:read"
  }
  ```
  — [Quickstart](https://docs.ramp.com/developer-api/v1/getting-started)
- "Client Credentials access tokens expire after 10 days. Client Credentials grants do not return a refresh token; request a new access token when the current token expires." and "If you omit it [`scope`] from a Client Credentials request, Ramp issues a token with no scopes" — [Quickstart](https://docs.ramp.com/developer-api/v1/getting-started)
- `POST /developer/v1/token` (spec, `security: []`): form fields `grant_type` REQUIRED enum `[authorization_code, client_credentials, refresh_token]` ("Refresh token and client credentials grants must be added to app's permitted Grant Types"), `scope` ("Space-separated list of scopes to be granted to the returned token. These must be a subset (including full set) of the app's allowed scopes. If omitted for grant_type=client_credentials, the returned token has no scopes."), `code`, `redirect_uri`, `refresh_token`. Response 200 fields: `access_token` ("Prefixed with ramp_business_tok_, ramp_user_tok_, ramp_business_jwt_, or ramp_user_jwt_"), `expires_in` (seconds), `id_token`, `refresh_token` ("won't appear in the response for client_credentials grant type"), `refresh_token_expires_in` (nullable), `scope`, `token_type` ("should always be 'Bearer'") — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)
- The spec's `securitySchemes.oauth2` lists two flows, `authorizationCode` (tokenUrl `https://api.ramp.com/developer/v1/token`, authorizationUrl `https://api.ramp.com/v1/authorize`) and `clientCredentials` (same tokenUrl), each with the full scope list, including `vendors:read`, `vendors:write`, `bills:read`, `bills:write`, `business:read`, `accounting:read`, `payments:read`, `payments:write`, `approvals:read`, `approvals:write`, `approval_chains:write`, `audit_logs:read`, `bank_accounts:read`, `bank_accounts:write`, `entities:read`, `users:read`, `webhooks` has no scope of its own — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)
- Scope descriptions: `bills:read` "Access bill data and payment history"; `bills:write` "Create, update, and pay bills"; `vendors:read` "View vendor information and payment details"; `vendors:write` "Create and update vendor information"; `business:read` "Access business information and settings"; `accounting:read` "Access accounting sync status and metadata"; `payments:read` "View vendor payments"; `approvals:write` "Approve or reject pending approvals". "Each endpoint's API reference lists its required scopes." "Some endpoints also require a Ramp plan feature and return `403` without it, even with the right scope." — [Authorization guide](https://docs.ramp.com/developer-api/v1/authorization) ([text](https://docs.ramp.com/llms-guides/authorization.txt))
- Where scopes are chosen: in the app, "Under **Scopes**, click **Configure allowed scopes** and select `transactions:read`" (Quickstart step 1.5); in the request, "Pass the scopes you need as a space-separated `scope` field. If you omit `scope`, the token has no scopes." — [Quickstart](https://docs.ramp.com/developer-api/v1/getting-started); [Authorization guide](https://docs.ramp.com/developer-api/v1/authorization)
- Basic auth vs body: "Send `client_id:client_secret`, base64-encoded, in an `Authorization: Basic ...` header. Some apps are configured to send `client_id` and `client_secret` in the form body instead." — [Authorization guide](https://docs.ramp.com/developer-api/v1/authorization)
- Attribution for client-credentials actions: "The token has business-wide access limited by those scopes, and there's no end user. Actions that need a user are attributed to the app's creator by default." — [Authorization guide](https://docs.ramp.com/developer-api/v1/authorization)
- Lifetimes: "Client Credentials access tokens last 10 days (864,000 seconds). Authorization Code and Refresh Token access tokens default to 1 hour (3,600 seconds)"; "Tokens work only in the environment that issued them. Sandbox and production need separate apps." — [Authorization guide](https://docs.ramp.com/developer-api/v1/authorization)
- Token request errors: "Failed requests to `/developer/v1/token` return HTTP `400` with error code `DEVELOPER_7012`" with messages "Client authentication failed.", "The provided authorization grant or refresh token is invalid.", "The requested scope is invalid, unknown, malformed, or exceeds the scope granted by the resource owner." (fix: "Request only scopes configured for your app"), "The authenticated client is not authorized to use this authorization grant type." (fix: "Enable the grant type in your app settings"); "If the client credentials are missing or malformed, the request returns HTTP `401`." — [Authorization guide](https://docs.ramp.com/developer-api/v1/authorization)
- API-call errors: 401 "The token is missing, expired, or invalid"; 403 "The token lacks a required scope"; 429 see rate limiting — [Authorization guide](https://docs.ramp.com/developer-api/v1/authorization)
- Revoke: `POST /developer/v1/token/revoke` with the same Basic credentials, form field `token` (REQUIRED) and optional `token_type_hint` enum `[access_token, refresh_token]`; returns 200 — [Authorization guide](https://docs.ramp.com/developer-api/v1/authorization); [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)
- Production token-issuance limit: "100 `POST /developer/v1/token` requests per 10 seconds per source IP. Reuse access tokens until they expire rather than requesting a token for each API call." — [Rate Limits guide](https://docs.ramp.com/developer-api/v1/rate-limiting)
- Authorization-code flow (only if you ever act for another business): redirect to `https://app.ramp.com/v1/authorize` (sandbox `https://demo.ramp.com/v1/authorize`) with `response_type=code`, `scope`, `client_id`, `redirect_uri` (exact match; `https://`, or `http://` only for localhost/127.0.0.1), `state` (required for apps created after June 15, 2023); codes expire after 5 minutes; exchange at `POST {environment-url}/developer/v1/token` with `grant_type=authorization_code`, `code`, `redirect_uri`; refresh with `grant_type=refresh_token` (requires the Refresh Token grant type enabled; no new refresh token is returned); "For apps that access the whole business, an Admin or Business Owner must approve." — [Authorization guide](https://docs.ramp.com/developer-api/v1/authorization)
- The spec's `authorizationUrl` is `https://api.ramp.com/v1/authorize`, whereas the guide's browser URLs are `https://app.ramp.com/v1/authorize` / `https://demo.ramp.com/v1/authorize` — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json) vs [Authorization guide](https://docs.ramp.com/developer-api/v1/authorization)

### Inferences
- For Callback, request exactly `scope=vendors:read vendors:write bills:read bills:write` (space-separated, URL-encoded in the form body) after enabling those four scopes on the sandbox app; add `business:read` only if a "who am I" call (`GET /developer/v1/business`) is wanted. Nothing in the vendor/bill endpoints used by Callback requires `accounting:read`.
- Because the token lasts 10 days and issuance is rate limited, cache the token in process (and on disk if the agent restarts often) and refresh on a 401 or when `expires_in` is near; do not mint a token per call.
- The `authorizationUrl` discrepancy is irrelevant for client credentials; if you ever do the browser flow, follow the guide's `app.ramp.com`/`demo.ramp.com` URL builder, not the spec.

### Gaps
- The docs do not state the sandbox app's allowed scope catalog differs from production; the Authorization guide notes only that "partners only" scopes appear in the partner console. Whether `vendors:write`/`bills:write` are selectable on a freshly provisioned sandbox app is not documented (it should be, since the Jan 17, 2025 changelog says all bills/vendors endpoints work in sandbox; see below).
- Whether the token request form body is `application/x-www-form-urlencoded` only (yes per spec) and whether JSON bodies are rejected (not documented; assume rejected).

## Sandbox: access, app creation, seed data, demo actions, base URLs

### Takeaway
Sandbox access is a self-serve form at `docs.ramp.com/developer-api/v1/sandbox-access` (Ramp says it is ready in about a minute); you then create a separate developer app inside `demo.ramp.com` to get sandbox credentials, call `https://demo-api.ramp.com`, and use the ⌘J demo-actions panel to simulate paying bills. Ramp documents sample transactions, cards, users, departments and reimbursements, but says nothing about seeded vendors, bills or vendor bank accounts, and nothing about hackathons.

### Cited Findings
- Sandbox environment table: "Ramp frontend https://demo.ramp.com — Access the Ramp user interface for testing"; "API https://demo-api.ramp.com — Base URL for API calls in the Sandbox environment"; key features include "No real money movement: Demo accounts cannot process actual financial transactions" — [Sandbox guide](https://docs.ramp.com/developer-api/v1/sandbox) ([text](https://docs.ramp.com/llms-guides/sandbox.txt))
- Access request: "Need a test environment for development? [Request sandbox access](/developer-api/v1/sandbox-access)" — [Quickstart](https://docs.ramp.com/developer-api/v1/getting-started). The sandbox-access page is a form; its plain-text export (`/llms-guides/sandbox-access.txt`, titled "Get Sandbox Access") has no body text — [Get Sandbox Access](https://docs.ramp.com/developer-api/v1/sandbox-access)
- Ramp Support: "Request access using the sandbox access form" at `docs.ramp.com/developer-api/v1/sandbox-access`; "the sandbox should be ready in about one minute, and you should check your inbox for an account access email"; Ramp provisions a sandbox business "pre-populated with sample data" that includes transactions, cards, users, departments, and reimbursements; "You need a separate developer app in the sandbox to get sandbox Client ID and Secret"; sandbox base URL `https://demo-api.ramp.com`, production `https://api.ramp.com`; the page does not mention hackathon or startup programs — [Accessing the Developer API (Ramp Support)](https://support.ramp.com/hc/en-us/articles/46681939909907-Accessing-the-Developer-API)
- App creation steps: "In your Ramp account, go to **Company → Developer** (you'll need admin access)"; "Click **Create New App**"; "Name your app ... and accept the Terms"; "Under **Grant types**, click **Add new grant type** → select **Client Credentials**"; "Under **Scopes**, click **Configure allowed scopes**"; "Copy your **Client ID** and **Client Secret**" — [Quickstart](https://docs.ramp.com/developer-api/v1/getting-started). Support gives the direct path `app.ramp.com/settings/ramp-developer/apps` ("Start building" for the first app, "Create new app" afterwards) and says "Admin access is required" — [Ramp Support](https://support.ramp.com/hc/en-us/articles/46681939909907-Accessing-the-Developer-API)
- Demo actions panel: "You can open the demo actions panel directly within the Ramp sandbox with ⌘ J." Actions: "Mark bill as paid — navigate to Bill Pay via the UX, then open the bill ... Press ⌘ J ... select 'pay current bill'. **Note:** This demo action depends on your Bill Pay payment release setting. When enabled, it only works for bills that are approved and scheduled. When disabled, it works for all bills."; "Mark reimbursement as paid"; "Create transaction — ... click on 'add transactions'". Permissions: mark bill as paid needs "Admin or Business Owner role, or the AP Clerk permission" — [Sandbox guide](https://docs.ramp.com/developer-api/v1/sandbox)
- Vendors/bills work in sandbox: changelog entry "January 17, 2025 — Bills and Vendors API Launch: ... All endpoints are accessible in both Sandbox and Production environments" — [Changelog](https://docs.ramp.com/developer-api/v1/changelog) ([text](https://docs.ramp.com/llms-guides/changelog.txt))
- Webhooks guide recommends sandbox for mock events and shows `https://demo-api.ramp.com/developer/v1/webhooks/mock-webhook-event` with `$RAMP_SANDBOX_API_TOKEN` — [Webhooks guide](https://docs.ramp.com/developer-api/v1/webhooks)
- Ramp CLI: "The CLI uses production by default. To use the Sandbox, pass `--env sandbox` or set it as your default" — [CLI guide](https://docs.ramp.com/developer-api/v1/cli)
- Hosted MCP sample data: "To try Ramp MCP with sample data, replace `https://mcp.ramp.com/mcp` with `https://demo-mcp.ramp.com/mcp` in any config below." — [Ramp MCP guide](https://docs.ramp.com/developer-api/v1/ramp-mcp)
- Reachability check (my probe, 2026-11-05): an unauthenticated `POST https://demo-api.ramp.com/developer/v1/token` returned HTTP 401, and `POST https://demo-mcp.ramp.com/mcp` returned 401 with OAuth protected-resource metadata whose `scopes_supported` include `vendors:read`, `vendors:write`, `bills:read`, `bills:write` — direct observation, not a doc.
- Bill Pay funding prerequisite (for creating bills): "Bill Pay must be enabled for your organization. For bank-funded payments, call `GET /entities` and confirm the selected entity has a bank account with `usage_type = BILL_PAY_BANK_ACCOUNT`." — [Bill Payments guide](https://docs.ramp.com/developer-api/v1/bill-payments)

### Inferences
- Create the sandbox developer app at `https://demo.ramp.com/settings/ramp-developer/apps` (the production path with the sandbox host); the docs only give the production path.
- The demo-actions note about the "payment release setting" matters for Callback demos: if payment release is enabled in the sandbox business, ⌘J "pay current bill" only works on approved and scheduled bills, so a held bill (descheduled) should not be payable, which is a nice demo of the hold.
- Because seeded vendors/bills are undocumented, plan to create your own fixtures on Nov 6: `POST /developer/v1/vendors` (auto-approved), `POST /developer/v1/vendors/{id}/update-bank-accounts` (ACH), then `POST /developer/v1/bills` (needs `entity_id` and, for ACH, `source_bank_account_id` from `GET /entities`).

### Gaps
- No documentation of which vendors, bills, or vendor bank accounts the sandbox is seeded with, whether Bill Pay is pre-enabled in the sandbox business, or whether an entity with a `BILL_PAY_BANK_ACCOUNT` exists there. Verify with `GET /developer/v1/vendors?page_size=100`, `GET /developer/v1/bills`, `GET /developer/v1/entities` on day one.
- No hackathon- or student-specific sandbox program appears in Ramp docs, Ramp Support, or search results; the generic form is the only documented path (a web search summary mentioned a Ramp support page saying sandbox is "not self-serve", but I could not locate that page; the Support article above describes the form as self-serve).
- Whether the sandbox business's approval policies are configured so that `update-bank-accounts` returns `approved` vs `pending_approval` (see Vendors) is not documented; it determines whether the "change bank account" tool applies immediately.

## Vendors: list/get/create/update, bank accounts, update-bank-accounts, hold/release

### Takeaway
All vendor endpoints live under `/developer/v1/vendors`; bank accounts are read via `GET /vendors/{id}/accounts` (last four only, with `status` and `proposal_id` as of Nov 4, 2026), added via `POST /vendors/{id}/update-bank-accounts` (response `status` is `approved` or `pending_approval`, with `proposal_id` when approval is needed), and holds via `POST /vendors/{id}/hold` / `/release` (204 applied, 202 pending approval; labeled beta in the July 15, 2026 changelog). The vendor object itself exposes no hold flag.

### Cited Findings

Endpoint inventory with scopes (spec):
- `GET /developer/v1/vendors` — `vendors:read` — List vendors
- `POST /developer/v1/vendors` — `vendors:write` — Create a new vendor
- `GET /developer/v1/vendors/{vendor_id}` — `vendors:read` — Fetch a vendor
- `PATCH /developer/v1/vendors/{vendor_id}` — `vendors:write` — Update a vendor
- `DELETE /developer/v1/vendors/{vendor_id}` — `vendors:write` — Delete a vendor ("A vendor cannot be deleted if it has associated transactions, bills, contracts, spend requests, or child vendors"; 204)
- `GET /developer/v1/vendors/{vendor_id}/accounts` — `vendors:read` — List vendor bank accounts
- `GET /developer/v1/vendors/{vendor_id}/accounts/{bank_account_id}` — `vendors:read` — Fetch a vendor bank account
- `POST /developer/v1/vendors/{vendor_id}/accounts/{bank_account_id}/archive` — `vendors:write` — Archive a vendor bank account
- `POST /developer/v1/vendors/{vendor_id}/update-bank-accounts` — `vendors:write` — "Add to a vendor's bank account details"
- `GET /developer/v1/vendors/{vendor_id}/payment-methods` — `vendors:read` — List available payment methods (`x-beta: true`)
- `POST /developer/v1/vendors/{vendor_id}/hold` — `vendors:write` — "Place a hold on a vendor"
- `POST /developer/v1/vendors/{vendor_id}/release` — `vendors:write` — "Release Vendor Hold"
- `GET /developer/v1/vendors/{vendor_id}/contacts` — `vendors:read`; `POST .../contacts` — `vendors:write`; `GET .../contacts/{vendor_contact_id}` — `vendors:read`
- `GET /developer/v1/vendors/{vendor_id}/children` — `vendors:read`; `GET /developer/v1/vendors/{vendor_id}/credits`, `GET /developer/v1/vendors/credits`, `GET /developer/v1/vendors/credits/{vendor_credit_id}` — `vendors:read`; vendor agreements endpoints under `/vendors/agreements` and `/vendors/{vendor_id}/agreements`
— [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json); human pages at [Vendors reference](https://docs.ramp.com/developer-api/v1/api/vendors)

`GET /developer/v1/vendors` (spec):
- Query params: `external_vendor_id` (string), `merchant_id` (uuid), `accounting_vendor_remote_ids` (comma-separated strings), `vendor_tracking_category_option_ids` (comma-separated UUIDs), `sk_category_ids` (comma-separated ints), `from_created_at`, `to_created_at`, `from_updated_at`, `to_updated_at` (ISO 8601), `start` (cursor: "The ID of the last entity of the previous page"), `page_size` (int, default 20, "must be between 2 and 100"), `vendor_owner_id` (uuid), `include_subsidiary` (bool, default false), `include_draft` (bool, default false: "include vendors that have not yet been fully approved and published: both plain drafts and vendors that were created via POST /developer/v1/vendors but are still waiting on creation approval"), `is_active` (bool), `name` (string) — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)
- Response `{ "data": [Vendor...], "page": { "next": string|null } }`. Vendor list item fields: `id` (uuid), `name`, `display_name` (nullable), `name_legal`, `description`, `is_active`, `is_deletable`, `country` (2-letter), `state`, `address` (default address: `address_line_1`, `address_line_2`, `city`, `country`, `postal_code`, `state`), `addresses[]` (`id`, `is_default`, `pay_to_name`, address fields), `tax_address`, `contacts[]` (array of contact UUIDs), `default_entity_id` (nullable), `default_payment_method` { `policy` oneOf { `kind` enum `[ADDRESS, BANK_ACCOUNT, CARD, DIRECT_DEBIT, MANUAL]`, `data` (address object or bank-account object), `card_payment_method`, `payment_method` }, `update_source` enum `[AUTO, MANUAL]` nullable }, `external_vendor_id` (nullable), `accounting_vendor_remote_id` (nullable), `merchant_id`, `parent_vendor_id` (nullable), `vendor_owner_id` (nullable), `vendor_type` enum `[BUSINESS, INDIVIDUAL]`, `federal_tax_classification`, `billing_frequency`, `net_payment_terms` (nullable int), `sk_category_id`, `sk_category_name`, `subsidiary[]` (only with `include_subsidiary=true`), `total_spend_all_time` / `total_spend_last_30_days` / `total_spend_last_365_days` / `total_spend_ytd` (each `{amount, currency_code, minor_unit_conversion_rate}`), `created_at`, `updated_at` (nullable). The single-vendor response (`GET /vendors/{vendor_id}`) adds `approval_status` enum `[APPROVED, PENDING, REJECTED]`, `custom_form_collection_response`, `custom_record_fields` — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)
- The Vendor schema contains no property whose name includes "hold", "blocked" or "frozen" (checked programmatically across the fully dereferenced schema) — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)
- `updated_at` was added Oct 8, 2026 ("Vendor update timestamps"); `display_name` Oct 28, 2026; `approval_status` July 30, 2026; `addresses[]` June 22, 2026 — [Changelog](https://docs.ramp.com/developer-api/v1/changelog)

`POST /developer/v1/vendors` (spec): "Vendors created in the API are approved by default, and are not subject to existing approval policies." Body: `name`, `business_vendor_contacts` REQUIRED (object with `email` REQUIRED, `first_name`, `last_name`, `phone`), `address` (`address_line_1` REQUIRED, `city` REQUIRED, `country` REQUIRED, `postal_code` REQUIRED, `address_line_2`, `state` ("Required for addresses in US, CA, and AU"), `pay_to_name`), `country` ("Defaults to US when omitted or null"), `state`, `default_entity_id`, `external_vendor_id`, `net_payment_terms`, `parent_vendor_id`, `request_payment_details` (bool), `request_tax_details` (bool), `requesting_user_id`, `vendor_owner_id`, `accounting_vendor_remote_id` or `vendor_tracking_category_option_id` (at most one), deprecated `acting_user_id`. Returns 200 with the Vendor object — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json). Changelog Oct 16, 2026: "Create vendors without `country` or `state`. Omitted or `null` `country` defaults to `US` for new vendors; US vendors without `state` remain not bill-ready." — [Changelog](https://docs.ramp.com/developer-api/v1/changelog)

`PATCH /developer/v1/vendors/{vendor_id}` (spec) body: `name`, `display_name` ("An empty string clears it"), `description`, `is_active`, `country`, `state`, `address` {...}, `default_bank_account_id` (uuid: "Unique identifier of an existing bank account belonging to this vendor to use as its default payment method. The account determines the payment method. Existing bills are not changed. Omit to leave the default unchanged."), `default_entity_id` (nullable), `external_vendor_id`, `net_payment_terms` (nullable, 0 = due on receipt), `request_payment_details`, `request_tax_details`, `accounting_vendor_remote_id` / `vendor_tracking_category_option_id`. Returns 200 Vendor — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)

`GET /developer/v1/vendors/{vendor_id}/accounts` (spec):
- Description: "Set include_pending=true to also return bank accounts awaiting approval."
- Query: `start`, `page_size` (default 20, 2–100), `include_pending` (bool, default false: "When true, also return bank accounts proposed in vendor edits that are pending approval. A pending account gets a new id when it is approved.")
- Response `{ "data": [...], "page": { "next": ... } }`; each item: `id` (uuid) REQUIRED, `account_number_last_four` (string, nullable) REQUIRED "Last four digits of the bank account number", `account_nickname` (string), `currency` (ISO 4217 enum, also stablecoin codes such as `USDC`) REQUIRED, `is_default` (bool) REQUIRED "Whether this is the payee's default bank account. For a pending account, whether its vendor edit makes it the default.", `payment_method` enum `[ACH, AUTOMATIC_CARD_PAYMENT, CARD, CHECK, CRYPTO_WALLET_TRANSFER, DIRECT_DEBIT, DOMESTIC_WIRE, EARLY_PAYMENT_DISCOUNT, FED_NOW, INTERNATIONAL, LOCAL_BANK_TRANSFER, ONE_TIME_CARD, ONE_TIME_CARD_DELIVERY, PAID_MANUALLY, RTP, SWIFT, UNSPECIFIED, VENDOR_CREDIT]` REQUIRED, `proposal_id` (uuid, nullable) REQUIRED "Current approval proposal for the vendor edit that adds this bank account. Further changes submitted to the same pending edit replace it, so it may differ from the proposal_id returned on submission. Null unless status is pending_approval.", `status` enum `[active, pending_approval, pending_deletion]` REQUIRED. There is no routing-number field and no full account number — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)
- `GET /vendors/{vendor_id}/accounts/{bank_account_id}` returns the same object; "Also returns bank accounts pending approval, using ids from the list endpoint's include_pending=true response." — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)
- Changelog Nov 4, 2026: "Pending vendor bank accounts — Set `include_pending=true` to list accounts awaiting approval. Bank account list and detail responses include `status` and `proposal_id`; a pending account gets a new ID when approved." Changelog July 21, 2026: "Vendor bank account responses now include `account_number_last_four`" — [Changelog](https://docs.ramp.com/developer-api/v1/changelog)
- `GET /developer/v1/vendors/{vendor_id}/payment-methods` (`x-beta: true`): "Returns ACH and CHECK payment methods with associated fee information." ACH item: `id`, `type` `ACH`, `account_number_last_four`, `routing_number_last_four`, `fee {amount, currency_code, minor_unit_conversion_rate}`; CHECK item: `id`, `type` `CHECK`, `mailing_address {...}`, `fee` — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json); changelog Oct 27, 2026 "Vendor payment methods (beta)" — [Changelog](https://docs.ramp.com/developer-api/v1/changelog)
- `POST /developer/v1/vendors/{vendor_id}/accounts/{bank_account_id}/archive`: body `{ "replacement_bank_account_id": uuid|null }` ("If the bank account has associated bills, drafts, or recurring templates, a replacement_bank_account_id must be provided"); 204 "Bank account archived" — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)

`POST /developer/v1/vendors/{vendor_id}/update-bank-accounts` (spec, `vendors:write`, `x-ramp-plus-required: false`, no `x-beta`):
- Description: "Adds payment details for the vendor through the approval workflow. The proposal may require approval depending on the business's approval policies. Supported payment methods: ACH: US bank account with routing and account numbers; Wire: US wire transfer with routing and account numbers; Crypto: stablecoin wallet with a wallet address and network"
- Request body (`application/json`, required): `account_nickname` (string, optional), `is_default` (bool, default false, "Whether this should be the default payment method"), and one of:
  - `ach_details` { `account_number` REQUIRED, `routing_number` REQUIRED, `account_name` (default "Checking Account"), `account_owner_type` enum `[BUSINESS, INDIVIDUAL]` default `BUSINESS`, `account_type` enum `[Checking, Savings]` default `Checking` }
  - `wire_details` { `account_number` REQUIRED, `routing_number` REQUIRED, `account_name`, `beneficiary_address` { `address_line_1` REQUIRED, `city` REQUIRED, `country` REQUIRED (ISO 3166-1 alpha-2), `postal_code` REQUIRED, `address_line_2`, `state` ("Required for addresses in US, CA, and AU") } }
  - `crypto_details` { `wallet_address` REQUIRED, `network` REQUIRED enum `[ARBITRUM_MAINNET, AVALANCHE_MAINNET, BASE_MAINNET, ETHEREUM_MAINNET, OPTIMISM_MAINNET, PLASMA_MAINNET, POLYGON_MAINNET, SOLANA_MAINNET, STELLAR_MAINNET, SUI_MAINNET, TEMPO_MAINNET]`, `account_name` (nullable) }
- Response 200 "Bank Account Update Response": `vendor_id` (uuid) REQUIRED; `status` REQUIRED enum `[approved, pending_approval]` "Status of the bank account update operation."; `payment_method_id` (uuid, nullable) "Unique identifier of the created payment method. Null when the update requires approval (status is pending_approval)."; `payment_method_type` (nullable) enum `[ach, crypto, wire]` "Will be null if bank validation fails."; `proposal_id` (uuid, nullable) "Unique identifier of the approval proposal, if approval is required." — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)
- Changelog June 25, 2026: "The `POST /developer/v1/vendors/{vendor_id}/update-bank-accounts` endpoint now returns a structured response body with `vendor_id`, `payment_method_id`, `payment_method_type` (`ach` or `wire`), `status` (`approved` or `pending_approval`), and `proposal_id`, giving you visibility into whether the update was applied immediately or requires approval." July 21, 2026: `crypto_details` option added, `payment_method_type` gains `crypto`. Sept 2, 2026: `wire_details.beneficiary_address` added — [Changelog](https://docs.ramp.com/developer-api/v1/changelog)
- Changelog Sept 22, 2026: "Set `default_bank_account_id` on `PATCH /developer/v1/vendors/{vendor_id}` to select an existing bank account belonging to the vendor as its default payment method. Omit it to keep the current default; existing bills are unaffected." — [Changelog](https://docs.ramp.com/developer-api/v1/changelog)
- Same approve-or-pend pattern on contacts: `POST /vendors/{vendor_id}/contacts` (body `email` REQUIRED, `first_name`, `last_name`, `phone`, `is_default`) returns `{ status: approved|pending_approval, contact_id|null, proposal_id|null, vendor_id }` — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)
- Audit log event types `New vendor approval policy updated` and `Vendor edit approval policy updated` exist (June 25, 2026), i.e., vendor edits are governed by a configurable "vendor edit approval policy" — [Changelog](https://docs.ramp.com/developer-api/v1/changelog)

`POST /developer/v1/vendors/{vendor_id}/hold` (spec, `vendors:write`):
- Description: "Blocks payments to this vendor by placing a vendor-scoped hold. Open, unpaid bills for the vendor that can still be modified are removed from scheduled payment flows. An optional `reason` can be supplied to record why the hold was placed."
- Request body (optional, `application/json`): `{ "reason": string|null }` ("Optional reason for placing the vendor on hold.")
- Responses: `202` "Vendor hold pending approval" (JSON body typed only as `object`, no documented fields); `204` "Vendor placed on hold" (no body)
- Spec extensions: `x-alias: "hold"`, `x-destructive: true`, `x-open-world: false`, `x-platforms: ["cli", "mcp", "third_party"]`, `x-ramp-plus-required: false`, `x-read-only: false`; no `x-beta` flag — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json); plain-text reference shows `request: { reason?: string | null }`, `response: Record<string, unknown>` — [llms-api.txt](https://docs.ramp.com/llms-api.txt)

`POST /developer/v1/vendors/{vendor_id}/release` (spec, `vendors:write`):
- Description: "Releases the active vendor-scoped hold and allows new payments to be scheduled for the vendor. Payments that were removed from scheduled payment flows while the hold was active must be scheduled again after the hold is released."
- No request body. Responses: `202` "Vendor hold release pending approval" (body `object`); `204` "Vendor hold released". Extensions: `x-alias: "hold-release"`, `x-destructive: true`, `x-platforms: ["cli", "mcp", "third_party"]` — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)
- Changelog July 15, 2026: "Vendor hold and release endpoints (beta) — Block or resume payments to a vendor via `POST /developer/v1/vendors/{vendor_id}/hold` (with an optional `reason`) and `POST /developer/v1/vendors/{vendor_id}/release`." — [Changelog](https://docs.ramp.com/developer-api/v1/changelog)
- Changelog April 30, 2026: audit log `event_type` values now include `Vendor hold released` and `Vendor placed on hold` (readable via `GET /developer/v1/audit-logs/events`, scope `audit_logs:read`) — [Changelog](https://docs.ramp.com/developer-api/v1/changelog); [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)
- Vendor webhooks: `vendors.activated` "Vendor has been activated", `vendors.approved` "Vendor has been approved", `vendors.updated` "Vendor has been updated" (scope to request `vendors:read`) — [Webhooks guide](https://docs.ramp.com/developer-api/v1/webhooks)
- Vendor contacts list item: `id`, `email` (nullable), `first_name` (nullable), `last_name` (nullable), `phone` (nullable), `is_default` — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)

### Inferences
- "Change a vendor's bank account" is really two Ramp operations: `update-bank-accounts` adds a new account (set `is_default: true` to make it the payout default), then optionally archive the old one with `POST .../accounts/{old_id}/archive` (passing `replacement_bank_account_id` if bills reference it). To merely switch the default among existing accounts, use `PATCH /vendors/{id}` with `default_bank_account_id`. The endpoint never edits an existing account in place.
- "Applied vs needs approval" is encoded by the 200 body: `status == "approved"` with a non-null `payment_method_id` means applied; `status == "pending_approval"` with `payment_method_id == null` and a `proposal_id` means it is sitting in the vendor-edit approval workflow. After a pending result, `GET /vendors/{id}/accounts?include_pending=true` will show the proposed account with `status: pending_approval` (and, per the docs, a different id once approved). Treat the account id as unstable until `status == "active"`.
- Request skeleton constructed from the schema (not a Ramp example):
  ```json
  POST /developer/v1/vendors/{vendor_id}/update-bank-accounts
  {
    "account_nickname": "Operating - Chase",
    "is_default": true,
    "ach_details": {
      "account_number": "000123456789",
      "routing_number": "021000021",
      "account_name": "Checking Account",
      "account_owner_type": "BUSINESS",
      "account_type": "Checking"
    }
  }
  ```
  Expected 200: `{"vendor_id": "...", "status": "approved" | "pending_approval", "payment_method_id": "..." | null, "payment_method_type": "ach" | null, "proposal_id": null | "..."}`.
- Vendor hold/release should be handled as "204 = done, 202 = queued for approval"; since the 202 body is an untyped object, log it but do not depend on fields. Read back the effect through bills (`GET /bills?vendor_id=...&status_summaries=ON_HOLD` or per-bill `status_summary`) and, if desired, the audit log, because the Vendor object has no hold flag.
- Both hold endpoints are marked `x-destructive: true` in Ramp's own tool metadata, so Ramp's MCP/CLI treat hold as a write that needs Full access; Callback's "hold is always allowed" policy is a product choice, not something Ramp distinguishes.
- Beta status: the bills hold/release operations carry `x-beta: true` in the spec; the vendor hold/release operations do not, but the July 15, 2026 changelog labels both pairs "(beta)". Expect either to change without a version bump.

### Gaps
- The 202 "pending approval" body for vendor hold/release has no documented fields (no `proposal_id` is promised). Not verified whether holding an already-held vendor returns 204 again or an error (idempotency undocumented).
- No documented way to read a vendor's current hold state directly (no field on the Vendor object; no "list holds" endpoint). Whether `vendors.updated` fires on hold/release is not documented.
- The approval policy that decides `approved` vs `pending_approval` for `update-bank-accounts` (and for holds) is configured in the Ramp dashboard ("Vendor edit approval policy"); the docs do not describe how to configure it or what the sandbox default is.
- Routing numbers: the bank-account read endpoints expose no routing number at all; only the beta `payment-methods` endpoint exposes `routing_number_last_four` for ACH. If the agent needs to read back "routing ending in", that is the only source.

## Bills: list/get/create/update/archive, hold/release, auto-approval

### Takeaway
`GET /developer/v1/bills` supports rich filters (`vendor_id`, `status_summaries`, `payment_status`, due-date ranges, amount bounds); the Bill object carries `status` (`OPEN`/`PAID`), `approval_status`, and `status_summary` (which includes `ON_HOLD`, `AWAITING_RELEASE`, `BLOCKED`); `POST /bills/{id}/hold` "Blocks all payments rails on a bill and deschedules payments" and `POST /bills/{id}/release` re-allows scheduling, both returning the full Bill; bills created with `POST /bills` are auto-approved (`approval_status=APPROVED`, `status=OPEN`), and there is no approval endpoint in the API.

### Cited Findings

Endpoint inventory with scopes (spec):
- `GET /developer/v1/bills` — `bills:read` — List bills
- `POST /developer/v1/bills` — `bills:write` — Create a bill ("Batch payments cannot be created in the API.")
- `GET /developer/v1/bills/{bill_id}` — `bills:read` — Fetch a bill
- `PATCH /developer/v1/bills/{bill_id}` — `bills:write` — Update a bill ("Only approved bills can be updated.")
- `DELETE /developer/v1/bills/{bill_id}` — `bills:write` — Archive a bill ("This is a destructive action. Associated inflight payments will be cancelled if possible or any attached one-time-card will be terminated. Paid bills and bills belonging to a batch payment cannot be deleted."; 204)
- `POST /developer/v1/bills/{bill_id}/hold` — `bills:write` — Hold Bill (`x-beta: true`)
- `POST /developer/v1/bills/{bill_id}/release` — `bills:write` — Release Bill Hold (`x-beta: true`)
- `POST /developer/v1/bills/{bill_id}/attachments` — `bills:write` — Upload a file attachment (multipart)
- `GET /developer/v1/bills/{bill_id}/remittance-receipt` — `bills:read`
- `POST /developer/v1/bills/{bill_id}/associations`, `DELETE /developer/v1/bills/{bill_id}/associations/{associated_bill_id}` — `bills:write` (beta, Sept 29, 2026)
- Drafts: `GET/POST /developer/v1/bills/drafts`, `GET/PATCH /developer/v1/bills/drafts/{draft_bill_id}`, `POST .../drafts/{draft_bill_id}/attachments`, `POST .../drafts/{draft_bill_id}/submit` (`bills:read`/`bills:write`)
- Payments API (beta): `GET /developer/v1/payments`, `POST /developer/v1/payments`, `GET/PATCH/DELETE /developer/v1/payments/{payment_id}`, `POST .../retry`, `GET .../remittance-receipt`, `POST /developer/v1/payments/check-preview` (`payments:read`/`payments:write`)
- Approvals (beta, Sept 15, 2026): `GET /developer/v1/approvals` (`approvals:read`, "List approval data for a set of Transactions, Reimbursements, or Bills"), `POST /developer/v1/approvals/{entity_type}/{entity_id}/add-step` (`approval_chains:write`)
— [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json); human page [Bills reference](https://docs.ramp.com/developer-api/v1/api/bills)

`GET /developer/v1/bills` query params (spec): `entity_id`, `customer_friendly_payment_id`, `draft_bill_id`, `invoice_number` (exact), `remote_id` (exact), `accounting_field_selection_id`, `status_summaries` (comma-separated list from the `status_summary` enum below), `payment_id`, `purchase_order_id`, `vendor_id`, `is_accounting_sync_enabled` (bool), `is_payment_sync_ready` (bool), `approval_status` enum `[APPROVED, INITIALIZED, PENDING, REJECTED, TERMINATED]` ("Note that this is separate from the approval status for payment release."), `payment_method` enum `[ACH, CARD, CHECK, CRYPTO_WALLET_TRANSFER, DOMESTIC_WIRE, INTERNATIONAL, LOCAL_BANK_TRANSFER, ONE_TIME_CARD, ONE_TIME_CARD_DELIVERY, PAID_MANUALLY, SWIFT]`, `payment_status` enum `[OPEN, PAID]`, `sync_status` enum `[BILL_AND_PAYMENT_SYNCED, BILL_SYNCED, NOT_SYNCED]`, `sync_ready` (bool), `payment_details_missing` (bool), `is_archived` (bool, default false), `from_created_at`/`to_created_at`, `from_due_date`/`to_due_date`, `from_issued_date`/`to_issued_date`, `from_paid_at`/`to_paid_at`, `from_payment_date`/`to_payment_date` (not supported for card payments), `min_amount`/`max_amount`, `start`, `page_size` (default 20, 2–100) — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)

Bill object (response of `GET /bills/{bill_id}`, also items of `GET /bills` `data[]`, and the body returned by hold/release) (spec):
- `id` (uuid); `status` enum `[OPEN, PAID]` REQUIRED; `status_summary` REQUIRED enum `[APPROVAL_PENDING, APPROVAL_REJECTED, ARCHIVED, AWAITING_PAYMENT, AWAITING_RELEASE, BLOCKED, HELD_BY_PROVIDER, ON_HOLD, PAYMENT_COMPLETED, PAYMENT_DETAILS_MISSING, PAYMENT_ERROR, PAYMENT_NOT_INSTRUCTED, PAYMENT_PROCESSING, PAYMENT_READY, PAYMENT_SCHEDULED, PENDING_VENDOR_APPROVAL, WAITING_FOR_TRANSACTION_MATCH, WAITING_FOR_TRANSFER_MATCH, WAITING_FOR_VENDOR]`; `approval_status` REQUIRED nullable enum `[APPROVED, INITIALIZED, PENDING, REJECTED, TERMINATED]` ("separate from the approval status for payment release")
- `amount` `{amount (int, minor units), currency_code, minor_unit_conversion_rate}`; `due_at` (date-time); `issued_at`; `paid_at` ("None if the bill's payment status is OPEN"); `created_at`; `archived_at` (nullable); `accounting_date`; `posting_date`
- `vendor` `{id, name, remote_code, remote_id, remote_name, type [BUSINESS, INDIVIDUAL]}`; `vendor_contact_id` (nullable); `entity_id`; `bill_owner {id, first_name, last_name}`
- `invoice_number`; `memo`; `vendor_memo`; `remote_id`; `deep_link_url`; `invoice_urls[]`; `attachments[]` `{id, attachment_type [EMAIL, FILE, INVOICE, VENDOR_CREDIT], url}` (Oct 7, 2026)
- `payment` REQUIRED ("For bills with multiple payments, this is the first non-vendor-credit payment."): `amount {...}`, `payment_method` (nullable, same 18-value enum as vendor accounts), `payment_date` (nullable; "For non-card payment methods, the scheduled initiation date"), `effective_date` (nullable; "the date the payment is actually initiated"), `id` (nullable; "Bills sharing the same payment ID belong to the same batch payment"), `customer_friendly_payment_id` (nullable), `details` (object, "Additional payment details specific to payment method", no properties defined in the spec), `trace_id {descriptor, trace_id}` ("only returned in the GET /bills/{bill_id} response body (not in the GET /bills paginated query)"), `bank_debit_trace_id {descriptor, trace_id}`
- `line_items[]` and `inventory_line_items[]` (`id`, `amount`, `memo`, `accounting_field_selections[]`, `purchase_order_line_item_id`, `item_receipt_line_item_ids[]`, `withholding_line_item`, inventory adds `quantity`, `unit_price`); `accounting_field_selections[]`; `applied_vendor_credits[] {id, amount}`; `related_bills[]`; `purchase_order_id`; `item_receipt_ids[]`; `sync_status`; `is_payment_sync_ready`; `enable_accounting_sync`; `fx_conversion_rate`; `draft_bill_id`; `draft_bill_created_at`; `is_withholding_release`
— [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)

Hold / release (spec):
- `POST /developer/v1/bills/{bill_id}/hold`: description "Blocks all payments rails on a bill and deschedules payments."; optional body `{ "reason": string|null }` ("Optional reason for placing the bill on hold."); response `200` "Bill" (the full Bill object); `x-beta: true`, `x-ramp-plus-required: false`; scope `bills:write`
- `POST /developer/v1/bills/{bill_id}/release`: description "Allows payments to be scheduled on a bill."; no body; response `200` "Bill"; `x-beta: true`; scope `bills:write`
— [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)
- Changelog July 15, 2026: "Bill hold and release endpoints (beta) — Block or resume scheduled payments on a bill via `POST /developer/v1/bills/{bill_id}/hold` (with an optional `reason`) and `POST /developer/v1/bills/{bill_id}/release`." — [Changelog](https://docs.ramp.com/developer-api/v1/changelog)
- Changelog March 25, 2026: "New bill payment status `ON_HOLD` — Added `ON_HOLD` enum value to the bill `payment_status` filter and response, representing bills that have been placed on hold." — [Changelog](https://docs.ramp.com/developer-api/v1/changelog). In the current spec, `ON_HOLD` appears only in `status_summary` / the `status_summaries` filter; the `payment_status` filter and `status` field enums are `[OPEN, PAID]` (checked programmatically) — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)
- Changelog Sept 15, 2026: "Awaiting-payment bill status — Bill responses and status filtering now support `AWAITING_PAYMENT`"; July 15, 2026: audit log event `Bill pay marked as unpaid` — [Changelog](https://docs.ramp.com/developer-api/v1/changelog)

Create / auto-approval:
- "Bills created with `POST /developer/v1/bills` are automatically approved and enter the workflow at the **Approved** status." Lifecycle: Draft → Pending approval → Approved → Paid; "In API responses, `approval_status` tracks review separately from payment `status` (`OPEN` or `PAID`)" — [Bill Pay guide](https://docs.ramp.com/developer-api/v1/bill-pay) ([text](https://docs.ramp.com/llms-guides/bill-pay.txt))
- "A bill created with `POST /bills` skips the draft phase and has `approval_status=APPROVED`, separately from payment `status=OPEN`. From there, Ramp processes the payment using the method you chose; the bill moves to **Paid** once the payment clears." "If your customer's policy requires review before payment, create a draft bill and submit it with `POST /bills/drafts/{draft_bill_id}/submit`. The Developer API does not expose bill approval; authorized reviewers approve in Ramp or through Ramp MCP." — [Bill Payments guide](https://docs.ramp.com/developer-api/v1/bill-payments) ([text](https://docs.ramp.com/llms-guides/bill-payments.txt))
- Required scopes for the create flow: "`bills:write` to create bills; `bills:read` to read; `entities:read` and `vendors:read` to resolve UUIDs." UUID sources: `GET /entities` → `entity_id`; `GET /entities` `bank_accounts[].id` where `usage_type = BILL_PAY_BANK_ACCOUNT` → `source_bank_account_id`; `GET /vendors` `id` → `vendor_id`; `GET /vendors/{id}/contacts` `id` → `vendor_contact_id`; `GET /vendors/{id}/accounts` `id` (ACH / wire / SWIFT / stablecoin only) → `vendor_account_id` — [Bill Payments guide](https://docs.ramp.com/developer-api/v1/bill-payments)
- Documented `POST /bills` example (ACH), copied:
  ```json
  {
    "vendor_id": "<uuid>",
    "vendor_contact_id": "<uuid>",
    "entity_id": "<uuid>",
    "invoice_number": "INV-001",
    "invoice_currency": "USD",
    "issued_at": "<YYYY-MM-DD>",
    "due_at": "<YYYY-MM-DD>",
    "payment_method": "ACH",
    "payment_details": {
      "payment_arrival_date": "<YYYY-MM-DD>",
      "source_bank_account_id": "<uuid>",
      "vendor_account_id": "<uuid>"
    }
  }
  ```
  — [Bill Payments guide](https://docs.ramp.com/developer-api/v1/bill-payments)
- `POST /bills` body (spec): `vendor_id` REQUIRED, `entity_id` REQUIRED, `invoice_number` REQUIRED, `invoice_currency` REQUIRED, `issued_at` REQUIRED (date), `due_at` REQUIRED (date), `vendor_contact_id` (required unless `use_default_vendor_contact: true`), `payment_method` (enum above; "Required unless use_default_payment_method is true"), `payment_details` (`ApiCreateBillPaymentRequestBody`; "Expected schema depends on payment_method"), `use_default_payment_method` (bool), `use_default_vendor_contact` (bool), `payment_date` (date; "the date the payment is scheduled to be initiated ... Ramp schedules the payment for the next available date" if unavailable), `target_payment_arrival_date` (with `use_default_payment_method=true`), `line_items[]`, `inventory_line_items[]`, `accounting_field_selections[]`, `attachment_id`, `memo`, `vendor_memo`, `remote_id` (needs an accounting connection and `enable_accounting_sync: true`), `enable_accounting_sync` (default true), `posting_date`, `purchase_order_ids[]` — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)
- `payment_details` on `POST /bills` (spec, schema `ApiCreateBillPaymentRequestBody`, a `oneOf` discriminated by `payment_method`): for `ACH`, `DOMESTIC_WIRE`, `INTERNATIONAL`, `LOCAL_BANK_TRANSFER`, `SWIFT`, `CRYPTO_WALLET_TRANSFER` → `ApiCreateBankAccountPaymentParamsRequestBody` { `payment_arrival_date` REQUIRED ("The expected date the payment will arrive in the vendor's bank account"), `source_bank_account_id` REQUIRED ("Must be associated with the passed business entity and have usage_type=BILL_PAY_BANK_ACCOUNT"), `vendor_account_id` REQUIRED ("Unique identifier of the vendor account to pay the bill to"), `is_same_day` (bool, default false, ACH only) }; for `CHECK` → `ApiCreateBillVendorPaymentParamsRequestBody` { `payment_arrival_date` REQUIRED, `source_bank_account_id` REQUIRED, `is_same_day` }; for `CARD` → `ApiCreateCardBillPaymentParamsRequestBody` { `spend_limit_id` REQUIRED, `transaction_id` ("If already paid, the unique identifier of the bill payment transaction") }; for `PAID_MANUALLY` → `ApiCreateManualBillPaymentParamsRequestBody` { `manual_payment_method` REQUIRED enum `[CASH, CHECK, CROSS_BORDER_PAYMENT, CRYPTO_WALLET_TRANSFER, DIRECT_DEBIT, DOMESTIC_WIRE_TRANSFER, NON_RAMP_CREDIT_CARD, OTHER, PAID_IN_ERP]`, `payment_date` REQUIRED ("the bill will be marked as paid") }. `ONE_TIME_CARD` / `ONE_TIME_CARD_DELIVERY` have no mapping entry in the discriminator — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)
- `PATCH /bills/{bill_id}` body (spec): `due_at`, `issued_at`, `invoice_number`, `memo`, `vendor_memo`, `remote_id`, `entity_id`, `vendor_id`, `vendor_contact_id`, `payment_method`, `payment_date` (nullable; "Send `null` on update to unschedule" per Oct 2025-era changelog), `posting_date`, `purchase_order_ids[]`, `line_items[]`/`inventory_line_items[]` (replace wholesale; pass existing `id` to keep a line's identity), `accounting_field_selections[]` — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)
- Idempotency on create: "`POST /bills` does not document an `idempotency_key` request field. Save the returned bill ID and, after a timeout, check whether the vendor/invoice already exists before creating again." — [Bill Payments guide](https://docs.ramp.com/developer-api/v1/bill-payments)
- API limitations: "The Developer API has no bill-approval endpoint. `POST /bills` creates an approved bill; draft-bill submission enters the configured approval workflow. Ramp MCP supports bill approvals for authorized reviewers." "Batch payments — one payment across multiple bills — aren't supported." "Entity bank accounts. Not manageable via API. Vendor bank accounts are" — [Bill Pay guide](https://docs.ramp.com/developer-api/v1/bill-pay)
- Attachments: `POST /developer/v1/bills/{bill_id}/attachments`, `multipart/form-data`, parts `file=@invoice.pdf`, `attachment_type=INVOICE` (or `FILE`), response `201 {"id": "d4ab27cc-18b9-429c-bdc4-60e5ec57fa39"}`; "An `INVOICE` upload fails if the bill already has one." — [Bill Payments guide](https://docs.ramp.com/developer-api/v1/bill-payments)
- Bill webhooks: `bills.approved`, `bills.archived`, `bills.created`, `bills.paid`, `bills.ready_to_sync`, `bills.rejected`, `bills.updated` ("Fires for invoice number updates, PO matching, accounting field updates, and payment detail changes. Does not fire for bill approvals or payments"), `draft_bills.created`; `payments.updated` "A bill payment has been updated." — [Webhooks guide](https://docs.ramp.com/developer-api/v1/webhooks)
- Incremental reads: `GET /developer/v1/bills` has `from_created_at` only ("No updated-at filter. Use webhooks or periodic full re-pulls to detect status changes such as payments and approvals.") — [Rate Limits guide](https://docs.ramp.com/developer-api/v1/rate-limiting)

### Inferences
- The "status of a hold" for a bill should be read from `status_summary == "ON_HOLD"` (and filtered with `status_summaries=ON_HOLD`); the March 25, 2026 changelog's wording about `payment_status` no longer matches the spec, so do not expect `payment_status=ON_HOLD` to validate. The hold endpoint returns the updated Bill, so the tool can confirm by inspecting `status_summary` in the 200 body without a second call. Verify once in the sandbox.
- A vendor hold presumably moves that vendor's schedulable open bills to `ON_HOLD` (the vendor hold description says they are "removed from scheduled payment flows"), but the docs do not say which `status_summary` value they take; the bill-level `ON_HOLD` is documented only for bill holds. Check in the sandbox before scripting the demo narration.
- For the Callback "read bills" tool, `GET /bills?vendor_id=<id>&page_size=100` plus `payment_status=OPEN` gives the open exposure for a vendor; `payment.payment_method` and `payment.payment_date` tell you whether a payment is scheduled, and `payment.details` is an opaque object (fields undocumented), so do not promise to read the destination account id from a bill.
- "Auto-approved" means the bill enters at `approval_status=APPROVED`, `status=OPEN`, and Ramp will initiate the payment on `payment_date`/arrival logic without any human approval, subject to the business's payment-release setting; it does not mean paid. In the sandbox nothing moves money; use ⌘J "pay current bill" to simulate `PAID`.

### Gaps
- The response-side `payment.details` has no documented schema in the spec (an empty object), so the destination account of an existing bill cannot be read reliably from the Bill; the request-side shapes are documented above. The `PAID_MANUALLY` path ("the bill will be marked as paid") is the only documented API way to make a bill `PAID` without the ⌘J demo action; whether that is acceptable for demo fixtures is a product call.
- Not documented: whether `POST /bills/{id}/hold` on a `PAID` or already `ON_HOLD` bill errors (and with which `error_code`), whether a bill hold survives a later `PATCH`, or whether `release` reschedules automatically (vendor release explicitly says payments "must be scheduled again"; bill release says only "Allows payments to be scheduled").
- No documented `AWAITING_RELEASE`/`BLOCKED`/`HELD_BY_PROVIDER` semantics beyond the enum names.

## Cross-cutting: rate limits, pagination, errors, idempotency, headers, versioning

### Takeaway
200 requests per 10 s per source IP (429 can lock you out for 60 s; honor `Retry-After`), keyset pagination via `page.next` with `page_size` 2–100 (default 20), structured errors under `error_v2` with an `x-trace-id` response header, no general idempotency header (only `X-Idempotency-Key` on one procurement upload endpoint), and no API-version header: the version is the `/developer/v1` path.

### Cited Findings
- "The standard Developer API limit is **200 requests per 10 seconds per source IP address**. Requests from integrations that share an outbound IP share this limit." "Exceeding a limit returns **429 Too Many Requests** and can block subsequent requests for **60 seconds**. A 10-second counting period does not mean access resumes 10 seconds after a burst." "Honor `Retry-After` when present. Otherwise, use exponential backoff with jitter" — [Rate Limits and Timeouts](https://docs.ramp.com/developer-api/v1/rate-limiting) ([text](https://docs.ramp.com/llms-guides/rate-limiting.txt))
- "A **504 Gateway Timeout** means a gateway did not receive an upstream response before its timeout." "Before retrying a write, check whether it completed and follow the endpoint's idempotency behavior; a timeout does not prove the operation failed." `504` = "Request exceeded the 60-second timeout" — [Rate Limits and Timeouts](https://docs.ramp.com/developer-api/v1/rate-limiting); [Errors guide](https://docs.ramp.com/developer-api/v1/error-handling)
- Pagination: "Ramp API uses a mechanism called keyset pagination"; the response includes `page.next`, "the complete URL for accessing the next page", e.g. `{"page": {"next": "https://api.ramp.com/developer/v1/cards/physical?page_size=3&start=2907e304-cac2-4abf-84c4-b3b454ae3b8c"}}`; "Treat the cursor as opaque"; "Repeat ... until the API returns a null value for `page.next`"; "the default page size is 20 and the maximum is 100" — [Pagination guide](https://docs.ramp.com/developer-api/v1/pagination) ([text](https://docs.ramp.com/llms-guides/pagination.txt)). The spec's `page_size` description on `/vendors` and `/bills`: "The value must be between 2 and 100" — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)
- Error format, copied from the Errors guide (HTTP 422):
  ```json
  {
     "error_v2": {
        "error_code": "DEVELOPER_7001",
        "additional_info": {
           "page_size": [
              "Value must be between 2 and 100"
           ]
        },
        "notes": "",
        "message": "Value must be between 2 and 100"
     },
     "error": {
        "page_size": [
           "Value must be between 2 and 100"
        ]
     }
  }
  ```
  Status codes: 400 Bad Request, 401 Unauthorized, 403 Forbidden, 404 Not Found, 422 Unprocessable Entity (validation), 429, 500, 504. "Log error responses, including the `x-trace-id` from the response headers". "Some endpoints ignore unknown query parameters or request body fields. Other endpoints reject unknown fields and return a validation error." — [Errors guide](https://docs.ramp.com/developer-api/v1/error-handling) ([text](https://docs.ramp.com/llms-guides/error-handling.txt))
- Spec error schema `DeveloperApiErrorResponse`: `{ "error_v2": { "error_code" (string, "Stable machine-readable error code"), "error_id" (string, "8 hex digit error identifier"), "message", "notes" (nullable), "additional_info" (object) } }`; the vendor/bill endpoints declare `400: raises InvalidInput` with this body — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)
- `x-trace-id`: "Ramp API responses include an `x-trace-id` header when the request reaches the application." Support tickets: open Developer Support at `https://app.ramp.com/chat?entryVariant=developerSupport` or Company → Developer → "Developer API Support" — [Support guide](https://docs.ramp.com/developer-api/v1/debugging)
- Idempotency: the only idempotency header in the docs is `X-Idempotency-Key` on `POST /developer/v1/unified-requests/{id}/attachments` (Oct 20, 2026 changelog); `POST /accounting/syncs` takes a body field `idempotency_key`; `POST /bills` "does not document an `idempotency_key` request field" — [Changelog](https://docs.ramp.com/developer-api/v1/changelog); [Bill Payments guide](https://docs.ramp.com/developer-api/v1/bill-payments)
- Versioning headers: none documented; the spec defines no `Ramp-Version`/`X-Ramp-*` request headers (the only `x-ramp-*` strings in the spec are the `x-ramp-plus-required` operation extension); all paths are prefixed `/developer/v1` — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)
- Content types (spec): JSON endpoints declare `application/json` request bodies; `POST /developer/v1/token` and `/token/revoke` declare `application/x-www-form-urlencoded`; attachment uploads use `multipart/form-data` — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)
- Monetary values: "Most Ramp API monetary values use integers in the smallest denomination of the relevant currency"; vendor/bill amounts use `{amount, currency_code, minor_unit_conversion_rate}` ("Divide amount by this value to get the real currency amount. For USD, this is 100 (e.g., 12034 / 100 = 120.34 USD)"); some other objects use `{value, currency}` — [Monetary Values guide](https://docs.ramp.com/developer-api/v1/monetary-values); [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)
- Read replicas: `GET` vendor/bill endpoints carry `x-uses-read-replica: true` in the spec — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)

### Inferences
- Send `Accept: application/json` and `Content-Type: application/json` on JSON calls; no `Accept` value is mandated, so the default is fine. Send `User-Agent` with the project name for support tracing.
- Because GETs are served from read replicas, a `GET /bills/{id}` immediately after `POST /bills/{id}/hold` could in principle lag; prefer the Bill body returned by the hold call itself.
- Rate limits are per IP, so a voice agent making a handful of calls per turn will never approach 200/10 s; the token-issuance limit is the one to respect (cache the token).
- For retries: retry idempotent GETs on 429/5xx/504 with backoff; for the hold/release POSTs, read back state (`status_summary`, or the 204/202 outcome) before retrying rather than blind-retrying.

### Gaps
- No documented `Retry-After` behaviour for the REST API beyond "when present"; no `X-RateLimit-*` headers are documented.
- The error-code catalogue (`DEVELOPER_7001`, `DEVELOPER_7012`, `DEVELOPER_7100`, ...) is not published as a list; only these three appear in the guides.

## Webhooks: `vendors.updated`, `tests.test_event`, registration, retries, signatures

### Takeaway
Register with `POST /developer/v1/webhooks` (`endpoint_url`, `event_types[]`, optional `additional_headers`), answer the `webhooks.verification` challenge and submit it to `POST /webhooks/{id}/verify`, verify `X-Ramp-Signature` (HMAC-SHA256 of the raw body with the subscription `secret`), and expect up to 10 delivery attempts with jittered exponential backoff for 429/5xx. `vendors.updated` is a real subscription event; `tests.test_event` exists only for the mock endpoint.

### Cited Findings
- Endpoints (spec, all `security: [{"oauth2": []}]`, i.e., no specific scope): `GET /developer/v1/webhooks` (array of subscriptions), `POST /developer/v1/webhooks` (201), `GET /developer/v1/webhooks/{webhook_id}`, `DELETE /developer/v1/webhooks/{webhook_id}` (204), `POST /developer/v1/webhooks/{webhook_id}/verify` (body `{ "challenge": string }` → `{ "success": boolean }`), `POST /developer/v1/webhooks/mock-webhook-event` (body `event_type` REQUIRED, `object_id` REQUIRED uuid, `object_metadata` REQUIRED object → 201 `{ id, type, created_at, business_id, object }`) — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)
- Create body (spec): `endpoint_url` REQUIRED (format url), `event_types[]` REQUIRED (enum of 50 values including `bills.approved`, `bills.archived`, `bills.created`, `bills.paid`, `bills.ready_to_sync`, `bills.rejected`, `bills.updated`, `draft_bills.created`, `payments.updated`, `vendors.activated`, `vendors.approved`, `vendors.updated`, `tests.test_event`, `webhooks.verification`), `additional_headers` (object, "Max 5 headers, header names max 100 chars, values max 1000 chars"). 201 response: `id`, `secret` ("Secret key for the webhook subscription"), `endpoint_url`, `event_types[]`, `additional_headers`, `status` enum `[active, inactive, pending_verification]`, `business_id`, `created_at`, `updated_at` — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)
- "Your access token must include the required read scope for each selected business event." (`vendors:read` for `vendors.*`, `bills:read` for `bills.*`, `payments:read` for `payments.updated`; `tests.test_event` needs none and is "Test event for mock webhook delivery only; not a business-event subscription.") — [Webhooks guide](https://docs.ramp.com/developer-api/v1/webhooks) ([text](https://docs.ramp.com/llms-guides/webhooks.txt))
- Registration example, copied:
  ```bash
  curl -X POST https://api.ramp.com/developer/v1/webhooks \
    -H "Authorization: Bearer $RAMP_API_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
      "endpoint_url": "https://your-domain.com/webhooks",
      "event_types": ["transactions.authorized", "transactions.cleared"]
    }'
  # Response includes a 'secret' field - save it securely to verify webhook signatures
  ```
  Reserved header names for `additional_headers`: `Host`, `User-Agent`, `Content-Length`, `Content-Type`, `Accept`, `Accept-Encoding`, `Connection`, `Upgrade`, and anything starting with `X-Forwarded`, `X-Real`, `X-Original` — [Webhooks guide](https://docs.ramp.com/developer-api/v1/webhooks)
- Verification payload, copied: `{"type": "webhooks.verification", "challenge": "ramp_challenge_a8f5f167f44f4964e6c998dee827110c"}`; "Read the `challenge` value ... return a `2xx` response", then `POST /developer/v1/webhooks/{webhook_id}/verify` with `{"challenge": "CHALLENGE_STRING_FROM_RAMP"}` → `{"success": true}` activates the subscription — [Webhooks guide](https://docs.ramp.com/developer-api/v1/webhooks)
- Business-event payload, copied: `{"id": "a461a658-...", "type": "transactions.cleared", "created_at": "2025-04-01T12:00:00Z", "business_id": "945c7d8f-...", "object": {"id": "<resource-id>"}}`; `object` "Contains the affected resource ID and any event-specific metadata supplied for that event"; "Business-event requests also include an `X-Ramp-Webhook-ID` header with the same event ID as the payload." — [Webhooks guide](https://docs.ramp.com/developer-api/v1/webhooks). Changelog Oct 29, 2026: "Entity-scoped webhook payloads include `object.entity_id` ... It is `null` when the resource has no entity" — [Changelog](https://docs.ramp.com/developer-api/v1/changelog)
- Signature: "Every webhook request from Ramp includes an `X-Ramp-Signature` header, which contains an HMAC-SHA256 hash of the raw request body, signed with your webhook secret." Sample header value `7f2a8b9c3d4e5f6a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a` (64 hex chars). "Verify the exact raw request bytes before parsing or reserializing the JSON." — [Webhooks guide](https://docs.ramp.com/developer-api/v1/webhooks)
- Retries: "**2xx** Delivery successful - no retries; **3xx, 4xx** (except 429) Instant delivery failure - no retries; **429, 5xx** (including timeouts and connection errors) Up to **10 total delivery attempts**, with exponential backoff". "The first retry delay is between 0 and 2 seconds, and later retry delays increase up to a maximum of 60 seconds. Webhook requests time out after 10 seconds." "The same event `id` is used for all retry attempts". "Webhooks may arrive out of order." If your endpoint returns 429, "Ramp retries and can pause deliveries to that receiver. Include a `Retry-After` header" — [Webhooks guide](https://docs.ramp.com/developer-api/v1/webhooks)
- Mock events: "Use `POST /developer/v1/webhooks/mock-webhook-event` to create a mock event for all matching active subscriptions within the token's business and integration ... Mock events use the normal event payload and signature, with no test marker added by Ramp." Example (sandbox):
  ```bash
  curl -X POST https://demo-api.ramp.com/developer/v1/webhooks/mock-webhook-event \
    -H "Authorization: Bearer $RAMP_SANDBOX_API_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
      "event_type": "transactions.cleared",
      "object_id": "3aa9eed1-c793-4b52-a502-e7f2a7e4c1b2",
      "object_metadata": {}
    }'
  ```
  "It returns an error if no active subscription matches the event type." — [Webhooks guide](https://docs.ramp.com/developer-api/v1/webhooks)
- Operational: "Missed events during extended downtime may not be recoverable"; to change a URL, delete and re-create; no delivery-log dashboard ("Keep delivery and processing logs in your integration") — [Webhooks guide](https://docs.ramp.com/developer-api/v1/webhooks)

### Inferences
- Python verification: `hmac.compare_digest(hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest(), request.headers["X-Ramp-Signature"])`. The sample is lowercase hex and the docs call it "an HMAC-SHA256 hash of the raw request body"; the encoding of the secret (UTF-8 string, not base64-decoded) is my assumption, so test once with a mock event and fall back to comparing against a base64 digest if hex fails.
- For Callback, webhooks are optional: the voice agent can poll `GET /bills/{id}` after acting. If you do wire a `vendors.updated` listener (e.g., to narrate "the bank account change was approved"), remember the payload carries only `object.id` (+ `entity_id`), so follow with `GET /vendors/{id}/accounts?include_pending=true`.
- `tests.test_event` is the right choice for an end-to-end plumbing test via the mock endpoint without needing a real resource UUID of a particular type (the mock endpoint still requires an `object_id` UUID).

### Gaps
- Whether `vendors.updated` fires on hold/release or on `update-bank-accounts` approval is not documented. No per-event `object_metadata` schema is documented for vendor or bill events.

## Tooling: Developer-docs MCP, hosted Ramp MCP, plain-text guides, OpenAPI, SDK

### Takeaway
The hosted account MCP is `https://mcp.ramp.com/mcp` (sandbox: `https://demo-mcp.ramp.com/mcp`; Claude Code: `claude mcp add --transport http ramp https://mcp.ramp.com/mcp`), OAuth in the browser, no scopes. The "Ramp Developer MCP" for docs that a third-party directory lists at `https://mcp.ramp.com/developer/mcp` returned 404 to an MCP `initialize` on 2026-11-05 and is absent from Ramp's current docs, which instead point coding agents at `https://docs.ramp.com/llms.txt`. There is no official Python SDK; use httpx against the OpenAPI spec.

### Cited Findings
- Ramp MCP (account data) — Claude Code setup, copied: `claude mcp add --transport http ramp https://mcp.ramp.com/mcp`; "Run `/mcp` in your session to verify the connection."; sandbox: "replace `https://mcp.ramp.com/mcp` with `https://demo-mcp.ramp.com/mcp` in any config below"; Cursor `~/.cursor/mcp.json` `{"mcpServers": {"ramp": {"url": "https://mcp.ramp.com/mcp"}}}`; Claude Desktop via `npx mcp-remote https://mcp.ramp.com/mcp`; VS Code Copilot `{"servers": {"ramp": {"type": "http", "url": "https://mcp.ramp.com/mcp"}}}`; per-business connections at `https://mcp.ramp.com/<business-identifier>/mcp` ("Don't use `developer`, `employee`, `mcp`, `ramp-data`, or `ramp-rate`; those paths are reserved.") — [Ramp MCP guide](https://docs.ramp.com/developer-api/v1/ramp-mcp) ([text](https://docs.ramp.com/llms-guides/ramp-mcp.txt))
- What Ramp MCP can do: "Read and analyze spend: search transactions, pull spend exports, and look up vendors ..."; "Process approvals: approve or reject transactions, bills, reimbursements, and requests"; "Submit and edit expenses ..."; "Make purchases: get Agent Card credentials"; "The tools you see depend on your Ramp role and permissions." Access: consent screen "Full access" vs "Read-only"; "Ramp MCP doesn't use OAuth scopes"; admins limit access at Company → Integrations → Ramp MCP → Manage Access — [Ramp MCP guide](https://docs.ramp.com/developer-api/v1/ramp-mcp)
- agents.ramp.com connection page: live `https://mcp.ramp.com/mcp`, "Use sample data instead at: `https://demo-mcp.ramp.com/mcp`"; Claude connector `https://claude.ai/directory/61bac03c-3f98-4b3c-affb-1b99533fa82c`; ChatGPT app `https://chatgpt.com/apps/ramp/asdk_app_69250fb6281c819195b52a1556b0060c`; Cursor deeplink; "MCP: File uploads and downloads not available, slower performance"; "The full list of tools available in the CLI and MCP can be found [here](/docs/reference/agent-tools)" — [agents.ramp.com Connecting](https://agents.ramp.com/docs/guides/connecting) (the `/docs/mcp/setup` URL serves this same page as Markdown)
- The agent-tools reference page is a build-time component (`<McpToolsReference />`); its text/markdown export is a 603-byte stub with no tool list; it says the list "is generated at build time and runtime access depends on your permissions, admin controls, scopes, connected client, environment, and CLI version" — [agents.ramp.com agent tools](https://agents.ramp.com/docs/reference/agent-tools)
- Read-only connections: "Write requests fail with `403 Forbidden` and error code `DEVELOPER_7100` (`These scopes are not allowed for this token: ...`)" — [Authorization guide](https://docs.ramp.com/developer-api/v1/authorization)
- Probe (mine, 2026-11-05): `POST https://demo-mcp.ramp.com/mcp` (MCP `initialize`, no auth) → 401 with `{"resource":"https://demo-mcp.ramp.com","authorization_servers":["https://demo-mcp.ramp.com"],"scopes_supported":[... "vendors:write", "bills:write", "vendors:read", "bills:read", "approvals:write" ...]}` — direct observation.
- MCP overview page: "Ramp ships two MCP servers, each scoped to a different audience. Coding agents that need Ramp's developer docs can read them directly from `https://docs.ramp.com/llms.txt`." (the page's two server sections are empty in the text export) — [MCP overview](https://docs.ramp.com/developer-api/v1/mcp) ([text](https://docs.ramp.com/llms-guides/mcp.txt))
- Ramp Data MCP (the other hosted server; public datasets, not account data): "**Server URL:** `https://mcp.ramp.com/ramp-data/mcp`"; "Ramp Data requires provisioned access"; Claude Code: `claude mcp add --transport http ramp-data https://mcp.ramp.com/ramp-data/mcp --header "Authorization: Bearer $RAMP_DATA_API_KEY"` — [Ramp Data MCP guide](https://docs.ramp.com/developer-api/v1/ramp-data-mcp)
- Developer-docs MCP, third-party claim: a directory page lists "Ramp Developer" as an "Unauthenticated hosted MCP server that gives coding assistants live access to Ramp Developer API guides, API reference and OpenAPI schemas", with the command `claude mcp add --transport http ramp-developer-mcp https://mcp.ramp.com/developer/mcp` ("copied from Ramp's docs"), tools `developer_docs_search`, `developer_docs_schema`, `developer_docs_resolve`, `developer_docs_read`, `developer_docs_list`, `developer_docs_root`, `developer_submit_feedback`, status Beta, list "as documented on October 17, 2026", and links to `https://docs.ramp.com/developer-api/v1/developer-mcp` and `https://docs.ramp.com/llms-guides/developer-mcp.txt` — [developersdigest.tech: Ramp Developer MCP](https://www.developersdigest.tech/mcp-servers/ramp-developer)
- Probes (mine, 2026-11-05): `POST https://mcp.ramp.com/developer/mcp` and `POST https://mcp.ramp.com/developer` with an MCP `initialize` → HTTP 404 `{"detail":"Not Found"}` (while `POST https://mcp.ramp.com/ramp-data/mcp` answered 200 with `serverInfo.name: "Ramp MCP Remote"`, so the host and method work); `https://docs.ramp.com/llms-guides/developer-mcp.txt` → the docs JS shell (not a guide), and "developer-mcp" appears nowhere in `llms.txt` or the guides index; the Wayback Machine has no snapshot of either URL — direct observation.
- Plain-text docs and spec: "LLM index: /llms.txt; Plain text guides index: /llms-guides.txt; Per-guide plain text docs: /llms-guides/<slug>.txt; Plain text API reference: /llms-api.txt; Combined plain text docs: /llms-full.txt; OpenAPI spec: /openapi/developer-api.json" (HTML comment in every docs page); guide slugs relevant here: `getting-started`, `authorization`, `sandbox`, `sandbox-access`, `bill-pay`, `bill-payments`, `webhooks`, `rate-limiting`, `pagination`, `error-handling`, `debugging`, `monetary-values`, `changelog`, `ramp-mcp`, `mcp`, `cli`, `data-relationships` — [docs.ramp.com llms.txt](https://docs.ramp.com/llms.txt)
- The OpenAPI spec declares `servers: [{"description": "Production", "url": "https://api.ramp.com"}]` only — [OpenAPI spec](https://docs.ramp.com/openapi/developer-api.json)
- Ramp CLI: "Use Ramp from your terminal or AI agent: search transactions, approve bills, upload receipts, and more."; runs as the logged-in user, `--env sandbox`, `ramp auth login --read-only` — [CLI guide](https://docs.ramp.com/developer-api/v1/cli); [Authorization guide](https://docs.ramp.com/developer-api/v1/authorization)
- SDKs: Ramp's changelog mentions "Ramp MCP Server ... Available as a Python package" at `https://github.com/ramp-public/ramp-mcp` (a self-hosted MCP server, Feb 2025) — [Changelog](https://docs.ramp.com/developer-api/v1/changelog). PyPI probes (mine): `ramp` is an unrelated ML package by Ken Van Haren; `ramp-api`, `ramp-sdk`, `ramp-python`, `rampapi`, `ramp-developer-api` do not exist (404). A third-party Rust crate `ramp-api` is "generated from Ramp's OpenAPI spec" — [docs.rs ramp-api](https://docs.rs/crate/ramp-api/0.2.1)

### Inferences
- Treat the Developer-docs MCP as unavailable (retired or moved since Oct 17, 2026). For Claude Code, the practical equivalent is: keep the downloaded spec (`developer-api.json`) and `llms-api.txt` in the repo (or scratch dir) and grep them; both are static files Ramp explicitly publishes for agents.
- If you want the hosted MCP in Claude Code for exploring sandbox data: `claude mcp add --transport http ramp-demo https://demo-mcp.ramp.com/mcp`, then complete the browser OAuth as a sandbox (`demo.ramp.com`) user. It cannot replace the Developer API for Callback because it acts as a user via browser OAuth, not as a server with client credentials, and its tool list is permission-dependent.
- httpx client pattern (my sketch, consistent with the docs above):
  ```python
  import base64, time, httpx

  class Ramp:
      def __init__(self, client_id, client_secret, base="https://demo-api.ramp.com",
                   scopes="vendors:read vendors:write bills:read bills:write"):
          self.base, self.scopes = base, scopes
          self._basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
          self._tok, self._exp = None, 0
          self.http = httpx.Client(base_url=base, timeout=30)

      def token(self):
          if self._tok and time.time() < self._exp - 300:
              return self._tok
          r = self.http.post("/developer/v1/token",
                             headers={"Authorization": f"Basic {self._basic}",
                                      "Content-Type": "application/x-www-form-urlencoded"},
                             data={"grant_type": "client_credentials", "scope": self.scopes})
          r.raise_for_status(); j = r.json()
          self._tok, self._exp = j["access_token"], time.time() + j["expires_in"]  # 864000 s
          return self._tok

      def call(self, method, path, **kw):
          h = {"Authorization": f"Bearer {self.token()}", "Accept": "application/json"}
          r = self.http.request(method, path, headers=h, **kw)
          if r.status_code >= 400:
              err = r.json().get("error_v2", {}) if r.headers.get("content-type","").startswith("application/json") else {}
              raise RuntimeError(f"{r.status_code} {err.get('error_code')} {err.get('message')} trace={r.headers.get('x-trace-id')}")
          return r

      def pages(self, path, **params):
          r = self.call("GET", path, params=params)
          while True:
              j = r.json(); yield from j["data"]
              nxt = j["page"]["next"]
              if not nxt: break
              r = self.call("GET", nxt)  # absolute URL from page.next; cursor is opaque

      # tools
      def vendors(self, **f): return list(self.pages("/developer/v1/vendors", page_size=100, **f))
      def vendor_accounts(self, vid, pending=False):
          return list(self.pages(f"/developer/v1/vendors/{vid}/accounts", page_size=100, include_pending=pending))
      def bills(self, **f): return list(self.pages("/developer/v1/bills", page_size=100, **f))
      def hold_vendor(self, vid, reason=None):
          r = self.call("POST", f"/developer/v1/vendors/{vid}/hold", json={"reason": reason} if reason else None)
          return {"applied": r.status_code == 204, "pending_approval": r.status_code == 202}
      def release_vendor(self, vid):
          r = self.call("POST", f"/developer/v1/vendors/{vid}/release")
          return {"applied": r.status_code == 204, "pending_approval": r.status_code == 202}
      def hold_bill(self, bid, reason=None):
          return self.call("POST", f"/developer/v1/bills/{bid}/hold", json={"reason": reason} if reason else None).json()
      def release_bill(self, bid):
          return self.call("POST", f"/developer/v1/bills/{bid}/release").json()
      def add_vendor_bank_account(self, vid, routing, account, nickname=None, default=True):
          body = {"is_default": default, "ach_details": {"routing_number": routing, "account_number": account}}
          if nickname: body["account_nickname"] = nickname
          return self.call("POST", f"/developer/v1/vendors/{vid}/update-bank-accounts", json=body).json()
          # -> {"vendor_id", "status": "approved"|"pending_approval", "payment_method_id", "payment_method_type", "proposal_id"}
  ```
  Note `httpx.Client.request` accepts the absolute `page.next` URL even with a `base_url` set; when `page.next` is used, Ramp returns `https://api.ramp.com/...` in production and (presumably) `https://demo-api.ramp.com/...` in sandbox.

### Gaps
- Could not obtain the hosted Ramp MCP's tool list (names/parameters) from any static source; it is only visible after connecting a client.
- Could not confirm whether the Developer-docs MCP still exists under another URL; Ramp's docs no longer mention it (as of the 2026-11-05 exports).
- No official Ramp Python SDK found anywhere (docs, PyPI, GitHub mention); not a confirmed absence from Ramp, but no evidence of one.

## Recent changelog entries (2026) touching vendors, bills, bank accounts, hold/release

### Takeaway
The hold/release endpoints shipped July 15, 2026 (beta); the structured `update-bank-accounts` response shipped June 25, 2026; `account_number_last_four` on vendor bank accounts July 21, 2026; pending-account visibility (`include_pending`, `status`, `proposal_id`) landed Nov 4, 2026, the day of this research; and `ON_HOLD` appeared as a bill status on March 25, 2026.

### Cited Findings
All from [Changelog](https://docs.ramp.com/developer-api/v1/changelog) ([text](https://docs.ramp.com/llms-guides/changelog.txt)), quoted:
- November 4, 2026 — "**Pending vendor bank accounts** — Set `include_pending=true` to list accounts awaiting approval. Bank account list and detail responses include `status` and `proposal_id`; a pending account gets a new ID when approved." Also "**Nullable vendor contact email** — Vendor contact responses ... can return `email: null`."
- October 29, 2026 — "**Custom check recipient names** — Set `address.pay_to_name` when creating or updating a vendor ... vendor responses include `addresses[].pay_to_name`." and "**Entity-aware webhook routing** — Entity-scoped webhook payloads include `object.entity_id`".
- October 28, 2026 — "**Vendor display names** — Vendor responses include nullable `display_name`. Set it with `PATCH /developer/v1/vendors/{vendor_id}`"; "**Applied vendor credit currency on bills**".
- October 27, 2026 — "**Vendor payment methods (beta)** — List a vendor's `ACH` and `CHECK` payment methods, including fee information, using `vendors:read`."
- October 16, 2026 — "**Optional vendor location details** — Create vendors without `country` or `state`."
- October 8, 2026 — "**Vendor update timestamps** — Vendor list and detail responses include nullable `updated_at`".
- October 7, 2026 — "**Typed bill attachments** — Bill responses include `attachments` with `id`, `attachment_type`, and download `url`."
- October 6, 2026 — "**Vendor payment terms** — Set and read `net_payment_terms`"; "**Draft bill comments (beta)**".
- October 2, 2026 — "**Stable bill line-item IDs** — Bill responses include `id` on `line_items` and `inventory_line_items`."
- September 29, 2026 — "**Bill associations (beta)**" and "**Related bills in responses**".
- September 28, 2026 — "**Vendor direct debit defaults**"; "**Draft bill creation webhook** — Subscribe to `draft_bills.created`"; "**Add-approver scope transition (beta)** — Request `approval_chains:write` for `POST /developer/v1/approvals/{entity_type}/{entity_id}/add-step`."
- September 22, 2026 — "**Vendor default bank account updates** — Set `default_bank_account_id` on `PATCH /developer/v1/vendors/{vendor_id}` ... Omit it to keep the current default; existing bills are unaffected."
- September 16, 2026 — "**Vendor address country requirement** — Vendor addresses now require a non-null `country` when creating or updating a vendor."
- September 15, 2026 — "**Awaiting-payment bill status** ... `AWAITING_PAYMENT`"; "**New approvals endpoints (beta)** — List approval data for transactions, reimbursements, and bills, or add an approver"; "**Bill payment sync readiness** — `is_payment_sync_ready`".
- September 2, 2026 — "**Vendor wire beneficiary addresses** — Set the optional `wire_details.beneficiary_address` field when adding wire details".
- August 27, 2026 — "**Vendor name updates** — Set the optional `name` field when updating a vendor".
- August 21, 2026 — "**New vendor contact creation endpoint** ... Responses indicate whether creation is `approved` or `pending_approval`."
- August 19, 2026 — "**Purchase order bill filtering** — `purchase_order_id` query parameter"; "**Vendor detail requestors** — `requesting_user_id`".
- August 10, 2026 — "**Submit draft bills (beta)** — `POST /developer/v1/bills/drafts/{draft_bill_id}/submit`"; "**Bank debit trace IDs for bill payments** — `bank_debit_trace_id`".
- July 30, 2026 — "**Default entity support for vendors**"; "**Vendor approval status** — Vendor details now include `approval_status` with an `APPROVED`, `PENDING`, or `REJECTED` value."; "**Request vendor payment and tax details** — `request_payment_details` or `request_tax_details`".
- July 27, 2026 — "**Parent vendor association** — `parent_vendor_id`".
- July 21, 2026 — "**New crypto wallet bill payment method** — `CRYPTO_WALLET_TRANSFER`"; "**Attach crypto wallets to vendors** — ... `crypto_details` ... The `payment_method_type` field now includes a `crypto` value."; "**Vendor bank account last four digits** — Vendor bank account responses now include `account_number_last_four`".
- July 16, 2026 — "**Fetch bill remittance receipts** — `GET /developer/v1/bills/{bill_id}/remittance-receipt`".
- July 15, 2026 — "**Bill hold and release endpoints (beta)**" and "**Vendor hold and release endpoints (beta)**" (quoted in full above); "**New FedNow payment method** — `FED_NOW`"; "**New bill pay audit log event** — `Bill pay marked as unpaid`".
- July 2, 2026 — "**Vendor memo support for draft bills**"; "**Target payment arrival date for bills** — `target_payment_arrival_date` ... with `use_default_payment_method=true`".
- June 29, 2026 — "**New `bank_accounts:write` OAuth scope** ... for manual ACH account creation workflows" (this is for the business's own bank accounts, not vendor accounts).
- June 25, 2026 — "**Vendor bank account update response**" (quoted in full above); audit log events including `New vendor approval policy updated`, `Vendor edit approval policy updated`.
- June 22, 2026 — "**Multiple vendor addresses** — Vendor responses now include an `addresses` array".
- June 11, 2026 — "**New `fund_id` field on bill card payment details**".
- June 8, 2026 — "**Retainage withholding on bills** — ... bills include an `is_withholding_release` field".
- April 30, 2026 — audit log `event_type` values added: `Vendor hold released`, `Vendor placed on hold` (among others).
- March 25, 2026 — "**New bill payment status `ON_HOLD`** — Added `ON_HOLD` enum value to the bill `payment_status` filter and response, representing bills that have been placed on hold."
- January 17, 2025 (for context) — "**Bills and Vendors API Launch** ... All endpoints are accessible in both Sandbox and Production environments".

### Inferences
- Hold functionality predates the API endpoints: bills could be `ON_HOLD` (March 2026) and vendor holds were audited (April 2026) before the July 15, 2026 endpoints exposed them, so the sandbox UI should already show holds placed via API.
- The vendor-account surface changed on Nov 4, 2026 (`status`, `proposal_id`, `include_pending`); any cached schema older than that day is stale, so regenerate from the live spec on Nov 6.

### Gaps
- The changelog does not state when (or whether) the hold/release endpoints will leave beta, nor any deprecation of the `payment_status=ON_HOLD` wording that conflicts with the current spec.
