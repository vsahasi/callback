# Callback dashboard

Live view of the detector's verdict on each call: decision, reason, next action,
every layer's score, the request timeline and the one-time-code attempts. Next.js
(App Router, TypeScript). It reads the detector's review API through server-side
route handlers, so the bearer token never reaches the browser.

## Environment

| Variable | Purpose |
| --- | --- |
| `DETECTOR_URL` | Detector base URL, for example `http://127.0.0.1:8000`. Empty serves fixtures (with `DASHBOARD_FIXTURES=1` to skip auth). |
| `DETECTOR_TOKEN` | The bearer token this server presents to the detector. Set it to the detector's `CALLBACK_REVIEW_TOKEN`, not its `DETECTOR_TOKEN`: the read-only one opens `/review` and nothing else. Server-side only. |
| `DASHBOARD_SOURCE` | Empty or `v1` reads `/review/api/v1/sessions`. `events` reads the older `/review/api/sessions`. |
| `DASHBOARD_USER` | Basic-auth username for the dashboard. Empty accepts any username. |
| `DASHBOARD_PASSWORD` | Basic-auth password. Required unless `DASHBOARD_FIXTURES=1` with no `DETECTOR_URL`; otherwise every route answers 503. |
| `DASHBOARD_FIXTURES` | `1` forces recorded sample data even when `DETECTOR_URL` is set. |

Copy `.env.example` to `.env.local` and fill it in. Never commit real values.

## Run against the mock detector

From the repo root:

```sh
uv sync --extra dev
CALLBACK_API_TOKEN=kit-test uv run uvicorn detector.app:app --port 8137
uv run python examples/drive_mock.py --base-url http://127.0.0.1:8137 --token kit-test
```

Then in `web/`:

```sh
npm install
DETECTOR_URL=http://127.0.0.1:8137 DETECTOR_TOKEN=kit-test DASHBOARD_PASSWORD=local npm run dev
```

`npm run dev` binds to 127.0.0.1 only (`next dev -H 127.0.0.1`), so the page is not
exposed on your network. Open http://127.0.0.1:3000 and sign in with any username and
the password. The v1 detector needs the review endpoints `/review/api/v1/sessions` and
`/review/api/v1/sessions/{id}`. The page polls every 1.5 s and follows the newest
call until you click an older one. Click the newest call, or "Jump to the
newest", to follow again.

## Access control

The routes read the detector with a server-side token, so the dashboard is a
proxy for the review API. Every route, pages and `/api/*`, requires HTTP basic
auth against `DASHBOARD_USER` and `DASHBOARD_PASSWORD` (constant-time compare,
`src/middleware.ts`). With no password set it answers 503 and never serves open.
The one exception is `DASHBOARD_FIXTURES=1` with no `DETECTOR_URL`, where only
sample data is reachable.

## Fixture mode

With `DASHBOARD_FIXTURES=1` or no `DETECTOR_URL`, the routes serve recorded
JSON and the header shows SAMPLE DATA instead of LIVE. Nothing else needs to run.

- `fixtures/v1/` is the default. It holds real responses captured from the mock
  detector's `/review/api/v1` endpoints after the five demo calls were run through
  the agent's conformance test.
- `fixtures/sessions.json` and `fixtures/sessions/` are captured from the mock
  detector's older review API and are used with `DASHBOARD_SOURCE=events`.

## Failure behaviour

A detector that is down, slow, rejects the token or sends garbage makes the
routes answer 502 or 504 with `{"error": ...}` and no decision. The page shows
NO DECISION and NOT LIVE; it never renders an error as an allow. If a refresh
fails after a good one, the last reading stays up with a warning.

## Masking

Strings from the API pass through `src/lib/mask.ts` on the server: any run of
five or more digits, and any separated number of seven or more, becomes dots.
Session ids and timestamps are left alone.

## Data source

All data access goes through `src/lib/source.ts` (`DetectorSource`). To read from
Supabase later, implement `listSessions` and `getSession` there and return it
from `getSource()`.

## Tests

```sh
npm test
npm run typecheck
```

## Deploy to Vercel

Import the repo, set the project's Root Directory to `web`, and add
`DETECTOR_URL`, `DETECTOR_TOKEN`, `DASHBOARD_USER` and `DASHBOARD_PASSWORD` as environment variables. Also turn on Vercel Deployment Protection for the project, so the page is not public even before basic auth (leave
`DASHBOARD_FIXTURES` unset). The detector must be reachable from Vercel's
servers. No code changes are needed. For a demo without a detector, set
`DASHBOARD_FIXTURES=1`.

## Follow-ups

- Done. The detector now takes `CALLBACK_REVIEW_TOKEN`, a read-only token that
  opens `/review` and its JSON and nothing else: not `/v1`, not `/events`, not
  `/metrics`. Put that value in this project's `DETECTOR_TOKEN` variable.
  Nothing here changes, because every call this dashboard makes is already a
  `/review` one. A leak of this environment then exposes the review surface
  instead of the credential that can authorize a payment.
- With `DASHBOARD_SOURCE=events` the evidence bundle is cut down to the fields the
  page renders (`eventsBundle` in `src/lib/source.ts`) before it leaves the server.
