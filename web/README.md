# Callback dashboard

Live view of the detector's verdict on each call: decision, reason, next action,
every layer's score, the request timeline and the one-time-code attempts. Next.js
(App Router, TypeScript). It reads the detector's review API through server-side
route handlers, so the bearer token never reaches the browser.

## Environment

| Variable | Purpose |
| --- | --- |
| `DETECTOR_URL` | Detector base URL, for example `http://127.0.0.1:8000`. Empty serves fixtures. |
| `DETECTOR_TOKEN` | The detector's `CALLBACK_API_TOKEN`. Server-side only. |
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
DETECTOR_URL=http://127.0.0.1:8137 DETECTOR_TOKEN=kit-test npm run dev
```

Open http://localhost:3000. The page polls every 1.5 s and follows the newest
call until you click an older one. Click the newest call, or "Jump to the
newest", to follow again.

## Fixture mode

With `DASHBOARD_FIXTURES=1` or no `DETECTOR_URL`, the routes serve the JSON in
`fixtures/`, captured from the mock detector (real_cfo, cloned_voice with a
correct code, cloned_voice with wrong codes, replay). The header shows
SAMPLE DATA. Nothing else needs to run.

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
`DETECTOR_URL` and `DETECTOR_TOKEN` as environment variables (leave
`DASHBOARD_FIXTURES` unset). The detector must be reachable from Vercel's
servers. No code changes are needed. For a demo without a detector, set
`DASHBOARD_FIXTURES=1`.
