// Route logic, kept apart from the Next route files so it can be tested.

import { maskDeep } from "./mask";
import { getSource, SourceError, usingFixtures, type DetectorSource } from "./source";

const HEADERS = { "Cache-Control": "no-store" };

function respond(body: unknown, status: number, fixtures: boolean): Response {
  return Response.json(body, {
    status,
    headers: { ...HEADERS, "X-Dashboard-Source": fixtures ? "fixtures" : "detector" },
  });
}

/**
 * Any failure becomes a non-2xx response whose body is `{error}` and never
 * contains a decision, so the page cannot mistake an outage for an allow.
 */
async function run<T>(
  fn: (s: DetectorSource) => Promise<T>,
  source?: DetectorSource,
): Promise<Response> {
  const fixtures = usingFixtures();
  try {
    const s = source ?? getSource();
    return respond(maskDeep(await fn(s)), 200, fixtures);
  } catch (e) {
    if (e instanceof SourceError)
      return respond({ error: { kind: e.kind, message: e.message } }, e.status, fixtures);
    return respond(
      { error: { kind: "bad_response", message: "Unexpected dashboard error." } },
      500,
      fixtures,
    );
  }
}

export const handleList = (source?: DetectorSource) => run((s) => s.listSessions(), source);
export const handleGet = (id: string, source?: DetectorSource) =>
  run((s) => s.getSession(id), source);
