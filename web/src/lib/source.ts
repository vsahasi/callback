// The one place the dashboard gets data from. Everything else imports
// `getSource()`. A Supabase source can be added by implementing
// `DetectorSource` and returning it from `getSource()`.
//
// Sources: "v1" (default) reads /review/api/v1, "events" reads the older
// /review/api. DASHBOARD_SOURCE=events selects the old one.

import { readFile } from "node:fs/promises";
import path from "node:path";
import type { ErrorKind, EvidenceBundle, SessionList } from "./types";
import type { CallView, ListView } from "./view";
import { mapV1Detail, mapV1List } from "./v1";

export interface DetectorSource {
  listSessions(): Promise<ListView>;
  getSession(id: string): Promise<CallView>;
}

export class SourceError extends Error {
  constructor(
    public kind: ErrorKind,
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

const TIMEOUT_MS = 4000;
const SAFE_ID = /^[A-Za-z0-9_-]{1,128}$/;

export function assertSafeId(id: string): void {
  if (!SAFE_ID.test(id)) throw new SourceError("not_found", "Unknown session.", 404);
}

type Env = Record<string, string | undefined>;

const arr = <T,>(v: unknown): T[] => (Array.isArray(v) ? (v as T[]) : []);

/** Copies only the keys in `keys`. Anything else the detector sent is dropped. */
function pick<T extends object>(src: unknown, keys: readonly string[]): T {
  const out: Record<string, unknown> = {};
  if (src && typeof src === "object")
    for (const k of keys) if (k in src) out[k] = (src as Record<string, unknown>)[k];
  return out as T;
}

/**
 * Allowlist of exactly the evidence fields the page renders. The raw bundle can
 * carry a transcript, recording links and per-request metadata, so it is never
 * forwarded whole.
 */
export function eventsBundle(raw: EvidenceBundle): EvidenceBundle {
  const b = pick<EvidenceBundle>(raw, [
    "generated_at", "session_id", "mock_mode", "headline", "decision", "state", "next_action",
    "reason", "why", "risk_score", "confidence", "claimed_user_id",
  ]);
  b.why = arr<string>(raw.why).filter((x) => typeof x === "string");
  b.thresholds = pick(raw.thresholds, ["allow_below", "deny_at_or_above"]);
  b.factors = arr<EvidenceBundle["factors"][number]>(raw.factors).map((f) =>
    pick(f, ["name", "verdict", "score", "risk", "weight", "risk_share", "detail", "fatal", "fatal_candidate"]),
  );
  b.otp = raw.otp
    ? {
        ...pick<NonNullable<EvidenceBundle["otp"]>>(raw.otp, [
          "challenged", "code_length", "attempts_used", "attempts_remaining", "max_attempts", "resolved",
        ]),
        attempts: arr<NonNullable<EvidenceBundle["otp"]>["attempts"][number]>(raw.otp.attempts).map((a) =>
          pick(a, ["attempt", "at", "outcome", "detail"]),
        ),
      }
    : null;
  b.timeline = arr<EvidenceBundle["timeline"][number]>(raw.timeline).map((t) =>
    pick(t, [
      "at", "request_id", "endpoint", "status", "outcome", "latency_ms", "call_event", "sequence",
      "decision", "state", "risk_score", "next_action", "reason", "error_message",
    ]),
  );
  return b;
}

function eventsList(l: SessionList): ListView {
  return {
    sessions: arr<SessionList["sessions"][number]>(l?.sessions).map((s) => ({
      session_id: s.session_id,
      decision: s.decision,
      who: s.claimed_user_id,
      at: s.first_seen,
      risk: s.risk_score,
    })),
  };
}

/** Fetches JSON from the detector with the bearer token. Every failure is a SourceError. */
class Detector {
  constructor(
    private base: string,
    private token: string,
    private fetchImpl: typeof fetch = fetch,
  ) {}

  async get(pathname: string): Promise<unknown> {
    let res: Response;
    try {
      res = await this.fetchImpl(this.base.replace(/\/+$/, "") + pathname, {
        headers: { Authorization: `Bearer ${this.token}`, Accept: "application/json" },
        cache: "no-store",
        signal: AbortSignal.timeout(TIMEOUT_MS),
      });
    } catch (e) {
      const timedOut = e instanceof Error && (e.name === "TimeoutError" || e.name === "AbortError");
      throw timedOut
        ? new SourceError("timeout", "The detector did not answer in time.", 504)
        : new SourceError("unreachable", "The detector cannot be reached.", 502);
    }
    if (res.status === 401 || res.status === 403)
      throw new SourceError("auth", "The detector rejected the dashboard's token.", 502);
    if (res.status === 404) throw new SourceError("not_found", "Unknown session.", 404);
    if (!res.ok) throw new SourceError("bad_response", `The detector answered ${res.status}.`, 502);
    try {
      return await res.json();
    } catch {
      throw new SourceError("bad_response", "The detector sent an unreadable answer.", 502);
    }
  }
}

const unreadable = () => new SourceError("bad_response", "The detector sent something unreadable.", 502);

class V1HttpSource implements DetectorSource {
  constructor(private d: Detector) {}
  async listSessions() {
    const raw = await this.d.get("/review/api/v1/sessions");
    if (!raw || typeof raw !== "object" || !Array.isArray((raw as { sessions?: unknown }).sessions))
      throw unreadable();
    return mapV1List(raw);
  }
  async getSession(id: string) {
    assertSafeId(id);
    const view = mapV1Detail(await this.d.get(`/review/api/v1/sessions/${encodeURIComponent(id)}`));
    if (!view) throw unreadable();
    return view;
  }
}

class EventsHttpSource implements DetectorSource {
  constructor(private d: Detector) {}
  async listSessions() {
    return eventsList((await this.d.get("/review/api/sessions")) as SessionList);
  }
  async getSession(id: string): Promise<CallView> {
    assertSafeId(id);
    const bundle = (await this.d.get(`/review/api/sessions/${encodeURIComponent(id)}`)) as EvidenceBundle;
    if (!bundle || typeof bundle !== "object" || !bundle.session_id) throw unreadable();
    return { kind: "events", bundle: eventsBundle(bundle) };
  }
}

async function readJson(file: string): Promise<unknown> {
  return JSON.parse(await readFile(file, "utf8"));
}

class FixtureSource implements DetectorSource {
  constructor(private flavor: "v1" | "events") {}
  private dir() {
    const base = path.join(process.cwd(), "fixtures");
    return this.flavor === "v1" ? path.join(base, "v1") : base;
  }
  async listSessions() {
    const raw = await readJson(path.join(this.dir(), "sessions.json"));
    return this.flavor === "v1" ? mapV1List(raw) : eventsList(raw as SessionList);
  }
  async getSession(id: string): Promise<CallView> {
    assertSafeId(id);
    let raw: unknown;
    try {
      raw = await readJson(path.join(this.dir(), "sessions", `${id}.json`));
    } catch {
      throw new SourceError("not_found", "Unknown session.", 404);
    }
    if (this.flavor === "events") return { kind: "events", bundle: eventsBundle(raw as EvidenceBundle) };
    const view = mapV1Detail(raw);
    if (!view) throw unreadable();
    return view;
  }
}

export function usingFixtures(env: Env = process.env): boolean {
  return env.DASHBOARD_FIXTURES === "1" || !env.DETECTOR_URL;
}

export function sourceFlavor(env: Env = process.env): "v1" | "events" {
  return env.DASHBOARD_SOURCE === "events" ? "events" : "v1";
}

export function getSource(env: Env = process.env, fetchImpl?: typeof fetch): DetectorSource {
  const flavor = sourceFlavor(env);
  if (usingFixtures(env)) return new FixtureSource(flavor);
  if (!env.DETECTOR_TOKEN) throw new SourceError("unconfigured", "DETECTOR_TOKEN is not set.", 500);
  const d = new Detector(env.DETECTOR_URL!, env.DETECTOR_TOKEN, fetchImpl);
  return flavor === "v1" ? new V1HttpSource(d) : new EventsHttpSource(d);
}
