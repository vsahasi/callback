// The one place the dashboard gets data from. Everything else imports
// `getSource()`. A Supabase source can be added by implementing
// `DetectorSource` and returning it from `getSource()`.

import { readFile } from "node:fs/promises";
import path from "node:path";
import type { ErrorKind, EvidenceBundle, SessionList } from "./types";

export interface DetectorSource {
  listSessions(): Promise<SessionList>;
  getSession(id: string): Promise<EvidenceBundle>;
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

class DetectorHttpSource implements DetectorSource {
  constructor(
    private base: string,
    private token: string,
    private fetchImpl: typeof fetch = fetch,
  ) {}

  private async get<T>(pathname: string): Promise<T> {
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
    if (!res.ok)
      throw new SourceError("bad_response", `The detector answered ${res.status}.`, 502);
    try {
      return (await res.json()) as T;
    } catch {
      throw new SourceError("bad_response", "The detector sent an unreadable answer.", 502);
    }
  }

  listSessions() {
    return this.get<SessionList>("/review/api/sessions");
  }
  getSession(id: string) {
    assertSafeId(id);
    return this.get<EvidenceBundle>(`/review/api/sessions/${encodeURIComponent(id)}`);
  }
}

class FixtureSource implements DetectorSource {
  private dir = path.join(process.cwd(), "fixtures");

  async listSessions() {
    return JSON.parse(await readFile(path.join(this.dir, "sessions.json"), "utf8")) as SessionList;
  }
  async getSession(id: string) {
    assertSafeId(id);
    try {
      return JSON.parse(
        await readFile(path.join(this.dir, "sessions", `${id}.json`), "utf8"),
      ) as EvidenceBundle;
    } catch {
      throw new SourceError("not_found", "Unknown session.", 404);
    }
  }
}

export function usingFixtures(env: Record<string, string | undefined> = process.env): boolean {
  return env.DASHBOARD_FIXTURES === "1" || !env.DETECTOR_URL;
}

export function getSource(
  env: Record<string, string | undefined> = process.env,
  fetchImpl?: typeof fetch,
): DetectorSource {
  if (usingFixtures(env)) return new FixtureSource();
  if (!env.DETECTOR_TOKEN)
    throw new SourceError("unconfigured", "DETECTOR_TOKEN is not set.", 500);
  return new DetectorHttpSource(env.DETECTOR_URL!, env.DETECTOR_TOKEN, fetchImpl);
}
