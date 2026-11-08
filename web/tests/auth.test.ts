import { afterEach, describe, expect, it } from "vitest";
import { NextRequest } from "next/server";
import { middleware } from "@/middleware";
import { safeEqual } from "@/lib/auth";

const KEYS = ["DASHBOARD_USER", "DASHBOARD_PASSWORD", "DETECTOR_URL", "DETECTOR_TOKEN", "DASHBOARD_FIXTURES"];
const saved = Object.fromEntries(KEYS.map((k) => [k, process.env[k]]));
afterEach(() => {
  for (const k of KEYS) saved[k] === undefined ? delete process.env[k] : (process.env[k] = saved[k]);
});

function setEnv(e: Record<string, string>) {
  for (const k of KEYS) delete process.env[k];
  Object.assign(process.env, e);
}
const basic = (u: string, p: string) => ({ authorization: "Basic " + Buffer.from(`${u}:${p}`).toString("base64") });
const req = (path: string, headers: Record<string, string> = {}) =>
  new NextRequest("http://dash.test" + path, { headers });
const authed = { DASHBOARD_USER: "demo", DASHBOARD_PASSWORD: "hunter2", DETECTOR_URL: "http://d", DETECTOR_TOKEN: "t" };

describe("middleware basic auth", () => {
  it("401 with a challenge, on a page and on /api", async () => {
    setEnv(authed);
    for (const p of ["/", "/api/sessions", "/api/sessions/ses_1"]) {
      const res = await middleware(req(p));
      expect(res.status).toBe(401);
      expect(res.headers.get("www-authenticate")).toMatch(/^Basic/);
    }
  });
  it("401 on wrong user or password or a malformed header", async () => {
    setEnv(authed);
    for (const h of [basic("demo", "nope"), basic("other", "hunter2"), { authorization: "Basic !!!" }, { authorization: "Bearer x" }])
      expect((await middleware(req("/api/sessions", h))).status).toBe(401);
  });
  it("passes through with the right credentials", async () => {
    setEnv(authed);
    for (const p of ["/", "/api/sessions"]) {
      const res = await middleware(req(p, basic("demo", "hunter2")));
      expect(res.status).toBe(200);
      expect(res.headers.get("x-middleware-next")).toBe("1");
    }
  });
  it("a password with a colon works", async () => {
    setEnv({ ...authed, DASHBOARD_PASSWORD: "a:b" });
    expect((await middleware(req("/", basic("demo", "a:b")))).status).toBe(200);
  });
  it("503 when a detector is configured and no password is set", async () => {
    setEnv({ DETECTOR_URL: "http://d", DETECTOR_TOKEN: "t" });
    const res = await middleware(req("/api/sessions", basic("demo", "")));
    expect(res.status).toBe(503);
    expect(await res.text()).toMatch(/DASHBOARD_PASSWORD/);
    // and the same when fixtures are forced but a detector URL still exists
    setEnv({ DETECTOR_URL: "http://d", DASHBOARD_FIXTURES: "1" });
    expect((await middleware(req("/"))).status).toBe(503);
  });
  it("503 on an empty password, even with a matching empty header", async () => {
    setEnv({ DETECTOR_URL: "http://d", DASHBOARD_PASSWORD: "" });
    expect((await middleware(req("/", basic("", "")))).status).toBe(503);
  });
  it("fixture-only mode serves without auth", async () => {
    setEnv({ DASHBOARD_FIXTURES: "1" });
    expect((await middleware(req("/api/sessions"))).status).toBe(200);
    // Fixtures are only implied by a missing URL, not forced: stay closed.
    setEnv({});
    expect((await middleware(req("/"))).status).toBe(503);
  });
});

describe("safeEqual", () => {
  it("compares strings of any length", async () => {
    expect(await safeEqual("abc", "abc")).toBe(true);
    expect(await safeEqual("abc", "abd")).toBe(false);
    expect(await safeEqual("abc", "abcd")).toBe(false);
    expect(await safeEqual("", "")).toBe(true);
  });
});
