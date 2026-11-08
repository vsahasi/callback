import { describe, expect, it } from "vitest";
import { handleGet, handleList } from "@/lib/handlers";
import { getSource } from "@/lib/source";

const env = { DETECTOR_URL: "http://detector.test", DETECTOR_TOKEN: "secret-token" };
const source = (f: typeof fetch) => getSource(env, f);

async function expectNoDecision(res: Response) {
  expect(res.ok).toBe(false);
  const body = await res.json();
  expect(body.error).toBeTruthy();
  expect(JSON.stringify(body)).not.toMatch(/allow|decision|risk_score/i);
  expect(JSON.stringify(body)).not.toContain("secret-token");
}

describe("route handler errors never look like an allow", () => {
  it("unreachable detector -> 502 with an error body", async () => {
    const res = await handleList(source(() => Promise.reject(new TypeError("fetch failed"))));
    expect(res.status).toBe(502);
    expect((await res.clone().json()).error.kind).toBe("unreachable");
    await expectNoDecision(res);
  });
  it("timeout -> 504", async () => {
    const e = Object.assign(new Error("t"), { name: "TimeoutError" });
    const res = await handleList(source(() => Promise.reject(e)));
    expect(res.status).toBe(504);
    await expectNoDecision(res);
  });
  it("rejected token -> 502 auth, not a pass-through 401", async () => {
    const res = await handleGet("sess-1", source(async () => new Response("no", { status: 401 })));
    expect(res.status).toBe(502);
    expect((await res.clone().json()).error.kind).toBe("auth");
    await expectNoDecision(res);
  });
  it("detector 500 and garbage bodies are errors", async () => {
    await expectNoDecision(await handleList(source(async () => new Response("x", { status: 500 }))));
    await expectNoDecision(await handleList(source(async () => new Response("<html>", { status: 200 }))));
  });
  it("missing token with a detector URL is an error, not fixtures", async () => {
    const saved = { ...process.env };
    process.env.DETECTOR_URL = "http://x";
    delete process.env.DETECTOR_TOKEN;
    delete process.env.DASHBOARD_FIXTURES;
    try {
      const res = await handleList();
      expect(res.status).toBe(500);
      await expectNoDecision(res);
    } finally {
      process.env = saved;
    }
  });
  it("sends the bearer token upstream and masks digits in good data", async () => {
    let auth = "";
    const res = await handleList(
      source(async (_u, init) => {
        auth = (init?.headers as Record<string, string>).Authorization;
        return Response.json({ count: 1, sessions: [{ claimed_user_id: "4155550123" }] });
      }),
    );
    expect(auth).toBe("Bearer secret-token");
    expect(JSON.stringify(await res.json())).not.toMatch(/\d{5}/);
  });
  it("rejects path-like session ids", async () => {
    const res = await handleGet("../etc/passwd", source(async () => Response.json({})));
    expect(res.status).toBe(404);
  });
});
