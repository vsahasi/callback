import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { handleGet, handleList } from "@/lib/handlers";
import { getSource } from "@/lib/source";
import { layersFromSignals, mapV1Detail, mapV1List } from "@/lib/v1";
import { decisionKind, decisionLabel, titleCase } from "@/lib/format";

const fx = (f: string) => JSON.parse(readFileSync(path.join(__dirname, "../fixtures/v1", f), "utf8"));
const env = { DETECTOR_URL: "http://detector.test", DETECTOR_TOKEN: "secret-token" };

describe("v1 list mapping", () => {
  it("maps summaries and keeps null decisions null", () => {
    const l = mapV1List({
      sessions: [
        { session_id: "a", created_at: "2026-11-08T18:00:00Z", last_decision: null, claimed_user_id: null, locked: false },
        { session_id: "b", created_at: "2026-11-08T18:01:00Z", last_decision: "step_up", locked: true },
        { nope: 1 },
      ],
    });
    expect(l.sessions.map((s) => s.session_id)).toEqual(["a", "b"]);
    expect(l.sessions[0].decision).toBeNull();
    expect(decisionLabel(null)).toBe("NO DECISION YET");
    expect(decisionKind(null)).not.toBe("allow");
    expect(decisionLabel("step_up")).toBe("STEP-UP");
    expect(l.sessions[1].locked).toBe(true);
  });
  it("garbage is an empty list, not a crash", () => {
    expect(mapV1List(null).sessions).toEqual([]);
    expect(mapV1List({ sessions: "x" }).sessions).toEqual([]);
  });
});

describe("v1 detail mapping", () => {
  it("real_cfo: allow after step-up, with all step-ups", () => {
    const v = mapV1Detail(fx("sessions/sess-demo-real-cfo.json"))!;
    expect(v.decision).toBe("allow");
    expect(v.reason).toBe("voiceprint_pass 0.83≥0.62");
    expect(v.decisions.map((d) => d.decision)).toEqual(["allow", "step_up", "step_up", "allow", "allow"]);
    expect(v.otp).toMatchObject({ sent: true, last_four: "0100", attempts: 1, verified: true });
    expect(v.callbacks[0].result).toBe("confirmed");
    expect(v.challenges[0]).toMatchObject({ status: "passed", phrase_match: true, speaker_score: 0.83 });
    const [df, vp, text, agent] = v.layers;
    expect(df).toMatchObject({ key: "deepfake", value: 0.04, tone: "good" });
    expect(vp).toMatchObject({ key: "voiceprint", value: 0.83, threshold: 0.62, tone: "good", verdict: "pass" });
    expect(text.chips).toEqual([]);
    expect(agent.value).toBe(0.05);
  });
  it("cloned_voice: deny, high spoof, voiceprint alone passes", () => {
    const v = mapV1Detail(fx("sessions/sess-demo-cloned.json"))!;
    expect(v.decision).toBe("deny");
    expect(v.locked).toBe(true);
    expect(v.layers[0]).toMatchObject({ tone: "bad", verdict: "high", value: 0.92 });
    expect(v.layers[1].verdict).toBe("pass");
  });
  it("human_impostor: failed speaker challenge, flag chips", () => {
    const v = mapV1Detail(fx("sessions/sess-demo-impostor.json"))!;
    expect(v.decision).toBe("deny");
    expect(v.challenges[0]).toMatchObject({ status: "failed", phrase_match: true, reason: "speaker_mismatch" });
    expect(v.layers[1]).toMatchObject({ tone: "bad", verdict: "fail" });
    expect(v.layers[2].chips).toEqual(["urgency", "verification_refusal"]);
  });
  it("a session with no decisions yet has no decision and empty layers", () => {
    const v = mapV1Detail({
      session_id: "s", created_at: "2026-11-08T18:00:00Z", last_decision: null, last_reasons: [],
      locked: false, ended: false, decisions: [], challenges: [], otp: null, factors: [], events: [],
    })!;
    expect(v.decision).toBeNull();
    expect(v.reason).toBeNull();
    expect(v.decisions).toEqual([]);
    expect(v.otp).toBeNull();
    expect(v.layers.every((l) => l.value === null && l.verdict === "no reading")).toBe(true);
  });
  it("missing lists entirely do not crash", () => {
    const v = mapV1Detail({ session_id: "s" })!;
    expect(v.decisions).toEqual([]);
    expect(v.challenges).toEqual([]);
  });
  it("null signals give empty layers", () => {
    expect(layersFromSignals(null).every((l) => l.value === null)).toBe(true);
    const v = mapV1Detail({
      session_id: "s", last_decision: "step_up",
      decisions: [{ at: "2026-11-08T18:00:00Z", action: "list_vendors", tier: 1, decision: "step_up", signals: null }],
    })!;
    expect(v.layers.every((l) => l.tone === "none")).toBe(true);
  });
  it("partial signals: only what is present is filled", () => {
    const l = layersFromSignals({ deepfake: { spoof_prob: 0.5 }, text: { risk: null, flags: [{ type: "urgency" }] } });
    expect(l[0]).toMatchObject({ value: 0.5, tone: "mid" });
    expect(l[1].value).toBeNull();
    expect(l[2].value).toBeNull();
    expect(l[2].chips).toEqual(["urgency"]);
    expect(l[3].value).toBeNull();
  });
  it("rejects a detail payload with no session id", () => {
    expect(mapV1Detail({ last_decision: "allow" })).toBeNull();
  });
});

describe("v1 source errors never become an allow", () => {
  const bad = async (res: Response) => {
    expect(res.ok).toBe(false);
    const body = await res.json();
    expect(body.error).toBeTruthy();
    expect(JSON.stringify(body)).not.toMatch(/allow|decision/i);
  };
  it("upstream 500, unreachable, garbage and wrong shape", async () => {
    await bad(await handleList(getSource(env, async () => new Response("x", { status: 500 }))));
    await bad(await handleList(getSource(env, () => Promise.reject(new TypeError("down")))));
    await bad(await handleList(getSource(env, async () => Response.json({ nope: true }))));
    await bad(await handleGet("sess-1", getSource(env, async () => Response.json({ last_decision: "allow" }))));
  });
  it("hits the v1 paths, and the old ones with DASHBOARD_SOURCE=events", async () => {
    const urls: string[] = [];
    const f = (async (u: string) => {
      urls.push(u);
      return Response.json({ sessions: [] });
    }) as unknown as typeof fetch;
    await handleList(getSource(env, f));
    await handleList(getSource({ ...env, DASHBOARD_SOURCE: "events" }, f));
    expect(urls).toEqual([
      "http://detector.test/review/api/v1/sessions",
      "http://detector.test/review/api/sessions",
    ]);
  });
  it("fixture mode serves v1 by default", async () => {
    const res = await handleList(getSource({ DASHBOARD_FIXTURES: "1" }));
    expect(res.headers.get("X-Dashboard-Source")).toBe("fixtures");
  });
});

describe("labels", () => {
  it("OTP and SMS read in capitals", () => {
    expect(titleCase("otp")).toBe("OTP");
    expect(titleCase("sms_otp")).toBe("SMS OTP");
  });
});
