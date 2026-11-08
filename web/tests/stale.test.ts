import { describe, expect, it } from "vitest";
import { isStale, STALE_AFTER_MS } from "@/lib/format";
import { getSource } from "@/lib/source";
import { handleGet, handleList } from "@/lib/handlers";

describe("isStale", () => {
  it("is stale on an error, with no reading, or when the reading is old", () => {
    expect(isStale(1000, 1500, false)).toBe(false);
    expect(isStale(1000, 1500, true)).toBe(true);
    expect(isStale(null, 1500, false)).toBe(true);
    expect(isStale(1000, 1000 + STALE_AFTER_MS + 1, false)).toBe(true);
    expect(isStale(1000, 1000 + STALE_AFTER_MS, false)).toBe(false);
  });
});

describe("events source is allowlisted before it reaches the browser (N5)", () => {
  const env = { DETECTOR_URL: "http://d", DETECTOR_TOKEN: "t", DASHBOARD_SOURCE: "events" };
  it("drops media, transcripts, metadata and unknown fields, and masks the rest", async () => {
    const bundle = {
      session_id: "sess-1", generated_at: "2026-11-08T18:00:00Z", mock_mode: true, headline: "h",
      decision: "allow", state: "allowed", next_action: "complete_allow", reason: "ok 12345678",
      why: [], risk_score: 0.1, confidence: 0.9, claimed_user_id: "u_eric",
      thresholds: { allow_below: 0.3, deny_at_or_above: 0.55, otp_weight: 0.4 },
      factors: [{ name: "voice_match", verdict: "pass", score: 0.9, risk: 0.1, weight: 0.2, risk_share: 0.1, detail: "fine", fatal: false, fatal_candidate: false, secret: "x" }],
      otp: { challenged: true, code_length: 6, attempts_used: 1, max_attempts: 3, resolved: true, attempts: [], code: "424242" },
      timeline: [{ at: "2026-11-08T18:00:00Z", request_id: "r1", endpoint: "/events", status: 200, outcome: "accepted", latency_ms: 3,
        call_event: "x", sequence: 1, decision: "allow", metadata: { transcript: "my account is twenty one thousand" } }],
      media: { transcript_text: "routing zero two one zero zero zero zero two one", recording_uri: "s3://x" },
      human_review: { guidance: "g" },
    };
    const res = await handleGet("sess-1", getSource(env, async () => Response.json(bundle)));
    const text = await res.text();
    expect(res.status).toBe(200);
    for (const bad of ["transcript", "media", "recording_uri", "human_review", "metadata", "424242", "secret", "otp_weight"])
      expect(text).not.toContain(bad);
    expect(text).not.toMatch(/\d{5}/);
    expect(JSON.parse(text).bundle.factors[0].name).toBe("voice_match");
    expect(JSON.parse(text).bundle.timeline[0].sequence).toBe(1);
  });
  it("the list view carries only rendered fields", async () => {
    const res = await handleList(
      getSource(env, async () => Response.json({ sessions: [{ session_id: "s", decision: "deny", claimed_user_id: "u", first_seen: "2026-11-08T18:00:00Z", risk_score: 0.7, headline: "leak" }] })),
    );
    expect(await res.text()).not.toContain("leak");
  });
});
