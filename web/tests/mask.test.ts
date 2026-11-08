import { describe, expect, it } from "vitest";
import { maskDeep, maskDigits } from "@/lib/mask";

describe("maskDigits", () => {
  it("masks runs of five or more digits", () => {
    expect(maskDigits("acct 12345678 ok")).not.toMatch(/\d{5}/);
    expect(maskDigits("12345")).not.toContain("1");
  });
  it("keeps runs of four or fewer", () => {
    expect(maskDigits("attempt 3 of 3, code 4242, score 0.04")).toBe(
      "attempt 3 of 3, code 4242, score 0.04",
    );
  });
  it("masks separated phone numbers", () => {
    for (const p of ["+1 (415) 555-0123", "415-555-0123", "415 555 0123"])
      expect(maskDigits(`call ${p} now`)).not.toMatch(/\d{3}/);
  });
  it("leaves ISO dates alone", () => {
    expect(maskDigits("on 2026-11-08 at noon")).toContain("2026-11-08");
  });
});

describe("maskDeep", () => {
  it("masks nested strings but keeps ids and timestamps", () => {
    const out = maskDeep({
      session_id: "sess-1928813db15349cc",
      at: "2026-11-08T03:45:21.341875Z",
      claimed_user_id: "+14155550123",
      factors: [{ detail: "routing 021000021 seen" }],
      score: 123456,
    });
    expect(out.session_id).toBe("sess-1928813db15349cc");
    expect(out.at).toBe("2026-11-08T03:45:21.341875Z");
    expect(out.claimed_user_id).not.toMatch(/\d{5}/);
    expect(out.factors[0].detail).not.toMatch(/\d{5}/);
    expect(out.score).toBe(123456);
  });
});
