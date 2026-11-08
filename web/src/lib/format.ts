import type { Decision } from "./types";

export const POLL_MS = 1500;

/** Maps any detector decision string onto a colour class. step_up shares challenge's. */
export function decisionKind(d: string | null | undefined): Decision | "unknown" {
  if (d === "step_up") return "challenge";
  return d === "pending" || d === "challenge" || d === "allow" || d === "deny" ? d : "unknown";
}

/** The big word. A call with no decision yet is not an allow, and says so. */
export function decisionLabel(d: string | null | undefined): string {
  if (d === "step_up") return "STEP-UP";
  if (d == null) return "NO DECISION YET";
  return DECISION_LABEL[decisionKind(d)];
}

export const DECISION_LABEL: Record<string, string> = {
  allow: "ALLOW",
  deny: "DENY",
  challenge: "CHALLENGE",
  pending: "LISTENING",
  unknown: "UNKNOWN",
};

export const DECISION_GLYPH: Record<string, string> = {
  allow: "✓",
  deny: "✕",
  challenge: "?",
  pending: "…",
  unknown: "–",
};

const ACRONYMS = /\b(otp|sms|ai|sid)\b/gi;

export function titleCase(s: string | null | undefined): string {
  return (s ?? "").replace(/_/g, " ").replace(ACRONYMS, (m) => m.toUpperCase());
}

export function time(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? ""
    : d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function pct(n: number | null | undefined): string {
  return n == null ? "n/a" : n.toFixed(2);
}
