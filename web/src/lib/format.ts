import type { Decision } from "./types";

export const POLL_MS = 1500;

export function decisionKind(d: string | null | undefined): Decision | "unknown" {
  return d === "pending" || d === "challenge" || d === "allow" || d === "deny" ? d : "unknown";
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

export function titleCase(s: string | null | undefined): string {
  return (s ?? "").replace(/_/g, " ");
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
