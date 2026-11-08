// Maps the detector's /review/api/v1 payloads onto the page's view types.
// Everything is read defensively: a missing list is empty, a missing signal is
// a layer with "no reading", and nothing is invented.

import type {
  CallbackView,
  ChallengeView,
  DecisionRow,
  Layer,
  ListView,
  OtpView,
  Row,
  Tone,
  V1View,
} from "./view";

type Raw = Record<string, unknown>;

const obj = (v: unknown): Raw | null => (v && typeof v === "object" && !Array.isArray(v) ? (v as Raw) : null);
const arr = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);
const str = (v: unknown): string | null => (typeof v === "string" ? v : null);
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const bool = (v: unknown): boolean | null => (typeof v === "boolean" ? v : null);
const strs = (v: unknown): string[] => arr(v).filter((x): x is string => typeof x === "string");

export function mapV1Row(raw: unknown): Row | null {
  const r = obj(raw);
  const id = r && str(r.session_id);
  if (!r || !id) return null;
  return {
    session_id: id,
    // A locked session has been denied at some point. A later tier 0 hold is
    // allowed even then, so never let it show as an allow.
    decision: r.locked === true && r.last_decision === "allow" ? "deny" : str(r.last_decision),
    who: str(r.claimed_user_id),
    at: str(r.created_at) ?? str(r.updated_at) ?? "",
    risk: null,
    locked: r.locked === true,
    ended: r.ended === true,
  };
}

export function mapV1List(raw: unknown): ListView {
  const sessions = arr(obj(raw)?.sessions)
    .map(mapV1Row)
    .filter((r): r is Row => r !== null);
  return { sessions };
}

// Bands are the settled contract's (ruling 2.10), used only to colour a bar.
function band(v: number, lowBelow: number, highAt: number): { tone: Tone; verdict: string } {
  if (v < lowBelow) return { tone: "good", verdict: "low" };
  if (v >= highAt) return { tone: "bad", verdict: "high" };
  return { tone: "mid", verdict: "mid" };
}

const clamp01 = (n: number) => Math.min(1, Math.max(0, n));

function empty(key: Layer["key"], label: string): Layer {
  return { key, label, value: null, bar: 0, tone: "none", verdict: "no reading", detail: null };
}

/** Per-layer rows from one Decision's `signals`. Null signals give empty rows. */
export function layersFromSignals(signalsRaw: unknown): Layer[] {
  const s = obj(signalsRaw) ?? {};

  // Deepfake: spoof_prob lives under signals.deepfake in the contract. A flat
  // signals.spoof_prob is accepted too.
  const deepfake = obj(s.deepfake);
  const spoof = num(deepfake?.spoof_prob) ?? num(s.spoof_prob);
  const windows = num(deepfake?.windows);
  const df: Layer =
    spoof === null
      ? empty("deepfake", "Deepfake")
      : {
          key: "deepfake",
          label: "Deepfake",
          value: spoof,
          bar: clamp01(spoof),
          ...band(spoof, 0.2, 0.6),
          detail: `Synthetic-speech probability. High is bad.${windows !== null ? ` ${windows} windows.` : ""}`,
        };

  const vp = obj(s.voiceprint);
  const score = num(vp?.score);
  const threshold = num(vp?.threshold);
  let vpl: Layer;
  if (score === null) vpl = empty("voiceprint", "Voiceprint");
  else {
    const passed = bool(vp?.passed);
    const pm = bool(vp?.phrase_match);
    vpl = {
      key: "voiceprint",
      label: "Voiceprint",
      value: score,
      bar: clamp01(score),
      tone: passed === true ? "good" : passed === false ? "bad" : "none",
      verdict: passed === true ? "pass" : passed === false ? "fail" : "no verdict",
      threshold: threshold ?? undefined,
      detail:
        [
          threshold !== null ? `Score ${score.toFixed(2)} against a threshold of ${threshold.toFixed(2)}.` : "",
          pm !== null ? `Phrase match: ${pm ? "yes" : "no"}.` : "",
        ]
          .filter(Boolean)
          .join(" ") || null,
    };
  }

  const text = obj(s.text);
  const risk = num(text?.risk) ?? num(s.text_risk);
  const flagSrc = text ? arr(text.flags) : arr(s.flags);
  const flags = flagSrc
    .map((f) => (typeof f === "string" ? f : str(obj(f)?.type)))
    .filter((f): f is string => Boolean(f));
  const tl: Layer =
    risk === null
      ? { ...empty("text", "Text risk"), chips: flags }
      : {
          key: "text",
          label: "Text risk",
          value: risk,
          bar: clamp01(risk),
          ...band(risk, 0.3, 0.7),
          detail: flags.length ? null : "No flags.",
          chips: flags,
        };

  const agent = num(s.agent_likelihood);
  const al: Layer =
    agent === null
      ? empty("agent", "AI caller")
      : {
          key: "agent",
          label: "AI caller",
          value: agent,
          bar: clamp01(agent),
          ...band(agent, 0.4, 0.75),
          detail: "Likelihood the caller is an automated agent.",
        };

  return [df, vpl, tl, al];
}

function mapDecision(raw: unknown): DecisionRow | null {
  const d = obj(raw);
  if (!d) return null;
  const decision = str(d.decision);
  if (!decision) return null;
  return {
    at: str(d.at) ?? "",
    action: str(d.action) ?? "",
    tier: num(d.tier),
    decision,
    reasons: strs(d.reasons),
    required_factors: strs(d.required_factors),
    ttl_s: num(d.ttl_s),
  };
}

export function mapV1Detail(raw: unknown): V1View | null {
  const r = obj(raw);
  const id = r && str(r.session_id);
  if (!r || !id) return null;

  const rawDecisions = arr(r.decisions);
  const decisions = rawDecisions.map(mapDecision).filter((d): d is DecisionRow => d !== null);
  // Tier 0 holds are background housekeeping that the detector always allows.
  // The headline is the newest decision that was not one, else the newest.
  const mapped = rawDecisions.map(obj);
  const latest = [...mapped].reverse().find((d) => d && d.tier !== 0) ?? mapped[mapped.length - 1] ?? null;
  const headline = latest ? mapDecision(latest) : null;

  const challenges: ChallengeView[] = arr(r.challenges)
    .map(obj)
    .filter((c): c is Raw => c !== null)
    .map((c) => ({
      kind: str(c.kind) ?? "",
      status: str(c.status) ?? "",
      phrase_match: bool(c.phrase_match),
      speaker_score: num(c.speaker_score),
      reason: str(c.reason),
    }));

  const o = obj(r.otp);
  const attempts = Array.isArray(o?.attempts) ? o.attempts.length : num(o?.attempts);
  const otp: OtpView | null = o
    ? {
        sent: o.sent === true,
        last_four: str(o.sent_to_last_four),
        attempts,
        verified: bool(o.verified),
      }
    : null;

  const callbacks: CallbackView[] = arr(r.factors)
    .map(obj)
    .filter((f): f is Raw => f !== null && f.kind === "callback")
    .map((f) => ({ result: str(f.result) ?? "", at: str(f.at) }));

  const locked = r.locked === true;
  let reasons = headline ? headline.reasons : strs(r.last_reasons);
  // On a locked call the first reason is "session_locked"; the cause is next.
  if (reasons[0] === "session_locked" && reasons.length > 1) reasons = reasons.slice(1);
  const lastDecision = headline?.decision ?? str(r.last_decision);
  return {
    kind: "v1",
    session_id: id,
    decision: locked && lastDecision === "allow" ? "deny" : lastDecision,
    reason: reasons[0] ?? null,
    more_reasons: reasons.slice(1),
    last_action: headline?.action ?? str(r.last_action),
    claimed_user_id: str(r.claimed_user_id),
    channel: str(r.channel),
    scenario: str(r.scenario),
    created_at: str(r.created_at),
    updated_at: str(r.updated_at),
    locked,
    ended: r.ended === true,
    layers: layersFromSignals(latest?.signals),
    decisions,
    challenges,
    otp,
    callbacks,
  };
}
