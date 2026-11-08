// What the page renders. Both detector sources are mapped onto these types on
// the server, so the browser never sees a raw upstream payload.

import type { EvidenceBundle } from "./types";

export interface Row {
  session_id: string;
  /** Raw decision string, or null when the call has none yet. */
  decision: string | null;
  who: string | null;
  at: string;
  /** Only the events source reports an overall risk. */
  risk: number | null;
  locked?: boolean;
  ended?: boolean;
}

export interface ListView {
  sessions: Row[];
}

export type Tone = "good" | "bad" | "mid" | "none";

export interface Layer {
  key: "deepfake" | "voiceprint" | "text" | "agent";
  label: string;
  /** Null when the detector sent no reading for this layer. */
  value: number | null;
  /** Bar length, 0..1. */
  bar: number;
  tone: Tone;
  verdict: string;
  detail: string | null;
  /** Voiceprint only: where the pass line sits on the bar. */
  threshold?: number;
  chips?: string[];
}

export interface DecisionRow {
  at: string;
  action: string;
  tier: number | null;
  decision: string;
  reasons: string[];
  required_factors: string[];
  ttl_s: number | null;
}

export interface ChallengeView {
  kind: string;
  status: string;
  phrase_match: boolean | null;
  speaker_score: number | null;
  reason: string | null;
}

export interface OtpView {
  sent: boolean;
  last_four: string | null;
  attempts: number | null;
  verified: boolean | null;
}

export interface CallbackView {
  result: string;
  at: string | null;
}

export interface V1View {
  kind: "v1";
  session_id: string;
  decision: string | null;
  reason: string | null;
  more_reasons: string[];
  last_action: string | null;
  claimed_user_id: string | null;
  channel: string | null;
  scenario: string | null;
  created_at: string | null;
  updated_at: string | null;
  locked: boolean;
  ended: boolean;
  layers: Layer[];
  decisions: DecisionRow[];
  challenges: ChallengeView[];
  otp: OtpView | null;
  callbacks: CallbackView[];
}

export interface EventsView {
  kind: "events";
  bundle: EvidenceBundle;
}

export type CallView = V1View | EventsView;
