// Mirrors the Pydantic models in detector/review.py. Only fields the page
// reads are listed; the API may carry more, and unknown fields are ignored.

export type Decision = "pending" | "challenge" | "allow" | "deny";

export interface SessionSummary {
  session_id: string;
  decision: string;
  state: string;
  headline: string;
  reason: string | null;
  risk_score: number | null;
  confidence: number | null;
  scenario: string | null;
  claimed_user_id: string | null;
  first_seen: string;
  last_seen: string;
  requests: number;
  otp_attempts: number;
  live: boolean;
}

export interface SessionList {
  generated_at: string;
  count: number;
  sessions: SessionSummary[];
}

export interface EvidenceFactor {
  name: string;
  verdict: string;
  score: number;
  risk: number;
  weight: number;
  risk_share: number;
  detail: string | null;
  fatal: boolean;
  fatal_candidate: boolean;
}

export interface EvidenceOtpAttempt {
  attempt: number;
  at: string;
  outcome: string;
  detail: string | null;
}

export interface EvidenceOtp {
  challenged: boolean;
  code_length: number | null;
  attempts_used: number;
  attempts_remaining: number | null;
  max_attempts: number;
  resolved: boolean | null;
  attempts: EvidenceOtpAttempt[];
}

export interface EvidenceTimelineEntry {
  at: string;
  request_id: string;
  endpoint: string;
  status: number;
  outcome: string;
  latency_ms: number;
  call_event: string | null;
  sequence: number | null;
  decision: string | null;
  state: string | null;
  risk_score: number | null;
  next_action: string | null;
  reason: string | null;
  error_message: string | null;
}

export interface EvidenceBundle {
  generated_at: string;
  session_id: string;
  mock_mode: boolean;
  headline: string;
  decision: string;
  state: string;
  next_action: string | null;
  reason: string | null;
  why: string[];
  risk_score: number;
  confidence: number;
  claimed_user_id: string | null;
  thresholds: { allow_below: number; deny_at_or_above: number };
  factors: EvidenceFactor[];
  otp: EvidenceOtp | null;
  timeline: EvidenceTimelineEntry[];
}

/** What a failing route returns. Never carries a decision. */
export interface ApiError {
  error: { kind: ErrorKind; message: string };
}

export type ErrorKind =
  | "unreachable"
  | "timeout"
  | "auth"
  | "not_found"
  | "bad_response"
  | "unconfigured";
