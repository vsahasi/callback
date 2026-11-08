// Defensive masking. The review API already redacts bank details and phone
// numbers, but the dashboard is shown on a projector, so it does not trust
// that. Any run of five or more digits is replaced, and so is any separated
// number (a phone written with spaces, dashes or brackets) of seven or more.

const MASK = "••••";

const CONTIGUOUS = /\d{5,}/g;
const SEPARATED = /\+?\(?\d[\d ()-]{5,}\d/g;
const ISO_DATE = /^\d{4}-\d{2}-\d{2}/;

export function maskDigits(text: string): string {
  const separated = text.replace(SEPARATED, (m) => {
    if (ISO_DATE.test(m)) return m;
    const digits = m.replace(/\D/g, "").length;
    return digits >= 7 ? MASK : m;
  });
  return separated.replace(CONTIGUOUS, MASK);
}

// Keys whose string values are opaque identifiers or timestamps, which
// legitimately contain digit runs and carry no caller data.
const PASSTHROUGH_KEYS = new Set(["session_id", "request_id"]);
const TIMESTAMP = /^\d{4}-\d{2}-\d{2}T[\d:.+Z-]+$/;

/** Mask every string in a JSON value, except ids and timestamps. */
export function maskDeep<T>(value: T, key = ""): T {
  if (typeof value === "string") {
    if (PASSTHROUGH_KEYS.has(key) || TIMESTAMP.test(value)) return value;
    return maskDigits(value) as T;
  }
  if (Array.isArray(value)) return value.map((v) => maskDeep(v, key)) as T;
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value)) out[k] = maskDeep(v, k);
    return out as T;
  }
  return value;
}
