// HTTP basic auth for the whole dashboard. The routes read the detector with a
// server-side token, so without this anyone who can load the page could read
// every call.

type Env = Record<string, string | undefined>;

export type AuthVerdict = "ok" | "unauthorized" | "unconfigured";

const enc = new TextEncoder();

async function digest(s: string): Promise<Uint8Array> {
  return new Uint8Array(await crypto.subtle.digest("SHA-256", enc.encode(s)));
}

/** Constant-time string compare: both sides are hashed so length does not leak. */
export async function safeEqual(a: string, b: string): Promise<boolean> {
  const [x, y] = await Promise.all([digest(a), digest(b)]);
  let diff = 0;
  for (let i = 0; i < x.length; i++) diff |= x[i] ^ y[i];
  return diff === 0;
}

/** Nothing secret is reachable only when fixtures are forced and no detector is configured. */
export function fixtureOnly(env: Env): boolean {
  return env.DASHBOARD_FIXTURES === "1" && !env.DETECTOR_URL;
}

function parseBasic(header: string | null): { user: string; pass: string } | null {
  const m = header?.match(/^Basic\s+([A-Za-z0-9+/=]+)\s*$/i);
  if (!m) return null;
  let decoded: string;
  try {
    decoded = atob(m[1]);
  } catch {
    return null;
  }
  const i = decoded.indexOf(":");
  if (i < 0) return null;
  return { user: decoded.slice(0, i), pass: decoded.slice(i + 1) };
}

export async function checkAuth(header: string | null, env: Env): Promise<AuthVerdict> {
  if (fixtureOnly(env)) return "ok";
  const password = env.DASHBOARD_PASSWORD;
  if (!password) return "unconfigured";
  const given = parseBasic(header);
  if (!given) return "unauthorized";
  // Evaluate both so a wrong user costs the same as a wrong password. If
  // DASHBOARD_USER is unset, any username is accepted and the password decides.
  const userOk = env.DASHBOARD_USER ? await safeEqual(given.user, env.DASHBOARD_USER) : true;
  const passOk = await safeEqual(given.pass, password);
  return userOk && passOk ? "ok" : "unauthorized";
}
