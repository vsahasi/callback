import type { ApiError } from "@/lib/types";

const COPY: Record<string, string> = {
  unreachable: "The detector cannot be reached.",
  timeout: "The detector is not answering.",
  auth: "The detector refused the dashboard's token.",
  unconfigured: "The dashboard is not configured. Set DETECTOR_URL and DETECTOR_TOKEN.",
  bad_response: "The detector sent something the dashboard cannot read.",
  not_found: "That call is no longer on the detector.",
};

export function ErrorPanel({ error }: { error: ApiError["error"] }) {
  return (
    <div className="state state-error" role="alert">
      <div className="nodata">NO DECISION</div>
      <h2>{COPY[error.kind] ?? error.message}</h2>
      <p>
        This is not an allow. Nothing here can be trusted until the detector is back. The page
        keeps retrying.
      </p>
      <code>{error.kind}</code>
    </div>
  );
}

export function Notice({ tone, children }: { tone: "bad" | "warn"; children: React.ReactNode }) {
  return (
    <div className={`notice notice-${tone}`} role="status">
      {children}
    </div>
  );
}
