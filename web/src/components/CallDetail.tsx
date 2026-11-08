import type { CallView } from "@/lib/view";
import { Detail } from "./Detail";
import { DetailV1 } from "./DetailV1";

interface Props {
  view: CallView;
  stale: boolean;
  asOf: number | null;
  following: boolean;
  onFollow: () => void;
}

// While stale, the whole reading is dimmed and marked NOT LIVE, so an old
// ALLOW cannot pass for the current state of the call.
export function CallDetail({ view, asOf, ...rest }: Props) {
  const body =
    view.kind === "v1" ? <DetailV1 v={view} {...rest} /> : <Detail b={view.bundle} {...rest} />;
  if (!rest.stale) return body;
  return (
    <>
      <div className="stale-banner" role="alert">
        <b>NOT LIVE</b>
        <span>
          Last reading {asOf ? new Date(asOf).toLocaleTimeString() : "never"}. This decision may no longer be
          current. Do not rely on it.
        </span>
      </div>
      <div className="is-stale" aria-label="Stale reading">
        {body}
      </div>
    </>
  );
}
