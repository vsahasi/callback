import type { CallView } from "@/lib/view";
import { Detail } from "./Detail";
import { DetailV1 } from "./DetailV1";

interface Props {
  view: CallView;
  stale: boolean;
  following: boolean;
  onFollow: () => void;
}

export function CallDetail({ view, ...rest }: Props) {
  return view.kind === "v1" ? <DetailV1 v={view} {...rest} /> : <Detail b={view.bundle} {...rest} />;
}
