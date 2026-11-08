import type { ApiError } from "@/lib/types";
import type { Row } from "@/lib/view";
import { DECISION_GLYPH, decisionKind, decisionLabel, pct, time } from "@/lib/format";

interface Props {
  sessions: Row[];
  selected: string | null;
  following: boolean;
  loading: boolean;
  error: ApiError["error"] | null;
  onPick: (id: string) => void;
}

export function SessionRail({ sessions, selected, following, loading, error, onPick }: Props) {
  return (
    <nav className="rail" aria-label="Calls">
      <div className="rail-head">
        <span>Calls</span>
        {sessions.length > 0 && (
          <span className={following ? "follow on" : "follow"}>
            {following ? "following latest" : "pinned"}
          </span>
        )}
      </div>
      <ul>
        {sessions.map((s, i) => {
          const k = decisionKind(s.decision);
          return (
            <li key={s.session_id}>
              <button
                className={`row d-${k} ${s.session_id === selected ? "sel" : ""}`}
                onClick={() => onPick(s.session_id)}
                aria-current={s.session_id === selected}
              >
                <span className="badge">
                  <b aria-hidden>{DECISION_GLYPH[k]}</b> {decisionLabel(s.decision)}
                </span>
                <span className="who">{s.who ?? "unknown caller"}</span>
                <span className="meta">
                  {time(s.at)}
                  {i === 0 && <em>newest</em>}
                </span>
                <span className="risk">
                  {s.risk !== null ? pct(s.risk) : s.locked ? "LOCKED" : ""}
                </span>
              </button>
            </li>
          );
        })}
      </ul>
      {sessions.length === 0 && !error && !loading && (
        <p className="rail-empty">No calls yet.</p>
      )}
      {sessions.length === 0 && error && <p className="rail-empty">Call list unavailable.</p>}
    </nav>
  );
}
