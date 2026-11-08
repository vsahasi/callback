import type { EvidenceBundle, EvidenceFactor } from "@/lib/types";
import { DECISION_GLYPH, DECISION_LABEL, decisionKind, pct, time, titleCase } from "@/lib/format";
import { Notice } from "./States";

export function Detail({
  b,
  following,
  onFollow,
}: {
  b: EvidenceBundle;
  following: boolean;
  onFollow: () => void;
}) {
  const k = decisionKind(b.decision);
  const factors = [...b.factors].sort((a, c) => c.risk_share - a.risk_share);
  const { allow_below, deny_at_or_above } = b.thresholds;
  return (
    <>
      {!following && (
        <Notice tone="warn">
          This is an older call. <button className="link" onClick={onFollow}>Jump to the newest</button>
        </Notice>
      )}
      <section className={`hero d-${k}`}>
        <div className="verdict">
          <span className="glyph" aria-hidden>{DECISION_GLYPH[k]}</span>
          <span className="word">{DECISION_LABEL[k]}</span>
        </div>
        <div className="why">
          <p className="reason">{b.reason ?? b.headline}</p>
          <p className="next">
            <span>Next action</span> <b>{titleCase(b.next_action) || "none"}</b>
            <span className="sep" /> <span>Caller claims</span> <b>{b.claimed_user_id ?? "unknown"}</b>
          </p>
        </div>
        <div className="score">
          <span className="label">Risk</span>
          <span className="big">{pct(b.risk_score)}</span>
          <div className="meter" title={`allow below ${allow_below}, deny at ${deny_at_or_above}`}>
            <div className="zone z-allow" style={{ width: `${allow_below * 100}%` }} />
            <div
              className="zone z-chal"
              style={{ left: `${allow_below * 100}%`, width: `${(deny_at_or_above - allow_below) * 100}%` }}
            />
            <div className="zone z-deny" style={{ left: `${deny_at_or_above * 100}%`, right: 0 }} />
            <div className="needle" style={{ left: `${Math.min(1, Math.max(0, b.risk_score)) * 100}%` }} />
          </div>
          <span className="scale">
            allow below {allow_below} / deny from {deny_at_or_above}
          </span>
        </div>
      </section>

      <div className="grid">
        <section className="panel factors">
          <h3>Layers <small>score: 1.00 consistent with the caller, 0.00 not</small></h3>
          {factors.length === 0 && <p className="dim">No layer has scored yet.</p>}
          {factors.map((f) => (
            <FactorRow key={f.name} f={f} />
          ))}
        </section>

        <div className="stack">
          {b.otp && b.otp.challenged && (
            <section className="panel">
              <h3>
                One-time code{" "}
                <small>
                  {b.otp.attempts_used} of {b.otp.max_attempts} attempts
                  {b.otp.code_length ? `, ${b.otp.code_length} digits` : ""}
                </small>
              </h3>
              {b.otp.attempts.length === 0 && <p className="dim">Waiting for the caller to answer.</p>}
              <ol className="otp">
                {b.otp.attempts.map((a) => (
                  <li key={a.attempt} className={a.outcome === "correct" ? "ok" : "bad"}>
                    <b>#{a.attempt}</b> <span>{titleCase(a.outcome)}</span>
                    <time>{time(a.at)}</time>
                  </li>
                ))}
              </ol>
            </section>
          )}
          <section className="panel timeline">
            <h3>Timeline <small>{b.timeline.length} requests</small></h3>
            <ol>
              {b.timeline.map((t, i) => (
                <li key={t.request_id + i} className={t.status >= 400 ? "err" : ""}>
                  <span className="seq">{t.sequence ?? "-"}</span>
                  <span className="ev">{titleCase(t.call_event ?? t.endpoint)}</span>
                  <span className={`tdec d-${decisionKind(t.decision)}`}>
                    {t.decision ? t.decision : t.status >= 400 ? `HTTP ${t.status}` : ""}
                  </span>
                  <time>{time(t.at)}</time>
                </li>
              ))}
            </ol>
          </section>
        </div>
      </div>
    </>
  );
}

function FactorRow({ f }: { f: EvidenceFactor }) {
  const v = f.verdict.toLowerCase();
  return (
    <div className={`factor v-${v}`}>
      <div className="f-top">
        <span className="f-name">{titleCase(f.name)}</span>
        <span className="f-verdict">{f.verdict}</span>
        <span className="f-weight">weight {f.weight.toFixed(2)}</span>
        <span className="f-score">{f.score.toFixed(2)}</span>
      </div>
      <div className="bar" role="meter" aria-valuenow={f.score} aria-valuemin={0} aria-valuemax={1} aria-label={`${titleCase(f.name)} score`}>
        <div className="fill" style={{ width: `${Math.min(1, Math.max(0, f.score)) * 100}%` }} />
      </div>
      {f.detail && <p className="f-detail">{f.detail}</p>}
    </div>
  );
}
