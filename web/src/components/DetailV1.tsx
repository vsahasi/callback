import type { Layer, V1View } from "@/lib/view";
import { DECISION_GLYPH, decisionKind, decisionLabel, time, titleCase } from "@/lib/format";
import { Notice } from "./States";

export function DetailV1({
  v,
  stale,
  following,
  onFollow,
}: {
  v: V1View;
  stale: boolean;
  following: boolean;
  onFollow: () => void;
}) {
  const k = decisionKind(v.decision);
  const status = v.ended ? "Ended" : "Open";
  const latest = v.decisions[v.decisions.length - 1];
  return (
    <>
      {stale && <Notice tone="warn">Refresh failed. Showing the last good reading.</Notice>}
      {!following && (
        <Notice tone="warn">
          This is an older call. <button className="link" onClick={onFollow}>Jump to the newest</button>
        </Notice>
      )}
      {v.locked && v.decision !== "deny" && (
        <Notice tone="bad">Session locked. The detector denies every further request on this call.</Notice>
      )}
      <section className={`hero d-${k}`}>
        <div className="verdict">
          <span className="glyph" aria-hidden>{DECISION_GLYPH[k]}</span>
          <span className={`word ${v.decision === null ? "word-sm" : ""}`}>{decisionLabel(v.decision)}</span>
        </div>
        <div className="why">
          <p className="reason">
            {v.reason ? titleCase(v.reason) : "The detector has not decided anything on this call yet."}
          </p>
          {v.more_reasons.length > 0 && (
            <ul className="chips">
              {v.more_reasons.map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          )}
          <p className="next">
            <span>Last action</span> <b>{titleCase(v.last_action) || "none"}</b>
            <span className="sep" /> <span>Caller claims</span> <b>{v.claimed_user_id ?? "unknown"}</b>
            {v.channel && (
              <>
                <span className="sep" /> <span>Channel</span> <b>{v.channel}</b>
              </>
            )}
          </p>
        </div>
        <div className="score">
          <span className="label">Session</span>
          <span className={`big big-sm ${v.locked ? "locked" : ""}`}>{v.locked ? "LOCKED" : status.toUpperCase()}</span>
          <span className="scale">
            {v.locked ? `${status}, locked. ` : ""}
            {v.created_at ? `Started ${time(v.created_at)}` : ""}
            {latest?.ttl_s ? `. Approval valid ${latest.ttl_s} s` : ""}
          </span>
        </div>
      </section>

      <div className="grid">
        <section className="panel factors">
          <h3>Layers <small>from the latest decision</small></h3>
          {v.decisions.length === 0 && <p className="dim">No decision yet, so no layer has a reading.</p>}
          {v.decisions.length > 0 && v.layers.map((l) => <LayerRow key={l.key} l={l} />)}
        </section>

        <div className="stack">
          <section className="panel">
            <h3>Step-ups</h3>
            <StepUps v={v} />
          </section>
          <section className="panel timeline">
            <h3>Decisions <small>{v.decisions.length}, oldest first</small></h3>
            {v.decisions.length === 0 && <p className="dim">None yet.</p>}
            <ol className="decs">
              {v.decisions.map((d, i) => (
                <li key={i}>
                  <span className="seq">{i + 1}</span>
                  <span className="ev">
                    {titleCase(d.action)} <small>tier {d.tier ?? "?"}</small>
                  </span>
                  <span className={`tdec d-${decisionKind(d.decision)}`}>{decisionLabel(d.decision)}</span>
                  <time>{time(d.at)}</time>
                  {d.required_factors.length > 0 && (
                    <p className="req">needs {d.required_factors.map(titleCase).join(", ")}</p>
                  )}
                  {d.reasons.length > 0 && <p className="rs">{d.reasons.join(" / ")}</p>}
                </li>
              ))}
            </ol>
          </section>
        </div>
      </div>
    </>
  );
}

function LayerRow({ l }: { l: Layer }) {
  return (
    <div className={`factor t-${l.tone}`}>
      <div className="f-top">
        <span className="f-name">{l.label}</span>
        <span className="f-verdict">{l.verdict}</span>
        <span className="f-score">{l.value === null ? "n/a" : l.value.toFixed(2)}</span>
      </div>
      <div
        className="bar"
        role="meter"
        aria-valuenow={l.value ?? undefined}
        aria-valuemin={0}
        aria-valuemax={1}
        aria-label={`${l.label} score`}
      >
        <div className="fill" style={{ width: `${l.bar * 100}%` }} />
        {l.threshold !== undefined && <div className="tick" style={{ left: `${l.threshold * 100}%` }} />}
      </div>
      {l.chips && l.chips.length > 0 && (
        <ul className="chips">
          {l.chips.map((c) => (
            <li key={c}>{titleCase(c)}</li>
          ))}
        </ul>
      )}
      {l.detail && <p className="f-detail">{l.detail}</p>}
    </div>
  );
}

function StepUps({ v }: { v: V1View }) {
  const any = v.challenges.length > 0 || v.otp !== null || v.callbacks.length > 0;
  if (!any) return <p className="dim">No step-up has been attempted.</p>;
  return (
    <div className="steps">
      {v.challenges.map((c, i) => (
        <div key={i} className={`step s-${c.status}`}>
          <b>Voice challenge ({c.kind})</b>
          <span className="res">{c.status}</span>
          <p>
            {c.phrase_match !== null && <>Phrase match: {c.phrase_match ? "yes" : "no"}. </>}
            {c.speaker_score !== null && <>Speaker score {c.speaker_score.toFixed(2)}. </>}
            {c.reason && <>Reason: {titleCase(c.reason)}.</>}
          </p>
        </div>
      ))}
      {v.otp && (
        <div className={`step ${v.otp.verified ? "s-passed" : v.otp.sent ? "s-pending" : "s-none"}`}>
          <b>SMS code</b>
          <span className="res">{v.otp.verified ? "verified" : v.otp.sent ? "not verified" : "not sent"}</span>
          <p>
            {v.otp.sent && v.otp.last_four && <>Sent to ending {v.otp.last_four}. </>}
            {v.otp.attempts !== null && <>Attempts: {v.otp.attempts}.</>}
          </p>
        </div>
      )}
      {v.callbacks.map((c, i) => (
        <div key={i} className={`step ${c.result === "confirmed" ? "s-passed" : "s-failed"}`}>
          <b>Callback on file</b>
          <span className="res">{titleCase(c.result)}</span>
          {c.at && <p>{time(c.at)}</p>}
        </div>
      ))}
    </div>
  );
}
