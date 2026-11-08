"use client";

import { useEffect, useMemo, useState } from "react";
import { usePolling } from "@/lib/usePolling";
import type { EvidenceBundle, SessionList } from "@/lib/types";
import { SessionRail } from "./SessionRail";
import { Detail } from "./Detail";
import { ErrorPanel, Notice } from "./States";

export function Dashboard() {
  const list = usePolling<SessionList>("/api/sessions");
  const [picked, setPicked] = useState<string | null>(null);
  const [theme, setTheme] = useState<"light" | "dark" | null>(null);

  const sessions = useMemo(
    () =>
      [...(list.data?.sessions ?? [])].sort(
        (a, b) => Date.parse(b.first_seen) - Date.parse(a.first_seen),
      ),
    [list.data],
  );
  const newest = sessions[0]?.session_id ?? null;
  const following = picked === null;
  const selected = picked ?? newest;

  const detail = usePolling<EvidenceBundle>(
    selected ? `/api/sessions/${encodeURIComponent(selected)}` : null,
  );

  useEffect(() => {
    if (theme) {
      document.documentElement.dataset.theme = theme;
      try {
        localStorage.setItem("theme", theme);
      } catch {}
    }
  }, [theme]);

  const toggleTheme = () => {
    const cur =
      document.documentElement.dataset.theme ??
      (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    setTheme(cur === "dark" ? "light" : "dark");
  };

  const listDown = list.error && !list.data;

  return (
    <div className="shell">
      <header className="top">
        <div className="brand">
          <span className="mark" aria-hidden>
            {"◐"}
          </span>
          <span className="name">Callback</span>
          <span className="sub">caller verification</span>
        </div>
        <div className="top-right">
          {list.fixtures && <span className="chip chip-warn">SAMPLE DATA</span>}
          {detail.data?.mock_mode && !list.fixtures && <span className="chip">MOCK DETECTOR</span>}
          <span className={`live ${list.error ? "live-bad" : ""}`}>
            <i /> {list.error ? "NOT LIVE" : "LIVE"}
          </span>
          <button className="btn" onClick={toggleTheme} aria-label="Toggle light and dark theme">
            Theme
          </button>
        </div>
      </header>

      <SessionRail
        sessions={sessions}
        selected={selected}
        following={following}
        loading={list.loading}
        error={list.error}
        onPick={(id) => setPicked(id === newest ? null : id)}
      />

      <main className="main">
        {list.error && list.data && (
          <Notice tone="bad">
            Lost the detector ({list.error.message}). What you see below may be out of date.
          </Notice>
        )}
        {listDown ? (
          <ErrorPanel error={list.error!} />
        ) : list.loading ? (
          <div className="state">Connecting to the detector...</div>
        ) : !selected ? (
          <div className="state">
            <h2>No calls yet</h2>
            <p>The next call will appear here on its own.</p>
          </div>
        ) : detail.error && !detail.data ? (
          <ErrorPanel error={detail.error} />
        ) : !detail.data ? (
          <div className="state">Loading call...</div>
        ) : (
          <Detail
            b={detail.data}
            stale={Boolean(detail.error)}
            following={following}
            onFollow={() => setPicked(null)}
          />
        )}
      </main>
    </div>
  );
}
