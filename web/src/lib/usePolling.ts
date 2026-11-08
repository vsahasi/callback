"use client";

import { useEffect, useRef, useState } from "react";
import { POLL_MS } from "./format";
import type { ApiError } from "./types";

export interface Polled<T> {
  data: T | null;
  /** Set when the latest attempt failed. `data` may still hold older data. */
  error: ApiError["error"] | null;
  loading: boolean;
  fixtures: boolean;
  updatedAt: number | null;
}

export function usePolling<T>(url: string | null): Polled<T> {
  const [state, setState] = useState<Polled<T>>({
    data: null,
    error: null,
    loading: true,
    fixtures: false,
    updatedAt: null,
  });
  const lastUrl = useRef<string | null>(null);

  useEffect(() => {
    if (!url) return;
    let stop = false;
    let timer: ReturnType<typeof setTimeout>;
    const ctl = new AbortController();
    if (lastUrl.current !== url) {
      lastUrl.current = url;
      setState((s) => ({ ...s, data: null, error: null, loading: true }));
    }

    const tick = async () => {
      try {
        const res = await fetch(url, { cache: "no-store", signal: ctl.signal });
        const fixtures = res.headers.get("X-Dashboard-Source") === "fixtures";
        const body = await res.json().catch(() => null);
        if (stop) return;
        if (res.ok && body && !body.error) {
          setState({ data: body as T, error: null, loading: false, fixtures, updatedAt: Date.now() });
        } else {
          const error = body?.error ?? { kind: "bad_response", message: `HTTP ${res.status}` };
          setState((s) => ({ ...s, error, loading: false, fixtures }));
        }
      } catch {
        if (stop) return;
        setState((s) => ({
          ...s,
          error: { kind: "unreachable", message: "The dashboard server cannot be reached." },
          loading: false,
        }));
      }
      if (!stop) timer = setTimeout(tick, POLL_MS);
    };
    tick();
    return () => {
      stop = true;
      ctl.abort();
      clearTimeout(timer);
    };
  }, [url]);

  return state;
}
