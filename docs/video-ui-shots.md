# Video: what we can put on screen

Every UI surface we already have, what it actually renders, and which beat it
belongs to. Checked against the code on 2026-11-09. The cut itself is in the
Plan tab in Notion and in `video-plan.md`; this is only the visual
inventory.

Three callers all open with "Hi, this is Eric Catalano". Only the third gets
through. The screens below are what makes that legible.

---

## 1. The review dashboard

`web/`, Next.js, deployed to Vercel. It polls the detector, so it updates
while a call is running, which means it can be filmed live rather than cut in
afterwards.

| Element | What it reads | Where |
| --- | --- | --- |
| Call list | "Calls", sessions appear on their own | `SessionRail.tsx` |
| Empty state | "No calls yet. The next call will appear here on its own." | `Dashboard.tsx` |
| Connecting state | "Connecting to the detector..." | `Dashboard.tsx` |
| Session header | "Session", "Caller claims", "Channel" | `DetailV1.tsx` |
| Decision panel | "Decisions", "Last action", "NO DECISION" before one lands | `DetailV1.tsx`, `States.tsx` |
| Signal panel | "Layers" | `DetailV1.tsx` |
| Step-up panel | "Step-ups", "Callback on file", "SMS code" | `DetailV1.tsx` |

Best shots:

- The empty state at the top of the video. "No calls yet. The next call will
  appear here on its own." is a good cold open under a ringing phone.
- The Layers panel the moment the clone is refused. Two rows carry the whole
  argument: voice match passes, synthetic speech fails.
- The Step-ups panel filling in on the real call. Three rows go from empty to
  satisfied: voice challenge, SMS code, callback on file.
- "NO DECISION" to a decision, if the poll catches the transition.

Two badges to know about:

- **MOCK DETECTOR** renders whenever the detector is in mock mode, which is
  how every demo call runs. Leave it in frame. It is doing the same honest
  work as the 2:10 line, for free, and cropping it would be hiding something
  the repo already admits.
- **SAMPLE DATA** renders when `DASHBOARD_FIXTURES=1`. That one must be off
  before filming, or the dashboard tells the viewer the numbers are fake.
  Check `web/README.md` for the environment it needs: `DETECTOR_URL`, a
  `DETECTOR_TOKEN` set to the detector review token, and `DASHBOARD_PASSWORD`.

## 2. The showcase page

`showcase/dist/showcase.html`, static, no server needed. Five sections:

1. "Four seconds of a phone call."
2. "Where these numbers come from"
3. "Why a clone is only challenged"
4. "Why a recording is denied at 0.17"
5. "What it cannot do"

This is the honest-detection screen. It runs on real audio, so it is the
right place to make the scoring claim, and section 5 is the limit stated in
our own words. Sections 1 and 3 belong to the "how the model works" beat.

## 3. The Ramp sandbox web app

The only screen that proves anything moved. Acme Supplies is the vendor.

- Default account ending **4417** before, ending **6789** after the real call.
- Vendor **hold** after a denied call, with the two open bills at `ON_HOLD` if
  the hold is pending approval instead of applied.

Show the hold once, after the human impostor, rather than after both denials.
Twice is repetitive and costs fifteen seconds we do not have.

## 4. The worker log

Plain terminal, and the most convincing screen for a technical judge because
nothing about it can be staged.

- The `authorize` line carries the decision and the reasons. The first reason
  is a contract code: `synthetic_speech` for the clone, `speaker_mismatch` for
  the impostor, `automated_caller` for the attacker agent.
- Score lines follow, numbers only, such as `spoof_high 0.92`.
- Text layer flags read like `text_flags: urgency, verification_refusal`,
  which is the screen to use when we talk about the impostor pushing back on
  the callback.
- At startup it logs `demo scenario for the detector:` and the name. Keep this
  one out of frame, since it shows the scenario was set by hand.

## 5. The phone itself

Film it directly, not through screen capture.

- The real SMS arriving with the six digit code.
- The call dropping, then ringing again from the Telnyx number. This is the
  single most persuasive shot in the video and it needs no UI at all.

## 6. The live risk stream, unconfirmed

The detector publishes a `RiskUpdate` on the `RISK_TOPIC` data topic about
every two seconds, and the agent receives it (`session_wiring.py:296` to
`:317`). A live risk number climbing during a call would be the best visual
in the whole video.

I could not confirm anything renders it today. The dashboard gets its data by
polling the detector API instead. Worth ten minutes to check before filming,
because if something does render it, it replaces two of the cuts above.

---

## Capture notes

- No speakerphone on any call. The agent speaks through Cartesia, one of the
  held out deepfake tools, so speaker echo can put synthetic audio on the
  caller track and flag a real caller as a fake. Headphones, and record the
  agent side separately.
- Film the dashboard at a window width that keeps the Layers panel and the
  Step-ups panel both visible, so the cuts do not need a zoom.
- The dashboard is the only surface that needs the detector reachable. The
  showcase page is a single static file and will survive a dead network.
