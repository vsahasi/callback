# Video plan

A three minute cut. The call scripts are in `demo-runbook.md` and the screens
are inventoried in `video-ui-shots.md`. This file is the shot list, the order
to shoot in, and the things that can ruin a take.

One line holds the video together: three callers all open with "Hi, this is
Eric Catalano", and only the third gets through. Fake Eric at the top, real
Eric at the bottom.

The calls are 2 (`cloned_voice`), 5 (`human_impostor`) and 1 (`real_cfo`). The
two refusals fail for different reasons, which is the point. `synthetic_speech`
catches the clone and `speaker_mismatch` catches the human: one detector, two
different attacks. Call 4 (`ai_agent`) is the swap if the clone will not
cooperate on the day, and it needs nobody in frame.

---

## The cut

| Time | Beat | Stat on screen |
| --- | --- | --- |
| 0:00 to 0:20 | A cloned voice of Eric asks to change Acme's bank details. Refused. | AI fraud: 22,364 complaints, $893M |
| 0:20 to 0:50 | How that decision was made. Dashboard, then the showcase page. | |
| 0:50 to 1:15 | A human, his own voice, same claim, same request. Refused. | BEC: $3.05B across 24,768 complaints |
| 1:15 to 1:40 | Why that one fails differently, and what the hold protects. | |
| 1:40 to 2:30 | The real Eric. Challenge, code, callback, the change lands. | |
| 2:30 to 3:00 | The model never authorises. A gate does. | $20.9B lost in 2025, up 26% |

## Beat by beat

### 0:00 to 0:20, the clone

Open on the dashboard empty state, "No calls yet. The next call will appear
here on its own.", under a ringing phone. The clone says the line, makes the
request, and is refused. End on the agent saying it has placed a hold.

### 0:20 to 0:50, how it works

Cut to the Layers panel and hold on two rows:

- voice match, 0.78, passes
- synthetic speech, 0.92, fails

The voiceprint matched and the deepfake check did not. A low average cannot
save a call when one signal fails outright, and that is the reason this is not
just a voiceprint. Then the showcase page for where the numbers come from, on
the recording tab and not the clone tab. The showcase scores real audio and
only challenges a clone, so cutting to that tab would contradict the call we
just watched. The recording tab denies at 0.17, below the allow line, which is
the same argument. See `video-ui-shots.md`.

Say the limit here, in one line: the scoring runs on real audio, the live
acoustic path is not wired to the room yet, so this demo scripts the signal.
The MOCK DETECTOR badge is already on the dashboard, so leave it in frame. The
repo says the same thing in `round2-architecture.md`, and the showcase page
has a "What it cannot do" section. A video implying live detection next to a
repo that says the listener is not built reads as a cover up. Volunteering it
costs eight seconds and makes everything else credible.

### 0:50 to 1:15, the human

Veer, his own voice, same opening line, pushing on urgency and refusing the
callback. Refused.

### 1:15 to 1:40, why that one is different

The voiceprint fails outright, the challenge fails, and the text layer flags
the pressure: `text_flags: urgency, verification_refusal` in the worker log.
Nothing about the audio was synthetic, and it still did not get through.

Then Ramp: Acme on hold, account untouched. A denied money move still protects
the vendor, because holding needs no verification. Show the hold here only,
not after both refusals. Twice is repetitive and costs fifteen seconds the cut
does not have.

### 1:40 to 2:30, the real one

The most room of any beat. Do not compress the callback: hanging up and
dialling the number on file is the thing a viewer does not expect, and it is
entirely real.

1. The challenge phrase, repeated word for word.
2. The real text landing on his phone, then the six digits.
3. "For your security I will hang up and call you back on the number we have
   on file." The call drops.
4. The phone rings from the Telnyx number. He confirms, takes a fresh phrase.
5. "Acme Supplies now pays to the account ending 6789."

Caption the three factors as they clear: voice challenge, SMS code, callback
on file. Then cut to Ramp, where 4417 becomes 6789.

### 2:30 to 3:00, the close

A voice agent makes changing a bank account as easy as asking, which is
exactly why the model is not allowed to do it. It can request; a Python gate
decides on the detector's answer, and calls Ramp only on allow. A prompt
injection reaches the request, never the decision.

## The stats

All of them are the FBI IC3 2025 Annual Report, ic3.gov. Put the source on
screen with the number.

| Figure | Where it goes |
| --- | --- |
| AI-related fraud: 22,364 complaints, nearly $893 million | 0:00, the clone |
| Business email compromise: about $3.05 billion across 24,768 complaints, second only to investment fraud | 0:50, the human |
| Over $30 million of that BEC total involved AI | spare, if the 0:20 beat wants one |
| All cyber-enabled crime: nearly $20.9 billion in 2025, up 26% on 2024, over a million complaints | 2:30, the close |

Pull the exact figures from the PDF before they go on screen. These were read
through the report summary, not the BEC table itself.

Do not use the deepfake growth percentages going around: 2,137% (Signicat),
1,300% (Pindrop), 442% (CrowdStrike), 1,633% (Right-Hand AI). They are
vendor-published, they conflict on what period they cover, and the same figure
gets attributed to different reports across roundups. They would cost us the
credibility the 0:20 line buys, for the same reason the unsourced 43 to 83
percent came out of `round2-architecture.md`.

## Before filming anything

1. **Answer sandbox fact 3.** `add_default_bank_account` returns either
   `approved` or `pending_approval`, and nobody has run it. If it is
   `pending_approval` then Acme still shows 4417 and the 2:20 shot becomes
   "change queued for approval" rather than "the account changed". Veer has
   the Ramp credentials and the seed script, so this one is his.
2. **Rehearse on `424242`.** Real Twilio texts are switched on in the deployed
   secret. Twilio caps sends per number, which surfaces as error 60203, which
   the detector turns into `sms_failed`, which the agent treats as a deny.
   Burn the cap in rehearsal and the recorded take denies a legitimate CFO on
   camera. Blank `CALLBACK_MOCK_SMS` for rehearsals, set it back to `twilio`
   for the take only.
3. **Check the scenario line before every take.** The worker logs
   `demo scenario for the detector: <name>` when it starts. With
   `DETECTOR_SCENARIO` empty the mock plays `real_cfo`, which allows the
   clone. Film the 0:00 beat with the variable unset and the attacker succeeds
   on camera.

## Shoot order

Three calls now, so `DETECTOR_SCENARIO` has to be set and the worker restarted
between each one.

1. Ramp before state on screen, Acme at 4417 with no hold.
2. Call 2 (`cloned_voice`) first, while the Twilio cap is untouched and
   `CALLBACK_MOCK_SMS` is still blank. The clone is denied before the SMS
   step, so it needs no code.
3. Reset Acme, restart the worker on `human_impostor`.
4. Call 5, then the Ramp hold shot.
5. Reset Acme, restart the worker on `real_cfo`.
6. Call 1 last, with real texts on, in as few takes as possible, so the one
   shot that depends on a rate limited provider has the fewest attempts in
   front of it.
7. Dashboard and showcase screens last, from the sessions the takes created.

Resetting Acme means releasing the hold and putting the default account back
to 4417: `RampClient.release_vendor` from a Python shell, the account in the
Ramp web app.

## Two production traps

- **No speakerphone on any call.** The agent speaks through Cartesia, and
  Cartesia is one of the held out deepfake tools, so speaker echo can put
  synthetic audio on the caller track and flag a real caller as a fake.
  Headphones, and capture the agent side separately rather than through a room
  mic.
- **All three calls need a real phone.** The browser refuses a bank change
  outright ("payment changes need a phone call"), so the browser shortcut that
  saves minutes on calls 2, 3 and 5 does not apply to any shot in this cut.

## Budget

- 50 inbound LiveKit minutes for the whole project, shared with every
  rehearsal, and this cut needs three phone calls per full take.
- Twilio trial: 40 successful verifications, to verified numbers only. Only
  call 1 sends one.
- Rehearse the dialogue without placing calls where you can. The wording is
  all in `demo-runbook.md`.

## If it runs long

Cut in this order: the closing card, then the 1:15 explainer compressed to a
single line over the refusal, then the readback step in call 1. Keep the
callback and keep the two Layers rows. Those two are the video.
