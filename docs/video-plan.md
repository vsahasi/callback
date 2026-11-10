# Video plan

A three minute cut. The call scripts are in `demo-runbook.md` and the screens
are inventoried in `video-ui-shots.md`. This file is the shot list, the order
to shoot in, and the things that can ruin a take.

One line holds the video together: three callers all open with "Hi, this is
Eric Catalano", and only the third gets through. Fake Eric at the top, real
Eric at the bottom.

The calls are 5 (the human impostor), 4 (the AI caller) and 1 (the real CFO),
by their runbook numbers. One refusal is a person and the other is a program,
which is the point. `speaker_mismatch` catches the human, because the voice
that answers the callback is not the voice that placed the call, and
`synthetic_speech` catches the machine, because a vocoder leaves marks a
larynx does not: one detector, two different attacks. Call 2 (the ElevenLabs
clone) is the swap if the AI caller will not cooperate on the day. It is the
swap and not the cut because catching a clone live over the phone is untested,
and the AI caller has the better claim to the slot anyway: nobody is in frame,
and it is the attack the first stat is about.

The detector runs in real mode for the recording. The agent posts the caller's
speech to the detector, and the decision is read off what the detector
measured (`detector/bridge.py`), not off a scenario the worker was started
with. There is nothing to set per call and nothing to restart between calls.

---

## The cut

| Time | Beat | Stat on screen |
| --- | --- | --- |
| 0:00 to 0:30 | A human, his own voice, says he is Eric and asks to change Acme's bank details. The code goes to Eric's phone, the callback goes to Eric, and the two voices do not match. Refused. | BEC: $3.05B across 24,768 complaints |
| 0:30 to 0:55 | How that decision was made. Dashboard, then the showcase page. | |
| 0:55 to 1:15 | An AI caller, same claim, same request. Refused before it reaches the code. | AI fraud: 22,364 complaints, $893M |
| 1:15 to 1:40 | Why that one fails differently, and what the hold protects. | |
| 1:40 to 2:30 | The real Eric. Challenge, code, callback, the change lands. | |
| 2:30 to 3:00 | The model never authorises. A gate does. | $20.9B lost in 2025, up 26% |

## Beat by beat

### 0:00 to 0:30, the impostor

Open on the dashboard empty state, "No calls yet. The next call will appear
here on its own.", under a ringing phone. Veer, his own voice, nothing
disguised, says the line and makes the request, pushing on urgency. Then give
the attacker everything: the agent texts the code to the number on file, which
is Eric's phone and not the one that called, and Eric reads it over so Veer
can say the digits; the agent hangs up and calls Eric back; Eric answers and
says yes. It is still refused, because the voice on the callback leg is not
the voice on the first leg. End on the agent saying it has placed a hold.

Two phones are in this beat and both get filmed: the text landing on Eric's
phone while Veer is the one on the call, and Eric's phone ringing for the
callback. The deny lands on the callback leg, so Eric hears the hold line, not
Veer. Cut the middle hard; the callback and the hold line are the thirty
seconds.

Veer says the phrase and supplies the code rather than refusing them. Refusing
ends the call earlier and for a less interesting reason, and the point of this
beat is that the impostor with the code and a yes from Eric still does not get
through.

### 0:30 to 0:55, how it works

Cut to the Layers panel and hold on two rows:

- voice match, fails
- synthetic speech, low, passes

Nothing about the audio was synthetic. A live human at a handset is exactly
what the deepfake check is looking for, and it waved him through. What failed
is the one signal with something to compare against, the voice on the other
leg of the same call, and one failed signal is enough: a low average cannot
save a call when one signal fails outright. That is the reason this is not
just a deepfake detector.

Then the showcase page for where the numbers come from, on the different
person tab and then the real CFO tab, for the red against green in two
clicks. The showcase scores a single clip against a voice on file and can only
earn a challenge from it (0.37); the call did the same thing in two legs, and
when the second leg did not match the first the challenge became a deny. That
is the same escalation the dashboard's DECISIONS panel shows, STEP-UP then
DENY. Do not cut to the clone tab here, whatever happens to the swap: it says
"Why a clone is only challenged" in its own heading, and a viewer who has just
watched a refusal does not need to be told the same attack is only challenged.
See `video-ui-shots.md`.

The dashboard's top right reads LIVE and nothing else. There is no badge to
explain because there is nothing to disclaim: the numbers on screen were
measured on the call we just watched.

### 0:55 to 1:15, the machine

`agent/scripts/attacker_agent.py`, a Cartesia voice, phones the line. Nobody
is in frame and nobody speaks. It waits for the disclosure, says it is Eric
Catalano, asks for the same change before the 5 PM payment run, stays polite
and never says it is an AI. Refused before it gets anywhere near a code.

Film the worker log on this one rather than a person: the `authorize` line
with `synthetic_speech` or `automated_caller` first, then `spoof_high` and a
number. Either code is fine; the policy checks the spoof gate before anything
else, and which name it lands on depends on whether the behaviour signal also
tipped.

### 1:15 to 1:40, why that one is different

This time the voice is not the problem. The caller never reached a callback,
so there was no second leg to compare; what failed is the synthesis check, on
the first few seconds of speech. Show the Layers panel again: synthetic speech
high, fails. Two refusals, two different rows red.

Say what that check is, in one line, because the write-up has to say it
plainly: it is signal processing, not a trained model. Jitter and shimmer that
a larynx cannot suppress, phase that a vocoder keeps too steady, a spectrum
over-smoothed towards the mean, the noise floor between words
(`detector/analysis/synthesis.py`). No corpus, no inference in the serving
path (`DETECTOR.md`, Status). It catches this caller. It would not catch
everything, which the same file says in its "Honest limits", and that is also
why the clone is the swap and not the cut.

Then Ramp: Acme on hold, account untouched. A denied money move still protects
the vendor, because holding needs no verification. Show the hold here only,
not after both refusals. Twice is repetitive and costs fifteen seconds the cut
does not have.

### 1:40 to 2:30, the real one

The most room of any beat. Do not compress the callback: hanging up and
dialling the number on file is the thing a viewer does not expect, and it is
entirely real. It is also the exact step that sank the impostor at 0:00, now
seen from the side of the person it is for.

1. The challenge phrase, repeated word for word.
2. The real text landing on his phone, then the six digits, bare, one after
   another.
3. "For your security I will hang up and call you back on the number we have
   on file." The call drops.
4. The phone rings from the Twilio number. He confirms, takes a fresh phrase.
   The detector compares this leg's voice with the first leg's, and this time
   they match.
5. "Acme Supplies now pays to the account ending 6789."

Caption the three factors as they clear: voice challenge, SMS code, callback
on file. Then cut to Ramp, where 4417 becomes 6789. Sandbox fact 3 is
`approved`, so it is the account changing and not a change queued for
approval.

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
| Business email compromise: about $3.05 billion across 24,768 complaints, second only to investment fraud | 0:00, the impostor |
| AI-related fraud: 22,364 complaints, nearly $893 million | 0:55, the machine |
| Over $30 million of that BEC total involved AI | spare, if the 0:30 beat wants one |
| All cyber-enabled crime: nearly $20.9 billion in 2025, up 26% on 2024, over a million complaints | 2:30, the close |

Pull the exact figures from the PDF before they go on screen. These were read
through the report summary, not the BEC table itself.

Do not use the deepfake growth percentages going around: 2,137% (Signicat),
1,300% (Pindrop), 442% (CrowdStrike), 1,633% (Right-Hand AI). They are
vendor-published, they conflict on what period they cover, and the same figure
gets attributed to different reports across roundups. They would cost us the
credibility the 1:15 line buys, for the same reason the unsourced 43 to 83
percent came out of `round2-architecture.md`.

## Before filming anything

1. **Confirm real mode, on a throwaway call.** The worker logs nothing about
   a scenario now, and the detector's `v1_session_created` line must show no
   `scenario`. The first `authorize` line of a call must not say `no_audio`;
   if it does, the detector is not seeing the agent's audio, the bridge is not
   deployed, and the only thing that runs is the mock fallback, which is a
   different video and has to say so on the take. Ask Eric to run his
   pre-demo list first: `/healthz` up, the bridge deployed, Twilio sending to
   a verified phone. On the dashboard the top right reads LIVE and nothing
   else; if MOCK DETECTOR appears, stop.
2. **Rehearse on `424242`.** Real Twilio texts are switched on in the deployed
   detector. Twilio caps sends per number, which surfaces as error 60203, which
   the detector turns into `sms_failed`, which the agent treats as a deny.
   Burn the cap in rehearsal and the recorded take denies a legitimate CFO on
   camera. Two of the three calls send a real text now, and both go to Eric's
   number: the impostor's code goes to the number on file, not the phone that
   called. Blank `CALLBACK_MOCK_SMS` for rehearsals, set it back to `twilio`
   for the take only.
3. **Eric's phone is in two calls.** It takes the text and the callback on the
   impostor call, and it is the caller on his own. It has to be the number on
   file and verified in Twilio, charged, and on a headset, because he is the
   voice on three of the five legs the detector compares.

## Shoot order

No scenario to set and no worker restart between calls; each call is decided
on its own audio. The order is about the Twilio cap and about who has to be
in the room.

1. Ramp before state on screen, Acme at 4417 with no hold.
2. Call 4 (the AI caller) first. It needs nobody in frame, it is denied before
   the SMS step so it sends no text, and if it misbehaves you know early
   whether the clone is coming out.
3. Reset Acme.
4. Call 5 (the impostor), then nothing else: the hold shot comes after the
   second refusal in the cut, but film it after whichever denied call is
   cleanest.
5. Reset Acme.
6. Call 1 last, with real texts on, in as few takes as possible, so the one
   shot that depends on a rate limited provider has the fewest attempts in
   front of it. The impostor take already spent one send on the same number.
7. Dashboard and showcase screens last, from the sessions the takes created.

Resetting Acme means releasing the hold and putting the default account back
to 4417: `RampClient.release_vendor` from a Python shell, the account in the
Ramp web app. Say the full vendor name on every call; the sandbox has an
`ACME` as well as `Acme Supplies`.

## Two production traps

- **No speakerphone on any call.** The agent speaks through Cartesia, and
  Cartesia is one of the held out deepfake tools, so speaker echo puts
  synthetic audio on the caller track. In mock mode that was a hypothetical.
  In real mode it is the exact cue the synthesis check scores, and it flags
  the real Eric as a fake at 1:40. Headphones, and capture the agent side
  separately rather than through a room mic.
- **All three calls need a real phone.** The browser refuses a bank change
  outright ("payment changes need a phone call"), the impostor's whole point
  is the callback, and the attacker phones in. The browser shortcut that saves
  minutes on the dry runs does not apply to any shot in this cut.

## Budget

- 50 inbound LiveKit minutes for the whole project, shared with every
  rehearsal, and this cut needs three phone calls per full take. The
  attacker's call uses both legs; it goes out through the Twilio trunk, which
  draws on LiveKit's separate SIP minutes rather than the 50.
- Twilio trial: 40 successful verifications, to verified numbers only. Two of
  the three calls send one, both to Eric.
- Rehearse the dialogue without placing calls where you can. The wording is
  all in `demo-runbook.md`.

## If it runs long

Cut in this order: the closing card, then the 1:15 explainer compressed to a
single line over the refusal, then the readback step in call 1. Keep the
callback and keep the two Layers rows, voice match red for the human and
synthetic speech red for the machine. Those two are the video.
